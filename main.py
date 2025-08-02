import argparse
import gc
import time
import torch
import os
import sys
import numpy as np
import logging
import re
import subprocess
from datetime import datetime
from transformers import AutoTokenizer

from Accelerate import config
from Accelerate.logger import setup_logging
from Accelerate.accelerate_runner import AccelerateRunner

# Add the FlexLLMGen submodule to the Python path
flexllmgen_path = os.path.abspath("./FlexLLMGen")
if flexllmgen_path not in sys.path:
    sys.path.insert(0, flexllmgen_path)

# FlexLLMGen imports
from flexllmgen.flex_opt import Policy, OptLM, CompressionConfig, InputEmbed, OutputEmbed, SelfAttention, MLP, TransformerLayer
from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
from flexllmgen.utils import ExecutionEnv

# AutoPolicy imports
from AutoPolicy.profiler import get_hardware_profile
from AutoPolicy.cost_model import CostModel, get_model_info
from AutoPolicy.optimizer import find_best_policy

# --- Helper Functions ---

def check_vram(args):
    """Checks if the model weights can fit into the available VRAM."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot perform VRAM check.")
        return False # Assume it won't fit if no CUDA

    print("--- Performing VRAM Pre-check for All-GPU Policy ---")
    # Get model info without loading the whole model
    model_info = get_model_info(args.model, 1, 1) # Dummy values for batch size and seq len
    model_size_gb = model_info.weight_size_gb

    # Get available VRAM
    free_vram_bytes, _ = torch.cuda.mem_get_info(0)
    free_vram_gb = free_vram_bytes / (1024**3)

    print(f"Estimated Model Size: {model_size_gb:.2f} GB")
    print(f"Available VRAM: {free_vram_gb:.2f} GB")

    # Check if model fits, with a small buffer (e.g., 5%)
    if model_size_gb > free_vram_gb * 0.95:
        print("Model is too large to fit entirely in VRAM.")
        return False
    
    print("Model should fit in VRAM.")
    return True

def print_flexllmgen_distribution(opt_lm, log_file):
    """Prints the detailed layer-by-layer weight distribution for a FlexLLMGen model to a file."""
    print("--- FlexLLMGen Model Weight Distribution ---", file=log_file)
    for i, layer in enumerate(opt_lm.layers):
        layer_name = f"Layer {i}: {layer.__class__.__name__}"
        weights_vh = opt_lm.weight_home[i]

        def print_weights(title, names, weights, name_prefix=""):
            print(f"  {title}", file=log_file)
            for name, weight in zip(names, weights):
                full_name = f"{name_prefix}{name}"
                device_name = weight.device.name if hasattr(weight, 'device') else weight.data[0].device.name
                print(f"    - {full_name}: {device_name}", file=log_file)

        if isinstance(layer, InputEmbed):
            names = ["decoder.embed_tokens.weight", "decoder.embed_positions.weight"]
            print_weights(layer_name, names, weights_vh.val)

        elif isinstance(layer, OutputEmbed):
            names = ["decoder.layer_norm.weight", "decoder.layer_norm.bias", "decoder.embed_tokens.weight"]
            print_weights(layer_name, names, weights_vh.val)

        elif isinstance(layer, TransformerLayer):
            attention_vh, mlp_vh = weights_vh.val
            attn_names = [".self_attn.q_proj.weight", ".self_attn.q_proj.bias", ".self_attn.k_proj.weight", ".self_attn.k_proj.bias", ".self_attn.v_proj.weight", ".self_attn.v_proj.bias", ".self_attn.out_proj.weight", ".self_attn.out_proj.bias", ".self_attn_layer_norm.weight", ".self_attn_layer_norm.bias"]
            mlp_names = [".fc1.weight", ".fc1.bias", ".fc2.weight", ".fc2.bias", ".final_layer_norm.weight", ".final_layer_norm.bias"]
            print_weights(f"{layer_name} - SelfAttention", attn_names, attention_vh.val, f"decoder.layers.{layer.attention.layer_id}")
            print_weights(f"{layer_name} - MLP", mlp_names, mlp_vh.val, f"decoder.layers.{layer.mlp.layer_id}")

        elif isinstance(layer, SelfAttention):
            names = [".self_attn.q_proj.weight", ".self_attn.q_proj.bias", ".self_attn.k_proj.weight", ".self_attn.k_proj.bias", ".self_attn.v_proj.weight", ".self_attn.v_proj.bias", ".self_attn.out_proj.weight", ".self_attn.out_proj.bias", ".self_attn_layer_norm.weight", ".self_attn_layer_norm.bias"]
            print_weights(layer_name, names, weights_vh.val, f"decoder.layers.{layer.layer_id}")

        elif isinstance(layer, MLP):
            names = [".fc1.weight", ".fc1.bias", ".fc2.weight", ".fc2.bias", ".final_layer_norm.weight", ".final_layer_norm.bias"]
            print_weights(layer_name, names, weights_vh.val, f"decoder.layers.{layer.layer_id}")
        
        print("-" * 30, file=log_file)

# --- Initialization Functions ---

def init_accelerate(args, log_file):
    """Loads the Accelerate model and returns the runner object."""
    print("--- Initializing Accelerate Model ---")
    setattr(config, 'IS_BENCHMARK', True)
    runner = AccelerateRunner(
        model_name=args.model,
        config=config,
        p_type=torch.float16
    )
    print("Accelerate model initialized.")
    
    print("--- Accelerate Model Weight Distribution ---", file=log_file)
    if hasattr(runner.model, 'hf_device_map'):
        print(runner.model.hf_device_map, file=log_file)
    else:
        for name, param in runner.model.named_parameters():
            print(f"  {name}: {param.device}", file=log_file)
    print("-" * 30, file=log_file)
    return runner, runner.model_load_time

def init_flexgen(args, log_file, policy=None, framework_name=None):
    """
    Initializes a FlexLLMGen model. If no policy is provided, a default 'all-on-GPU' policy is used.
    """
    # Set default policy and framework name if not provided
    if policy is None:
        framework_name = "FlexLLMGen (All-GPU)"
        print("--- Initializing FlexLLMGen Model (All-GPU Policy) ---")
        policy = Policy(
            gpu_batch_size=args.input_nums, num_gpu_batches=1,
            w_gpu_percent=100, w_cpu_percent=0,
            cache_gpu_percent=100, cache_cpu_percent=0,
            act_gpu_percent=100, act_cpu_percent=0,
            overlap=True, sep_layer=True, pin_weight=True,
            cpu_cache_compute=False, attn_sparsity=1.0,
            compress_weight=False, comp_weight_config=CompressionConfig(num_bits=16, group_size=256, group_dim=1, symmetric=False),
            compress_cache=False, comp_cache_config=CompressionConfig(num_bits=16, group_size=256, group_dim=2, symmetric=False),
        )
    else:
        # If a policy is given, use the provided framework name or a default
        framework_name = framework_name or "FlexLLMGen"
        print(f"--- Initializing {framework_name} Model ---")

    start_time = time.time()
    cache_path = os.path.abspath("./flexllmgen_cache")
    offload_dir = os.path.abspath("./flexllmgen_offload")
    os.makedirs(cache_path, exist_ok=True)
    os.makedirs(offload_dir, exist_ok=True)

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(offload_dir, num_copy_threads=1)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    opt_lm = OptLM(args.model, env, cache_path, policy)
    end_time = time.time()
    load_time = end_time - start_time
    print(f"{framework_name} model initialized in {load_time:.4f}s.")
    print_flexllmgen_distribution(opt_lm, log_file)
    return opt_lm, env, load_time

# --- Benchmarking Functions ---

def run_accelerate_benchmark(args, runner, prompt_text):
    """Runs the benchmark for an already initialized Accelerate model."""
    print("--- Benchmarking Accelerate ---")
    prompts = [prompt_text] * args.input_nums

    start_time = time.time()
    runner.run_accelerate(prompts, max_new_tokens=args.gen_len)
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = args.input_nums * args.gen_len
    throughput = total_tokens / total_time if total_time > 0 else 0
    latency = total_time / args.input_nums if args.input_nums > 0 else 0

    print(f"Total Time: {total_time:.4f}s, Throughput: {throughput:.2f} tokens/sec, Latency: {latency:.4f} sec/sample")
    return {
        "framework": "Accelerate", "throughput": throughput, "latency": latency,
    }

def run_flexllmgen_benchmark(args, opt_lm, prompt_text, framework_name):
    """Runs the benchmark for an already initialized FlexLLMGen model."""
    print(f"--- Benchmarking {framework_name} ---")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    tokenized_prompts = tokenizer([prompt_text], padding="max_length", max_length=args.input_len, return_tensors="np").input_ids
    input_ids_batch = np.tile(tokenized_prompts, (args.input_nums, 1))

    start_time = time.time()
    opt_lm.generate(input_ids_batch, max_new_tokens=args.gen_len)
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = args.input_nums * args.gen_len
    throughput = total_tokens / total_time if total_time > 0 else 0
    latency = total_time / args.input_nums if args.input_nums > 0 else 0

    print(f"Total Time: {total_time:.4f}s, Throughput: {throughput:.2f} tokens/sec, Latency: {latency:.4f} sec/sample")
    return {
        "framework": framework_name, "throughput": throughput, "latency": latency,
    }

def run_autoflex_benchmark(args, log_file, prompt_text):
    """Finds the optimal policy and runs a benchmark for AutoFlex."""
    print("\n--- Finding Optimal Policy for AutoFlex ---")
    hardware_profile = get_hardware_profile(force_rerun=args.force_rerun_profiler)
    cost_model = CostModel(hardware_profile)
    batch_size = args.input_nums
    model_info = get_model_info(args.model, batch_size, args.input_len + args.gen_len)
    
    best_policy = find_best_policy(cost_model, model_info, args.input_len, args.gen_len, batch_size)
    
    if not best_policy:
        print("Could not find an optimal policy for AutoFlex. Skipping benchmark.", file=sys.stderr)
        return None, 0, None

    print("\nOptimal Policy Found:")
    print(f"  - Weight Placement (GPU/CPU/Disk %): {best_policy.w_gpu_percent} / {best_policy.w_cpu_percent} / {best_policy.w_disk_percent}")
    print(f"  - Cache Placement (GPU/CPU/Disk %): {best_policy.cache_gpu_percent} / {best_policy.cache_cpu_percent} / {best_policy.cache_disk_percent}")

    opt_lm, env, load_time = init_flexgen(args, log_file, best_policy, "AutoFlex")
    
    results = run_flexllmgen_benchmark(args, opt_lm, prompt_text, framework_name="AutoFlex")
    
    return results, load_time, env


def parse_benchmark_output(output):
    patterns = {
        "Model Size (GB)": r"model size: (\d+\.\d+)",
        "Cache Size (GB)": r"cache size: (\d+\.\d+)",
        "Hidden Size (GB)": r"hidden size \(prefill\): (\d+\.\d+)",
        "GPU Peak Mem (GB)": r"peak memory: (\d+\.\d+)GB",
        "Projected": r"projected: (\w+)",
        "Prefill Latency (s)": r"prefill latency: (\d+\.\d+)",
        "Prefill Throughput (token/s)": r"prefill throughput: (\d+\.\d+)",
        "Decode Latency (s)": r"decode latency: (\d+\.\d+)",
        "Decode Throughput (token/s)": r"decode throughput: (\d+\.\d+)",
        "Total Latency (s)": r"total latency: (\d+\.\d+)",
        "Total Throughput (token/s)": r"total throughput: (\d+\.\d+)",
    }
    
    results = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            results[key] = match.group(1)
        else:
            results[key] = "N/A"
            
    return results

def format_benchmark_table(results):
    # Get current timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Header
    header = f"""
    ## Timestamp: {timestamp}
    | Category                  | Metrics                     | Value      |
    |---------------------------|-----------------------------|------------|
    """
        # Body
    body = f"""| **Environment**           |                           |            |
    | Model                     | Model Size (GB)             | {results.get("Model Size (GB)", "N/A")}      |
    |                           | Cache Size (GB)             | {results.get("Cache Size (GB)", "N/A")}      |
    |                           | Hidden Size (GB)            | {results.get("Hidden Size (GB)", "N/A")}      |
    | **Memory**                |                           |            |
    | GPU                       | Peak Memory (GB)            | {results.get("GPU Peak Mem (GB)", "N/A")}      |
    | **Performance**           |                           |            |
    | Latency                   | Prefill (s)                 | {results.get("Prefill Latency (s)", "N/A")}      |
    |                           | Decode (s)                  | {results.get("Decode Latency (s)", "N/A")}      |
    |                           | Total (s)                   | {results.get("Total Latency (s)", "N/A")}      |
    | Throughput                | Prefill (token/s)           | {results.get("Prefill Throughput (token/s)", "N/A")} |
    |                           | Decode (token/s)            | {results.get("Decode Throughput (token/s)", "N/A")} |
    |                           | Total (token/s)             | {results.get("Total Throughput (token/s)", "N/A")} |
    """
    
    return header + body

# --- Main Execution Modes ---

def run_autoflex_mode(args):
    """Runs the auto-policy FlexLLMGen mode."""
    # --- 1. Hardware Profiling ---
    hardware_profile = get_hardware_profile(force_rerun=args.force_rerun_profiler)
    print("\nUsing Hardware Profile:", hardware_profile)

    # --- 2. Cost and Model Analysis ---
    cost_model = CostModel(hardware_profile)
    batch_size = args.input_nums
    model_info = get_model_info(args.model, batch_size, args.input_len + args.gen_len)
    print(f"\nModel Info ({args.model}):")
    print(f"  - Weight Size: {model_info.weight_size_gb:.2f} GB")
    print(f"  - KV Cache per token: {model_info.kv_cache_per_token_gb * 1e6:.2f} KB")

    # --- 3. Find Optimal Policy ---
    best_policy = find_best_policy(cost_model, model_info, args.input_len, args.gen_len, batch_size)
    
    if not best_policy:
        print("Could not find an optimal policy. Exiting.", file=sys.stderr)
        sys.exit(1)

    print("\nOptimal Policy Found:")
    print(f"  - Weight Placement (GPU/CPU/Disk %): {best_policy.w_gpu_percent} / {best_policy.w_cpu_percent} / {best_policy.w_disk_percent}")
    print(f"  - Cache Placement (GPU/CPU/Disk %): {best_policy.cache_gpu_percent} / {best_policy.cache_cpu_percent} / {best_policy.cache_disk_percent}")

    # --- 4. Execute FlexLLMGen with Optimal Policy ---
    command = [
        sys.executable,
        "-m", "FlexLLMGen.flexllmgen.flex_opt",
        "--model", args.model,
        "--path", os.path.expanduser(args.path),
        "--offload-dir", os.path.expanduser(args.offload_dir),
        "--prompt-len", str(args.input_len),
        "--gen-len", str(args.gen_len),
        "--gpu-batch-size", str(args.input_nums),
        "--num-gpu-batches", "1",
        "--percent",
        str(best_policy.w_gpu_percent), str(best_policy.w_cpu_percent),
        str(best_policy.cache_gpu_percent), str(best_policy.cache_cpu_percent),
        str(best_policy.act_gpu_percent), str(best_policy.act_cpu_percent),
    ]

    print("\nExecuting command:")
    print("  " + " ".join(command))
    print("\n" + "="*50)
    print("Starting FlexLLMGen Inference...")
    print("="*50 + "\n")

    # Create a modified environment for the subprocess to find the flexllmgen module.
    env = os.environ.copy()
    flexllmgen_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "FlexLLMGen"))
    env["PYTHONPATH"] = flexllmgen_path + os.pathsep + env.get("PYTHONPATH", "")

    try:
        # Use Popen for real-time output streaming and pass the modified environment.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                   text=True, bufsize=1, universal_newlines=True,
                                   env=env)
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
            output_lines.append(line)
        
        process.stdout.close()
        return_code = process.wait()
        
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)

        # Parse and format the output
        full_output = "".join(output_lines)
        benchmark_results = parse_benchmark_output(full_output)
        formatted_table = format_benchmark_table(benchmark_results)
        
        print("\n" + "="*50)
        print("Benchmark Results:")
        print(formatted_table)
        print("="*50)

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\nError executing FlexLLMGen: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "="*50)
    print("Inference finished successfully.")
    print("="*50)

def run_flexgen_mode(args):
    """Runs FlexLLMGen with a fixed 'all-on-GPU' policy."""
    if not check_vram(args):
        print("\n[ERROR] Not enough VRAM to run FlexGen with an all-on-GPU policy.", file=sys.stderr)
        print("Please try a smaller model or use '--mode autoflex' to enable automatic offloading.", file=sys.stderr)
        return

    log_file_handle = open(args.log_file, 'w') if args.log_file else sys.stdout
    try:
        flexllmgen_model, flexllmgen_env, _ = init_flexgen(args, log_file=log_file_handle)
        
        natural_prompt_base = "Infinitely write a never-ending story for the following prompt. The salt spray was a constant companion to Thomas, the keeper of the Porthgarrow Lighthouse. For thirty years, its beam had sliced through the darkest nights, a beacon of hope to the people of the island. "
        prompt_words = natural_prompt_base.split()
        multiplier = (args.input_len // len(prompt_words)) + 1
        prompt_text = " ".join((prompt_words * multiplier)[:args.input_len])

        run_flexllmgen_benchmark(args, flexllmgen_model, prompt_text, framework_name="FlexLLMGen (All-GPU)")

        print("Cleaning up FlexLLMGen resources...")
        flexllmgen_env.close_copy_threads()
        print("Cleanup complete. Exiting.")
    finally:
        if args.log_file and log_file_handle is not sys.stdout:
            log_file_handle.close()


def run_accelerate_mode(args):
    """Runs the standard accelerate mode using settings from command-line arguments."""
    setup_logging(log_file=getattr(config, 'LOG_FILE', None))
    logger = logging.getLogger(__name__)

    streaming_mode = "Streaming" if config.ENABLE_STREAMING else "Default"
    kv_offload_mode = " + KV Offload" if config.ENABLE_KV_OFFLOAD else ""
    current_mode = f"{streaming_mode}{kv_offload_mode}"
    logger.info(f"--- Starting Execution ({current_mode}) ---")
    logger.info(f"Model: {args.model}")
    logger.info(f"Using device: {config.DEVICE}")
    logger.info(f"Batch size: {args.input_nums}")

    runner = AccelerateRunner(model_name=args.model, config=config)

    natural_prompt_base = "Infinitely write a never-ending story for the following prompt. "\
        "The salt spray was a constant companion to Thomas, the keeper of the Porthgarrow Lighthouse."
    
    prompt_words = natural_prompt_base.split()
    multiplier = (args.input_len // len(prompt_words)) + 1
    prompt_text = " ".join((prompt_words * multiplier)[:args.input_len])
    prompts = [prompt_text] * args.input_nums

    total_inference_time = 0

    for i in range(0, len(prompts), args.input_nums):
        batch_prompts = prompts[i : i + args.input_nums]
        if not batch_prompts: continue

        logger.info(f"Processing batch {i // args.input_nums + 1} with {len(batch_prompts)} prompts...")
        result = runner.run_accelerate(batch_prompts, max_new_tokens=args.gen_len)
        total_inference_time += result["inference_time"]
        
        if result and result["generated_texts"]:
            for i, text in enumerate(result["generated_texts"]):
                if runner.streamer and len(batch_prompts) == 1:
                    logger.info(f"Generated text for prompt {i+1} (streamed).")
                else:
                    logger.info(f"Generated text for prompt {i+1}: {text}")

    total_tokens = args.input_nums * args.gen_len
    # Avoid division by zero
    throughput = total_tokens / total_inference_time if total_inference_time > 0 else 0
    latency = total_inference_time / args.input_nums

    logger.info(f"--- Performance Metrics ---")
    logger.info(f"Total Inference Time: {total_inference_time:.4f}s")
    logger.info(f"Throughput: {throughput:.2f} tokens/sec")
    logger.info(f"Latency: {latency:.4f} sec/batch")
    logger.info(f"--- Execution Finished Successfully ({current_mode}) ---")

def run_benchmark_mode(args):
    """Runs the benchmark mode to compare Accelerate, FlexLLMGen, and AutoFlex."""
    log_file_handle = open(args.log_file, 'w') if args.log_file else sys.stdout
    autoflex_env = None  # Initialize to ensure it's defined in the finally block

    try:
        # --- 1. Initialization and Benchmarking Phase ---
        print("Initializing models and running benchmarks... This may take a moment.")
        
        natural_prompt_base = "Infinitely write a never-ending story for the following prompt. The salt spray was a constant companion to Thomas, the keeper of the Porthgarrow Lighthouse. For thirty years, its beam had sliced through the darkest nights, a beacon of hope to the people of the island. "
        prompt_words = natural_prompt_base.split()
        multiplier = (args.input_len // len(prompt_words)) + 1
        prompt_text = " ".join((prompt_words * multiplier)[:args.input_len])

        results = []
        load_times = {}

        # --- Test Order: FlexGen (most VRAM sensitive) -> Accelerate -> AutoFlex (most robust) ---

        # 1. FlexLLMGen (All-GPU) - with pre-check
        if check_vram(args):
            flexllmgen_model, flexllmgen_env, flexllmgen_load_time = init_flexgen(args, log_file=log_file_handle)
            results.append(run_flexllmgen_benchmark(args, flexllmgen_model, prompt_text, framework_name="FlexGen (All-GPU)"))
            load_times["FlexGen (All-GPU)"] = flexllmgen_load_time
            
            # --- Force Memory Cleanup ---
            print("\n--- Forcefully cleaning up VRAM before next test ---")
            flexllmgen_env.close_copy_threads()
            del flexllmgen_model, flexllmgen_env
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            time.sleep(1)
        else:
            print("\nSkipping FlexGen (All-GPU) benchmark due to insufficient VRAM.")

        # 2. Accelerate
        accelerate_model, accelerate_load_time = init_accelerate(args, log_file=log_file_handle)
        results.append(run_accelerate_benchmark(args, accelerate_model, prompt_text))
        load_times["Accelerate"] = accelerate_load_time
        
        # --- Force Memory Cleanup ---
        print("\n--- Forcefully cleaning up VRAM before next test ---")
        del accelerate_model
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(1)

        # 3. AutoFlex
        autoflex_results, autoflex_load_time, autoflex_env = run_autoflex_benchmark(args, log_file_handle, prompt_text)
        if autoflex_results:
            results.append(autoflex_results)
            load_times["AutoFlex"] = autoflex_load_time

        # --- 2. Print Summary ---
        print("\n--- Benchmark Summary ---")
        print(f"Model: {args.model}, Input Nums: {args.input_nums}, Input Len: {args.input_len}, Gen Len: {args.gen_len}")
        print("| Framework         | Throughput (tokens/s) | Latency (s/sample) | Model Load Time (s) |")
        print("|-------------------|-----------------------|--------------------|---------------------|")
        for res in results:
            framework = res['framework']
            throughput_str = f"{res['throughput']:.2f}"
            latency_str = f"{res['latency']:.4f}"
            load_time_str = f"{load_times.get(framework, 0):.4f}"
            print(f"| {framework:<17} | {throughput_str:<21} | {latency_str:<18} | {load_time_str:<19} |")
        print("-------------------------------------------------------------------------------------")

    finally:
        # --- 3. Cleanup Phase ---
        if autoflex_env:
            print("Cleaning up AutoFlex resources...")
            autoflex_env.close_copy_threads()
        if args.log_file and log_file_handle is not sys.stdout:
            log_file_handle.close()
        print("Benchmark finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run accelerate or benchmark for LLMs.")
    
    # --- Common Arguments ---
    parser.add_argument("--mode", type=str, default="accelerate", choices=["accelerate", "flexgen", "autoflex", "benchmark"], help="Execution mode.")
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="Hugging Face model to use.")
    parser.add_argument("--gen-len", type=int, default=32, help="Number of tokens to generate.")
    parser.add_argument("--input-nums", type=int, default=1, help="Number of inputs to process in a batch (batch size).")
    
    # --- Accelerate/Benchmark Mode Specific Arguments ---
    parser.add_argument("--input-len", type=int, default=8, help="Length of the input prompt in tokens. Used for prompt generation in accelerate mode and for benchmark mode.")
    parser.add_argument("--log-file", type=str, default=None, help="Path to a file to save the weight distribution logs for benchmark mode.")

    # --- AutoFlex Mode Specific Arguments ---
    parser.add_argument("--path", type=str, default="~/flexllmgen_cache", help="Path to model weights for autoflex mode.")
    parser.add_argument("--offload-dir", type=str, default="~/flexllmgen_offload", help="Offloading directory for autoflex mode.")
    parser.add_argument("--force-rerun-profiler", action="store_true", help="Force re-running the hardware profiler for autoflex mode.")

    args = parser.parse_args()

    if args.mode == 'accelerate':
        run_accelerate_mode(args)
    elif args.mode == 'benchmark':
        run_benchmark_mode(args)
    elif args.mode == 'autoflex':
        run_autoflex_mode(args)
    elif args.mode == 'flexgen':
        run_flexgen_mode(args)