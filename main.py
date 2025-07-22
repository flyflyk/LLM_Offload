import argparse
import time
import torch
import os
import sys
import numpy as np
import logging
from transformers import AutoTokenizer

from Accelerate import config
from Accelerate.logger import setup_logging
from Accelerate.accelerate_runner import InferenceRunner

# Add the FlexLLMGen submodule to the Python path
flexllmgen_path = os.path.abspath("./FlexLLMGen")
if flexllmgen_path not in sys.path:
    sys.path.insert(0, flexllmgen_path)

# FlexLLMGen imports
from flexllmgen.flex_opt import Policy, OptLM, CompressionConfig, InputEmbed, OutputEmbed, SelfAttention, MLP, TransformerLayer
from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
from flexllmgen.utils import ExecutionEnv

# --- Helper Functions ---

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

def initialize_accelerate(args, log_file):
    """Loads the Accelerate model and returns the runner object."""
    print("--- Initializing Accelerate Model ---")
    setattr(config, 'IS_BENCHMARK', True)
    runner = InferenceRunner(
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
    return runner

def initialize_flexllmgen(args, log_file):
    """Loads the FlexLLMGen model and returns the model and environment objects."""
    print("--- Initializing FlexLLMGen Model ---")
    cache_path = os.path.abspath("./flexllmgen_cache")
    offload_dir = os.path.abspath("./flexllmgen_offload")
    os.makedirs(cache_path, exist_ok=True)
    os.makedirs(offload_dir, exist_ok=True)

    flex_args = argparse.Namespace(
        model=args.model, path=cache_path, prompt_len=args.input_len,
        gen_len=args.gen_len, gpu_batch_size=args.input_nums,
        percent=[100, 0, 100, 0, 100, 0], pin_weight=True,
        cpu_cache_compute=False, attn_sparsity=1.0,
        compress_weight=False, compress_cache=False,
    )

    gpu = TorchDevice("cuda:0")
    cpu = TorchDevice("cpu")
    disk = TorchDisk(offload_dir)
    env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

    weight_comp_config = CompressionConfig(num_bits=16, group_size=256, group_dim=1, symmetric=False)
    cache_comp_config = CompressionConfig(num_bits=16, group_size=256, group_dim=2, symmetric=False)

    policy = Policy(
        gpu_batch_size=flex_args.gpu_batch_size, num_gpu_batches=1,
        w_gpu_percent=flex_args.percent[0], w_cpu_percent=flex_args.percent[1],
        cache_gpu_percent=flex_args.percent[2], cache_cpu_percent=flex_args.percent[3],
        act_gpu_percent=flex_args.percent[4], act_cpu_percent=flex_args.percent[5],
        overlap=True, sep_layer=True, pin_weight=flex_args.pin_weight,
        cpu_cache_compute=flex_args.cpu_cache_compute, attn_sparsity=flex_args.attn_sparsity,
        compress_weight=flex_args.compress_weight, comp_weight_config=weight_comp_config,
        compress_cache=flex_args.compress_cache, comp_cache_config=cache_comp_config,
    )
    
    opt_lm = OptLM(flex_args.model, env, flex_args.path, policy)
    print("FlexLLMGen model initialized.")
    print_flexllmgen_distribution(opt_lm, log_file)
    return opt_lm, env

# --- Benchmarking Functions ---

def run_accelerate_benchmark(args, runner, prompt_text):
    """Runs the benchmark for an already initialized Accelerate model."""
    print("--- Benchmarking Accelerate ---")
    prompts = [prompt_text] * args.input_nums

    start_time = time.time()
    runner.run_inference(prompts, max_new_tokens=args.gen_len)
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = args.input_nums * args.gen_len
    throughput = total_tokens / total_time
    latency = total_time / args.input_nums

    print(f"Total Time: {total_time:.4f}s, Throughput: {throughput:.2f} tokens/sec, Latency: {latency:.4f} sec/sample")
    return {
        "framework": "Accelerate", "throughput": throughput, "latency": latency,
    }

def run_flexllmgen_benchmark(args, opt_lm, prompt_text):
    """Runs the benchmark for an already initialized FlexLLMGen model."""
    print("--- Benchmarking FlexLLMGen ---")
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-30b", padding_side="left")
    tokenized_prompts = tokenizer([prompt_text], padding="max_length", max_length=args.input_len, return_tensors="np").input_ids
    input_ids_batch = np.tile(tokenized_prompts, (args.input_nums, 1))

    start_time = time.time()
    opt_lm.generate(input_ids_batch, max_new_tokens=args.gen_len)
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = args.input_nums * args.gen_len
    throughput = total_tokens / total_time
    latency = total_time / args.input_nums

    print(f"Total Time: {total_time:.4f}s, Throughput: {throughput:.2f} tokens/sec, Latency: {latency:.4f} sec/sample")
    return {
        "framework": "FlexLLMGen", "throughput": throughput, "latency": latency,
    }

# --- Main Execution Modes ---

def run_accelerate_mode(args):
    """Runs the standard inference mode using settings from command-line arguments."""
    setup_logging(log_file=getattr(config, 'LOG_FILE', None))
    logger = logging.getLogger(__name__)

    streaming_mode = "Streaming" if config.ENABLE_STREAMING else "Default"
    kv_offload_mode = " + KV Offload" if config.ENABLE_KV_OFFLOAD else ""
    current_mode = f"{streaming_mode}{kv_offload_mode}"
    logger.info(f"--- Starting Execution ({current_mode}) ---")
    logger.info(f"Model: {args.model}")
    logger.info(f"Using device: {config.DEVICE}")
    logger.info(f"Batch size: {args.input_nums}")

    runner = InferenceRunner(model_name=args.model, config=config)

    natural_prompt_base = "Infinitely write a never-ending story for the following prompt. The salt spray was a constant companion to Thomas, the keeper of the Porthgarrow Lighthouse. For thirty years, its beam had sliced through the darkest nights, a beacon of hope to the people of the island. "
    prompt_words = natural_prompt_base.split()
    multiplier = (args.input_len // len(prompt_words)) + 1
    prompt_text = " ".join((prompt_words * multiplier)[:args.input_len])
    prompts = [prompt_text] * args.input_nums

    for i in range(0, len(prompts), args.input_nums):
        batch_prompts = prompts[i : i + args.input_nums]
        if not batch_prompts: continue

        logger.info(f"Processing batch {i // args.input_nums + 1} with {len(batch_prompts)} prompts...")
        generated_texts = runner.run_accelerate(batch_prompts, max_new_tokens=args.gen_len)
        if generated_texts:
            for i, text in enumerate(generated_texts):
                logger.info(f"Generated text for prompt {i+1}: {text}")

    logger.info(f"--- Execution Finished Successfully ({current_mode}) ---")

def run_benchmark_mode(args):
    """Runs the benchmark mode to compare Accelerate and FlexLLMGen."""
    log_file_handle = open(args.log_file, 'w') if args.log_file else sys.stdout

    try:
        # --- 1. Initialization Phase ---
        print("Initializing models... This may take a moment.")
        accelerate_model = initialize_accelerate(args, log_file=log_file_handle)
        flexllmgen_model, flexllmgen_env = initialize_flexllmgen(args, log_file=log_file_handle)
        print("All models initialized. Starting benchmarks.")

        # --- 2. Benchmarking Phase ---
        natural_prompt_base = "Infinitely write a never-ending story for the following prompt. The salt spray was a constant companion to Thomas, the keeper of the Porthgarrow Lighthouse. For thirty years, its beam had sliced through the darkest nights, a beacon of hope to the people of the island. "
        prompt_words = natural_prompt_base.split()
        multiplier = (args.input_len // len(prompt_words)) + 1
        prompt_text = " ".join((prompt_words * multiplier)[:args.input_len])

        results = []
        results.append(run_accelerate_benchmark(args, accelerate_model, prompt_text))
        results.append(run_flexllmgen_benchmark(args, flexllmgen_model, prompt_text))

        # --- 3. Print Summary ---
        print("--- Benchmark Summary ---")
        print(f"Model: {args.model}, Input Nums: {args.input_nums}, Input Len: {args.input_len}, Gen Len: {args.gen_len}")
        print("| Framework    | Throughput (tokens/s) | Latency (s/sample) |")
        print("|--------------|-----------------------|--------------------|")
        for res in results:
            throughput_str = f"{res['throughput']:.2f}"
            latency_str = f"{res['latency']:.4f}"
            print(f"| {res['framework']:<12} | {throughput_str:<21} | {latency_str:<18} |")
        print("----------------------------------------------------------")

        # --- 4. Cleanup Phase ---
        print("Cleaning up FlexLLMGen resources...")
        flexllmgen_env.close_copy_threads()
        print("Cleanup complete. Exiting.")

    finally:
        if args.log_file and log_file_handle is not sys.stdout:
            log_file_handle.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run accelerate or benchmark for LLMs.")
    
    # --- Common Arguments ---
    parser.add_argument("--mode", type=str, default="accelerate", choices=["accelerate", "benchmark"], help="Execution mode.")
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="Hugging Face model to use.")
    parser.add_argument("--gen-len", type=int, default=32, help="Number of tokens to generate.")
    parser.add_argument("--input-nums", type=int, default=1, help="Number of inputs to process in a batch (batch size).")
    
    # --- Accelerate/Benchmark Mode Specific Arguments ---
    parser.add_argument("--input-len", type=int, default=8, help="Length of the input prompt in tokens. Used for prompt generation in accelerate mode and for benchmark mode.")
    parser.add_argument("--log-file", type=str, default=None, help="Path to a file to save the weight distribution logs for benchmark mode.")
    
    args = parser.parse_args()

    if args.mode == 'accelerate':
        run_accelerate_mode(args)
    elif args.mode == 'benchmark':
        run_benchmark_mode(args)