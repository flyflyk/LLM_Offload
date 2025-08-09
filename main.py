import argparse
import gc
import time
import torch
import os
import sys
import logging

from src.accelerate import config
from src.accelerate.logger import setup_logging
from src.runners.accelerate_runner import AccelerateRunner
from src.runners.flex_runner import FlexRunner
from src.auto_policy.profiler import get_hardware_profile
from src.auto_policy.cost_model import CostModel, get_model_info
from src.auto_policy.optimizer import find_best_policy

# Add the FlexLLMGen submodule to the Python path
flexllmgen_path = os.path.abspath("./FlexLLMGen")
if flexllmgen_path not in sys.path:
    sys.path.insert(0, flexllmgen_path)
from flexllmgen.flex_opt import Policy, CompressionConfig


# --- Helper Functions ---

def check_vram(args):
    """Checks if the model weights can fit into the available VRAM."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot perform VRAM check.")
        return False

    print("--- Performing VRAM Pre-check for All-GPU Policy ---")
    model_info = get_model_info(args.model, 1, 1)
    model_size_gb = model_info.weight_size_gb
    free_vram_bytes, _ = torch.cuda.mem_get_info(0)
    free_vram_gb = free_vram_bytes / (1024**3)

    print(f"Estimated Model Size: {model_size_gb:.2f} GB")
    print(f"Available VRAM: {free_vram_gb:.2f} GB")

    if model_size_gb > free_vram_gb * 0.95:
        print("Model is too large to fit entirely in VRAM.")
        return False
    
    print("Model should fit in VRAM.")
    return True

def generate_prompt(input_len):
    """Generates a prompt of a specific token length."""
    base_prompt = "Infinitely write a never-ending story for the following prompt. The salt spray was a constant companion to Thomas, the keeper of the Porthgarrow Lighthouse."
    prompt_words = base_prompt.split()
    multiplier = (input_len // len(prompt_words)) + 1
    return " ".join((prompt_words * multiplier)[:input_len])

# --- Main Execution Modes ---

def run_accelerate_mode(args):
    """Runs the standard accelerate mode."""
    setup_logging(log_file=getattr(config, 'LOG_FILE', None))
    logger = logging.getLogger(__name__)

    streaming_mode = "Streaming" if config.ENABLE_STREAMING else "Default"
    kv_offload_mode = " + KV Offload" if config.ENABLE_KV_OFFLOAD else ""
    current_mode = f"{streaming_mode}{kv_offload_mode}"
    logger.info(f"--- Starting Execution (Accelerate - {current_mode}) ---")
    logger.info(f"Model: {args.model}, Batch size: {args.input_nums}")

    runner = AccelerateRunner(model_name=args.model, config=config)
    prompt_text = generate_prompt(args.input_len)
    prompts = [prompt_text] * args.input_nums

    result = runner.run_accelerate(prompts, max_new_tokens=args.gen_len)
    
    total_tokens = args.input_nums * args.gen_len
    throughput = total_tokens / result["inference_time"] if result["inference_time"] > 0 else 0
    latency = result["inference_time"] / args.input_nums

    logger.info("--- Performance Metrics ---")
    logger.info(f"Model Load Time: {runner.model_load_time:.4f}s")
    logger.info(f"Total Inference Time: {result['inference_time']:.4f}s")
    logger.info(f"Throughput: {throughput:.2f} tokens/sec")
    logger.info(f"Latency: {latency:.4f} sec/batch")
    logger.info(f"--- Execution Finished Successfully (Accelerate) ---")
    return {"framework": "Accelerate", "throughput": throughput, "latency": latency, "load_time": runner.model_load_time}

def run_flex_mode(args, use_autoflex=False):
    """
    Runs FlexLLMGen with a specific policy.
    If use_autoflex is True, it finds the optimal policy first.
    Otherwise, it uses a default all-on-GPU policy.
    """
    framework_name = "AutoFlex" if use_autoflex else "FlexGen (All-GPU)"
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting Execution ({framework_name}) ---")

    policy = None
    if use_autoflex:
        logger.info("Finding optimal policy for AutoFlex...")
        hardware_profile = get_hardware_profile(force_rerun=args.force_rerun_profiler)
        cost_model = CostModel(hardware_profile)
        model_info = get_model_info(args.model, args.input_nums, args.input_len + args.gen_len)
        policy = find_best_policy(cost_model, model_info, args.input_len, args.gen_len, args.input_nums)
        if not policy:
            logger.error("Could not find an optimal policy for AutoFlex. Exiting.")
            return None
        logger.info(f"Optimal Policy Found: W: {policy.w_gpu_percent}/{policy.w_cpu_percent}, C: {policy.cache_gpu_percent}/{policy.cache_cpu_percent}")
    else:
        if not check_vram(args):
            logger.error("Not enough VRAM for FlexGen (All-GPU). Use '--mode autoflex'.")
            return None
        logger.info("Using default All-GPU policy.")
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

    runner = FlexRunner(
        model_name=args.model,
        policy=policy,
        offload_dir=os.path.expanduser(args.offload_dir),
        cache_dir=os.path.expanduser(args.path)
    )
    
    prompt_text = generate_prompt(args.input_len)
    prompts = [prompt_text] * args.input_nums

    result = runner.run(prompts, input_len=args.input_len, max_new_tokens=args.gen_len)
    runner.cleanup()

    total_tokens = args.input_nums * args.gen_len
    throughput = total_tokens / result["inference_time"] if result["inference_time"] > 0 else 0
    latency = result["inference_time"] / args.input_nums

    logger.info("--- Performance Metrics ---")
    logger.info(f"Model Load Time: {result['load_time']:.4f}s")
    logger.info(f"Total Inference Time: {result['inference_time']:.4f}s")
    logger.info(f"Throughput: {throughput:.2f} tokens/sec")
    logger.info(f"Latency: {latency:.4f} sec/batch")
    logger.info(f"--- Execution Finished Successfully ({framework_name}) ---")
    return {"framework": framework_name, "throughput": throughput, "latency": latency, "load_time": result['load_time']}


def run_benchmark_mode(args):
    """Runs a comparative benchmark between Accelerate, FlexGen, and AutoFlex."""
    print("--- Starting Benchmark Mode ---")
    print(f"Model: {args.model}, Batch Size: {args.input_nums}, Input Len: {args.input_len}, Gen Len: {args.gen_len}")
    
    results = []

    # 1. Accelerate
    print("--- Benchmarking Accelerate ---")
    accelerate_results = run_accelerate_mode(args)
    if accelerate_results:
        results.append(accelerate_results)
    
    # --- Force Memory Cleanup ---
    print("--- Forcefully cleaning up VRAM before next test ---")
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    # 2. FlexGen (All-GPU)
    print("--- Benchmarking FlexGen (All-GPU) ---")
    flexgen_results = run_flex_mode(args, use_autoflex=False)
    if flexgen_results:
        results.append(flexgen_results)

    # --- Force Memory Cleanup ---
    print("--- Forcefully cleaning up VRAM before next test ---")
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    # 3. AutoFlex
    print("--- Benchmarking AutoFlex ---")
    autoflex_results = run_flex_mode(args, use_autoflex=True)
    if autoflex_results:
        results.append(autoflex_results)

    # --- Print Summary ---
    print("--- Benchmark Summary ---")
    print(f"Model: {args.model}, Batch Size: {args.input_nums}, Input Len: {args.input_len}, Gen Len: {args.gen_len}")
    print("| Framework         | Throughput (tokens/s) | Latency (s/sample) | Model Load Time (s) |")
    print("|-------------------|-----------------------|--------------------|---------------------|")
    for res in sorted(results, key=lambda x: x['framework']):
        framework = res['framework']
        throughput_str = f"{res['throughput']:.2f}"
        latency_str = f"{res['latency']:.4f}"
        load_time_str = f"{res['load_time']:.4f}"
        print(f"| {framework:<17} | {throughput_str:<21} | {latency_str:<18} | {load_time_str:<19} |")
    print("-------------------------------------------------------------------------------------")
    print("Benchmark finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run accelerate or benchmark for LLMs.")
    
    # --- Common Arguments ---
    parser.add_argument("--mode", type=str, default="accelerate", choices=["accelerate", "flexgen", "autoflex", "benchmark"], help="Execution mode.")
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="Hugging Face model to use.")
    parser.add_argument("--gen-len", type=int, default=32, help="Number of tokens to generate.")
    parser.add_argument("--input-nums", type=int, default=1, help="Number of inputs to process in a batch (batch size).")
    parser.add_argument("--input-len", type=int, default=8, help="Length of the input prompt in tokens.")
    
    # --- FlexGen/AutoFlex Specific Arguments ---
    parser.add_argument("--path", type=str, default="~/.cache/flexllmgen_cache", help="Path to model weights cache for FlexLLMGen.")
    parser.add_argument("--offload-dir", type=str, default="~/flexllmgen_offload", help="Offloading directory for FlexLLMGen.")
    parser.add_argument("--force-rerun-profiler", action="store_true", help="Force re-running the hardware profiler for autoflex mode.")

    args = parser.parse_args()

    if args.mode == 'accelerate':
        run_accelerate_mode(args)
    elif args.mode in ['flexgen', 'flexllmgen']:
        run_flex_mode(args, use_autoflex=False)
    elif args.mode == 'autoflex':
        run_flex_mode(args, use_autoflex=True)
    elif args.mode == 'benchmark':
        run_benchmark_mode(args)
    else:
        print(f"Invalid mode: {args.mode}", file=sys.stderr)
        sys.exit(1)
