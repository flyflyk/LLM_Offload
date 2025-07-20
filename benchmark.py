import argparse
import time
import torch
import os
import sys
import numpy as np
from transformers import AutoTokenizer

from inference_runner import InferenceRunner

# Add the FlexLLMGen submodule to the Python path
flexllmgen_path = os.path.abspath("./FlexLLMGen")
if flexllmgen_path not in sys.path:
    sys.path.insert(0, flexllmgen_path)

# FlexLLMGen imports
from flexllmgen.flex_opt import Policy, OptLM, CompressionConfig
from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
from flexllmgen.utils import ExecutionEnv

# --- Initialization Functions ---

def initialize_accelerate(args):
    """Loads the Accelerate model and returns the runner object."""
    print("--- Initializing Accelerate Model ---")
    runner = InferenceRunner(
        model_name=args.model,
        p_type=torch.float16,
        use_accelerate=True,
        offload_dir="offload_dir"
    )
    print("Accelerate model initialized.")
    if hasattr(runner.model, 'hf_device_map'):
        print("Accelerate Model Weight Distribution:")
        print(runner.model.hf_device_map)
    return runner

def initialize_flexllmgen(args):
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
    print(f"FlexLLMGen Model Weight Distribution: GPU={opt_lm.policy.w_gpu_percent}%, CPU={opt_lm.policy.w_cpu_percent}%")
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

# --- Main Execution ---

def main():
    parser = argparse.ArgumentParser(description="Benchmark Accelerate vs. FlexLLMGen")
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="Hugging Face model to benchmark.")
    parser.add_argument("--input-nums", type=int, default=4, help="Number of inputs (batch size).")
    parser.add_argument("--input-len", type=int, default=8, help="Length of the input prompt in tokens.")
    parser.add_argument("--gen-len", type=int, default=32, help="Number of tokens to generate.")
    
    args = parser.parse_args()

    # --- 1. Initialization Phase ---
    print("Initializing models... This may take a moment.")
    accelerate_model = initialize_accelerate(args)
    flexllmgen_model, flexllmgen_env = initialize_flexllmgen(args)
    print("All models initialized. Starting benchmarks.\n")

    # --- 2. Benchmarking Phase ---
    natural_prompt_base = "Infinitely write a never-ending story for the following prompt. The salt spray was a constant companion to Thomas, the keeper of the Porthgarrow Lighthouse. For thirty years, its beam had sliced through the darkest nights, a beacon of hope to the people of the island. "
    prompt_words = natural_prompt_base.split()
    multiplier = (args.input_len // len(prompt_words)) + 1
    prompt_text = " ".join((prompt_words * multiplier)[:args.input_len])

    results = []
    results.append(run_accelerate_benchmark(args, accelerate_model, prompt_text))
    results.append(run_flexllmgen_benchmark(args, flexllmgen_model, prompt_text))

    # --- 3. Print Summary ---
    print("\n--- Benchmark Summary ---")
    print(f"Model: {args.model}, Input Nums: {args.input_nums}, Input Len: {args.input_len}, Gen Len: {args.gen_len}")
    print("| Framework    | Throughput (tokens/s) | Latency (s/sample) |")
    print("|--------------|-----------------------|--------------------|")
    for res in results:
        throughput_str = f"{res['throughput']:.2f}"
        latency_str = f"{res['latency']:.4f}"
        print(f"| {res['framework']:<12} | {throughput_str:<21} | {latency_str:<18} |")
    print("----------------------------------------------------------")

    # --- 4. Cleanup Phase ---
    print("\nCleaning up FlexLLMGen resources...")
    flexllmgen_env.close_copy_threads()
    print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()