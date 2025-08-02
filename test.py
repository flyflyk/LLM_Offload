# C:/Users/flyfl/Documents/CodeProjects/py/inference/test.py
"""
Main entry point for running FlexLLMGen with an automated offloading policy.

This script acts as a high-level orchestrator that:
1. Profiles the hardware to understand system capabilities.
2. Uses a cost model to find the optimal offloading policy.
3. Generates and executes the appropriate `flex_opt.py` command.
"""

import argparse
import os
import subprocess
import sys

# --- Local Imports ---
from AutoPolicy.profiler import get_hardware_profile
from AutoPolicy.cost_model import CostModel, get_model_info
from AutoPolicy.optimizer import find_best_policy

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Automated inference runner for FlexLLMGen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, default="facebook/opt-6.7b", help="The model name.")
    parser.add_argument("--prompt-len", type=int, default=512, help="Prompt length.")
    parser.add_argument("--gen-len", type=int, default=32, help="Generation length.")
    parser.add_argument("--gpu-batch-size", type=int, default=4)
    parser.add_argument("--num-gpu-batches", type=int, default=1)
    parser.add_argument("--path", type=str, default="~/FlexLLMGen/model_weights", help="Path to model weights.")
    parser.add_argument("--offload-dir", type=str, default="~/FlexLLMGen/offload_dir", help="Offloading directory.")
    parser.add_argument("--force-rerun-profiler", action="store_true", help="Force re-running the hardware profiler.")
    
    args = parser.parse_args()

    # --- 1. Hardware Profiling ---
    hardware_profile = get_hardware_profile(force_rerun=args.force_rerun_profiler)
    print("\nUsing Hardware Profile:", hardware_profile)

    # --- 2. Cost and Model Analysis ---
    cost_model = CostModel(hardware_profile)
    total_batch_size = args.gpu_batch_size * args.num_gpu_batches
    model_info = get_model_info(args.model, total_batch_size, args.prompt_len + args.gen_len)
    print(f"\nModel Info ({args.model}):")
    print(f"  - Weight Size: {model_info.weight_size_gb:.2f} GB")
    print(f"  - KV Cache per token: {model_info.kv_cache_per_token_gb * 1e6:.2f} KB")

    # --- 3. Find Optimal Policy ---
    best_policy = find_best_policy(cost_model, model_info, args.prompt_len, args.gen_len)
    
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
        "--prompt-len", str(args.prompt_len),
        "--gen-len", str(args.gen_len),
        "--gpu-batch-size", str(args.gpu_batch_size),
        "--num-gpu-batches", str(args.num_gpu_batches),
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

    try:
        # Use Popen for real-time output streaming.
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
        for line in iter(process.stdout.readline, ''):
            print(line, end='')
        process.stdout.close()
        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
            
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"\nError executing FlexLLMGen: {e}", file=sys.stderr)
        sys.exit(1)

    print("\n" + "="*50)
    print("Inference finished successfully.")
    print("="*50)

if __name__ == "__main__":
    main()