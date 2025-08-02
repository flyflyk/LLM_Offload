import argparse
import os
import subprocess
import sys
import re
from datetime import datetime

# Add the FlexLLMGen submodule to the Python path
flexllmgen_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "FlexLLMGen"))
if flexllmgen_path not in sys.path:
    sys.path.insert(0, flexllmgen_path)

from AutoPolicy.profiler import get_hardware_profile
from AutoPolicy.cost_model import CostModel, get_model_info
from AutoPolicy.optimizer import find_best_policy

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


def main():
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

if __name__ == "__main__":
    main()