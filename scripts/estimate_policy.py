import argparse
import warnings
import json
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*Using a non-tuple sequence for multidimensional indexing.*")

from flexllmgen.opt_config import get_opt_config
from src.auto_policy.profiler import HardwareProfile
from src.auto_policy.cost_model import CostModel

def estimate_peak_memory(args):
    print("--- Starting Peak Memory Estimation for a Given Policy ---")

    # 1. Get model configuration
    opt_config = get_opt_config(args.model)

    # 2. Load hardware profile
    try:
        with open("hardware_profile.json", "r") as f:
            hardware_data = json.load(f)
        hardware_profile = HardwareProfile(**hardware_data)
    except FileNotFoundError:
        print("Error: hardware_profile.json not found. Please run the profiler first.", file=sys.stderr)
        sys.exit(1)

    # 3. Instantiate the CostModel
    cost_model = CostModel(opt_config, hardware_profile, args.input_len, args.gen_len)

    # 4. Construct the policy dictionary from command-line arguments
    w_gpu = args.w_gpu_percent / 100.0
    w_cpu = args.w_cpu_percent / 100.0
    c_gpu = args.cache_gpu_percent / 100.0
    c_cpu = args.cache_cpu_percent / 100.0
    a_gpu = args.act_gpu_percent / 100.0
    a_cpu = args.act_cpu_percent / 100.0

    policy = {
        'w_g': w_gpu, 'w_c': w_cpu, 'w_d': 1.0 - w_gpu - w_cpu,
        'c_g': c_gpu, 'c_c': c_cpu, 'c_d': 1.0 - c_gpu - c_cpu,
        'h_g': a_gpu, 'h_c': a_cpu, 'h_d': 1.0 - a_gpu - a_cpu,
    }

    # 5. get_peak_memory()
    gpu_mem_expr_list, cpu_mem_expr_list, _ = cost_model.get_peak_memory(policy, args.batch_size)

    # 6. Find max
    peak_gpu_mem_bytes = max(gpu_mem_expr_list) if gpu_mem_expr_list else 0
    peak_cpu_mem_bytes = max(cpu_mem_expr_list) if cpu_mem_expr_list else 0
    peak_gpu_mem_gb = peak_gpu_mem_bytes / (1024**3)
    peak_cpu_mem_gb = peak_cpu_mem_bytes / (1024**3)

    # 7. Print the results
    print("\n--- Memory Estimation for Manual Policy ---")
    print(f"Model: {args.model}, Batch Size: {args.batch_size}")
    print(f"Sequence Length (Input + Gen): {args.input_len + args.gen_len}")
    print("-" * 45)
    print("Policy (GPU/CPU/Disk)%")
    print(f"  - Weights:     {args.w_gpu_percent}/{args.w_cpu_percent}/{100 - args.w_gpu_percent - args.w_cpu_percent}")
    print(f"  - Cache:       {args.cache_gpu_percent}/{args.cache_cpu_percent}/{100 - args.cache_gpu_percent - args.cache_cpu_percent}")
    print(f"  - Activations: {args.act_gpu_percent}/{args.act_cpu_percent}/{100 - args.act_gpu_percent - args.act_cpu_percent}")
    print("-" * 45)
    print(f"Estimated Peak GPU VRAM: {peak_gpu_mem_gb:.3f} GB")
    print(f"Estimated Peak CPU RAM:  {peak_cpu_mem_gb:.3f} GB")
    print("-------------------------------------------")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate the peak memory usage for a given model and offloading policy.")

    # --- Model and Generation Arguments ---
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="The Hugging Face model to use.")
    parser.add_argument("--input-len", type=int, default=512, help="The length of the input prompt in tokens.")
    parser.add_argument("--gen-len", type=int, default=32, help="The number of tokens to generate.")
    parser.add_argument("--batch-size", type=int, default=1, help="The batch size for the estimation.")

    # --- Manual Policy Arguments ---
    parser.add_argument("--w-gpu-percent", type=float, default=100, help="Percentage of weights on GPU.")
    parser.add_argument("--w-cpu-percent", type=float, default=0, help="Percentage of weights on CPU.")
    parser.add_argument("--cache-gpu-percent", type=float, default=100, help="Percentage of KV cache on GPU.")
    parser.add_argument("--cache-cpu-percent", type=float, default=0, help="Percentage of KV cache on CPU.")
    parser.add_argument("--act-gpu-percent", type=float, default=100, help="Percentage of activations on GPU.")
    parser.add_argument("--act-cpu-percent", type=float, default=0, help="Percentage of activations on CPU.")

    args = parser.parse_args()

    estimate_peak_memory(args)