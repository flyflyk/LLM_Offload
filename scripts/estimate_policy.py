import argparse
import yaml
import warnings
from types import SimpleNamespace
import sys
import os

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Suppress the specific UserWarning from FlexLLMGen
warnings.filterwarnings("ignore", category=UserWarning, message=".*Using a non-tuple sequence for multidimensional indexing.*",)

from src.runners.flex_runner import FlexRunner

def load_config(mode) -> SimpleNamespace:
    config_path = os.path.join(project_root, f"src/configs/{mode}.yaml")
    with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            return SimpleNamespace(**config_dict)

def estimate_policy(args):
    print("--- Starting Policy Estimation ---")
    
    flexgen_config = load_config("flexgen")
    autoflex_config = load_config("autoflex")

    runner = FlexRunner(
        model_name=args.model,
        use_autoflex=True,
        args=args,
        config=flexgen_config,
        force_rerun=autoflex_config.force_rerun_profiler,
    )
    policy = runner.policy

    print("\n--- Estimated Resource Allocation ---")
    print(f"Model: {args.model}")
    print(f"Input Length: {args.input_len}, Generation Length: {args.gen_len}")
    print(f"GPU Batch Size: {policy.gpu_batch_size}")
    print(f"Estimated GPU VRAM: {policy.gpu_gb:.2f} GB")
    print(f"Estimated CPU RAM: {policy.cpu_gb:.2f} GB")
    print("------------------------------------")

    # Clean up
    if hasattr(runner, 'cleanup'):
        runner.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Estimate the resource policy for a given model using AutoFlex.")
    
    # --- Arguments from main.py (relevant for flexgen/autoflex) ---
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="The Hugging Face model to use.")
    parser.add_argument("--input-len", type=int, default=512, help="The length of the input prompt in tokens.")
    parser.add_argument("--gen-len", type=int, default=32, help="The number of tokens to generate.")
    parser.add_argument("--batch-size", type=int, default=1, help="The number of prompts to process in a batch (used for policy search).")
    parser.add_argument("--offload-dir", type=str, default="/mnt/ssd/offload_dir", help="The common directory for offloading tensors to disk.")
    parser.add_argument("--log-file", type=str, default="logs/log", help="Path to the log file.")

    args = parser.parse_args()
    estimate_policy(args)
