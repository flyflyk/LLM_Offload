import argparse
import time
import torch
import os
import sys
import numpy as np
import logging
from transformers import AutoTokenizer

from inference_engine import config
from inference_engine.logger import setup_logging
from inference_engine.inference_runner import InferenceRunner

# Add the FlexLLMGen submodule to the Python path
flexllmgen_path = os.path.abspath("./FlexLLMGen")
if flexllmgen_path not in sys.path:
    sys.path.insert(0, flexllmgen_path)

# FlexLLMGen imports
from flexllmgen.flex_opt import Policy, OptLM, CompressionConfig
from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
from flexllmgen.utils import ExecutionEnv

# --- Helper Functions ---

def print_flexllmgen_distribution(opt_lm, log_file):
    """Prints the detailed layer-by-layer weight distribution for a FlexLLMGen model to a file."""
    # ... (rest of the function is the same as in benchmark.py)

# --- Initialization Functions ---

def initialize_accelerate(args, log_file):
    """Loads the Accelerate model and returns the runner object."""
    print("--- Initializing Accelerate Model ---")
    # Set IS_BENCHMARK to True in config for benchmark mode
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
    # ... (this function is the same as in benchmark.py)

# --- Benchmarking Functions ---

def run_accelerate_benchmark(args, runner, prompt_text):
    """Runs the benchmark for an already initialized Accelerate model."""
    # ... (this function is the same as in benchmark.py)

def run_flexllmgen_benchmark(args, opt_lm, prompt_text):
    """Runs the benchmark for an already initialized FlexLLMGen model."""
    # ... (this function is the same as in benchmark.py)

# --- Main Execution Modes ---

def run_inference_mode():
    """Runs the standard inference mode using settings from config.py."""
    setup_logging(log_file=getattr(config, 'LOG_FILE', None))
    logger = logging.getLogger(__name__)

    streaming_mode = "Streaming" if config.ENABLE_STREAMING else "Default"
    kv_offload_mode = " + KV Offload" if config.ENABLE_KV_OFFLOAD else ""
    current_mode = f"{streaming_mode}{kv_offload_mode}"
    logger.info(f"--- Starting Execution ({current_mode}) ---")
    logger.info(f"Using device: {config.DEVICE}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")

    runner = InferenceRunner(model_name=config.CHOSEN_MODEL, config=config)

    for i in range(0, len(config.PROMPT_LIST), config.BATCH_SIZE):
        batch_prompts = config.PROMPT_LIST[i : i + config.BATCH_SIZE]
        if not batch_prompts: continue

        logger.info(f"Processing batch {i // config.BATCH_SIZE + 1} with {len(batch_prompts)} prompts...")
        runner.run_inference(batch_prompts, max_new_tokens=config.MAX_TOKENS)

    logger.info(f"
--- Execution Finished Successfully ({current_mode}) ---")

def run_benchmark_mode(args):
    """Runs the benchmark mode to compare Accelerate and FlexLLMGen."""
    log_file_handle = open(args.log_file, 'w') if args.log_file else sys.stdout

    try:
        # --- 1. Initialization Phase ---
        print("Initializing models... This may take a moment.")
        accelerate_model = initialize_accelerate(args, log_file=log_file_handle)
        flexllmgen_model, flexllmgen_env = initialize_flexllmgen(args, log_file=log_file_handle)
        print("All models initialized. Starting benchmarks.
")

        # --- 2. Benchmarking Phase ---
        # ... (this section is the same as in benchmark.py)

        # --- 3. Print Summary ---
        # ... (this section is the same as in benchmark.py)

        # --- 4. Cleanup Phase ---
        # ... (this section is the same as in benchmark.py)

    finally:
        if args.log_file and log_file_handle is not sys.stdout:
            log_file_handle.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference or benchmark for LLMs.")
    parser.add_argument("--mode", type=str, default="inference", choices=["inference", "benchmark"], help="Execution mode.")
    # Add arguments from benchmark.py for benchmark mode
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="Hugging Face model to benchmark.")
    parser.add_argument("--input-nums", type=int, default=4, help="Number of inputs (batch size).")
    parser.add_argument("--input-len", type=int, default=8, help="Length of the input prompt in tokens.")
    parser.add_argument("--gen-len", type=int, default=32, help="Number of tokens to generate.")
    parser.add_argument("--log-file", type=str, default=None, help="Path to a file to save the weight distribution logs.")
    
    args = parser.parse_args()

    if args.mode == 'inference':
        run_inference_mode()
    elif args.mode == 'benchmark':
        run_benchmark_mode(args)
