import argparse
import time
import torch
import os
import sys
import logging

from src.accelerate import config
from src.utils.logger import setup_logging
from src.runners.accelerate_runner import AccelerateRunner
from src.runners.flex_runner import FlexRunner
from src.utils.memory import get_device_limit
from src.utils.benchmark import log_metrics, cleanup_mem
from accelerate import infer_auto_device_map
from transformers import AutoModelForCausalLM
from src.utils.prompts import generate_prompt
from flexllmgen.opt_config import get_opt_config

def run_accelerate_mode(args):
    setup_logging(log_file=getattr(config, 'LOG_FILE', None))
    logger = logging.getLogger(__name__)
    p_type = torch.float16

    # Get model info
    opt_config = get_opt_config(args.model)
    cache_size = opt_config.cache_bytes(args.batch_size, args.input_len + args.gen_len)
    hidden_size = opt_config.hidden_bytes(args.batch_size, args.input_len + args.gen_len)
    
    # Generate device map
    meta_model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=p_type, device_map="meta")
    max_memory = get_device_limit()
    device_map = infer_auto_device_map(meta_model, max_memory=max_memory, no_split_module_classes=meta_model._no_split_modules)
    logger.info(f"Inferred device map: {device_map}")
    del meta_model
    cleanup_mem()

    offload_mode = "Offload" if config.ENABLE_OFFLOAD else "Default"
    kv_offload_mode = " + KV Offload" if config.ENABLE_KV_OFFLOAD else ""
    current_mode = f"{offload_mode}{kv_offload_mode}"
    logger.info(f"--- Starting Execution (Accelerate - {current_mode}) ---")

    runner = AccelerateRunner(model_name=args.model, config=config, device_map=device_map, p_type=p_type)
    prompt_text = generate_prompt(args.input_len)
    prompts = [prompt_text] * args.batch_size

    result = runner.run_accelerate(prompts, max_new_tokens=args.gen_len)
    
    total_tokens = args.batch_size * args.gen_len
    throughput = total_tokens / result["inference_time"] if result["inference_time"] > 0 else 0

    log_metrics(
        framework="Accelerate",
        throughput=throughput,
        infer_time=result["inference_time"],
        model_load_time=runner.model_load_time,
        model_name=args.model,
        flex_allocation_info={
            "device_sizes": {},
            "cache_size_gb": cache_size / (1024**3),
            "hidden_size_gb": hidden_size / (1024**3),
        }
    )
    return {"framework": "Accelerate", "throughput": throughput, "load_time": runner.model_load_time}

def run_flex_mode(args, use_autoflex=False):
    framework_name = "AutoFlex" if use_autoflex else "FlexGen"
    setup_logging()
    logger = logging.getLogger(__name__)
    logger.info(f"--- Starting Execution ({framework_name}) ---")

    runner = FlexRunner(
        model_name=args.model,
        use_autoflex=use_autoflex,
        args=args,
        offload_dir=os.path.expanduser(args.offload_dir),
        cache_dir=os.path.expanduser(args.path)
    )
    
    prompt = generate_prompt(args.input_len)
    prompts = [prompt] * args.batch_size

    result = runner.run(prompts, input_len=args.input_len, max_new_tokens=args.gen_len)

    # Get policy and allocation info
    policy_info = runner.get_policy_info()
    allocation_info = runner.get_model_info()

    runner.cleanup()

    total_tokens = args.batch_size * args.gen_len
    throughput = total_tokens / result["inference_time"] if result["inference_time"] > 0 else 0

    log_metrics(
        framework=framework_name,
        throughput=throughput,
        infer_time=result["inference_time"],
        model_load_time=result["load_time"],
        model_name=args.model,
        flex_policy_info=policy_info,
        flex_allocation_info=allocation_info
    )
    return {"framework": framework_name, "throughput": throughput, "load_time": result['load_time']}


def run_benchmark_mode(args):
    print("--- Starting Benchmark Mode ---")
    print(f"Model: {args.model}, Batch Size: {args.batch_size}, Input Len: {args.input_len}, Gen Len: {args.gen_len}")
    
    results = []

    # 1. Accelerate
    print("--- Benchmarking Accelerate ---")
    accelerate_results = run_accelerate_mode(args)
    if accelerate_results:
        results.append(accelerate_results)
    
    cleanup_mem()
    time.sleep(2)

    # 2. FlexGen (All-GPU)
    print("--- Benchmarking FlexGen (All-GPU) ---")
    flexgen_results = run_flex_mode(args, use_autoflex=False)
    if flexgen_results:
        results.append(flexgen_results)

    cleanup_mem()
    time.sleep(2)

    # 3. AutoFlex
    print("--- Benchmarking AutoFlex ---")
    autoflex_results = run_flex_mode(args, use_autoflex=True)
    if autoflex_results:
        results.append(autoflex_results)

    print("--- Benchmark Summary ---")
    print(f"Model: {args.model}, Batch Size: {args.batch_size}, Input Len: {args.input_len}, Gen Len: {args.gen_len}")
    print("| Framework         | Throughput (tokens/s) | Model Load Time (s) |")
    print("|-------------------|-----------------------|---------------------|")
    for res in sorted(results, key=lambda x: x['framework']):
        framework = res['framework']
        throughput_str = f"{res['throughput']:.2f}"
        load_time_str = f"{res['load_time']:.4f}"
        print(f"| {framework:<17} | {throughput_str:<21} | {load_time_str:<19} |")
    print("-------------------------------------------------------------------")
    print("Benchmark finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run accelerate or benchmark for LLMs.")
    
    # Common arguments
    parser.add_argument("--mode", type=str, default="accelerate", choices=["accelerate", "flexgen", "autoflex", "benchmark"], help="Execution mode.")
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="Hugging Face model to use.")
    parser.add_argument("--gen-len", type=int, default=32, help="Number of tokens to generate.")
    parser.add_argument("--batch-size", type=int, default=1, help="Number of inputs to process in a batch (batch size).")
    parser.add_argument("--input-len", type=int, default=8, help="Length of the input prompt in tokens.")
    
    # FlexGen/AutoFlex specific arguments
    parser.add_argument("--path", type=str, default="/mnt/ssd/flexgen_cache", help="Path to model weights cache for FlexGen.")
    parser.add_argument("--offload-dir", type=str, default="/mnt/ssd/flexgen_offload", help="Offloading directory for FlexGen.")
    parser.add_argument("--pin-weight", type=lambda x: (str(x).lower() == 'true'), default=True, help="Use pinned memory for weights in FlexGen/AutoFlex modes. Set to False to increase RAM offload capacity.")

    # AutoFlex specific arguments
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
