import argparse
import time
import torch
from inference_runner import InferenceRunner
import subprocess
import os
# FlexLLMGen imports
from flexllmgen.flex_opt import Policy, OptLM, TorchDevice, CompressionConfig

def benchmark_accelerate(args, prompt_text):
    """
    Benchmarks the existing accelerate-based inference pipeline.
    """
    print("--- Benchmarking Accelerate ---")
    runner = InferenceRunner(
        model_name=args.model,
        p_type=torch.float16,
        use_accelerate=True,
        offload_dir="offload_dir"
    )

    prompts = [prompt_text] * args.input_nums

    start_time = time.time()
    outputs = runner.run_inference(
        prompts,
        max_new_tokens=args.gen_len
    )
    end_time = time.time()

    total_time = end_time - start_time
    # Calculate throughput in tokens/sec
    total_tokens = args.input_nums * args.gen_len
    throughput = total_tokens / total_time
    latency = total_time / args.input_nums

    print(f"Model: {args.model}")
    print(f"Input Nums: {args.input_nums}")
    print(f"Input Length: {args.input_len}")
    print(f"Generation Length: {args.gen_len}")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Throughput: {throughput:.4f} tokens/sec")
    print(f"Latency: {latency:.4f} sec/sample")
    print("---------------------------------")
    return {
        "framework": "Accelerate",
        "model": args.model,
        "input_nums": args.input_nums,
        "input_len": args.input_len,
        "gen_len": args.gen_len,
        "throughput": throughput,
        "latency": latency,
    }


def benchmark_flexllmgen(args, prompt_text):
    """
    Benchmarks the FlexLLMGen framework using a direct library call.
    """
    print("--- Benchmarking FlexLLMGen ---")

    # 1. Set up environment and arguments for FlexLLMGen
    flexllmgen_path = os.path.abspath("./FlexLLMGen")
    cache_path = os.path.abspath("./flexllmgen_cache")
    os.makedirs(cache_path, exist_ok=True)

    # Add FlexLLMGen to python path to ensure imports work
    flexllmgen_env = os.environ.copy()
    current_pythonpath = flexllmgen_env.get('PYTHONPATH', '')
    if flexllmgen_path not in current_pythonpath:
        if current_pythonpath:
            flexllmgen_env['PYTHONPATH'] = f"{flexllmgen_path}:{current_pythonpath}"
        else:
            flexllmgen_env['PYTHONPATH'] = flexllmgen_path
        os.environ['PYTHONPATH'] = flexllmgen_env['PYTHONPATH']


    # Mimic the argparse Namespace that FlexLLMGen's components expect
    flex_args = argparse.Namespace(
        model=args.model,
        path=cache_path,
        prompt_len=args.input_len,
        gen_len=args.gen_len,
        gpu_batch_size=args.input_nums,
        percent=[100, 0, 100, 0, 100, 0],
        pin_weight=True,
        cpu_cache_compute=False,
        attn_sparsity=1.0,
        compress_weight=False,
        compress_cache=False,
    )

    # 2. Initialize the model (outside the timer)
    env = TorchDevice(torch.device("cuda:0"))
    policy = Policy(
        flex_args.gpu_batch_size,
        1,  # num_gpu_batches
        flex_args.percent,
        flex_args.pin_weight,
        flex_args.cpu_cache_compute,
        flex_args.attn_sparsity,
        flex_args.compress_weight,
        CompressionConfig(num_bits=4, group_size=64), # weight_comp_config
        flex_args.compress_cache,
        CompressionConfig(num_bits=4, group_size=64), # cache_comp_config
    )
    
    print("Initializing FlexLLMGen model...")
    opt_lm = OptLM(flex_args.model, env, flex_args.path, policy)
    print("Initialization complete.")

    prompts = [prompt_text] * args.input_nums

    # 3. Run benchmark (time only the generation part)
    start_time = time.time()
    # The generate function in the library does not return the output text directly,
    # but it prints the performance metrics we need. We will capture those.
    # For a fair comparison, we run the generation.
    # Note: The internal timer of FlexLLMGen will also run and print its own stats.
    outputs, _ = opt_lm.generate(
        prompts,
        max_new_tokens=flex_args.gen_len,
        debug=False, # Set to False to avoid excessive printing during timing
        show_progress=False
    )
    end_time = time.time()

    total_time = end_time - start_time
    total_tokens = args.input_nums * args.gen_len
    throughput = total_tokens / total_time
    latency = total_time / args.input_nums

    print(f"Model: {args.model}")
    print(f"Input Nums: {args.input_nums}")
    print(f"Input Length: {args.input_len}")
    print(f"Generation Length: {args.gen_len}")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Throughput: {throughput:.4f} tokens/sec")
    print(f"Latency: {latency:.4f} sec/sample")
    print("---------------------------------")

    return {
        "framework": "FlexLLMGen",
        "model": args.model,
        "input_nums": args.input_nums,
        "input_len": args.input_len,
        "gen_len": args.gen_len,
        "throughput": throughput,
        "latency": latency,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Accelerate vs. FlexLLMGen")
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="Hugging Face model to benchmark.")
    parser.add_argument("--input-nums", type=int, default=4, help="Number of inputs (batch size).")
    parser.add_argument("--input-len", type=int, default=8, help="Length of the input prompt in tokens.")
    parser.add_argument("--gen-len", type=int, default=32, help="Number of tokens to generate.")
    
    args = parser.parse_args()

    # Generate a shared, more natural prompt for both frameworks
    natural_prompt_base = "Infinitely write a never-ending story for the following prompt. The salt spray was a constant companion to Thomas, the keeper of the Porthgarrow Lighthouse. For thirty years, its beam had sliced through the darkest nights, a beacon of hope to the people of the island. "
    prompt_words = natural_prompt_base.split()
    # Repeat the base prompt to be long enough and then truncate to the desired length
    multiplier = (args.input_len // len(prompt_words)) + 1
    prompt_text = " ".join((prompt_words * multiplier)[:args.input_len])

    results = []
    results.append(benchmark_accelerate(args, prompt_text))
    results.append(benchmark_flexllmgen(args, prompt_text))

    # --- Print Summary ---
    print("\n--- Benchmark Summary ---")
    print("| Framework    | Model             | Input Nums | Input Len | Gen Len | Throughput (tokens/s) | Latency (s/sample) |")
    print("|--------------|-------------------|------------|-----------|---------|-----------------------|--------------------|")
    for res in results:
        throughput_str = f"{res['throughput']:.2f}"
        print(f"| {res['framework']:<12} | {res['model']:<17} | {res['input_nums']:<10} | {res['input_len']:<9} | {res['gen_len']:<7} | {throughput_str:<21} | {res['latency']:.4f}             |")
    print("---------------------------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
