import argparse
import time
import torch
from inference_runner import InferenceRunner
import subprocess
import os

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
    Benchmarks the FlexLLMGen framework by calling its CLI.
    """
    print("--- Benchmarking FlexLLMGen ---")
    
    flexllmgen_path = os.path.abspath("./FlexLLMGen")
    
    command = [
        "python",
        "-m",
        "flexllmgen.flex_opt",
        "--model", args.model,
        "--prompt-text", prompt_text,
        "--gpu-batch-size", str(args.input_nums),
        "--prompt-len", str(args.input_len),
        "--gen-len", str(args.gen_len),
        "--percent", "0", "100", "100", "0", "100", "0"
    ]

    # Prepare environment for subprocess to include FlexLLMGen in PYTHONPATH
    flexllmgen_env = os.environ.copy()
    current_pythonpath = flexllmgen_env.get('PYTHONPATH', '')
    if current_pythonpath:
        flexllmgen_env['PYTHONPATH'] = f"{flexllmgen_path}:{current_pythonpath}"
    else:
        flexllmgen_env['PYTHONPATH'] = flexllmgen_path

    start_time = time.time()
    process = subprocess.run(command, cwd=flexllmgen_path, capture_output=True, text=True, env=flexllmgen_env)
    end_time = time.time()

    total_time = end_time - start_time
    
    throughput = 0
    output_text = process.stdout + process.stderr
    for line in output_text.splitlines():
        if "total throughput" in line.lower():
            try:
                throughput = float(line.split(":")[1].strip().split()[0])
                break
            except (IndexError, ValueError) as e:
                print(f"Could not parse throughput from line: '{line}'. Error: {e}")

    latency = total_time / args.input_nums if throughput > 0 else float('inf')

    print(f"Model: {args.model}")
    print(f"Input Nums: {args.input_nums}")
    print(f"Input Length: {args.input_len}")
    print(f"Generation Length: {args.gen_len}")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Reported Throughput: {throughput} tokens/s")
    print(f"Latency: {latency:.4f} sec/sample")
    print(f"--- Raw Output ---")
    print(f"{output_text}")
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
