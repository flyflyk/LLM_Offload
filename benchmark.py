import argparse
import time
import torch
from inference_runner import InferenceRunner
import subprocess
import os

def benchmark_accelerate(args):
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

    prompts = ["Hello, my name is"] * args.batch_size

    start_time = time.time()
    outputs = runner.run_inference(
        prompts,
        max_new_tokens=args.gen_len
    )
    end_time = time.time()

    total_time = end_time - start_time
    throughput = args.batch_size / total_time
    latency = total_time / args.batch_size

    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Generation Length: {args.gen_len}")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Throughput: {throughput:.4f} samples/sec")
    print(f"Latency: {latency:.4f} sec/sample")
    print("---------------------------------")
    return {
        "framework": "Accelerate",
        "model": args.model,
        "batch_size": args.batch_size,
        "gen_len": args.gen_len,
        "throughput": throughput,
        "latency": latency,
    }


def benchmark_flexllmgen(args):
    """
    Benchmarks the FlexLLMGen framework by calling its CLI.
    """
    print("--- Benchmarking FlexLLMGen ---")
    
    # Note: Adjust the flexllmgen_path to your actual location
    flexllmgen_path = os.path.abspath("../FlexLLMGen")
    
    command = [
        "python",
        "-m",
        "flexllmgen.flex_opt",
        "--model", args.model,
        "--gpu-batch-size", str(args.batch_size),
        "--gen-len", str(args.gen_len),
        "--prompt-len", "8", # Assuming a small prompt length for now
        # FlexLLMGen's --percent argument needs to be configured based on your hardware.
        # This example assumes all on GPU: [weight-gpu, weight-cpu, cache-gpu, cache-cpu, act-gpu, act-cpu]
        "--percent", "100", "0", "100", "0", "100", "0"
    ]

    start_time = time.time()
    # We need to execute this from within the FlexLLMGen directory to ensure imports work correctly.
    process = subprocess.run(command, cwd=flexllmgen_path, capture_output=True, text=True)
    end_time = time.time()

    total_time = end_time - start_time
    
    # FlexLLMGen outputs throughput directly, so we parse it.
    # This is an approximation. A more robust way would be to parse the log file.
    throughput = 0
    output_text = process.stdout + process.stderr
    for line in output_text.splitlines():
        if "Throughput" in line:
            try:
                # Example line: "Throughput: 53.29 token/s" or similar
                throughput = float(line.split(":")[1].strip().split()[0])
                break
            except (IndexError, ValueError) as e:
                print(f"Could not parse throughput from line: '{line}'. Error: {e}")


    latency = total_time / args.batch_size if throughput > 0 else float('inf')


    print(f"Model: {args.model}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Generation Length: {args.gen_len}")
    print(f"Total Time: {total_time:.4f}s")
    # Note: FlexLLMGen often reports in token/s, while our Accelerate bench is samples/sec.
    # This will need to be normalized for a fair comparison.
    print(f"Reported Throughput: {throughput} (unit may vary)")
    print(f"Latency: {latency:.4f} sec/sample")
    print(f"--- Raw Output ---")
    print(f"{output_text}")
    print("---------------------------------")

    return {
        "framework": "FlexLLMGen",
        "model": args.model,
        "batch_size": args.batch_size,
        "gen_len": args.gen_len,
        "throughput": throughput, # Note: Unit might be different
        "latency": latency,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Accelerate vs. FlexLLMGen")
    parser.add_argument("--model", type=str, default="facebook/opt-1.3b", help="Hugging Face model to benchmark.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for inference.")
    parser.add_argument("--gen-len", type=int, default=32, help="Number of tokens to generate.")
    
    args = parser.parse_args()

    results = []
    results.append(benchmark_accelerate(args))
    results.append(benchmark_flexllmgen(args))

    # --- Print Summary ---
    print("\n--- Benchmark Summary ---")
    print("| Framework    | Model             | Batch Size | Gen Len | Throughput      | Latency (s/sample) |")
    print("|--------------|-------------------|------------|---------|-----------------|--------------------|")
    for res in results:
        # Basic formatting, can be improved
        throughput_str = f"{res['throughput']:.2f}"
        if res['framework'] == 'FlexLLMGen':
            throughput_str += " (tokens/s?)" # Highlight potential unit difference
        else:
            throughput_str += " (samples/s)"

        print(f"| {res['framework']:<12} | {res['model']:<17} | {res['batch_size']:<10} | {res['gen_len']:<7} | {throughput_str:<15} | {res['latency']:.4f}             |")
    print("-------------------------------------------------------------------------------------------------")


if __name__ == "__main__":
    main()
