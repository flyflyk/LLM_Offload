import torch
import time
import traceback
from model_loader import load_model
from config import CHOSEN_MODEL, MAX_TOKENS, ENABLE_STREAMING, DEVICE

def _run_inference(
    prompt: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 50
) -> tuple[str | None, float | None, dict | None]:
    if not prompt:
        print("[main] Prompt cannot be empty.")
        return None, None, None

    inference_vram_info = {}
    is_cuda = (device == torch.device("cuda"))

    try:
        print(f"\n--- [main] Generating for prompt: '{prompt}' ---")

        # 1. Tokenize the input
        inputs = tokenizer(prompt, return_tensors='pt')
        print("[main] Inputs tokenized on CPU.")

        try:
            input_embeddings_device = model.get_input_embeddings().weight.device
            print(f"[main] Model expects inputs on device: {input_embeddings_device}")
        except AttributeError:
            print("[main] Warning: Could not determine input embedding device automatically. Using config DEVICE.")
            input_embeddings_device = device

        inputs = inputs.to(input_embeddings_device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        print(f"[main] Inputs moved to device: {input_ids.device}")

        # 2. VRAM Monitor：Before inference
        start_mem_allocated = 0
        if is_cuda:
            torch.cuda.synchronize()
            start_mem_allocated = torch.cuda.memory_allocated(device)
            start_mem_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            print(f"[main] VRAM Before Generate - Allocated: {start_mem_allocated / (1024**3):.2f} GB, Reserved: {start_mem_reserved / (1024**3):.2f} GB")
        
        # 3. Generate with the model
        if is_cuda:
            torch.cuda.synchronize()
        start_gen_time = time.time()

        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=True,
                temperature=1.0,
                top_k=50,
            )

        if is_cuda:
            torch.cuda.synchronize()
        end_gen_time = time.time()
        total_gen_time = end_gen_time - start_gen_time

        # 4. VRAM monitor：After inference
        if is_cuda:
             peak_mem_allocated = torch.cuda.max_memory_allocated(device)
             peak_mem_reserved = torch.cuda.max_memory_reserved(device)
             end_mem_allocated = torch.cuda.memory_allocated(device)
             print(f"[main] VRAM After Generate - Allocated: {end_mem_allocated / (1024**3):.2f} GB")
             print(f"[main] VRAM Peak During Generate - Allocated: {peak_mem_allocated / (1024**3):.2f} GB, Reserved: {peak_mem_reserved / (1024**3):.2f} GB")
             allocated_increase = peak_mem_allocated - start_mem_allocated
             inference_vram_info = {
                 "peak_allocated_gb": peak_mem_allocated / (1024**3),
                 "peak_reserved_gb": peak_mem_reserved / (1024**3),
                 "allocated_increase_gb": allocated_increase / (1024**3)
             }

        # 5. Decode the generated token IDs
        generated_ids = generation_output[0].cpu()
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 6. Calculate Token Latency
        input_token_count = input_ids.shape[1]
        output_token_count = generated_ids.shape[-1]
        new_tokens_generated = output_token_count - input_token_count
        token_latency = None
        if new_tokens_generated > 0:
            token_latency = total_gen_time / new_tokens_generated
            print(f"[main] Generated {new_tokens_generated} new tokens.")
            print(f"[main] Total generation time: {total_gen_time:.4f} seconds.")
            print(f"[main] Average latency per new token: {token_latency:.4f} seconds/token.")
        else:
            print(f"[main] No new tokens were generated. Input length: {input_token_count}, Output length: {output_token_count}")
            print(f"[main] Processing time (no new tokens): {total_gen_time:.4f} seconds.")

        print(f"[main] Full generated text:\n{generated_text}")
        print("-" * 30)

        return generated_text, token_latency, inference_vram_info

    except Exception as e:
        print(f"[main] An error occurred during generation: {e}")
        traceback.print_exc()
        return None, None, None

def _print_summary(results: dict):
    print("\n--- [main] Results Summary ---")
    if not results:
        print("[main] No results collected.")
        return

    if "model_vram" in results:
         print("Model VRAM Footprint:")
         for key, value in results["model_vram"].items():
             if "gb" in key:
                 print(f"  {key}: {value:.4f} GB")
         print("-" * 15)

    print("Inference Metrics per Prompt:")
    for prompt_label, metrics in results.items():
         if prompt_label == "model_vram": continue

         print(f"{prompt_label}:")
         latency = metrics.get("latency")
         if latency is not None and latency != float('inf'):
             print(f"  Latency (sec/token): {latency:.4f}")
         else:
             print(f"  Latency: No valid data")

         inf_vram = metrics.get("inference_vram")
         if inf_vram:
             print(f"  Inference Peak VRAM Allocated Increase: {inf_vram.get('allocated_increase_gb', 'N/A'):.4f} GB")
             print(f"  Inference Peak VRAM Allocated Total: {inf_vram.get('peak_allocated_gb', 'N/A'):.4f} GB")
         else:
              print(f"  Inference VRAM: No data (likely CPU)")

    print("-" * 30)


# --- Main execution function ---
def main():
    mode = "Streaming (Disk Offload)" if ENABLE_STREAMING else "Baseline (Full Load)"
    print(f"--- [main] Starting Execution ({mode}) ---")

    # 1. Load model and tokenizer
    print(f"[main] Loading model '{CHOSEN_MODEL}'...")
    model, tokenizer, device, model_vram_info = load_model(model_name=CHOSEN_MODEL)
    if model is None or tokenizer is None or device is None:
        print("[main] Failed to load model or tokenizer. Exiting script.")
        return
    print(f"[main] Model '{CHOSEN_MODEL}' loaded. Main computation device: {device}.")
    if model_vram_info and device == torch.device("cuda"):
         print(f"[main] Initial Model VRAM (After Load) - Allocated: {model_vram_info['after_load_allocated_gb']:.2f} GB, Initial Increase: {model_vram_info['model_footprint_gb']:.2f} GB")

    # 2. Define a list of prompts
    prompt_list = [
        "Once upon a time, in a land far, far away",
        "The secret ingredient to the world's best chocolate cake is",
        "Explain the concept of quantum entanglement in simple terms:",
        "Write a short poem about a rainy day in the city.",
    ]

    all_results = {}
    is_cuda = (device == torch.device("cuda"))
    if model_vram_info and is_cuda:
        all_results["model_vram"] = model_vram_info

    # 3. Inference for each prompt
    print("\n--- [main] Running Measured Inference Examples ---")
    for i, prompt in enumerate(prompt_list):
        prompt_label = f"Prompt {i+1}"
        generated_text, token_latency, inference_vram = _run_inference(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=MAX_TOKENS
        )
        prompt_metrics = {}
        if token_latency is not None:
             prompt_metrics["latency"] = token_latency
        if inference_vram and is_cuda:
             prompt_metrics["inference_vram"] = inference_vram
        all_results[prompt_label] = prompt_metrics


    # 4. Print summary
    _print_summary(all_results)
    print(f"\n--- [main] Execution Finished ({mode}) ---")


if __name__ == "__main__":
    main()