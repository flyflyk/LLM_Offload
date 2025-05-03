import torch
import time
import traceback
from model_loader import load_model
from config import CHOSEN_MODEL, MAX_TOKENS, PROMPT_LIST

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

    try:
        print(f"\n--- [main] Generating for prompt: '{prompt}' ---")

        # 1. Tokenize the input
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # 2. VRAM Monitor：Before inference
        if device == torch.device("cuda"):
            torch.cuda.synchronize()
            start_mem_allocated = torch.cuda.memory_allocated(device)
            start_mem_reserved = torch.cuda.memory_reserved(device)
            torch.cuda.reset_peak_memory_stats(device)
            print(f"[main] VRAM Before Generate - Allocated: {start_mem_allocated / (1024**3):.2f} GB, Reserved: {start_mem_reserved / (1024**3):.2f} GB")

        # 3. Generate with the model
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

        if device == torch.device("cuda"):
            torch.cuda.synchronize()
        end_gen_time = time.time()
        total_gen_time = end_gen_time - start_gen_time

        # 4. VRAM monitor：After inference
        if device == torch.device("cuda"):
             peak_mem_allocated = torch.cuda.max_memory_allocated(device)
             peak_mem_reserved = torch.cuda.max_memory_reserved(device)
             end_mem_allocated = torch.cuda.memory_allocated(device)
             print(f"[main] VRAM After Generate - Allocated: {end_mem_allocated / (1024**3):.2f} GB")
             print(f"[main] VRAM Peak During Generate - Allocated: {peak_mem_allocated / (1024**3):.2f} GB, Reserved: {peak_mem_reserved / (1024**3):.2f} GB")
             inference_vram_info = {
                 "peak_allocated_gb": peak_mem_allocated / (1024**3),
                 "peak_reserved_gb": peak_mem_reserved / (1024**3),
                 "allocated_increase_gb": (peak_mem_allocated - start_mem_allocated) / (1024**3)
             }

        # 5. Decode the generated token IDs
        generated_ids = generation_output[0]
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
    print("--- [main] Starting Execution ---")

    # 1. Load the model, tokenizer, and device
    print(f"[main] Loading model '{CHOSEN_MODEL}'...")
    model, tokenizer, device, model_vram_info = load_model(model_name=CHOSEN_MODEL)

    # 2. Check if loading was successful
    if model is None or tokenizer is None or device is None:
        print("[main] Failed to load model or tokenizer. Exiting script.")
        return
    print(f"[main] Model '{CHOSEN_MODEL}' loaded successfully on device: {device}.")
    if model_vram_info:
         print(f"[main] Initial Model VRAM - Allocated: {model_vram_info['after_load_allocated_gb']:.2f} GB, Model Footprint: {model_vram_info['model_footprint_gb']:.2f} GB")

    # 3. Define the list of prompts for inference
    
    all_results = {}
    if model_vram_info and device == torch.device("cuda"):
        all_results["model_vram"] = model_vram_info

    # 4. Run Inference for each prompt
    print("\n--- [main] Running Measured Inference Examples ---")
    for i, prompt in enumerate(PROMPT_LIST):
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
        if inference_vram and device == torch.device("cuda"):
             prompt_metrics["inference_vram"] = inference_vram
        all_results[prompt_label] = prompt_metrics


    # 5. Print Final Summary
    _print_summary(all_results)
    print("\n--- [main] Execution Finished ---")


if __name__ == "__main__":
    main()