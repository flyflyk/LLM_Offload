import torch
import time
import traceback
from model_loader import load_model
from config import CHOSEN_MODEL, MAX_TOKENS

def _run_inference(
    prompt: str,
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 50
) -> tuple[str | None, float | None]:
    if not prompt:
        print("[main] Prompt cannot be empty.")
        return None, None

    try:
        print(f"\n--- [main] Generating for prompt: '{prompt}' ---")

        # 1. Tokenize the input
        inputs = tokenizer(prompt, return_tensors='pt').to(device)
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Record start time for generation
        start_gen_time = time.time()

        # 2. Generate with the model
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

        # Record end time for generation
        end_gen_time = time.time()
        total_gen_time = end_gen_time - start_gen_time

        # 3. Decode the generated token IDs
        generated_ids = generation_output[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 4. Calculate Token Latency
        input_token_count = input_ids.shape[1]
        output_token_count = generated_ids.shape[-1]
        new_tokens_generated = output_token_count - input_token_count

        token_latency = float('inf') # Default to infinity if no new tokens
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

        return generated_text, token_latency

    except Exception as e:
        print(f"[main] An error occurred during generation: {e}")
        traceback.print_exc()
        return None, None

def _print_latency_summary(results: dict):
    print("\n--- [main] Latency Summary (seconds/token) ---")
    if results:
        for name, lat in results.items():
            if lat is not None and lat != float('inf'):
                print(f"{name}: {lat:.4f}")
            else:
                print(f"{name}: No valid latency data")
    else:
        print("[main] No latency data collected.")
    print("-" * 30)


# --- Main execution function ---
def main():
    print("--- [main] Starting Execution ---")

    # 1. Load the model, tokenizer, and device
    print(f"[main] Loading model '{CHOSEN_MODEL}'...")
    model, tokenizer, device = load_model(model_name=CHOSEN_MODEL)

    # 2. Check if loading was successful
    if model is None or tokenizer is None or device is None:
        print("[main] Failed to load model or tokenizer. Exiting script.")
        return

    print(f"[main] Model '{CHOSEN_MODEL}' loaded successfully on device: {device}.")

    # 3. Define the list of prompts for inference
    prompt_list = [
        "Once upon a time",
        "The recipe for a perfect pizza starts with",
        "Artificial intelligence is",
        "To be or not to be, that"
    ]

    latency_results = {}

    # 4. Run Inference for each prompt
    print("\n--- [main] Running Inference Examples ---")
    for i, prompt in enumerate(prompt_list):
        prompt_label = f"Prompt {i+1}"
        generated_text, token_latency = _run_inference(
            prompt=prompt,
            model=model,
            tokenizer=tokenizer,
            device=device,
            max_new_tokens=MAX_TOKENS
        )
        if token_latency is not None:
             latency_results[prompt_label] = token_latency


    # 5. Print Final Summary
    _print_latency_summary(latency_results)

    print("\n--- [main] Execution Finished ---")


if __name__ == "__main__":
    main()