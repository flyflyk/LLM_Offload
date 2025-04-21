import torch
import time
from model_loader import load_model
from config import CHOSEN_MODEL

# --- Step 1: Load the model ---

model, tokenizer, device = load_model(model_name=CHOSEN_MODEL)

# --- Step 2: Check if loading was successful ---
if model is None or tokenizer is None or device is None:
    print("[main] Failed to load model or tokenizer. Exiting script.")
    exit()

# --- Step 3: Define the inference function ---
def generate(prompt: str, max_new_tokens: int = 50):
    """
    Generate text using the preloaded model and tokenizer, and calculate token latency.
    """
    if not prompt:
        print("[main - generate] Prompt cannot be empty.")
        return None, None

    # Double-check if model, tokenizer, device exist (extra safety)
    if 'model' not in globals() or 'tokenizer' not in globals() or 'device' not in globals():
         print("[main - generate] Error: Model, tokenizer, or device not found in global scope!")
         return None, None

    try:
        print(f"\n--- [main - generate] Generating for prompt: '{prompt}' ---")

        # 1. Tokenize the input
        inputs = tokenizer(prompt, return_tensors='pt').to(device) # Tokenize and move inputs to device
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask

        # Record start time
        start_time = time.time()

        # 2. Generate with the model
        with torch.no_grad():
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

        # Record end time
        end_time = time.time()
        total_time = end_time - start_time

        # 3. Decode the generated token IDs
        generated_ids = generation_output[0]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        # 4. Calculate Token Latency
        input_token_count = input_ids.shape[1]
        # Note: generated_ids shape can be (1, seq_len) or (seq_len,)
        output_token_count = generated_ids.shape[-1]
        new_tokens_generated = output_token_count - input_token_count

        if new_tokens_generated > 0: # Avoid division by zero
            token_latency = total_time / new_tokens_generated
            print(f"[main - generate] Generated {new_tokens_generated} new tokens.")
            print(f"[main - generate] Total generation time: {total_time:.4f} seconds.")
            print(f"[main - generate] Average latency per new token: {token_latency:.4f} seconds/token.")
        else:
            token_latency = float('inf')
            print(f"[main - generate] No new tokens were generated. Input length: {input_token_count}, Output length: {output_token_count}")
            print(f"[main - generate] Processing time (no new tokens): {total_time:.4f} seconds.")


        print(f"[main - generate] Full generated text:\n{generated_text}")
        print("-" * 30)

        return generated_text, token_latency

    except Exception as e:
        print(f"[main - generate] An error occurred during generation: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# --- Step 4: Run Inference Prompts ---
if __name__ == "__main__":
    print("\n--- [main] Running Inference Examples ---")

    prompt_list = [
        "Once upon a time",
        "The recipe for a perfect pizza starts with",
        "Artificial intelligence is",
        "To be or not to be, that"
    ]
    latencies = {}

    # Iterate through prompts
    for i, p in enumerate(prompt_list):
        _, latency = generate(p, max_new_tokens=40)
        # Record latency if valid
        if latency is not None and latency != float('inf'):
            latencies[f"Prompt {i+1}"] = latency

    # Print latency summary
    print("\n--- [main] Latency Summary (seconds/token) ---")
    if latencies:
        for name, lat in latencies.items():
            print(f"{name}: {lat:.4f}")
    else:
        print("No valid latency data collected.")

    print("\n--- [main] Main script finished ---")