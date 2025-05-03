import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

def load_model(model_name: str = "gpt2-medium"):
    print(f"\n--- [model_loader] Attempting to load model: {model_name} ---")
    start_load_time = time.time()

    # 1. Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[model_loader] Using device: {device}")

    try:
        # 2. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("[model_loader] Tokenizer loaded successfully.")

        # 3. Prepare model loading arguments
        model_kwargs = {}
        if device == torch.device("cuda"):
            try:
                compute_capability = torch.cuda.get_device_capability(device)
                if compute_capability[0] >= 7: # Compute capability 7.0 and above support float16
                    print("[model_loader] GPU supports float16. Using torch_dtype=torch.float16.")
                    model_kwargs["torch_dtype"] = torch.float16
                else:
                    print("[model_loader] GPU compute capability < 7.0. Using default float32.")
            except Exception as cap_e:
                 print(f"[model_loader] Warning: Could not determine GPU compute capability: {cap_e}. Using default float32.")


        # 4. Load model weights
        print("[model_loader] Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        print("[model_loader] Model weights loaded successfully.")

        # 5. Move model to the target device (GPU or CPU)
        model.to(device)
        print(f"[model_loader] Model moved to {device}.")

        # 6. Set model to evaluation mode
        model.eval()
        print("[model_loader] Model set to evaluation mode.")

        # --- 7. Loading complete ---
        end_load_time = time.time()
        print(f"[model_loader] Model and tokenizer ready! Loading took {end_load_time - start_load_time:.2f} seconds.")
        print("-" * 30)

        # --- 8. Return the loaded objects ---
        return model, tokenizer, device

    except Exception as e:
        print(f"[model_loader] Error loading model or tokenizer: {e}")
        import traceback
        traceback.print_exc()
        print("-" * 30)
        return None, None, None