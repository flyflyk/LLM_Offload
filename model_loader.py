import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import traceback

def load_model(model_name: str):
    print(f"\n--- [model_loader] Attempting to load model: {model_name} ---")
    initial_vram_allocated = 0
    initial_vram_reserved = 0
    model_vram_allocated = 0
    model_vram_reserved = 0
    model_only_vram = 0

    try:
        start_load_time = time.time()

        # 1. Determine device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[model_loader] Using device: {device}")
        if device == torch.device("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            initial_vram_allocated = torch.cuda.memory_allocated(device)
            initial_vram_reserved = torch.cuda.memory_reserved(device)
            print(f"[model_loader] Initial VRAM - Allocated: {initial_vram_allocated / (1024**3):.2f} GB, Reserved: {initial_vram_reserved / (1024**3):.2f} GB")


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

        # 6. Record VRAM status after loaded model
        if device == torch.device("cuda"):
            model_vram_allocated = torch.cuda.memory_allocated(device)
            model_vram_reserved = torch.cuda.memory_reserved(device)
            model_only_vram = model_vram_allocated - initial_vram_allocated
            print(f"[model_loader] After Model Load VRAM - Allocated: {model_vram_allocated / (1024**3):.2f} GB, Reserved: {model_vram_reserved / (1024**3):.2f} GB")
            print(f"[model_loader] Estimated Model VRAM Footprint: {model_only_vram / (1024**3):.2f} GB")

        # 7. Set model to evaluation mode
        model.eval()
        print("[model_loader] Model set to evaluation mode.")

        # 8. Loading complete
        end_load_time = time.time()
        print(f"[model_loader] Model and tokenizer ready! Loading took {end_load_time - start_load_time:.2f} seconds.")
        print("-" * 30)

        # 9. Return the loaded objects and VRAM info
        vram_info = {
            "initial_allocated_gb": initial_vram_allocated / (1024**3),
            "initial_reserved_gb": initial_vram_reserved / (1024**3),
            "after_load_allocated_gb": model_vram_allocated / (1024**3),
            "after_load_reserved_gb": model_vram_reserved / (1024**3),
            "model_footprint_gb": model_only_vram / (1024**3) if device == torch.device("cuda") else 0
        }
        return model, tokenizer, device, vram_info

    except Exception as e:
        print(f"[model_loader] Error loading model or tokenizer: {e}")
        traceback.print_exc()
        print("-" * 30)
        return None, None, None, None