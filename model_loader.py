import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import traceback
import os
import shutil
from config import ENABLE_STREAMING, OFFLOAD_FOLDER, MAX_CPU_OFFLOAD_RAM_GB, DEVICE

def load_model(model_name: str):
    mode = "Streaming (Disk Offload)" if ENABLE_STREAMING else "Baseline (Full Load)"
    print(f"\n--- [model_loader] Attempting to load model: {model_name} ({mode}) ---")

    initial_vram_allocated = 0
    initial_vram_reserved = 0
    model_vram_allocated = 0
    model_vram_reserved = 0
    model_only_vram = 0
    start_load_time = time.time()

    try:
        # 1. Get Device
        device = torch.device(DEVICE)
        print(f"[model_loader] Using device: {device}")

        # --- VRAM State Monitor ---
        if device == torch.device("cuda"):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            initial_vram_allocated = torch.cuda.memory_allocated(device)
            initial_vram_reserved = torch.cuda.memory_reserved(device)
            print(f"[model_loader] Initial VRAM - Allocated: {initial_vram_allocated / (1024**3):.2f} GB, Reserved: {initial_vram_reserved / (1024**3):.2f} GB")
        # ------------------------------------

        # 2. Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        print("[model_loader] Tokenizer loaded successfully.")

        # 3. Load Model
        model_kwargs = {}
        if ENABLE_STREAMING:
            print("[model_loader] Streaming enabled. Using accelerate for disk offload.")
            if os.path.exists(OFFLOAD_FOLDER):
                print(f"[model_loader] Cleaning up existing offload directory: {OFFLOAD_FOLDER}")
                shutil.rmtree(OFFLOAD_FOLDER)
            os.makedirs(OFFLOAD_FOLDER, exist_ok=True)
            print(f"[model_loader] Created offload directory: {OFFLOAD_FOLDER}")

            max_memory_dict = {}
            if device == torch.device("cuda"):
                total_vram = torch.cuda.get_device_properties(device).total_memory
                gpu_limit_bytes = max(1 * (1024**3), total_vram * 0.9)
                gpu_limit_gb = gpu_limit_bytes / (1024**3)
                max_memory_dict[0] = f"{gpu_limit_gb:.0f}GiB" # GPU device index is 0
                print(f"[model_loader] Explicitly setting max VRAM for GPU 0: {max_memory_dict[0]}")
            else:
                 print("[model_loader] CUDA not detected, cannot set GPU memory limit.")

            max_memory_dict['cpu'] = f"{MAX_CPU_OFFLOAD_RAM_GB}GiB"
            print(f"[model_loader] Setting max CPU RAM for offload: {max_memory_dict['cpu']}")

            model_kwargs = {
                "device_map": "auto",
                "offload_folder": OFFLOAD_FOLDER,
                "max_memory": max_memory_dict,
                "low_cpu_mem_usage": True,
                "torch_dtype": torch.float16
            }
            print(f"[model_loader] Accelerate arguments: {model_kwargs}")

        else:
            print("[model_loader] Streaming disabled. Loading model fully to device.")
            if device == torch.device("cuda"):
                try:
                    compute_capability = torch.cuda.get_device_capability(device)
                    if compute_capability[0] >= 7:
                        print("[model_loader] GPU supports float16. Using torch_dtype=torch.float16.")
                        model_kwargs["torch_dtype"] = torch.float16
                    else:
                        print("[model_loader] GPU compute capability < 7.0. Using default float32.")
                except Exception as cap_e:
                     print(f"[model_loader] Warning: Could not determine GPU compute capability: {cap_e}. Using default float32.")

        # 4. Load Model Weights
        print("[model_loader] Loading model weights...")
        model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        print("[model_loader] Model object created.")

        # 5. Move model to device
        if not ENABLE_STREAMING:
            model.to(device)
            print(f"[model_loader] Model moved to {device}.")
        else:
            print("[model_loader] Model placement managed by Accelerate.")
            try:
                 print(f"[model_loader] Model device map:\n{model.hf_device_map}")
            except AttributeError:
                 print("[model_loader] Could not retrieve model.hf_device_map (Might be fully on one device).")


        # --- VRAM State Monitor: After loaded ---
        if device == torch.device("cuda"):
            torch.cuda.synchronize()
            model_vram_allocated = torch.cuda.memory_allocated(device)
            model_vram_reserved = torch.cuda.memory_reserved(device)
            model_only_vram = model_vram_allocated - initial_vram_allocated
            print(f"[model_loader] After Model Load VRAM - Allocated: {model_vram_allocated / (1024**3):.2f} GB, Reserved: {model_vram_reserved / (1024**3):.2f} GB")
            print(f"[model_loader] Estimated Initial VRAM Footprint (Allocated Increase): {model_only_vram / (1024**3):.2f} GB")
        # -------------------------------------------

        # 6. Set model to evaluation mode
        model.eval()
        print("[model_loader] Model set to evaluation mode.")

        # 7. Load Complete
        end_load_time = time.time()
        print(f"[model_loader] Model and tokenizer ready! Loading took {end_load_time - start_load_time:.2f} seconds.")
        print("-" * 30)

        # 8. Return objects and VRAM info
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