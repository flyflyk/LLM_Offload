import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os
import shutil
import logging
from config import ENABLE_STREAMING, OFFLOAD_FOLDER, MAX_CPU_OFFLOAD_RAM_GB, DEVICE

logger = logging.getLogger(__name__)


def load_model(model_name: str):
    mode = "Streaming (Layer Offload)" if ENABLE_STREAMING else "Baseline (Full Load)"
    logger.info(f"Attempting to load model: {model_name} ({mode})")

    initial_vram_allocated = 0
    initial_vram_reserved = 0
    model_vram_allocated = 0
    model_only_vram = 0
    start_load_time = time.time()
    
    preferred_device = torch.device(DEVICE)
    logger.info(f"Preferred device from config: {preferred_device}")

    actual_computation_device = None
    if preferred_device.type == "cuda":
        actual_computation_device = preferred_device
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(actual_computation_device)
        initial_vram_allocated = torch.cuda.memory_allocated(actual_computation_device)
        initial_vram_reserved = torch.cuda.memory_reserved(actual_computation_device)
        logger.info(f"Initial VRAM on {actual_computation_device} - Allocated: {initial_vram_allocated / (1024**3):.2f} GB, Reserved: {initial_vram_reserved / (1024**3):.2f} GB")

    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    logger.info("Tokenizer loaded successfully.")

    model_kwargs = {"low_cpu_mem_usage": True}
    
    if ENABLE_STREAMING:
        logger.info("Accelerate layer offloading (device_map='auto') enabled.")
        if os.path.exists(OFFLOAD_FOLDER):
            logger.info(f"Cleaning up existing offload directory: {OFFLOAD_FOLDER}")
            shutil.rmtree(OFFLOAD_FOLDER)
        os.makedirs(OFFLOAD_FOLDER, exist_ok=True)
        logger.info(f"Created offload directory: {OFFLOAD_FOLDER}")

        max_memory_dict = {}
        if preferred_device.type == "cuda" and torch.cuda.is_available():
            gpu_target_for_max_memory = 0
            try:
                total_vram = torch.cuda.get_device_properties(gpu_target_for_max_memory).total_memory
                gpu_limit_bytes = total_vram - min(1 * (1024**3), int(total_vram * 0.1)) # 1GB or 10% headroom
                gpu_limit_gb = gpu_limit_bytes / (1024**3)
                max_memory_dict[gpu_target_for_max_memory] = f"{gpu_limit_gb:.0f}GiB"
                logger.info(f"Setting max VRAM for GPU {gpu_target_for_max_memory}: {max_memory_dict[gpu_target_for_max_memory]}")
            except Exception as e_mem_prop:
                logger.warning(f"Could not determine VRAM for GPU {gpu_target_for_max_memory} to set max_memory: {e_mem_prop}. Accelerate will use defaults.")
        
        max_memory_dict['cpu'] = f"{MAX_CPU_OFFLOAD_RAM_GB}GiB"
        logger.info(f"Setting max CPU RAM for offload: {max_memory_dict['cpu']}")

        model_kwargs.update({
            "device_map": "auto",
            "offload_folder": OFFLOAD_FOLDER,
            "max_memory": max_memory_dict,
            "torch_dtype": torch.float16
        })
        logger.info(f"Accelerate arguments: {model_kwargs}")
    else:
        logger.info(f"Layer offloading disabled. Loading model fully to preferred device: {preferred_device}.")
        if preferred_device.type == "cuda" and torch.cuda.is_available():
            model_kwargs["torch_dtype"] = torch.float16
    
    logger.info(f"Loading model weights for {model_name} with kwargs: {model_kwargs}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    logger.info("Model object created.")

    model_actual_device = model.device
    logger.info(f"Model's primary device after load: {model_actual_device}")

    if not ENABLE_STREAMING:
        if model_actual_device != preferred_device:
            logger.info(f"Moving model from {model_actual_device} to {preferred_device}...")
            model.to(preferred_device)
            model_actual_device = model.device
            logger.info(f"Model moved to {model_actual_device}.")
    else:
        logger.info("Model placement managed by Accelerate.")
        if hasattr(model, 'hf_device_map'):
             logger.info(f"Model device map (first few layers):\n{str(model.hf_device_map)[:200]}...")
        if model_actual_device.type == "cuda":
            actual_computation_device = model_actual_device
        elif preferred_device.type == "cuda" and torch.cuda.is_available() and actual_computation_device is None:
            actual_computation_device = preferred_device
        elif actual_computation_device is None:
            actual_computation_device = torch.device("cpu")


    final_device = actual_computation_device if actual_computation_device else preferred_device
    
    if final_device.type == "cuda" and torch.cuda.is_available():
        if actual_computation_device and actual_computation_device.type == 'cuda':
            torch.cuda.synchronize(actual_computation_device)
            model_vram_allocated = torch.cuda.memory_allocated(actual_computation_device)
            model_only_vram = model_vram_allocated - initial_vram_allocated
            logger.info(f"After Model Load VRAM on {actual_computation_device} - Allocated: {model_vram_allocated / (1024**3):.2f} GB")
            logger.info(f"Estimated Model VRAM Footprint (Allocated Increase on {actual_computation_device}): {model_only_vram / (1024**3):.2f} GB")
        else:
            logger.warning(f"VRAM reporting mismatch: final_device is {final_device}, but actual_computation_device is {actual_computation_device}")
            model_only_vram = 0
    else:
        logger.info("Preferred/Actual device is CPU or CUDA not available. No GPU VRAM metrics for model load.")
        model_only_vram = 0

    model.eval()
    logger.info("Model set to evaluation mode.")

    end_load_time = time.time()
    logger.info(f"Model and tokenizer ready! Loading took {end_load_time - start_load_time:.2f} seconds.")
    logger.info("-" * 30)

    returned_device = model.device

    vram_info = {
        "initial_allocated_gb": initial_vram_allocated / (1024**3) if final_device.type == "cuda" else 0,
        "initial_reserved_gb": initial_vram_reserved / (1024**3) if final_device.type == "cuda" else 0,
        "after_load_allocated_gb": model_vram_allocated / (1024**3) if final_device.type == "cuda" else 0,
        "model_footprint_gb": model_only_vram / (1024**3)
    }
    return model, tokenizer, returned_device, vram_info