import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
import os
import shutil
import logging
from config import ENABLE_OFFLOAD, OFFLOAD_FOLDER, MAX_CPU_OFFLOAD_RAM_GB, DEVICE

logger = logging.getLogger(__name__)


def load_model(model_name: str):
    mode = "Streaming (Layer Offload)" if ENABLE_STREAMING else "Default (Full Load)"
    logger.info(f"Attempting to load model: {model_name} ({mode})")

    initial_vram_allocated = 0
    initial_vram_reserved = 0
    model_vram_allocated = 0
    model_only_vram = 0
    start_load_time = time.time()
    gpu = torch.device(DEVICE)
    
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(gpu)
    initial_vram_allocated = torch.cuda.memory_allocated(gpu)
    initial_vram_reserved = torch.cuda.memory_reserved(gpu)
    logger.info(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')

    model_kwargs = {
        "low_cpu_mem_usage": True,
        "torch_dtype": torch.float16
    }
    
    if ENABLE_STREAMING:
        logger.info("Accelerate layer offloading (device_map='auto') enabled.")
        if os.path.exists(OFFLOAD_FOLDER):
            logger.info(f"Cleaning up existing offload directory: {OFFLOAD_FOLDER}")
            shutil.rmtree(OFFLOAD_FOLDER)
        os.makedirs(OFFLOAD_FOLDER, exist_ok=True)
        logger.info(f"Created offload directory: {OFFLOAD_FOLDER}")

        max_memory_dict = {}
        total_vram = torch.cuda.get_device_properties(0).total_memory
        gpu_limit_bytes = total_vram - min(1 * (1024**3), int(total_vram * 0.1)) # reserve 1GB or 10% 
        gpu_limit_gb = gpu_limit_bytes / (1024**3)
        max_memory_dict[0] = f"{gpu_limit_gb:.0f}GiB"    
        max_memory_dict['cpu'] = f"{MAX_CPU_OFFLOAD_RAM_GB} GiB"

        model_kwargs.update({
            "device_map": "auto",
            "offload_folder": OFFLOAD_FOLDER,
            "max_memory": max_memory_dict
        })
    else:
        logger.info(f"Layer offloading disabled.")
    
    logger.info(f"Loading model weights for {model_name} with kwargs: {model_kwargs}...")
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    model.to(gpu)
    if ENABLE_STREAMING:
        logger.info(f"Model device map (first few layers):\n{str(model.hf_device_map)[:200]}...")
    
    torch.cuda.synchronize(gpu)
    model_vram_allocated = torch.cuda.memory_allocated(gpu)
    model_only_vram = model_vram_allocated - initial_vram_allocated
    model.eval()
    logger.info("Model set to evaluation mode.")

    end_load_time = time.time()
    logger.info(f"Model and tokenizer ready! Loading took {end_load_time - start_load_time:.2f} seconds.")
    logger.info("-" * 30)

    vram_info = {
        "initial_allocated_gb": initial_vram_allocated / (1024**3),
        "initial_reserved_gb": initial_vram_reserved / (1024**3),
        "after_load_allocated_gb": model_vram_allocated / (1024**3),
        "model_footprint_gb": model_only_vram / (1024**3)
    }
    return model, tokenizer, gpu, vram_info
