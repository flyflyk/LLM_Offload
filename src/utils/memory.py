import torch
import psutil
from src.accelerate import config

def calc_mem_per_device(model: torch.nn.Module, device_map: dict) -> dict:
    """
    Calculates the memory usage (in GB) of a model on each device based on a given device_map.
    """
    device_sizes = {}
    for device in set(device_map.values()):
        device_sizes[device] = 0

    for layer_name, device in device_map.items():
        module = model.get_submodule(layer_name)
        module_size = sum(p.numel() * p.element_size() for p in module.parameters())
        device_sizes[device] += module_size
    
    # Convert bytes to GB
    for device, size_in_bytes in device_sizes.items():
        device_sizes[device] = f"{size_in_bytes / (1024**3):.4f} GB"
        
    return device_sizes

def get_device_limit():
    max_memory = {}
    # VRAM
    if torch.cuda.is_available():
        total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
        # Leave a small buffer for OS and other processes
        gpu_mem_gb = int((total_vram_bytes / (1024**3)) - 1)
        max_memory[torch.cuda.current_device()] = f"{gpu_mem_gb}GB"

    # RAM
    cpu_mem_config_gb = getattr(config, 'MAX_CPU_OFFLOAD', 0)
    max_ram = 0
    if cpu_mem_config_gb == -1:  # Auto-detect available RAM
        available_ram_bytes = psutil.virtual_memory().available
        max_ram = int((available_ram_bytes / (1024**3)) * 0.95)
    elif cpu_mem_config_gb > 0:  # Use specified RAM limit
        max_ram = cpu_mem_config_gb
    max_memory["cpu"] = f"{max_ram}GB"

    return max_memory