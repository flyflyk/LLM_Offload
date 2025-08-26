import torch
import psutil
import numpy as np
from src.accelerate import config
from flexllmgen.flex_opt import ValueHolder

def calc_mem_per_device(model) -> dict:
    """
    Calculates the memory usage of model on each device.
    """
    device_sizes = {}

    def get_tensor_size(tensor):
        if hasattr(tensor, 'data') and isinstance(tensor.data, list):
            total_size = 0
            for sub_tensor in tensor.data:
                if hasattr(sub_tensor, 'shape') and hasattr(sub_tensor, 'dtype'):
                    total_size += np.prod(sub_tensor.shape) * torch.tensor([], dtype=sub_tensor.dtype).element_size()
            return total_size
        elif hasattr(tensor, 'shape') and hasattr(tensor, 'dtype'):
            return np.prod(tensor.shape) * torch.tensor([], dtype=tensor.dtype).element_size()
        return 0

    def get_all_tensors(value_holder):
        all_tensors = []
        items = value_holder.val
        if not isinstance(items, (list, tuple)):
            items = [items]

        for item in items:
            if hasattr(item, 'device'): # It's a tensor-like object
                all_tensors.append(item)
            elif isinstance(item, ValueHolder): # It's a nested ValueHolder
                all_tensors.extend(get_all_tensors(item))
        return all_tensors

    for i, _ in enumerate(model.layers):
        all_weight_tensors = get_all_tensors(model.weight_home[i])
        
        for weight_tensor in all_weight_tensors:
            device = weight_tensor.device.name
            if device not in device_sizes:
                device_sizes[device] = 0
            
            tensor_size = get_tensor_size(weight_tensor)
            device_sizes[device] += tensor_size

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