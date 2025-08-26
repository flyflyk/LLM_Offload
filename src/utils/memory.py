import torch
import psutil
import numpy as np
from src.accelerate import config
from flexllmgen.flex_opt import InputEmbed, SelfAttention, MLP, OutputEmbed

def calc_mem_per_device(model, device_map: dict) -> dict:
    """
    Calculates the memory usage (in GB) of a FlexGen OptLM model on each device
    based on a given device_map.
    """
    name_to_idx = {}
    for i, layer in enumerate(model.layers):
        if isinstance(layer, InputEmbed):
            name_to_idx["embed_tokens"] = i
        elif isinstance(layer, SelfAttention):
            name_to_idx[f"decoder.layer.{layer.layer_id}.self_attn"] = i
        elif isinstance(layer, MLP):
            name_to_idx[f"decoder.layer.{layer.layer_id}.mlp"] = i
        elif isinstance(layer, OutputEmbed):
            name_to_idx["lm_head"] = i
        else:
            if hasattr(layer, 'layer_id'):
                name_to_idx[f"decoder.layer.{layer.layer_id}"] = i
            else:
                name_to_idx[f"layer.{i}"] = i

    device_sizes = {}
    for device in set(device_map.values()):
        device_sizes[device] = 0

    def get_tensor_size_bytes(tensor):
        # CompressedTorchTensor has a .data attribute which is a list of tensors
        if hasattr(tensor, 'data') and isinstance(tensor.data, list):
            total_size = 0
            for sub_tensor in tensor.data:
                if hasattr(sub_tensor, 'shape') and hasattr(sub_tensor, 'dtype'):
                    total_size += np.prod(sub_tensor.shape) * torch.tensor([], dtype=sub_tensor.dtype).element_size()
            return total_size
        # TorchTensor
        elif hasattr(tensor, 'shape') and hasattr(tensor, 'dtype'):
            return np.prod(tensor.shape) * torch.tensor([], dtype=tensor.dtype).element_size()
        return 0

    for layer_name, device in device_map.items():
        if layer_name not in name_to_idx:
            continue
        
        layer_idx = name_to_idx[layer_name]
        # The weights are in a ValueHolder, need to get .val
        weights = model.weight_home[layer_idx].val
        
        layer_size_bytes = 0
        for weight_tensor in weights:
            layer_size_bytes += get_tensor_size_bytes(weight_tensor)
            
        device_sizes[device] += layer_size_bytes

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