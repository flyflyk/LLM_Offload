import torch
import psutil
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map
from src.accelerate import config

def oom_check(model_name: str, device_map: dict, batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.float16):
    print("INFO - [Pre-check] - Starting OOM risk analysis...")
    torch.cuda.empty_cache()
    available_vram = torch.cuda.mem_get_info()[0]
    model_config = AutoConfig.from_pretrained(model_name)

    # Calculate static weights based on the provided device_map
    meta_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="meta")
    
    vram_weights = 0
    other_layers_sizes = []
    for layer_name, device in device_map.items():
        module = meta_model.get_submodule(layer_name)
        module_size = sum(p.numel() * p.element_size() for p in module.parameters())
        if isinstance(device, int) or 'cuda' in str(device):
            vram_weights += module_size
        else:
            other_layers_sizes.append(module_size)
    
    # The space needed on VRAM is the sum of layers on VRAM plus the largest layer that needs to be loaded in.
    max_other_layer = max(other_layers_sizes) if other_layers_sizes else 0
    static_weights = vram_weights + max_other_layer

    # Calculate the budget and requirement
    # Subtract 1GB for safety margin and other overheads
    vram_budget = (available_vram - static_weights) - 1 * 1e9 
    p = 2 if dtype == torch.float16 else 4
    h = model_config.hidden_size
    num_heads = model_config.num_attention_heads

    # Peak activation memory
    attention_size = batch_size * num_heads * max_seq_len * max_seq_len * p
    ffn_size = batch_size * max_seq_len * (h * 4) * p
    peak_activ = ffn_size + attention_size

    # Calculate KV Cache size
    num_layers = model_config.num_hidden_layers
    kv_cache_size = num_layers * 2 * batch_size * max_seq_len * h * p
    vram_need = peak_activ + kv_cache_size

    # Final Decision
    if vram_need > vram_budget:
        print("="*60)
        print("⚠️OOM Risk Warning")
        print("="*60)
        print(f"Current model '{model_name}' has a high risk of OOM errors with the given configuration.")
        print(f"\n- VRAM Budget (Available - Static Weights - Safety Margin): {vram_budget / 1e9:.2f} GiB")
        print(f"  - Available VRAM: {available_vram / 1e9:.2f} GiB")
        print(f"  - Static Weights (on VRAM + largest offloaded layer): {static_weights / 1e9:.2f} GiB")
        print(f"- VRAM Dynamic Required (Activations + KV Cache): {vram_need / 1e9:.2f} GiB")
        print(f"  - Peak Activations: {peak_activ / 1e9:.2f} GiB")
        print(f"  - KV Cache: {kv_cache_size / 1e9:.2f} GiB")
        print("\n---")
        print("💡Suggestions:")
        print("---")
        if batch_size > 1:
            print(f"1. Decrease batch size from {batch_size} to 1 (--batch-size 1)")
        print(f"2. Shorten sequence length (Current {max_seq_len})")
        print("="*60)
        return False

    print(f"INFO - [Pre-check] - VRAM usage analysis passed. VRAM budget: {vram_budget / 1e9:.2f} GiB. Estimated requirement: {vram_need / 1e9:.2f} GiB (including KV Cache).")
    return True

def check_vram(args, get_model_info):
    """Checks if the model weights can fit into the available VRAM."""
    print("--- Performing VRAM Pre-check for All-GPU Policy ---")
    model_info = get_model_info(args.model, 1, 1)
    model_size_gb = model_info.weight_size_gb
    free_vram_bytes, _ = torch.cuda.mem_get_info(0)
    free_vram_gb = free_vram_bytes / (1024**3)

    print(f"Estimated Model Size: {model_size_gb:.2f} GB")
    print(f"Available VRAM: {free_vram_gb:.2f} GB")

    if model_size_gb > free_vram_gb * 0.95:
        print("Model is too large to fit entirely in VRAM.")
        return False
    
    print("Model should fit in VRAM.")
    return True

def get_model_device_mem(model: torch.nn.Module, device_map: dict) -> dict:
    """
    Calculates the memory usage of a model on each device based on a given device_map.
    """
    device_sizes = {}
    for device in set(device_map.values()):
        device_sizes[device] = 0

    for layer_name, device in device_map.items():
        module = model.get_submodule(layer_name)
        module_size = sum(p.numel() * p.element_size() for p in module.parameters())
        device_sizes[device] += module_size
        
    return device_sizes

def get_auto_memory_map():
    """
    Generates a memory map for Accelerate's device_map='auto'.
    It uses the total memory of each device as a guideline.
    """
    max_memory = {}
    # VRAM
    if torch.cuda.is_available():
        total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
        # Leave a small buffer (e.g., 1 GB) for OS and other processes
        gpu_mem_gb = int((total_vram_bytes / (1024**3)) - 1)
        max_memory[torch.cuda.current_device()] = f"{gpu_mem_gb}GB"

    # RAM
    cpu_mem_config_gb = getattr(config, 'OFFLOAD_FOLDER_MAX_CPU_OFFLOAD_RAM_GB', 0)
    max_ram = 0
    if cpu_mem_config_gb == -1:  # Auto-detect available RAM
        available_ram_bytes = psutil.virtual_memory().available
        max_ram = int((available_ram_bytes / (1024**3)) * 0.95)
    elif cpu_mem_config_gb > 0:  # Use specified RAM limit
        max_ram = cpu_mem_config_gb
    max_memory["cpu"] = f"{max_ram}GB"

    return max_memory