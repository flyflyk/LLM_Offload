import torch
import psutil
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map
from src.accelerate import config

def oom_check(model_name: str, batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.float16):
    print("INFO - [Pre-check] - Starting OOM risk analysis (device map simulation)...")
    torch.cuda.empty_cache()
    available_vram = torch.cuda.mem_get_info()[0]
    config = AutoConfig.from_pretrained(model_name)

    # Simulate accelerate's device map generation
    static_weights = 0
    meta_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="meta")
    max_memory = get_max_mem_dict()

    # Get the device map
    device_map = infer_auto_device_map(meta_model, max_memory=max_memory, no_split_module_classes=meta_model._no_split_modules)
    device_sizes = get_model_device_mem(meta_model, device_map)
    static_weights = sum(size for device, size in device_sizes.items() if device == 0 or (isinstance(device, str) and 'cuda' in device))

    # Calculate the budget and requirement, adding a 20% safety margin for fragmentation
    vram_budget = (available_vram - static_weights) * 0.8
    p = 2 if dtype == torch.float16 else 4
    h = config.hidden_size
    num_heads = config.num_attention_heads

    attention_size = batch_size * num_heads * max_seq_len * max_seq_len * p
    ffn_size = batch_size * max_seq_len * (h * 4) * p
    peak_activ = ffn_size + attention_size
    max_layer_size = (h * h * 4 + h * 4 * h) * p
    vram_need = peak_activ + max_layer_size

    # Final Decision
    if vram_need > vram_budget:
        print("="*60)
        print("⚠️OOM Risk Warning")
        print("="*60)
        print(f"Current model '{model_name}' has a high risk of OOM errors with the given configuration.")
        print(f"\n- Available VRAM: {available_vram / 1e9:.2f} GiB")
        print(f"- Static weights: {static_weights / 1e9:.2f} GiB")
        print(f"- VRAM remaining: {vram_budget / 1e9:.2f} GiB")
        print(f"- Total budget: {vram_need / 1e9:.2f} GiB (Activations ~{peak_activ / 1e9:.2f} GiB)")
        print("\n---")
        print("💡Suggestions:")
        print("---")
        if batch_size > 1:
            print(f"1. Decrease batch size from {batch_size} to 1 (--batch-size 1)")
        print(f"2. Shorten sequence length (Current {max_seq_len})")
        print("="*60)
        return False

    print(f"INFO - [Pre-check] - VRAM usage analysis passed. VRAM budget: {vram_budget / 1e9:.2f} GiB. Estimated requirement: {vram_need / 1e9:.2f} GiB.")
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

def get_max_mem_dict():
    max_memory = {}
    # VRAM
    if torch.cuda.is_available():
        free_vram_bytes, _ = torch.cuda.mem_get_info(0)
        gpu_mem_gb = int((free_vram_bytes / (1024**3)) * 0.95)
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
