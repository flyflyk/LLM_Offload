import torch
from transformers import AutoConfig

def oom_check(model_name: str, batch_size: int, max_seq_len: int, dtype: torch.dtype = torch.float16):
    """
    Performs a pre-check to estimate VRAM usage and prevent OOM errors.
    """
    if not torch.cuda.is_available():
        print("INFO - [Pre-check] - No CUDA device found. Skipping VRAM check.")
        return True

    print("INFO - [Pre-check] - Starting OOM risk analysis...")

    torch.cuda.empty_cache()
    available_vram = torch.cuda.mem_get_info()[0]

    try:
        config = AutoConfig.from_pretrained(model_name)
    except Exception as e:
        print(f"WARNING - [Pre-check] - Could not fetch model config for '{model_name}'. Skipping check. Error: {e}")
        return True

    p = 2 if dtype == torch.float16 else 4
    h = config.hidden_size
    num_heads = config.num_attention_heads

    attention_scores_size = batch_size * num_heads * max_seq_len * max_seq_len * p
    ffn_intermediate_size = batch_size * max_seq_len * (h * 4) * p
    peak_activations_estimate = ffn_intermediate_size + attention_scores_size

    largest_layer_size = (h * h * 4 + h * 4 * h) * p
    required_dynamic_vram = peak_activations_estimate + largest_layer_size
    vram_dynamics_budget = available_vram * 0.60

    if required_dynamic_vram > vram_dynamics_budget:
        print("="*60)
        print("⚠️OOM Risk Warning")
        print("="*60)
        print(f"Current model '{model_name}' has a high risk of OOM errors with the given configuration.")
        print(f"\n- Available VRAM: {available_vram / 1e9:.2f} GiB")
        print(f"- Estimated requirements for dynamic caculation: {required_dynamic_vram / 1e9:.2f} GiB (Activations ~{peak_activations_estimate / 1e9:.2f} GiB)")
        print(f"- VRAM dynamic budget: {vram_dynamics_budget / 1e9:.2f} GiB (60% of available VRAM")
        print("\n---")
        print("💡Suggestions:")
        print("---")
        if batch_size > 1:
            print(f"1. Decrease batch size from {batch_size} to 1 (--batch-size 1)")
        print(f"2. Shorten sequence length (Current {max_seq_len})。")
        print("="*60)
        return False

    print(f"INFO - [Pre-check] - VRAM usage analysis passed. Estimated dynamic requirement: {required_dynamic_vram / 1e9:.2f} GiB. Budget: {vram_dynamics_budget / 1e9:.2f} GiB.")
    return True

def check_vram(args, get_model_info):
    """Checks if the model weights can fit into the available VRAM."""
    if not torch.cuda.is_available():
        print("CUDA is not available. Cannot perform VRAM check.")
        return False

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
