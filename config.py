import torch

CHOSEN_MODEL = "facebook/opt-6.7b"
MAX_TOKENS = 40
ENABLE_STREAMING = True
OFFLOAD_FOLDER = "offload_dir"
MAX_CPU_OFFLOAD_RAM_GB = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"