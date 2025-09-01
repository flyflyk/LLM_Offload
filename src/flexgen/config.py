# FlexGen Configuration

# --- Offloading Paths ---
# Path to the model weights cache. This is where FlexGen stores downloaded and converted weights.
PATH = "~/flexgen_cache"

# --- Performance Settings ---
# Use pinned memory for weights on CPU. Setting this to True can speed up CPU-GPU transfers,
# but may fail on systems with limited pinned memory. Set to False if you encounter OOM errors
# when offloading large amounts of weights to RAM.
PIN_WEIGHT = True

# --- Weight Placement ---
# W_GPU_PERCENT + W_CPU_PERCENT <= 100
W_GPU_PERCENT = 100
W_CPU_PERCENT = 0

# --- KV Cache Placement ---
# CACHE_GPU_PERCENT + CACHE_CPU_PERCENT <= 100
CACHE_GPU_PERCENT = 100
CACHE_CPU_PERCENT = 0

# --- Activations Placement ---
# ACT_GPU_PERCENT + ACT_CPU_PERCENT <= 100
ACT_GPU_PERCENT = 100
ACT_CPU_PERCENT = 0
