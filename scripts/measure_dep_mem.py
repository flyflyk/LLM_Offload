import sys
import os
from memory_profiler import profile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

@profile
def measure_startup_memory():
    # --- 1. Init ---
    print("Phase 1: Initial state")
    pass

    # --- 2. Load core dependencies ---
    print("\nPhase 2: Load core dependencies")
    import time
    import logging
    import torch
    import argparse
    import json
    import psutil
    import dataclasses
    from collections import Counter
    from typing import List
    from transformers import AutoTokenizer
    from types import SimpleNamespace

    from src.auto_policy.profiler import get_hardware_profile
    from src.auto_policy.optimizer import Optimizer
    from src.utils.memory import calc_mem_per_device
    
    # --- 3. Load framework ---
    print("\nPhase3: Load framework(flexgen)")
    from flexllmgen.flex_opt import OptLM, Policy, SelfAttention, InputEmbed, MLP, OutputEmbed, ValueHolder, CompressionConfig
    from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
    from flexllmgen.utils import ExecutionEnv
    from flexllmgen.opt_config import get_opt_config
 
    print("\nMeasurement complete.")

if __name__ == "__main__":
    measure_startup_memory()
