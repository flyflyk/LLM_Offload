# C:/Users/flyfl/Documents/CodeProjects/py/inference/AutoPolicy/cost_model.py
"""
Cost model for predicting inference latency based on hardware and policy.

This module defines a simplified cost model to estimate the performance of
FlexLLMGen under different offloading policies. It considers I/O time for moving
weights/cache and computation time.
"""

import dataclasses

# --- Safe Imports from FlexLLMGen ---
# These are used for their data structures and do not modify FlexLLMGen behavior.
from FlexLLMGen.flexllmgen.flex_opt import Policy
from FlexLLMGen.flexllmgen.opt_config import get_opt_config, OptConfig

# --- Data Structures ---

@dataclasses.dataclass
class ModelInfo:
    """A dataclass to hold size information about the LLM."""
    name: str
    config: OptConfig
    weight_size_gb: float
    kv_cache_per_token_gb: float

def get_model_info(model_name: str, batch_size: int, max_seq_len: int) -> ModelInfo:
    """
    Calculates and returns size information for a given model.

    Args:
        model_name: The Hugging Face name of the model.
        batch_size: The total batch size for inference.
        max_seq_len: The maximum sequence length (prompt + generation).

    Returns:
        A ModelInfo object containing size details.
    """
    config = get_opt_config(model_name)
    # Note: These are simplified calculations for the cost model.
    weight_size_gb = config.model_bytes() / 1e9
    # KV cache size per token = num_layers * 2 (k/v) * batch_size * hidden_dim * bytes_per_element
    kv_cache_per_token_gb = (config.n_layers * 2 * batch_size * config.input_dim * 2) / 1e9
    
    return ModelInfo(
        name=model_name,
        config=config,
        weight_size_gb=weight_size_gb,
        kv_cache_per_token_gb=kv_cache_per_token_gb
    )

# --- Cost Model Class ---

class CostModel:
    """
    A simplified cost model to predict inference latency.
    """
    def __init__(self, hardware_profile: dict):
        """
        Initializes the cost model with hardware performance data.

        Args:
            hardware_profile: A dictionary from the profiler module.
        """
        self.profile = hardware_profile

    def _get_io_time(self, size_gb: float, placement: str) -> float:
        """Calculates the time to move a given amount of data to the GPU."""
        if placement == 'gpu':
            return 0.0
        elif placement == 'cpu':
            return size_gb / self.profile['cpu_gpu_bandwidth']
        elif placement == 'disk':
            # Assumes a Disk -> CPU -> GPU path.
            return (size_gb / self.profile['disk_cpu_bandwidth']) + \
                   (size_gb / self.profile['cpu_gpu_bandwidth'])
        return float('inf')

    def predict_latency(self, policy: Policy, model_info: ModelInfo, prompt_len: int, gen_len: int) -> float:
        """
        Predicts the total inference latency for a given policy and task.

        Note: This is a simplified model that sums I/O and compute time. A more
        advanced model would use max(io, compute) to account for pipelining overlap.

        Args:
            policy: The FlexLLMGen policy to evaluate.
            model_info: The size information for the model.
            prompt_len: The length of the input prompt.
            gen_len: The number of tokens to generate.

        Returns:
            The predicted total latency in seconds.
        """
        num_layers = model_info.config.n_layers
        
        # --- 1. Prefill Phase Latency ---
        w_size = model_info.weight_size_gb
        w_cpu_part = w_size * (policy.w_cpu_percent / 100.0)
        w_disk_part = w_size * (policy.w_disk_percent / 100.0)
        
        prefill_weight_io_time = self._get_io_time(w_cpu_part, 'cpu') + self._get_io_time(w_disk_part, 'disk')
        
        # A very rough proxy for prefill computation based on weights and prompt length.
        prefill_compute_flops = w_size * 1e9 * prompt_len
        prefill_compute_time = prefill_compute_flops / (self.profile['gpu_tflops'] * 1e12)

        prefill_latency = prefill_weight_io_time + prefill_compute_time

        # --- 2. Decoding Phase Latency (per token) ---
        layer_weight_size = w_size / num_layers
        lw_cpu_part = layer_weight_size * (policy.w_cpu_percent / 100.0)
        lw_disk_part = layer_weight_size * (policy.w_disk_percent / 100.0)
        decode_weight_io_time = self._get_io_time(lw_cpu_part, 'cpu') + self._get_io_time(lw_disk_part, 'disk')

        total_kv_size = model_info.kv_cache_per_token_gb * (prompt_len + gen_len)
        layer_kv_size = total_kv_size / num_layers
        lk_cpu_part = layer_kv_size * (policy.cache_cpu_percent / 100.0)
        lk_disk_part = layer_kv_size * (policy.cache_disk_percent / 100.0)
        decode_kv_io_time = self._get_io_time(lk_cpu_part, 'cpu') + self._get_io_time(lk_disk_part, 'disk')
        
        # A rough proxy for single-token decoding computation.
        decode_compute_flops = layer_weight_size * 1e9 * 2
        decode_compute_time = decode_compute_flops / (self.profile['gpu_tflops'] * 1e12)
        
        per_token_decode_latency = decode_weight_io_time + decode_kv_io_time + decode_compute_time
        total_decode_latency = per_token_decode_latency * gen_len
        
        return prefill_latency + total_decode_latency