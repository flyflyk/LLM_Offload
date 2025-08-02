import dataclasses

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

class CostModel:
    def __init__(self, hardware_profile: dict):
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