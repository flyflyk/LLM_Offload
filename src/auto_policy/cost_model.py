import pulp
from src.auto_policy.profiler import HardwareProfile

class CostModel:
    def __init__(self, model_config, hardware: HardwareProfile, input_len: int, gen_len: int):
        self.model_config = model_config
        self.hardware = hardware
        self.input_len = input_len
        self.gen_len = gen_len

    def estimate_latency(self, policy, batch_size, compress_weight: bool, compress_cache: bool):
        # Policy variables
        _, w_c, w_d = policy['w_g'], policy['w_c'], policy['w_d']
        _, c_c, c_d = policy['c_g'], policy['c_c'], policy['c_d']
        _, h_c, h_d = policy['h_g'], policy['h_c'], policy['h_d']
        
        # Model and prompt parameters
        l = self.model_config.num_hidden_layers
        n = self.gen_len
        s = self.input_len
        h1 = self.model_config.hidden_size
        h2 = self.model_config.ffn_embed_dim

        # Hardware bandwidths from profiler (bytes/s)
        cg_bw = self.hardware.cpu_gpu_bandwidth
        cd_bw = self.hardware.disk_cpu_bandwidth

        # Compute TFLOPs based on the linear model from profiler
        eff_tflops = self.hardware.tflops_slope * batch_size + self.hardware.tflops_intercept
        # Convert TFLOPs to FLOPs/s
        flops_per_second = eff_tflops * 1e12

        # Average sizes of components for one layer (bytes, FP16)
        weight_size = (2 * h1**2 + h1 * h2) * 2 * 2
        activation_size = s * h1 * 2 * batch_size
        kv_cache_size = 2 * (s + n/2) * h1 * 2 * batch_size

        if compress_weight:
            weight_size *= 0.25
        if compress_cache:
            kv_cache_size *= 0.25

        # --- Prefill Stage Latency (single layer) ---
        # IO latencies
        ctog_pre = ((w_c + w_d) * weight_size + (h_c + h_d) * activation_size) / cg_bw
        gtoc_pre = ((c_c + c_d) * kv_cache_size + (h_c + h_d) * activation_size) / cg_bw
        dtoc_pre = (w_d * weight_size + h_d * activation_size) / cd_bw
        ctod_pre = (c_d * kv_cache_size + h_d * activation_size) / cd_bw

        # Compute latency for prefill
        prefill_flops = 2 * s * h1**2 * 2 + 2 * s * h1 * h2 # Simplified estimate
        compp = (prefill_flops * batch_size) / flops_per_second
        T_pre = max(ctog_pre, gtoc_pre, dtoc_pre, ctod_pre, compp)

        # --- Decode Stage Latency (single layer, single token) ---
        activation_size = activation_size / s

        # IO latencies
        ctog_gen = ((w_c + w_d) * weight_size + (c_c + c_d) * kv_cache_size + (h_c + h_d) * activation_size) / cg_bw
        gtoc_gen = (h_c + h_d) * activation_size / cg_bw
        dtoc_gen = (w_d * weight_size + c_d * kv_cache_size + h_d * activation_size) / cd_bw
        ctod_gen = 0

        # Compute latency for decode
        decode_flops = (2 * 1 * h1**2 * 2 + 2 * 1 * h1 * h2) * 2 # Matmuls for q,k,v,o and 2 MLP layers
        compg = (decode_flops * batch_size) / flops_per_second
        T_gen = max(ctog_gen, gtoc_gen, dtoc_gen, ctod_gen, compg)

        # Total latency
        total_latency = T_pre * l + T_gen * (n - 1) * l
        
        return total_latency

    def get_peak_memory(self, policy, batch_size, compress_weight: bool, compress_cache: bool):
        # Policy variables
        w_g, w_c, _ = policy['w_g'], policy['w_c'], policy['w_d']
        c_g, c_c, _ = policy['c_g'], policy['c_c'], policy['c_d']
        h_g, h_c, _ = policy['h_g'], policy['h_c'], policy['h_d']

        # Model parameters
        l = self.model_config.num_hidden_layers
        s = self.input_len
        n = self.gen_len
        h1 = self.model_config.hidden_size
        h2 = self.model_config.ffn_dim

        # Component sizes (total for all layers, FP16)
        weight_size = (2 * h1**2 + h1 * h2) * 2 * 2 * l
        # Peak activation size during prefill
        activation_size = s * h1 * 2 * l * batch_size
        # Peak KV cache size at the end of generation
        kv_cache_size = 2 * l * (s + n) * h1 * 2 * batch_size

        # Peak memory estimation (Bytes)
        gpu_mem = w_g * weight_size + c_g * kv_cache_size + h_g * activation_size + (kv_cache_size + activation_size) / l
        cpu_mem = w_c * weight_size + c_c * kv_cache_size + h_c * activation_size + (weight_size + kv_cache_size + activation_size) / l
        
        return gpu_mem, cpu_mem
