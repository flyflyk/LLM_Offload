import pulp
from src.auto_policy.profiler import HardwareProfile

class CostModel:
    def __init__(self, model_config, hardware: HardwareProfile, input_len: int, gen_len: int):
        self.model_config = model_config
        self.hardware = hardware
        self.input_len = input_len
        self.gen_len = gen_len

    def estimate_latency(self, prob, policy, batch_size, compress_weight: bool, compress_cache: bool):
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
        eff_tflops = self.hardware.tflops_slope * batch_size + self.hardware.tflops_bias
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
        T_pre = pulp.LpVariable(f"T_pre_bs{batch_size}_cw{compress_weight}_cc{compress_cache}", 0)
        ctog_pre = ((w_c + w_d) * weight_size + (h_c + h_d) * activation_size) / cg_bw
        gtoc_pre = ((c_c + c_d) * kv_cache_size + (h_c + h_d) * activation_size) / cg_bw
        dtoc_pre = (w_d * weight_size + h_d * activation_size) / cd_bw
        ctod_pre = (c_d * kv_cache_size + h_d * activation_size) / cd_bw
        prefill_flops = 2 * s * h1**2 * 2 + 2 * s * h1 * h2 # Simplified estimate
        compp = (prefill_flops * batch_size) / flops_per_second
        
        prob += T_pre >= ctog_pre, f"T_pre_constraint_1_bs{batch_size}_cw{compress_weight}_cc{compress_cache}"
        prob += T_pre >= gtoc_pre, f"T_pre_constraint_2_bs{batch_size}_cw{compress_weight}_cc{compress_cache}"
        prob += T_pre >= dtoc_pre, f"T_pre_constraint_3_bs{batch_size}_cw{compress_weight}_cc{compress_cache}"
        prob += T_pre >= ctod_pre, f"T_pre_constraint_4_bs{batch_size}_cw{compress_weight}_cc{compress_cache}"
        prob += T_pre >= compp,    f"T_pre_constraint_5_bs{batch_size}_cw{compress_weight}_cc{compress_cache}"

        # --- Decode Stage Latency (single layer, single token) ---
        T_gen = pulp.LpVariable(f"T_gen_bs{batch_size}_cw{compress_weight}_cc{compress_cache}", 0)
        activation_size_gen = s * h1 * 2 * batch_size / s

        ctog_gen = ((w_c + w_d) * weight_size + (c_c + c_d) * kv_cache_size + (h_c + h_d) * activation_size_gen) / cg_bw
        gtoc_gen = (h_c + h_d) * activation_size_gen / cg_bw
        dtoc_gen = (w_d * weight_size + c_d * kv_cache_size + h_d * activation_size_gen) / cd_bw
        ctod_gen = 0
        decode_flops = (2 * 1 * h1**2 * 2 + 2 * 1 * h1 * h2) * 2 # Matmuls for q,k,v,o and 2 MLP layers
        compg = (decode_flops * batch_size) / flops_per_second

        prob += T_gen >= ctog_gen, f"T_gen_constraint_1_bs{batch_size}_cw{compress_weight}_cc{compress_cache}"
        prob += T_gen >= gtoc_gen, f"T_gen_constraint_2_bs{batch_size}_cw{compress_weight}_cc{compress_cache}"
        prob += T_gen >= dtoc_gen, f"T_gen_constraint_3_bs{batch_size}_cw{compress_weight}_cc{compress_cache}"
        prob += T_gen >= ctod_gen, f"T_gen_constraint_4_bs{batch_size}_cw{compress_weight}_cc{compress_cache}"
        prob += T_gen >= compg,    f"T_gen_constraint_5_bs{batch_size}_cw{compress_weight}_cc{compress_cache}"

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
        h2 = self.model_config.ffn_embed_dim

        # --- GPU Memory Calculation ---
        # Always use uncompressed sizes for GPU peak memory estimation
        weight_size_gpu = (2 * h1**2 + h1 * h2) * 2 * 2 * l
        kv_cache_size_gpu = 2 * l * (s + n) * h1 * 2 * batch_size
        activation_size_gpu = (s * h1 * 2 * batch_size) + (s * h2 * 2 * batch_size) # Input + Output for Residual

        # Attention matrix size for prefill stage (bs, num_heads, seq_len, seq_len)
        attn_matrix_size = batch_size * self.model_config.n_head * s * s * 2 # fp16

        gpu_mem = (w_g * weight_size_gpu +                # Permanently stored weights
                   c_g * kv_cache_size_gpu +                # Permanently stored cache
                   h_g * activation_size_gpu +                # Activations
                   kv_cache_size_gpu / l +                  # Transient KV cache buffer
                   attn_matrix_size +                       # Transient Attention matrix
                   weight_size_gpu / l)                     # Transient buffer for one layer's weights

        # --- CPU Memory Calculation ---
        # Use compressed sizes for components stored on CPU
        weight_size_cpu = (2 * h1**2 + h1 * h2) * 2 * 2 * l
        if compress_weight:
            weight_size_cpu *= 0.25

        kv_cache_size_cpu = 2 * l * (s + n) * h1 * 2 * batch_size
        if compress_cache:
            kv_cache_size_cpu *= 0.25
            
        # Activation size is the same for CPU
        activation_size_cpu = (s * h1 * 2 * batch_size) + (s * h2 * 2 * batch_size) # Input + Output for Residual

        # Transient buffer on CPU for moving data (assume uncompressed)
        transient_buffer_cpu = (weight_size_gpu + kv_cache_size_gpu) / l

        cpu_mem = (w_c * weight_size_cpu + 
                   c_c * kv_cache_size_cpu + 
                   h_c * activation_size_cpu + 
                   transient_buffer_cpu)
                   
        return gpu_mem, cpu_mem
