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
        w_g, w_c, w_d = policy['w_g'], policy['w_c'], policy['w_d']
        c_g, c_c, c_d = policy['c_g'], policy['c_c'], policy['c_d']
        h_g, h_c, h_d = policy['h_g'], policy['h_c'], policy['h_d']
        
        # Model and prompt parameters
        l = self.model_config.num_hidden_layers
        n = self.gen_len
        s = self.input_len
        h1 = self.model_config.hidden_size
        h2 = self.model_config.ffn_embed_dim
        bls = batch_size

        # Hardware parameters from profiler
        cpu_flops = self.hardware.cpu_flops

        # NOTE: Assuming symmetric bandwidths.
        ctog_bdw = self.hardware.cpu_gpu_bandwidth
        gtoc_bdw = self.hardware.cpu_gpu_bandwidth
        dtoc_bdw = self.hardware.disk_cpu_bandwidth
        ctod_bdw = self.hardware.disk_cpu_bandwidth

        # NOTE: Using the same GPU FLOPs model for both mm and bmm.
        eff_tflops = self.hardware.tflops_slope * batch_size + self.hardware.tflops_bias
        mm_flops = eff_tflops * 1e12
        bmm_flops = eff_tflops * 1e12

        # Component sizes
        weight_size_one_layer = (8 * h1**2 + 4 * h1 * h2)
        if compress_weight:
            weight_size_one_layer *= 0.25
        cache_compression_factor = 0.25 if compress_cache else 1.0

        # --- T_pre ---
        T_pre = pulp.LpVariable(f"T_pre_bs{batch_size}_cw{compress_weight}_cc{compress_cache}", 0)

        ctog_p = ((w_c + w_d) * weight_size_one_layer + 2 * (h_c + h_d) * s * h1 * bls) / ctog_bdw
        gtoc_p = (4 * (c_c + c_d) * (s + 1) * h1 * bls * cache_compression_factor + 2 * (h_c + h_d) * s * h1 * bls) / gtoc_bdw
        dtoc_p = (w_d * weight_size_one_layer + 2 * h_d * s * h1 * bls) / dtoc_bdw
        ctod_p = (4 * c_d * bls * (s + 1) * h1 * cache_compression_factor + 2 * h_d * s * h1 * bls) / ctod_bdw
        comp_p = (bls * (8 * s * h1**2 + 4 * s * h1 * h2)) / mm_flops + (4 * bls * s**2 * h1) / bmm_flops

        prob += T_pre >= ctog_p, f"T_pre_ctog_p_{batch_size}_c{compress_weight}_{compress_cache}"
        prob += T_pre >= gtoc_p, f"T_pre_gtoc_p_{batch_size}_c{compress_weight}_{compress_cache}"
        prob += T_pre >= dtoc_p, f"T_pre_dtoc_p_{batch_size}_c{compress_weight}_{compress_cache}"
        prob += T_pre >= ctod_p, f"T_pre_ctod_p_{batch_size}_c{compress_weight}_{compress_cache}"
        prob += T_pre >= comp_p, f"T_pre_comp_p_{batch_size}_c{compress_weight}_{compress_cache}"

        # --- T_gen ---
        T_gen = pulp.LpVariable(f"T_gen_bs{batch_size}_cw{compress_weight}_cc{compress_cache}", 0)

        ctog_g = ((w_c + w_d) * weight_size_one_layer + 2 * (h_c + h_d) * h1 * bls) / ctog_bdw
        gtoc_g = (2 * (h_c + h_d) * h1 * bls) / gtoc_bdw
        dtoc_g = (4 * c_d * bls * (s + n / 2) * h1 * cache_compression_factor + w_d * weight_size_one_layer + 2 * h_d * h1 * bls) / dtoc_bdw
        ctod_g = (4 * c_d * bls * h1 * cache_compression_factor + 2 * h_d * h1 * bls) / ctod_bdw
        gpu_comp_g = (bls * (8 * h1**2 + 4 * h1 * h2)) / mm_flops + (4 * c_g * bls * (s + n / 2) * h1 * cache_compression_factor) / bmm_flops
        cpu_comp_g = (4 * (c_c + c_d) * bls * (s + n / 2) * h1 * cache_compression_factor) / cpu_flops
        comp_g = gpu_comp_g + cpu_comp_g

        prob += T_gen >= ctog_g, f"T_gen_ctog_g_{batch_size}_c{compress_weight}_{compress_cache}"
        prob += T_gen >= gtoc_g, f"T_gen_gtoc_g_{batch_size}_c{compress_weight}_{compress_cache}"
        prob += T_gen >= dtoc_g, f"T_gen_dtoc_g_{batch_size}_c{compress_weight}_{compress_cache}"
        prob += T_gen >= ctod_g, f"T_gen_ctod_g_{batch_size}_c{compress_weight}_{compress_cache}"
        prob += T_gen >= comp_g, f"T_gen_comp_g_{batch_size}_c{compress_weight}_{compress_cache}"

        # Total latency
        total_latency = T_pre * l + T_gen * (n - 1) * l
        return total_latency

    def get_peak_memory(self, policy, batch_size):
        # Policy variables
        w_g, w_c, w_d = policy['w_g'], policy['w_c'], policy['w_d']
        c_g, c_c, c_d = policy['c_g'], policy['c_c'], policy['c_d']
        h_g, h_c, h_d = policy['h_g'], policy['h_c'], policy['h_d']

        # Model parameters
        l = self.model_config.num_hidden_layers
        s = self.input_len
        n = self.gen_len
        h1 = self.model_config.hidden_size
        h2 = self.model_config.ffn_embed_dim
        nh = self.model_config.n_head
        bls = batch_size
        gbs = batch_size # assume global batch size is the same as local batch size

        # --- GPU Peak Memory Expressions ---
        # Prefill
        gpu_home_p = w_g * (8 * h1**2 + 4 * h1 * h2) * l + h_g * 2 * s * h1 * bls + 4 * (s + n) * h1 * c_g * bls * l
        qkv_p = gbs * 8 * s * h1
        att_p_1 = c_g * gbs * (4 * s * h1) + gbs * (2 * nh * s**2)
        att_p_2 = att_p_1
        embed_p = gbs * 4 * s * h1
        mlp_p_1 = 2 * gbs * s * (h1 + h2)
        mlp_p_2 = 2 * gbs * s * (h1 + h2)
        gpu_w_p_base = 2 * (1 - w_g) * (8 * h1**2 + 4 * h1 * h2) + (1 - h_g) * 2 * s * h1 * gbs
        
        working_mem_p_list = [qkv_p, att_p_1, att_p_2, embed_p, mlp_p_1, mlp_p_2]
        gpu_peak_p_list = [gpu_home_p + gpu_w_p_base + item for item in working_mem_p_list]

        # Generation
        gpu_home_g = w_g * (8 * h1**2 + 4 * h1 * h2) * l + h_g * 2 * h1 * bls + 4 * (s + n) * h1 * c_g * bls * l
        qkv_g = 8 * gbs * h1
        att_g_1 = c_g * gbs * (2 * h1 + 2 * (s + n) * h1) + gbs * (2 * nh * (s + n))
        att_g_2 = att_g_1
        embed_g = 4 * gbs * h1
        mlp_g_1 = 2 * gbs * (h1 + h2)
        mlp_g_2 = 2 * gbs * (h2 + h1)
        gpu_w_g_base = 2 * (1 - w_g) * (8 * h1**2 + 4 * h1 * h2) + (1 - h_g) * 2 * s * h1 * gbs
        
        working_mem_g_list = [qkv_g, att_g_1, att_g_2, embed_g, mlp_g_1, mlp_g_2]
        gpu_peak_g_list = [gpu_home_g + gpu_w_g_base + item for item in working_mem_g_list]

        all_gpu_peak_expressions = gpu_peak_p_list + gpu_peak_g_list

        # --- CPU Peak Memory Expressions ---
        # Prefill
        cpu_home_p = w_c * (8 * h1**2 + 4 * h1 * h2) * l + h_c * 2 * s * h1 * bls + 4 * (s + n) * h1 * c_c * bls * l
        cpu_w_p = (1 - w_g) * (8 * h1**2 + 4 * h1 * h2) + (1 - h_g) * 2 * s * h1 * gbs
        cpu_peak_p = cpu_home_p + cpu_w_p

        # Generation
        cpu_home_g = w_c * (8 * h1**2 + 4 * h1 * h2) * l + h_c * 2 * h1 * bls + 4 * (s + n) * h1 * c_c * bls * l
        cpu_w_g = w_d * (8 * h1**2 + 4 * h1 * h2) + h_d * 2 * 2 * h1 * gbs + c_d * 2 * 4 * (s + n) * h1 * gbs + 2 * nh * (s + n) * gbs + 2 * h1 * gbs
        cpu_peak_g = cpu_home_g + cpu_w_g

        all_cpu_peak_expressions = [cpu_peak_p, cpu_peak_g]

        # --- NVMe Peak Memory ---
        nvme_peak = (8 * h1**2 + 4 * h1 * h2) * w_d * l + h_d * 2 * s * h1 * bls + c_d * 4 * (s + n) * h1 * bls * l
        
        return all_gpu_peak_expressions, all_cpu_peak_expressions, nvme_peak