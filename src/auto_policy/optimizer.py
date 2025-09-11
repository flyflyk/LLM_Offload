import logging
import psutil
import pulp
from tqdm import tqdm

from FlexLLMGen.flexllmgen.flex_opt import Policy, CompressionConfig
from .cost_model import get_model_info
from .profiler import HardwareProfile

logger = logging.getLogger(__name__)

def get_optimial_policy(
    model_name: str,
    hardware_profile: HardwareProfile,
    input_len: int,
    gen_len: int,
    max_batch_size: int = 4096,
) -> Policy:
    logger.info("Searching for the optimal policy using Linear Programming...")

    # --- Throughput Prediction Correction ---
    # Data: (BS=4, TP=11.96), (BS=120, TP=282.06)
    eff_tflops_slope = 0.0305
    eff_tflops_intercept = 0.035

    best_policy = None
    max_throughput = 0.0
    best_batch_size = 0
    best_num_copy_threads = 4  # Default value

    # --- 修正：更保守的壓縮係數和記憶體使用率 ---
    compression_factors = {
        'weight': 2.2,  # 更保守的權重壓縮係數
        'cache': 2.0,   # KV cache 壓縮係數通常較低
    }
    gpu_memory_safety_factor = 0.80  # 降低到 80% 以留更多緩衝
    
    # --- 修正：從小批次開始搜尋，找到第一個可行解後再優化 ---
    feasible_batch_sizes = []
    
    # 第一階段：找出所有可行的批次大小
    for batch_size in tqdm(
        range(4, max_batch_size + 1, 4), desc="Finding Feasible Batch Sizes"
    ):
        if is_batch_size_feasible(batch_size, model_name, hardware_profile, 
                                 input_len, gen_len, compression_factors, 
                                 gpu_memory_safety_factor):
            feasible_batch_sizes.append(batch_size)
    
    if not feasible_batch_sizes:
        logger.error("No feasible batch size found. Model too large for available hardware.")
        return None, 4
    
    logger.info(f"Found {len(feasible_batch_sizes)} feasible batch sizes: {feasible_batch_sizes[:10]}...")
    
    # 第二階段：在可行批次大小中找最優解
    for batch_size in tqdm(feasible_batch_sizes, desc="Optimizing Among Feasible Sizes"):
        model_info = get_model_info(model_name)
        config = model_info.config
        num_layers = config.num_hidden_layers

        # --- Define Base Model Component Sizes ---
        base_weight_size = config.model_bytes()
        base_hidden_state_size = batch_size * config.input_dim * (input_len + gen_len) * 2  # FP16
        base_kv_cache_size = (
            batch_size * num_layers * config.n_head *
            (input_len + gen_len) * (config.input_dim // config.n_head) * 2 * 2 # k/v, FP16
        )

        # --- Try strategies in order of: None -> Cache-only -> Full Compression ---
        for strategy_idx, (compress_w, compress_c) in enumerate([(False, False), (False, True), (True, True), (True, False)]):
            # --- Setup sizes with corrected compression factors ---
            total_weight_size = base_weight_size / compression_factors['weight'] if compress_w else base_weight_size
            total_kv_cache_size = base_kv_cache_size / compression_factors['cache'] if compress_c else base_kv_cache_size
            total_hidden_state_size = base_hidden_state_size

            size_w = total_weight_size / num_layers
            size_h = total_hidden_state_size
            size_c = total_kv_cache_size / num_layers

            # --- Define Linear Programming Problem ---
            prob = pulp.LpProblem(f"FlexGen_Offloading_{batch_size}_{strategy_idx}", pulp.LpMinimize)
            var_names = ["w_gpu", "w_cpu", "w_disk", "c_gpu", "c_cpu", "c_disk", "h_gpu", "h_cpu", "h_disk"]
            cache_cat = pulp.LpBinary if compress_c else pulp.LpContinuous
            vars = {
                name: pulp.LpVariable(
                    f"placement_{batch_size}_{strategy_idx}_{name}",
                    lowBound=0, upBound=1,
                    cat=cache_cat if name.startswith('c_') else pulp.LpContinuous
                ) for name in var_names
            }

            # --- Objective & Constraints ---
            T_cpu_to_gpu = 1 / hardware_profile.cpu_gpu_bandwidth
            T_disk_to_gpu = 1 / hardware_profile.disk_cpu_bandwidth + T_cpu_to_gpu
            prob += (size_w * (vars["w_cpu"] * T_cpu_to_gpu + vars["w_disk"] * T_disk_to_gpu) +
                     size_c * (vars["c_cpu"] * T_cpu_to_gpu + vars["c_disk"] * T_disk_to_gpu) +
                     size_h * (vars["h_cpu"] * T_cpu_to_gpu + vars["h_disk"] * T_disk_to_gpu)), "Total_Transfer_Time"
            prob += vars["w_gpu"] + vars["w_cpu"] + vars["w_disk"] == 1, f"Weight_Completeness_{strategy_idx}"
            prob += vars["c_gpu"] + vars["c_cpu"] + vars["c_disk"] == 1, f"Cache_Completeness_{strategy_idx}"
            prob += vars["h_gpu"] + vars["h_cpu"] + vars["h_disk"] == 1, f"Hidden_State_Completeness_{strategy_idx}"

            # --- 修正：更精確的 GPU 記憶體使用估算 ---
            weight_buffer = base_weight_size / num_layers
            # 考慮 MLP 層的記憶體需求（通常是隱藏維度的 4-8 倍）
            mlp_expansion_ratio = getattr(config, 'intermediate_size', 4 * config.input_dim) / config.input_dim
            mlp_buffer = batch_size * config.input_dim * mlp_expansion_ratio * 2  # FP16
            
            # 額外的運算緩衝區和碎片
            attention_buffer = batch_size * config.n_head * (input_len + gen_len) * (input_len + gen_len) * 2  # attention scores
            misc_buffer = 0.1 * (weight_buffer + mlp_buffer)  # 10% 額外緩衝
            
            peak_buffer = weight_buffer + mlp_buffer + attention_buffer + misc_buffer

            gpu_memory_constraint = ((total_weight_size * vars["w_gpu"]) +
                                   (size_c * vars["c_gpu"]) +
                                   (total_hidden_state_size * vars["h_gpu"]) +
                                   peak_buffer
            ) <= hardware_profile.gpu_mem * gpu_memory_safety_factor
            
            prob += gpu_memory_constraint, f"GPU_Capacity_{strategy_idx}"
            
            prob += ((total_weight_size * vars["w_cpu"]) +
                     (total_kv_cache_size * vars["c_cpu"]) +
                     (total_hidden_state_size * vars["h_cpu"])
            ) <= hardware_profile.cpu_mem * 0.9, f"CPU_Capacity_{strategy_idx}"  # CPU 也加入安全係數

            # --- Solve and Evaluate ---
            prob.solve(pulp.PULP_CBC_CMD(msg=False))

            if pulp.LpStatus[prob.status] == "Optimal":
                min_transfer_time = pulp.value(prob.objective)
                H = config.input_dim
                S = input_len + gen_len
                layer_flops = batch_size * (24 * H**2 + 4 * S * H)
                effective_tflops = eff_tflops_slope * batch_size + eff_tflops_intercept
                T_compute_gpu = layer_flops / (effective_tflops * 1e12 + 1e-10)
                total_latency = (T_compute_gpu + min_transfer_time) * num_layers
                throughput = batch_size / total_latency if total_latency > 0 else 0

                if throughput > max_throughput:
                    max_throughput = throughput
                    best_batch_size = batch_size
                    physical_cores = psutil.cpu_count(logical=False)
                    best_num_copy_threads = max(1, min(physical_cores // 2, 4))
                    
                    best_policy = Policy(
                        gpu_batch_size=batch_size,
                        num_gpu_batches=1,
                        w_gpu_percent=vars["w_gpu"].varValue * 100,
                        w_cpu_percent=vars["w_cpu"].varValue * 100,
                        cache_gpu_percent=vars["c_gpu"].varValue * 100,
                        cache_cpu_percent=vars["c_cpu"].varValue * 100,
                        act_gpu_percent=vars["h_gpu"].varValue * 100,
                        act_cpu_percent=vars["h_cpu"].varValue * 100,
                        overlap=True, sep_layer=True, pin_weight=True,
                        cpu_cache_compute=False, attn_sparsity=1.0,
                        compress_weight=compress_w,
                        comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
                        compress_cache=compress_c,
                        comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False),
                    )

    if best_policy:
        logger.info(f"Found best policy with a throughput of {max_throughput:.2f} tokens/sec "
              f"at a batch size of {best_batch_size}.")
        
        # --- 修正：添加記憶體使用驗證 ---
        estimated_gpu_usage = estimate_actual_gpu_usage(best_policy, model_name, input_len, gen_len, compression_factors)
        gpu_usage_percent = (estimated_gpu_usage / hardware_profile.gpu_mem) * 100
        logger.info(f"Estimated GPU memory usage: {estimated_gpu_usage/1e9:.2f}GB ({gpu_usage_percent:.1f}%)")
        
        if gpu_usage_percent > 95:
            logger.error(f"Critical: GPU memory usage is {gpu_usage_percent:.1f}%, this will cause OOM!")
            return None, 4
        elif gpu_usage_percent > 85:
            logger.warning(f"GPU memory usage is high ({gpu_usage_percent:.1f}%), consider reducing batch size.")
            
    else:
        logger.warning("Could not find a feasible policy. The model may be too large for the available hardware.")

    return best_policy, best_num_copy_threads


def is_batch_size_feasible(batch_size, model_name, hardware_profile, input_len, gen_len, 
                          compression_factors, gpu_memory_safety_factor):
    """快速檢查給定批次大小是否在記憶體限制內可行"""
    model_info = get_model_info(model_name)
    config = model_info.config
    
    # 估算最小記憶體需求（假設最激進的卸載策略）
    base_weight_size = config.model_bytes()
    base_hidden_state_size = batch_size * config.input_dim * (input_len + gen_len) * 2
    base_kv_cache_size = (
        batch_size * config.num_hidden_layers * config.n_head *
        (input_len + gen_len) * (config.input_dim // config.n_head) * 2 * 2
    )
    
    # 最少需要保留在 GPU 的內容（考慮最激進壓縮）
    min_gpu_weight = base_weight_size / (config.num_hidden_layers * compression_factors['weight'])  # 至少一層權重
    min_gpu_cache = base_kv_cache_size / (config.num_hidden_layers * compression_factors['cache'])   # 至少一層 cache
    min_gpu_hidden = base_hidden_state_size  # 隱藏狀態通常需要全部在 GPU
    
    # 運算緩衝區
    mlp_expansion_ratio = getattr(config, 'intermediate_size', 4 * config.input_dim) / config.input_dim
    mlp_buffer = batch_size * config.input_dim * mlp_expansion_ratio * 2
    attention_buffer = batch_size * config.n_head * (input_len + gen_len) * (input_len + gen_len) * 2
    misc_buffer = 0.1 * (min_gpu_weight + mlp_buffer)
    
    total_min_gpu_memory = min_gpu_weight + min_gpu_cache + min_gpu_hidden + mlp_buffer + attention_buffer + misc_buffer
    
    return total_min_gpu_memory <= hardware_profile.gpu_mem * gpu_memory_safety_factor


def estimate_strategy_memory_usage(batch_size, config, input_len, gen_len, 
                                  total_weight_size, total_kv_cache_size, total_hidden_state_size,
                                  compress_w, compress_c, num_layers):
    """估算特定策略的最小 GPU 記憶體需求"""
    # 在最激進的卸載策略下，GPU 至少需要的記憶體
    min_weight_gpu = total_weight_size / num_layers  # 至少一層權重
    min_cache_gpu = total_kv_cache_size / (num_layers * 4)  # 假設可以卸載 75% 的 cache
    min_hidden_gpu = total_hidden_state_size  # 隱藏狀態通常必須在 GPU
    
    # 運算緩衝區
    mlp_expansion_ratio = getattr(config, 'intermediate_size', 4 * config.input_dim) / config.input_dim
    mlp_buffer = batch_size * config.input_dim * mlp_expansion_ratio * 2
    attention_buffer = batch_size * config.n_head * (input_len + gen_len) * (input_len + gen_len) * 2
    misc_buffer = (min_weight_gpu + mlp_buffer) * 0.15
    
    return min_weight_gpu + min_cache_gpu + min_hidden_gpu + mlp_buffer + attention_buffer + misc_buffer


def estimate_actual_gpu_usage(policy, model_name, input_len, gen_len, compression_factors):
    """估算策略的實際 GPU 記憶體使用量"""
    model_info = get_model_info(model_name)
    config = model_info.config
    
    base_weight_size = config.model_bytes()
    batch_size = policy.gpu_batch_size
    base_hidden_state_size = batch_size * config.input_dim * (input_len + gen_len) * 2
    base_kv_cache_size = (
        batch_size * config.num_hidden_layers * config.n_head *
        (input_len + gen_len) * (config.input_dim // config.n_head) * 2 * 2
    )
    
    # 根據策略計算實際使用量
    weight_gpu = base_weight_size * (policy.w_gpu_percent / 100)
    if policy.compress_weight:
        weight_gpu /= compression_factors['weight']
        
    cache_gpu = base_kv_cache_size * (policy.cache_gpu_percent / 100)
    if policy.compress_cache:
        cache_gpu /= compression_factors['cache']
    
    hidden_gpu = base_hidden_state_size * (policy.act_gpu_percent / 100)
    
    # 運算緩衝區
    mlp_expansion_ratio = getattr(config, 'intermediate_size', 4 * config.input_dim) / config.input_dim
    mlp_buffer = batch_size * config.input_dim * mlp_expansion_ratio * 2
    attention_buffer = batch_size * config.n_head * (input_len + gen_len) * (input_len + gen_len) * 2
    misc_buffer = 0.1 * (weight_gpu + mlp_buffer)
    
    return weight_gpu + cache_gpu + hidden_gpu + mlp_buffer + attention_buffer + misc_buffer