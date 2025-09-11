import logging
import psutil
import pulp
from tqdm import tqdm

from FlexLLMGen.flexllmgen.flex_opt import Policy, CompressionConfig
from .cost_model import get_model_info
from .profiler import HardwareProfile

logger = logging.getLogger(__name__)

def to_gb(byte_size):
    return byte_size / (1024**3)

def get_optimial_policy(
    model_name: str,
    hardware_profile: HardwareProfile,
    input_len: int,
    gen_len: int,
    max_batch_size: int = 4096,
) -> Policy:
    logger.info("Searching for the optimal policy using Linear Programming...")
    best_policy = None
    max_throughput = 0.0
    best_batch_size = 0
    best_num_copy_threads = 4
    compression_factors = {'weight': 2.2, 'cache': 2.0}
    gpu_memory_safety_factor = 0.85

    feasible_batch_sizes = []
    printed_feasibility_debug = False

    for batch_size in tqdm(
        range(4, max_batch_size + 1, 4), desc="Finding Feasible Batch Sizes"
    ):
        is_feasible, _ = is_batch_size_feasible(batch_size, model_name, hardware_profile,
                                                input_len, gen_len, compression_factors,
                                                gpu_memory_safety_factor,
                                                debug_print=not printed_feasibility_debug)
        if is_feasible:
            feasible_batch_sizes.append(batch_size)
            printed_feasibility_debug = True
    
    if not feasible_batch_sizes:
        logger.error("No feasible batch size found. Model too large for available hardware.")
        return None, 4
    
    logger.info(f"Found {len(feasible_batch_sizes)} feasible batch sizes. First 10: {feasible_batch_sizes[:10]}...")
    
    best_bs_debug_info = {}
    seq_len = input_len + gen_len

    for batch_size in tqdm(feasible_batch_sizes, desc="Optimizing Among Feasible Sizes"):
        model_info = get_model_info(model_name)
        config = model_info.config
        num_layers = config.num_hidden_layers

        base_weight_size = config.model_bytes()
        base_hidden_state_size = batch_size * config.input_dim * seq_len * 2
        base_kv_cache_size = (
            batch_size * num_layers * config.n_head *
            seq_len * (config.input_dim // config.n_head) * 2 * 2
        )

        for strategy_idx, (compress_w, compress_c) in enumerate([(False, False), (False, True), (True, True), (True, False)]):
            total_weight_size = base_weight_size / compression_factors['weight'] if compress_w else base_weight_size
            total_kv_cache_size = base_kv_cache_size / compression_factors['cache'] if compress_c else base_kv_cache_size
            total_hidden_state_size = base_hidden_state_size
            
            # Variable definitions
            prob = pulp.LpProblem(f"FlexGen_Offloading_{batch_size}_{strategy_idx}", pulp.LpMinimize)
            var_names = ["w_gpu", "w_cpu", "w_disk", "c_gpu", "c_cpu", "c_disk", "h_gpu", "h_cpu", "h_disk"]
            vars = {name: pulp.LpVariable(f"placement_{batch_size}_{strategy_idx}_{name}", lowBound=0, upBound=1) for name in var_names}

            # Constraints and Objective
            T_cpu_to_gpu = 1 / hardware_profile.cpu_gpu_bandwidth
            T_disk_to_gpu = 1 / hardware_profile.disk_cpu_bandwidth + T_cpu_to_gpu
            prob += ((total_weight_size / num_layers) * (vars["w_cpu"] * T_cpu_to_gpu + vars["w_disk"] * T_disk_to_gpu) +
                     (total_kv_cache_size / num_layers) * (vars["c_cpu"] * T_cpu_to_gpu + vars["c_disk"] * T_disk_to_gpu) +
                     total_hidden_state_size * (vars["h_cpu"] * T_cpu_to_gpu + vars["h_disk"] * T_disk_to_gpu)), "Total_Transfer_Time_Per_Layer"
            prob += vars["w_gpu"] + vars["w_cpu"] + vars["w_disk"] == 1
            prob += vars["c_gpu"] + vars["c_cpu"] + vars["c_disk"] == 1
            prob += vars["h_gpu"] + vars["h_cpu"] + vars["h_disk"] == 1

            # Memory constraints
            weight_buffer_for_compute = total_weight_size / num_layers
            mlp_expansion_ratio = getattr(config, 'intermediate_size', 4 * config.input_dim) / config.input_dim
            mlp_buffer = batch_size * seq_len * config.input_dim * mlp_expansion_ratio * 2
            attention_buffer = batch_size * config.n_head * seq_len * seq_len * 2
            mha_intermediate_buffer = 4 * base_hidden_state_size
            misc_buffer = 0.1 * hardware_profile.gpu_mem

            stored_components_mem = (total_weight_size * vars["w_gpu"] +
                                     total_kv_cache_size * vars["c_gpu"] +
                                     total_hidden_state_size * vars["h_gpu"])
            
            peak_compute_buffer = (weight_buffer_for_compute + 
                                   mlp_buffer + 
                                   attention_buffer + 
                                   mha_intermediate_buffer +
                                   misc_buffer)

            prob += stored_components_mem + peak_compute_buffer <= hardware_profile.gpu_mem * gpu_memory_safety_factor, f"GPU_Capacity_{strategy_idx}"
            prob += ((total_weight_size * vars["w_cpu"]) + (total_kv_cache_size * vars["c_cpu"]) + (total_hidden_state_size * vars["h_cpu"])
            ) <= hardware_profile.cpu_mem * 0.9, f"CPU_Capacity_{strategy_idx}"

            prob.solve(pulp.PULP_CBC_CMD(msg=False))

            if pulp.LpStatus[prob.status] == "Optimal":
                min_transfer_time_per_layer = pulp.value(prob.objective)
                H = config.input_dim; S = input_len + gen_len
                latency_per_layer = (min_transfer_time_per_layer - total_hidden_state_size * (vars["h_cpu"].varValue * T_cpu_to_gpu + vars["h_disk"].varValue * T_disk_to_gpu))
                layer_flops = batch_size * (24 * H**2 + 4 * S * H)
                effective_tflops = (hardware_profile.tflops_slope * batch_size +
                                    hardware_profile.tflops_intercept)
                T_compute_gpu_per_layer = layer_flops / (effective_tflops * 1e12 + 1e-10)
                total_latency = (T_compute_gpu_per_layer + latency_per_layer) * num_layers + (total_hidden_state_size * (vars["h_cpu"].varValue * T_cpu_to_gpu + vars["h_disk"].varValue * T_disk_to_gpu))
                throughput = batch_size / total_latency if total_latency > 0 else 0

                if throughput > max_throughput:
                    max_throughput = throughput
                    best_batch_size = batch_size
                    physical_cores = psutil.cpu_count(logical=False)
                    best_num_copy_threads = max(1, min(physical_cores // 2, 4))
                    best_policy = Policy(gpu_batch_size=batch_size, num_gpu_batches=1,
                                         w_gpu_percent=vars["w_gpu"].varValue * 100, w_cpu_percent=vars["w_cpu"].varValue * 100,
                                         cache_gpu_percent=vars["c_gpu"].varValue * 100, cache_cpu_percent=vars["c_cpu"].varValue * 100,
                                         act_gpu_percent=vars["h_gpu"].varValue * 100, act_cpu_percent=vars["h_cpu"].varValue * 100,
                                         overlap=True, sep_layer=True, pin_weight=True, cpu_cache_compute=False, attn_sparsity=1.0,
                                         compress_weight=compress_w, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
                                         compress_cache=compress_c, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False))
                    
                    # <--- DEBUGGING LOG --- >
                    stored_mem_val = pulp.value(stored_components_mem)
                    total_estimated_mem = stored_mem_val + peak_compute_buffer
                    best_bs_debug_info = {
                        "batch_size": batch_size,
                        "throughput": throughput,
                        "strategy": f"W_comp: {compress_w}, C_comp: {compress_c}",
                        "total_gpu_mem_limit_gb": to_gb(hardware_profile.gpu_mem * gpu_memory_safety_factor),
                        "total_estimated_mem_gb": to_gb(total_estimated_mem),
                        "--- Breakdown (GB) ---": "",
                        "1. Stored Components": to_gb(stored_mem_val),
                        "  - Stored Weights (gpu_%)": f"{to_gb(total_weight_size * vars['w_gpu'].varValue)} ({vars['w_gpu'].varValue*100:.1f}%)",
                        "  - Stored Cache (gpu_%)": f"{to_gb(total_kv_cache_size * vars['c_gpu'].varValue)} ({vars['c_gpu'].varValue*100:.1f}%)",
                        "  - Stored Hidden (gpu_%)": f"{to_gb(total_hidden_state_size * vars['h_gpu'].varValue)} ({vars['h_gpu'].varValue*100:.1f}%)",
                        "2. Peak Compute Buffer": to_gb(peak_compute_buffer),
                        "  - Weight Buffer (1 layer)": to_gb(weight_buffer_for_compute),
                        "  - MLP Buffer": to_gb(mlp_buffer),
                        "  - Attention Score Buffer": to_gb(attention_buffer),
                        "  - MHA Intermediate Buffer": to_gb(mha_intermediate_buffer),
                        "  - Misc/Overhead Buffer": to_gb(misc_buffer),
                    }


    if best_policy:
        logger.info(f"Found best policy with a throughput of {max_throughput:.2f} tokens/sec "
              f"at a batch size of {best_batch_size}.")
        
        # --- Log the detailed memory breakdown for the BEST policy found ---
        logger.info("--- [DEBUG] Memory Calculation for Best Policy ---")
        for key, value in best_bs_debug_info.items():
            logger.info(f"{key}: {value}")
        logger.info("-------------------------------------------------")

        # --- Post-hoc validation using estimate_actual_gpu_usage ---
        estimated_gpu_usage, usage_breakdown = estimate_actual_gpu_usage(
            best_policy, model_name, input_len, gen_len, compression_factors, hardware_profile
        )
        gpu_usage_percent = (estimated_gpu_usage / hardware_profile.gpu_mem) * 100
        
        logger.info(f"--- [DEBUG] Post-Hoc Validation Breakdown ---")
        logger.info(f"Final Estimated GPU usage: {to_gb(estimated_gpu_usage):.2f}GB ({gpu_usage_percent:.1f}%)")
        for key, value in usage_breakdown.items():
            logger.info(f"  - {key}: {to_gb(value):.2f} GB")
        logger.info("-------------------------------------------")

        if gpu_usage_percent > 98:
            logger.error(f"Critical: GPU memory usage is {gpu_usage_percent:.1f}%, this will cause OOM!")
            return None, 4
        elif gpu_usage_percent > 90:
            logger.warning(f"GPU memory usage is high ({gpu_usage_percent:.1f}%), OOM risk is high.")
            
    else:
        logger.warning("Could not find a feasible policy.")

    return best_policy, best_num_copy_threads


def is_batch_size_feasible(batch_size, model_name, hardware_profile, input_len, gen_len, 
                          compression_factors, gpu_memory_safety_factor, debug_print=False):
    model_info = get_model_info(model_name)
    config = model_info.config
    num_layers = config.num_hidden_layers
    seq_len = input_len + gen_len
    
    base_weight_size = config.model_bytes()
    base_hidden_state_size = batch_size * config.input_dim * seq_len * 2
    base_kv_cache_size = (batch_size * num_layers * config.n_head * seq_len * (config.input_dim // config.n_head) * 2 * 2)
    
    # Minimized stored components (most aggressive offloading)
    min_gpu_weight = (base_weight_size / compression_factors['weight']) / num_layers
    min_gpu_cache = (base_kv_cache_size / compression_factors['cache']) / num_layers
    min_gpu_hidden = base_hidden_state_size
    min_stored_components = min_gpu_weight + min_gpu_cache + min_gpu_hidden
    
    # Peak compute buffer (this is fixed for a given batch size)
    weight_buffer_for_compute = min_gpu_weight
    mlp_expansion_ratio = getattr(config, 'intermediate_size', 4 * config.input_dim) / config.input_dim
    mlp_buffer = batch_size * seq_len * config.input_dim * mlp_expansion_ratio * 2
    attention_buffer = batch_size * config.n_head * seq_len * seq_len * 2
    mha_intermediate_buffer = 4 * base_hidden_state_size
    misc_buffer = 0.1 * hardware_profile.gpu_mem
    peak_compute_buffer = (weight_buffer_for_compute + mlp_buffer + attention_buffer + 
                           mha_intermediate_buffer + misc_buffer)

    total_min_gpu_memory = min_stored_components + peak_compute_buffer
    
    is_feasible = total_min_gpu_memory <= hardware_profile.gpu_mem * gpu_memory_safety_factor

    if debug_print:
        logger.info(f"--- [DEBUG] Feasibility Check (BS={batch_size}) ---")
        logger.info(f"GPU Memory Limit: {to_gb(hardware_profile.gpu_mem * gpu_memory_safety_factor):.2f} GB")
        logger.info(f"Estimated Minimum Memory: {to_gb(total_min_gpu_memory):.2f} GB -> Feasible: {is_feasible}")
        logger.info("--- Breakdown (GB) ---")
        logger.info(f"1. Min Stored Components: {to_gb(min_stored_components):.2f}")
        logger.info(f"  - Stored Weights (min): {to_gb(min_gpu_weight):.2f}")
        logger.info(f"  - Stored Cache (min): {to_gb(min_gpu_cache):.2f}")
        logger.info(f"  - Stored Hidden (min): {to_gb(min_gpu_hidden):.2f}")
        logger.info(f"2. Peak Compute Buffer: {to_gb(peak_compute_buffer):.2f}")
        logger.info(f"  - Weight Buffer (1 layer): {to_gb(weight_buffer_for_compute):.2f}")
        logger.info(f"  - MLP Buffer: {to_gb(mlp_buffer):.2f}")
        logger.info(f"  - Attention Score Buffer: {to_gb(attention_buffer):.2f}")
        logger.info(f"  - MHA Intermediate Buffer: {to_gb(mha_intermediate_buffer):.2f}")
        logger.info(f"  - Misc/Overhead Buffer: {to_gb(misc_buffer):.2f}")
        logger.info("-------------------------------------------------")

    return is_feasible, total_min_gpu_memory

def estimate_actual_gpu_usage(policy, model_name, input_len, gen_len, compression_factors, hardware_profile):
    model_info = get_model_info(model_name)
    config = model_info.config
    num_layers = config.num_hidden_layers
    batch_size = policy.gpu_batch_size
    seq_len = input_len + gen_len
    
    base_weight_size = config.model_bytes()
    base_hidden_state_size = batch_size * config.input_dim * seq_len * 2
    base_kv_cache_size = (batch_size * num_layers * config.n_head * seq_len * (config.input_dim // config.n_head) * 2 * 2)
    
    # Stored components based on policy
    weight_gpu = base_weight_size * (policy.w_gpu_percent / 100)
    if policy.compress_weight:
        weight_gpu /= compression_factors['weight']
        
    cache_gpu = base_kv_cache_size * (policy.cache_gpu_percent / 100)
    if policy.compress_cache:
        cache_gpu /= compression_factors['cache']
    
    hidden_gpu = base_hidden_state_size * (policy.act_gpu_percent / 100)
    
    stored_total = weight_gpu + cache_gpu + hidden_gpu

    # Peak compute buffer (must be consistent with optimizer)
    weight_buffer_for_compute = (base_weight_size / compression_factors['weight'] if policy.compress_weight else base_weight_size) / num_layers
    mlp_expansion_ratio = getattr(config, 'intermediate_size', 4 * config.input_dim) / config.input_dim
    mlp_buffer = batch_size * seq_len * config.input_dim * mlp_expansion_ratio * 2
    attention_buffer = batch_size * config.n_head * seq_len * seq_len * 2
    mha_intermediate_buffer = 4 * base_hidden_state_size
    misc_buffer = 0.1 * hardware_profile.gpu_mem
    
    compute_total = weight_buffer_for_compute + mlp_buffer + attention_buffer + mha_intermediate_buffer + misc_buffer

    total_usage = stored_total + compute_total

    breakdown = {
        "Stored Weights": weight_gpu,
        "Stored Cache": cache_gpu,
        "Stored Hidden": hidden_gpu,
        "Compute Weight Buf": weight_buffer_for_compute,
        "Compute MLP Buf": mlp_buffer,
        "Compute Attention Score Buf": attention_buffer,
        "Compute MHA Intermediate Buf": mha_intermediate_buffer,
        "Compute Misc Buf": misc_buffer,
    }
    
    return total_usage, breakdown