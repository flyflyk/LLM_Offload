import psutil
import pulp
from tqdm import tqdm

from FlexLLMGen.flexllmgen.flex_opt import Policy, CompressionConfig
from .cost_model import get_model_info
from .profiler import HardwareProfile

def get_optimial_policy(
    model_name: str,
    hardware_profile: HardwareProfile,
    input_len: int,
    gen_len: int,
    max_batch_size: int = 128,
) -> Policy:
    print("Searching for the optimal policy using Linear Programming...")

    best_policy = None
    max_throughput = 0.0
    best_batch_size = 0
    best_num_copy_threads = 4  # Default value

    # --- Iterate through batch sizes (multiples of 4) ---
    for batch_size in tqdm(
        range(4, max_batch_size + 1, 4), desc="Optimizing Batch Size"
    ):
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
        for strategy_idx, (compress_w, compress_c) in enumerate([(False, False), (False, True), (True, True)]):
            # --- Setup sizes ---
            compression_factor = 3.0 # Conservative estimate for 4-bit quantization
            total_weight_size = base_weight_size / compression_factor if compress_w else base_weight_size
            total_kv_cache_size = base_kv_cache_size / compression_factor if compress_c else base_kv_cache_size
            total_hidden_state_size = base_hidden_state_size

            size_w = total_weight_size / num_layers
            size_h = total_hidden_state_size
            size_c = total_kv_cache_size / num_layers

            # --- Define Linear Programming Problem ---
            prob = pulp.LpProblem(f"FlexGen_Offloading_{batch_size}_{strategy_idx}", pulp.LpMinimize)
            var_names = ["w_gpu", "w_cpu", "w_disk", "c_gpu", "c_cpu", "c_disk", "h_gpu", "h_cpu", "h_disk"]
            vars = pulp.LpVariable.dicts(f"placement_{batch_size}_{strategy_idx}", var_names, lowBound=0, upBound=1)

            # --- Objective & Constraints ---
            T_cpu_to_gpu = 1 / hardware_profile.cpu_gpu_bandwidth
            T_disk_to_gpu = 1 / hardware_profile.disk_cpu_bandwidth + T_cpu_to_gpu
            prob += (size_w * (vars["w_cpu"] * T_cpu_to_gpu + vars["w_disk"] * T_disk_to_gpu) +
                     size_c * (vars["c_cpu"] * T_cpu_to_gpu + vars["c_disk"] * T_disk_to_gpu) +
                     size_h * (vars["h_cpu"] * T_cpu_to_gpu + vars["h_disk"] * T_disk_to_gpu)), "Total_Transfer_Time"
            prob += vars["w_gpu"] + vars["w_cpu"] + vars["w_disk"] == 1, f"Weight_Completeness_{strategy_idx}"
            prob += vars["c_gpu"] + vars["c_cpu"] + vars["c_disk"] == 1, f"Cache_Completeness_{strategy_idx}"
            prob += vars["h_gpu"] + vars["h_cpu"] + vars["h_disk"] == 1, f"Hidden_State_Completeness_{strategy_idx}"
            peak_buffer = (base_weight_size / num_layers) + (base_kv_cache_size / num_layers)
            prob += ((total_weight_size * vars["w_gpu"]) +
                     (total_kv_cache_size * vars["c_gpu"]) +
                     (total_hidden_state_size * vars["h_gpu"]) +
                     peak_buffer
            ) <= hardware_profile.gpu_mem * 0.95, f"GPU_Capacity_{strategy_idx}"
            prob += ((total_weight_size * vars["w_cpu"]) +
                     (total_kv_cache_size * vars["c_cpu"]) +
                     (total_hidden_state_size * vars["h_cpu"])
            ) <= hardware_profile.cpu_mem, f"CPU_Capacity_{strategy_idx}"

            # --- Solve and Evaluate ---
            prob.solve(pulp.PULP_CBC_CMD(msg=False))

            if pulp.LpStatus[prob.status] == "Optimal":
                min_transfer_time = pulp.value(prob.objective)
                H = config.input_dim
                S = input_len + gen_len
                layer_flops = batch_size * (24 * H**2 + 4 * S * H)
                T_compute_gpu = layer_flops / (hardware_profile.peak_gpu_tflops * 1e12 + 1e-10)
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
                break

    if best_policy:
        print(f"\nFound best policy with a throughput of {max_throughput:.2f} tokens/sec "
              f"at a batch size of {best_batch_size}.")
    else:
        print("\nCould not find a feasible policy. The model may be too large for the available hardware.")

    return best_policy, best_num_copy_threads