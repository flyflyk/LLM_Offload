import psutil
import pulp
from tqdm import tqdm

from FlexLLMGen.flexllmgen.flex_opt import Policy, CompressionConfig
from .cost_model import get_model_info
from .profiler import HardwareProfile

def get_optimial_policy(
    model_name: str,
    hardware_profile: HardwareProfile,
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

        # --- 1. Define Model Component Sizes (for one layer) ---
        total_weight_size = config.model_bytes()
        total_hidden_state_size = batch_size * config.input_dim * 2  # FP16
        total_kv_cache_size = (
            batch_size * num_layers * config.n_head *
            config.max_seq_len * (config.input_dim // config.n_head) * 2 * 2 # k/v, FP16
        )

        size_w = total_weight_size / num_layers
        size_h = total_hidden_state_size
        size_c = total_kv_cache_size / num_layers

        # --- 2. Define Linear Programming Problem ---
        prob = pulp.LpProblem("FlexGen_Offloading", pulp.LpMinimize)

        # --- 3. Decision Variables ---
        var_names = ["w_gpu", "w_cpu", "w_disk", "c_gpu", "c_cpu", "c_disk", "h_gpu", "h_cpu", "h_disk"]
        vars = pulp.LpVariable.dicts("placement", var_names, lowBound=0, upBound=1)

        # --- 4. Objective Function (Minimize Data Transfer Time) ---
        T_cpu_to_gpu = 1 / hardware_profile.cpu_gpu_bandwidth
        T_disk_to_gpu = 1 / hardware_profile.disk_cpu_bandwidth + T_cpu_to_gpu # Disk -> CPU -> GPU

        prob += (
            size_w * (vars["w_cpu"] * T_cpu_to_gpu + vars["w_disk"] * T_disk_to_gpu) +
            size_c * (vars["c_cpu"] * T_cpu_to_gpu + vars["c_disk"] * T_disk_to_gpu) +
            size_h * (vars["h_cpu"] * T_cpu_to_gpu + vars["h_disk"] * T_disk_to_gpu)
        ), "Total_Transfer_Time"

        # --- 5. Constraints ---
        # a) Completeness Constraints (percentages must sum to 1)
        prob += vars["w_gpu"] + vars["w_cpu"] + vars["w_disk"] == 1, "Weight_Completeness"
        prob += vars["c_gpu"] + vars["c_cpu"] + vars["c_disk"] == 1, "Cache_Completeness"
        prob += vars["h_gpu"] + vars["h_cpu"] + vars["h_disk"] == 1, "Hidden_State_Completeness"

        # b) Storage Capacity Constraints
        # Buffer is for pre-loading the next layer's weights.
        buffer = size_w
        prob += (
            (total_weight_size * vars["w_gpu"]) +
            (total_kv_cache_size * vars["c_gpu"]) +
            (total_hidden_state_size * vars["h_gpu"]) +
            buffer
        ) <= hardware_profile.gpu_mem, "GPU_Capacity"

        prob += (
            (total_weight_size * vars["w_cpu"]) +
            (total_kv_cache_size * vars["c_cpu"]) +
            (total_hidden_state_size * vars["h_cpu"])
        ) <= hardware_profile.cpu_mem, "CPU_Capacity"
        # Disk capacity is assumed to be sufficient, so no constraint is added.

        # --- 6. Solve the LP Problem ---
        prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # --- 7. Evaluate the Solution ---
        if pulp.LpStatus[prob.status] == "Optimal":
            min_transfer_time = pulp.value(prob.objective)

            l_step = min_transfer_time
            throughput = batch_size / l_step

            if throughput > max_throughput:
                max_throughput = throughput
                best_batch_size = batch_size
                
                # Auto-detect settings for the policy
                physical_cores = psutil.cpu_count(logical=False)
                num_copy_threads = max(1, min(physical_cores // 2, 4))
                best_num_copy_threads = num_copy_threads
                
                model_name_lower = model_name.lower()
                large_model_keywords = ["13b", "17.5b", "30b", "66b", "175b"]
                is_large_model = any(keyword in model_name_lower for keyword in large_model_keywords)

                best_policy = Policy(
                    gpu_batch_size=batch_size,
                    num_gpu_batches=1,
                    w_gpu_percent=vars["w_gpu"].varValue * 100,
                    w_cpu_percent=vars["w_cpu"].varValue * 100,
                    cache_gpu_percent=vars["c_gpu"].varValue * 100,
                    cache_cpu_percent=vars["c_cpu"].varValue * 100,
                    act_gpu_percent=vars["h_gpu"].varValue * 100,
                    act_cpu_percent=vars["h_cpu"].varValue * 100,
                    overlap=True,
                    sep_layer=True,
                    pin_weight=True,
                    cpu_cache_compute=False,
                    attn_sparsity=1.0,
                    compress_weight=is_large_model,
                    comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
                    compress_cache=is_large_model,
                    comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False),
                )

    if best_policy:
        print(f"\nFound best policy with a throughput of {max_throughput:.2f} tokens/sec "
              f"at a batch size of {best_batch_size}.")
    else:
        print("\nCould not find a feasible policy. The model may be too large for the available hardware.")

    return best_policy, best_num_copy_threads