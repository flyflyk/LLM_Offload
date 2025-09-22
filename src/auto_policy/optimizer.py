import pulp
import itertools
from flexllmgen.flex_opt import Policy, CompressionConfig
from src.auto_policy.cost_model import CostModel
from src.auto_policy.profiler import HardwareProfile

class Optimizer:
    def __init__(self, model_config, hardware: HardwareProfile, input_len: int, gen_len: int):
        self.cost_model = CostModel(model_config, hardware, input_len, gen_len)
        self.gpu_capacity = hardware.gpu_mem
        self.cpu_capacity = hardware.cpu_mem

    def search(self):
        best_policy = None
        min_latency = float('inf')
        for bs in itertools.count(start=4, step=4):
            oom_cnt = 0
            for compress_weight, compress_cache in [
                (False, False),
                (False, True),
                (True, False),
                (True, True),
            ]:
                # Set up LP problem
                prob = pulp.LpProblem(f"Policy_Search_bs_{bs}_cw_{compress_weight}_cc_{compress_cache}", pulp.LpMinimize)
                w_vars = {
                    'w_g': pulp.LpVariable("w_g", 0, 1),
                    'w_c': pulp.LpVariable("w_c", 0, 1),
                    'w_d': pulp.LpVariable("w_d", 0, 1),
                }
                if compress_cache:
                    c_vars = {
                        'c_g': pulp.LpVariable("c_g", cat=pulp.LpBinary),
                        'c_c': pulp.LpVariable("c_c", cat=pulp.LpBinary),
                        'c_d': pulp.LpVariable("c_d", cat=pulp.LpBinary),
                    }
                else:
                    c_vars = {
                        'c_g': pulp.LpVariable("c_g", 0, 1),
                        'c_c': pulp.LpVariable("c_c", 0, 1),
                        'c_d': pulp.LpVariable("c_d", 0, 1),
                    }
                h_vars = {
                    'h_g': pulp.LpVariable("h_g", cat=pulp.LpBinary),
                    'h_c': pulp.LpVariable("h_c", cat=pulp.LpBinary),
                    'h_d': pulp.LpVariable("h_d", cat=pulp.LpBinary),
                }
                p = {**w_vars, **c_vars, **h_vars}
                total_latency = self.cost_model.estimate_latency(prob, p, bs, compress_weight, compress_cache)
                prob += total_latency / bs

                # Memory constraints
                gpu_peak_exprs, cpu_peak_exprs, nvme_peak_expr = self.cost_model.get_peak_memory(p, bs)
                for i, expr in enumerate(gpu_peak_exprs):
                    prob += expr <= self.gpu_capacity, f"GPU_Memory_Constraint_{i}"
                for i, expr in enumerate(cpu_peak_exprs):
                    prob += expr <= self.cpu_capacity, f"CPU_Memory_Constraint_{i}"
                # Not using NVMe currently
                _ = nvme_peak_expr

                # Sum-to-one constraints
                prob += p['w_g'] + p['w_c'] + p['w_d'] == 1, "Weight_Sum_Constraint"
                prob += p['c_g'] + p['c_c'] + p['c_d'] == 1, "Cache_Sum_Constraint"
                prob += p['h_g'] + p['h_c'] + p['h_d'] == 1, "Activation_Sum_Constraint"
                
                # Solve the LP
                prob.solve(pulp.PULP_CBC_CMD(msg=0))
                if pulp.LpStatus[prob.status] != 'Optimal':
                    oom_cnt += 1
                    continue
                current_latency = pulp.value(prob.objective)
                if current_latency < min_latency:
                    min_latency = current_latency
                    solved_p = {v.name: v.varValue for v in prob.variables()}
                    best_policy = Policy(
                        gpu_batch_size=bs,
                        num_gpu_batches=1,
                        w_gpu_percent=solved_p['w_g'] * 100,
                        w_cpu_percent=solved_p['w_c'] * 100,
                        cache_gpu_percent=solved_p['c_g'] * 100,
                        cache_cpu_percent=solved_p['c_c'] * 100,
                        act_gpu_percent=solved_p['h_g'] * 100,
                        act_cpu_percent=solved_p['h_c'] * 100,
                        overlap=True, 
                        sep_layer=True, 
                        pin_weight=True,
                        cpu_cache_compute=False,
                        attn_sparsity=1.0,
                        compress_weight=compress_weight,
                        comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
                        compress_cache=compress_cache,
                        comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False),
                    )

            # If all compression strategies resulted in OOM for this batch size, stop.
            if oom_cnt == 4:
                print(f"Stopping search at batch size {bs} as all configurations resulted in OOM.")
                break

        return best_policy