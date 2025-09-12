import pulp
from flexllmgen.flex_opt import Policy, CompressionConfig
from src.auto_policy.cost_model import CostModel
from src.auto_policy.profiler import HardwareProfile

class Optimizer:
    def __init__(self, model_config, hardware: HardwareProfile, input_len: int, gen_len: int):
        self.cost_model = CostModel(model_config, hardware, input_len, gen_len)
        self.gpu_capacity = hardware.gpu_mem
        self.cpu_capacity = hardware.cpu_mem

    def search(self, batch_size_options):
        best_policy = None
        min_latency = float('inf')

        for bs in batch_size_options:
            prob = pulp.LpProblem(f"Policy_Search_bs_{bs}", pulp.LpMinimize)

            p = {
                'w_g': pulp.LpVariable("w_g", 0, 1),
                'w_c': pulp.LpVariable("w_c", 0, 1),
                'w_d': pulp.LpVariable("w_d", 0, 1),
                'c_g': pulp.LpVariable("c_g", 0, 1),
                'c_c': pulp.LpVariable("c_c", 0, 1),
                'c_d': pulp.LpVariable("c_d", 0, 1),
                'h_g': pulp.LpVariable("h_g", 0, 1),
                'h_c': pulp.LpVariable("h_c", 0, 1),
                'h_d': pulp.LpVariable("h_d", 0, 1),
            }

            total_latency = self.cost_model.estimate_latency(p, bs)
            prob += total_latency

            gpu_mem, cpu_mem = self.cost_model.get_peak_memory(p, bs)
            prob += gpu_mem <= self.gpu_capacity, "GPU_Memory_Constraint"
            prob += cpu_mem <= self.cpu_capacity, "CPU_Memory_Constraint"
            
            prob += p['w_g'] + p['w_c'] + p['w_d'] == 1, "Weight_Sum_Constraint"
            prob += p['c_g'] + p['c_c'] + p['c_d'] == 1, "Cache_Sum_Constraint"
            prob += p['h_g'] + p['h_c'] + p['h_d'] == 1, "Activation_Sum_Constraint"

            prob.solve(pulp.PULP_CBC_CMD(msg=0))

            if pulp.LpStatus[prob.status] == 'Optimal':
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
                        compress_weight=False,
                        comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
                        compress_cache=False,
                        comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False),
                    )

        return best_policy