# C:/Users/flyfl/Documents/CodeProjects/py/inference/AutoPolicy/optimizer.py
"""
Policy optimizer to find the best offloading strategy.

This module uses the cost model to search for the optimal FlexLLMGen policy
that minimizes the predicted inference latency.
"""

import itertools
from tqdm import tqdm

# --- Safe Imports from FlexLLMGen & AutoPolicy ---
from FlexLLMGen.flexllmgen.flex_opt import Policy, CompressionConfig
from AutoPolicy.cost_model import CostModel, ModelInfo

def find_best_policy(cost_model: CostModel, model_info: ModelInfo, prompt_len: int, gen_len: int) -> Policy:
    """
    Finds the best offloading policy using a grid search over placement percentages.

    Args:
        cost_model: An initialized CostModel instance.
        model_info: An initialized ModelInfo instance.
        prompt_len: The length of the input prompt.
        gen_len: The number of tokens to generate.

    Returns:
        The FlexLLMGen Policy object predicted to yield the lowest latency.
    """
    print("Searching for the optimal policy...")
    
    best_policy = None
    min_latency = float('inf')
    
    # Define the search space for placement percentages.
    # A smaller step will be more accurate but take longer.
    search_step = 25
    percent_choices = list(range(0, 101, search_step))
    
    # Generate all valid combinations for weight and cache placements.
    # A valid combination is where the sum of GPU and CPU percentages is <= 100.
    weight_combinations = [p for p in itertools.product(percent_choices, repeat=2) if sum(p) <= 100]
    cache_combinations = [p for p in itertools.product(percent_choices, repeat=2) if sum(p) <= 100]
    
    total_searches = len(weight_combinations) * len(cache_combinations)
    pbar = tqdm(total=total_searches, desc="Policy Search")

    for w_gpu, w_cpu in weight_combinations:
        for c_gpu, c_cpu in cache_combinations:
            # For simplicity, we assume activations are always on GPU for best performance.
            act_gpu, act_cpu = 100, 0
            
            # Create a policy object to be evaluated by the cost model.
            # We use default values for other policy settings as they are not part of this simplified search.
            policy = Policy(
                gpu_batch_size=1, num_gpu_batches=1, # Placeholder values
                w_gpu_percent=w_gpu, w_cpu_percent=w_cpu,
                cache_gpu_percent=c_gpu, cache_cpu_percent=c_cpu,
                act_gpu_percent=act_gpu, act_cpu_percent=act_cpu,
                overlap=True, sep_layer=True, pin_weight=True,
                cpu_cache_compute=False, attn_sparsity=1.0,
                compress_weight=False, comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
                compress_cache=False, comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False)
            )
            
            latency = cost_model.predict_latency(policy, model_info, prompt_len, gen_len)
            
            if latency < min_latency:
                min_latency = latency
                best_policy = policy
            
            pbar.update(1)
            
    pbar.close()
    print(f"Found best policy with predicted latency: {min_latency:.4f}s")
    return best_policy