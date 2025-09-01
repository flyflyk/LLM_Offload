import gc
import torch
import logging

def log_metrics(framework: str, throughput: float, infer_time: float, model_load_time: float, model_name: str, flex_policy_info=None, flex_allocation_info=None):
    logger = logging.getLogger(__name__)
    logger.info("--- Performance Metrics ---")
    logger.info(f"Model: {model_name}")
    logger.info(f"Framework: {framework}")
    
    if flex_policy_info:
        logger.info("--- FlexGen Policy ---")
        for key, value in flex_policy_info.items():
            logger.info(f"{key}: {value}") 
    if flex_allocation_info:
        logger.info("--- Allocation ---")
        logger.info(f"Memory Distribution Summary (GB): {flex_allocation_info['device_sizes']}")      
        logger.info(f"Cache Size (GB): {flex_allocation_info['cache_size_gb']:.3f}")
        logger.info(f"Hidden Size (GB): {flex_allocation_info['hidden_size_gb']:.3f}")

    logger.info("--- Timing ---")
    logger.info(f"Model Load Time: {model_load_time:.4f}s")
    logger.info(f"Total Inference Time: {infer_time:.4f}s")
    logger.info(f"Throughput: {throughput:.2f} tokens/sec")
    logger.info(f"--- Execution Finished Successfully ({framework}) ---")

def cleanup_mem():
    print("--- Cleaning up VRAM before next test ---")
    gc.collect()
    torch.cuda.empty_cache()