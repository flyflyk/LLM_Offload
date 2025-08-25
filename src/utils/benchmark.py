import gc
import torch
import logging

def log_metrics(framework: str, throughput: float, infer_time: float, model_load_time: float):
    logger = logging.getLogger(__name__)
    logger.info("--- Performance Metrics ---")
    logger.info(f"Framework: {framework}")
    logger.info(f"Model Load Time: {model_load_time:.4f}s")
    logger.info(f"Total Inference Time: {infer_time:.4f}s")
    logger.info(f"Throughput: {throughput:.2f} tokens/sec")
    logger.info(f"--- Execution Finished Successfully ({framework}) ---")

def cleanup_mem():
    print("--- Cleaning up VRAM before next test ---")
    gc.collect()
    torch.cuda.empty_cache()
