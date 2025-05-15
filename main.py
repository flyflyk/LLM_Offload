import time
import logging
from logger import setup_logging
from model_loader import load_model
from config import (
    CHOSEN_MODEL, MAX_TOKENS, ENABLE_STREAMING, PROMPT_LIST,
    ENABLE_KV_OFFLOAD, BATCH_SIZE, DEVICE, LOG_FILE
)
from inference_runner import run_inference

setup_logging(log_file=LOG_FILE if 'LOG_FILE' in globals() and LOG_FILE else None)
logger = logging.getLogger(__name__)

def _print_summary(results: dict, total_prompts_processed: int, total_time: float, total_tokens: int):
    logger.info("\n--- Results Summary ---")
    if not results:
        logger.info("No results collected.")
        return

    if "model_vram" in results:
         logger.info("Model VRAM Footprint:")
         for key, value in results["model_vram"].items():
             if "gb" in key:
                 logger.info(f"  {key}: {value:.4f} GB")
         logger.info("-" * 15)

    logger.info("Inference Metrics per Batch:")
    batch_latencies = []
    for batch_label, metrics in results.items():
         if batch_label == "model_vram": continue

         logger.info(f"{batch_label}:")
         prompts_in_batch = metrics.get("num_prompts_in_batch", "N/A")
         latency = metrics.get("latency")
         tokens_generated = metrics.get("tokens_generated_in_batch", "N/A")
         batch_time = metrics.get("batch_time", "N/A")
         
         logger.info(f"  Prompts in batch: {prompts_in_batch}")
         logger.info(f"  Tokens generated: {tokens_generated}")
         logger.info(f"  Batch processing time: {batch_time:.4f} s" if isinstance(batch_time, float) else f"  Batch processing time: {batch_time}")

         if latency is not None and latency != float('inf'):
             logger.info(f"  Avg Latency: {latency:.4f} sec/token")
             if isinstance(latency, (int, float)) and latency != float('inf'):
                 batch_latencies.append(latency)
         else:
             logger.info(f"  Avg Latency: No valid data or no new tokens")
        
         if isinstance(tokens_generated, int) and tokens_generated > 0 and isinstance(batch_time, float) and batch_time > 0:
             throughput = tokens_generated / batch_time
             logger.info(f"  Batch Throughput: {throughput:.2f} tokens/sec")


         inf_vram = metrics.get("inference_vram")
         if inf_vram:
             logger.info(f"  Inference Peak VRAM Allocated Increase: {inf_vram.get('allocated_increase_gb', 'N/A'):.4f} GB")
             logger.info(f"  Inference Peak VRAM Allocated Total: {inf_vram.get('peak_allocated_gb', 'N/A'):.4f} GB")
         else:
              logger.info(f"  Inference VRAM: No data (likely CPU or error)")
         logger.info("-" * 10)
    
    logger.info("\n--- Overall Performance ---")
    logger.info(f"Total prompts processed: {total_prompts_processed}")
    logger.info(f"Total new tokens generated across all batches: {total_tokens}")
    logger.info(f"Total processing time for all batches: {total_time:.4f} seconds")
    
    if total_tokens > 0 and total_time > 0:
        overall_avg_latency = total_time / total_tokens
        overall_throughput = total_tokens / total_time
        logger.info(f"Overall Average Latency: {overall_avg_latency:.4f} sec/token")
        logger.info(f"Overall Throughput: {overall_throughput:.2f} tokens/sec")
    else:
        logger.info("Overall latency/throughput not calculable.")

    if batch_latencies:
        avg_batch_latency = sum(batch_latencies) / len(batch_latencies)
        logger.info(f"Average of Batch Latencies: {avg_batch_latency:.4f} sec/token")
    
    logger.info("-" * 30)


def main():
    streaming_mode = "Streaming" if ENABLE_STREAMING else "Default"
    kv_offload_mode = " + KV Offload" if ENABLE_KV_OFFLOAD else ""
    current_mode = f"{streaming_mode}{kv_offload_mode}"
    logger.info(f"--- Starting Execution ({current_mode}) ---")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Batch size: {BATCH_SIZE}")
    logger.info(f"Loading model '{CHOSEN_MODEL}'...")
    model, tokenizer, gpu, model_vram_info = load_model(model_name=CHOSEN_MODEL)

    if tokenizer.pad_token is None:
        logger.warning("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None :
            model.config.pad_token_id = model.config.eos_token_id
    
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.info(f"Set model.config.pad_token_id to {model.config.pad_token_id}")

    all_results = {
        "model_vram": model_vram_info
    }

    logger.info("\n--- Running Measured Inference Examples in Batches ---")
    
    total_prompts_processed = 0
    total_time_sum = 0
    total_tokens = 0

    for i in range(0, len(PROMPT_LIST), BATCH_SIZE):
        batch_prompts = PROMPT_LIST[i : i + BATCH_SIZE]
        if not batch_prompts: continue

        batch_label = f"Batch {i // BATCH_SIZE + 1}"
        logger.info(f"Processing {batch_label} with {len(batch_prompts)} prompts...")
        start_time = time.time()
        generated_texts, avg_token_latency, batch_inference_vram, new_tokens_in_batch = run_inference(
            prompts=batch_prompts,
            model=model,
            tokenizer=tokenizer,
            device=gpu,
            max_new_tokens=MAX_TOKENS
        )
        end_time = time.time()
        process_time = end_time - start_time

        prompt_metrics = {
            "num_prompts_in_batch": len(batch_prompts),
            "batch_time": process_time,
            "latency": avg_token_latency,
            "tokens_generated_in_batch": new_tokens_in_batch,
            "inference_vram": batch_inference_vram
        }
        all_results[batch_label] = prompt_metrics

        total_prompts_processed += len(batch_prompts)
        total_tokens += new_tokens_in_batch
        total_time_sum += process_time

    _print_summary(all_results, total_prompts_processed, total_time_sum, total_tokens)
    logger.info(f"\n--- Execution Finished Successfully ({current_mode}) ---")

if __name__ == "__main__":
    main()