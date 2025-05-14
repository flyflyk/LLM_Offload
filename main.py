import torch
import time
import logging
import sys
from typing import List, Tuple, Dict
from logger import setup_logging
from model_loader import load_model
from config import (
    CHOSEN_MODEL, MAX_TOKENS, ENABLE_STREAMING, PROMPT_LIST,
    ENABLE_KV_OFFLOAD, PROMPT_LOG, BATCH_SIZE, DEVICE, LOG_FILE
)

setup_logging(log_file=LOG_FILE if 'LOG_FILE' in globals() and LOG_FILE else None)
logger = logging.getLogger(__name__)

def _run_inference_batch(
    prompts: List[str],
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 50
) -> Tuple[List[str], float, Dict[str, float], int]:
    if not prompts or not all(prompts):
        raise ValueError("Prompt list cannot be empty or contain empty prompts.")

    inference_vram_info = {}
    is_cuda = (device.type == "cuda")
    batch_actual_size = len(prompts)

    kv_offload_status = "Enabled" if ENABLE_KV_OFFLOAD else "Disabled"
    if PROMPT_LOG:
        logger.info(f"Generating for batch of {batch_actual_size} prompts (KV Offload: {kv_offload_status})")

    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=model.config.max_position_embeddings
    )
    logger.info(f"Input IDs shape after tokenization (batch_size, seq_len): {inputs.input_ids.shape}")

    try:
        input_embeddings_device = model.get_input_embeddings().weight.device
    except AttributeError:
        logger.warning("Could not determine input embedding device automatically. Using model's main device.")
        input_embeddings_device = model.device
    
    inputs = inputs.to(input_embeddings_device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    logger.info(f"Inputs moved to device: {input_ids.device}")

    start_mem_allocated = 0
    if is_cuda:
        torch.cuda.synchronize(device=input_ids.device if input_ids.device.type == 'cuda' else None)
        start_mem_allocated = torch.cuda.memory_allocated(device)
        start_mem_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.reset_peak_memory_stats(device)
        logger.info(f"VRAM Before Generate (Device {device}) - Allocated: {start_mem_allocated / (1024**3):.2f} GB, Reserved: {start_mem_reserved / (1024**3):.2f} GB")

    generation_config_kwargs = {"use_cache": True}
    if ENABLE_KV_OFFLOAD:
        generation_config_kwargs["cache_implementation"] = "offloaded"
        logger.info("Attempting to set cache_implementation to 'offloaded'.")
    else:
        logger.info("KV Offload disabled by config.")

    logger.info("Running batched inference...")
    if input_ids.device.type == 'cuda':
        torch.cuda.synchronize(device=input_ids.device)
    start_gen_time = time.time()

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None \
                else (tokenizer.eos_token_id if tokenizer.eos_token_id is not None \
                      else model.config.eos_token_id),
            do_sample=True,
            temperature=1.0,
            top_k=50,
            **generation_config_kwargs
        )

    if input_ids.device.type == 'cuda':
        torch.cuda.synchronize(device=input_ids.device)
    end_gen_time = time.time()
    total_gen_time = end_gen_time - start_gen_time

    if is_cuda:
         peak_mem_allocated = torch.cuda.max_memory_allocated(device)
         peak_mem_reserved = torch.cuda.max_memory_reserved(device)
         end_mem_allocated = torch.cuda.memory_allocated(device)
         logger.info(f"VRAM After Generate (Device {device}) - Allocated: {end_mem_allocated / (1024**3):.2f} GB")
         logger.info(f"VRAM Peak During Generate (Device {device}) - Allocated: {peak_mem_allocated / (1024**3):.2f} GB, Reserved: {peak_mem_reserved / (1024**3):.2f} GB")
         allocated_increase = peak_mem_allocated - start_mem_allocated
         inference_vram_info = {
             "peak_allocated_gb": peak_mem_allocated / (1024**3),
             "peak_reserved_gb": peak_mem_reserved / (1024**3),
             "allocated_increase_gb": allocated_increase / (1024**3)
         }

    generated_texts = []
    tokens_generated = 0
    input_token_counts = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]

    for i in range(batch_actual_size):
        generated_ids_for_item = generation_output[i].cpu()
        input_len = input_token_counts[i]    
        start_of_generation_index = inputs.input_ids.shape[1]
        newly_generated_ids = generated_ids_for_item[start_of_generation_index:]
        generated_text_for_item = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
        generated_texts.append(generated_text_for_item.strip())
        
        current_new_tokens = len(newly_generated_ids)
        tokens_generated += current_new_tokens

        if PROMPT_LOG:
            logger.info(f"Batch item {i+1} - Prompt: '{prompts[i][:100]}...'")
            logger.info(f"Batch item {i+1} - Generated part: '{generated_text_for_item.strip()}'")
            logger.info(f"Batch item {i+1} - Input tokens (padded): {start_of_generation_index}, Original prompt tokens: {input_len}, Output tokens (incl. input): {len(generated_ids_for_item)}, New tokens: {current_new_tokens}")


    avg_token_latency = float('inf')
    if tokens_generated > 0:
        avg_token_latency = total_gen_time / tokens_generated
        logger.info(f"Batch generated {tokens_generated} new tokens in total.")
        logger.info(f"Total generation time for batch: {total_gen_time:.4f} seconds.")
        logger.info(f"Average latency per new token in batch: {avg_token_latency:.4f} seconds/token.")
        logger.info(f"Batch throughput: {tokens_generated / total_gen_time:.2f} tokens/second.")
    else:
        logger.info(f"No new tokens were generated for the batch.")
        logger.info(f"Processing time for batch (no new tokens): {total_gen_time:.4f} seconds.")
    
    logger.info("-" * 30)
    return generated_texts, avg_token_latency, inference_vram_info, tokens_generated


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
    model, tokenizer, device_from_loader, model_vram_info = load_model(model_name=CHOSEN_MODEL)
    logger.info(f"Model '{CHOSEN_MODEL}' loaded. Main computation device determined by loader: {device_from_loader}.")
    effective_device = device_from_loader
    is_cuda = (effective_device.type == "cuda")

    if model_vram_info and is_cuda:
         logger.info(f"Initial Model VRAM (After Load) - Allocated: {model_vram_info['after_load_allocated_gb']:.2f} GB, Footprint: {model_vram_info['model_footprint_gb']:.2f} GB")

    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            logger.warning("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            if model.config.pad_token_id is None :
                 model.config.pad_token_id = model.config.eos_token_id
        else:
            raise ValueError("Tokenizer has no pad_token and no eos_token. Cannot proceed with batching that requires padding.")
    
    if model.config.pad_token_id is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        logger.info(f"Set model.config.pad_token_id to {model.config.pad_token_id}")


    all_results = {}
    if model_vram_info and is_cuda:
        all_results["model_vram"] = model_vram_info

    logger.info("\n--- Running Measured Inference Examples in Batches ---")
    
    total_prompts_processed = 0
    total_time_sum = 0
    total_tokens = 0

    for i in range(0, len(PROMPT_LIST), BATCH_SIZE):
        batch_prompts = PROMPT_LIST[i : i + BATCH_SIZE]
        if not batch_prompts: continue

        batch_label = f"Batch {i // BATCH_SIZE + 1}"
        logger.info(f"Processing {batch_label} with {len(batch_prompts)} prompts...")
        batch_processing_start_time = time.time()
        generated_texts, avg_token_latency, batch_inference_vram, new_tokens_in_batch = _run_inference_batch(
            prompts=batch_prompts,
            model=model,
            tokenizer=tokenizer,
            device=effective_device,
            max_new_tokens=MAX_TOKENS
        )
        batch_processing_end_time = time.time()
        current_batch_processing_time = batch_processing_end_time - batch_processing_start_time

        total_prompts_processed += len(batch_prompts)
        
        prompt_metrics = {
            "num_prompts_in_batch": len(batch_prompts),
            "batch_time": current_batch_processing_time,
            "latency": avg_token_latency,
            "tokens_generated_in_batch": new_tokens_in_batch
        }

        if new_tokens_in_batch is not None and new_tokens_in_batch > 0:
            total_tokens += new_tokens_in_batch
            total_time_sum += current_batch_processing_time


        if batch_inference_vram and is_cuda:
             prompt_metrics["inference_vram"] = batch_inference_vram
        
        all_results[batch_label] = prompt_metrics

    _print_summary(all_results, total_prompts_processed, total_time_sum, total_tokens)
    logger.info(f"\n--- Execution Finished Successfully ({current_mode}) ---")

if __name__ == "__main__":
    main()