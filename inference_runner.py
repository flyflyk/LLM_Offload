import torch
import time
import logging
from typing import List, Tuple, Dict
from config import ENABLE_KV_OFFLOAD, PROMPT_LOG

logger = logging.getLogger(__name__)

def run_inference(
    prompts: List[str],
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 50
) -> Tuple[List[str], float, Dict[str, float], int]:
    if not prompts or not all(prompts):
        raise ValueError("Prompt list cannot be empty or contain empty prompts.")

    inference_vram_info = {}
    batch_size = len(prompts)

    kv_offload_status = "Enabled" if ENABLE_KV_OFFLOAD else "Disabled"
    if PROMPT_LOG:
        logger.info(f"Generating for batch of {batch_size} prompts (KV Offload: {kv_offload_status})")

    inputs = tokenizer(
        prompts,
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=model.config.max_position_embeddings
    )
    logger.info(f"Input IDs shape after tokenization (batch_size, seq_len): {inputs.input_ids.shape}")
    
    inputs = inputs.to(device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    logger.info(f"Inputs moved to device: {input_ids.device}")

    start_mem_allocated = 0
    torch.cuda.synchronize(device)
    start_mem_allocated = torch.cuda.memory_allocated(device)
    start_mem_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    logger.info(f"VRAM Before Generate (Device {device}) - Allocated: {start_mem_allocated / (1024**3):.2f} GB, Reserved: {start_mem_reserved / (1024**3):.2f} GB")

    generation_config_kwargs = {"use_cache": True}
    if ENABLE_KV_OFFLOAD:
        generation_config_kwargs["cache_implementation"] = "offloaded"

    logger.info("Running batched inference...")
    torch.cuda.synchronize(device)
    start_gen_time = time.time()

    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=True,
            temperature=1.0,
            top_k=50,
            **generation_config_kwargs
        )

    torch.cuda.synchronize(device)
    end_gen_time = time.time()
    total_gen_time = end_gen_time - start_gen_time

    peak_mem_allocated = torch.cuda.max_memory_allocated(device)
    peak_mem_reserved = torch.cuda.max_memory_reserved(device)
    end_mem_allocated = torch.cuda.memory_allocated(device)
    logger.info(f"VRAM After Generate (Device {device}) - Allocated: {end_mem_allocated / (1024**3):.2f} GB")
    allocated_increase = peak_mem_allocated - start_mem_allocated
    inference_vram_info = {
        "peak_allocated_gb": peak_mem_allocated / (1024**3),
        "peak_reserved_gb": peak_mem_reserved / (1024**3),
        "allocated_increase_gb": allocated_increase / (1024**3)
    }
         
    generated_texts = []
    tokens_generated = 0
    input_token_counts = [len(tokenizer.encode(p, add_special_tokens=False)) for p in prompts]

    for i in range(batch_size):
        generated_ids = generation_output[i].cpu()
        input_len = input_token_counts[i]    
        start_of_generation_index = inputs.input_ids.shape[1]
        newly_generated_ids = generated_ids[start_of_generation_index:]
        generated_text_for_item = tokenizer.decode(newly_generated_ids, skip_special_tokens=True)
        generated_texts.append(generated_text_for_item.strip())
        
        current_new_tokens = len(newly_generated_ids)
        tokens_generated += current_new_tokens

        if PROMPT_LOG:
            logger.info(f"Batch item {i+1} - Prompt: '{prompts[i][:100]}...'")
            logger.info(f"Batch item {i+1} - Generated part: '{generated_text_for_item.strip()}'")
            logger.info(f"Batch item {i+1} - Input tokens (padded): {start_of_generation_index}, Original prompt tokens: {input_len}, Output tokens (incl. input): {len(generated_ids)}, New tokens: {current_new_tokens}")


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