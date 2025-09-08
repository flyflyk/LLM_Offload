import torch
import logging
import time
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from src.utils.memory import calc_mem_per_device

logger = logging.getLogger(__name__)

class AccelerateRunner:
    def __init__(self, model_name: str, config: object, device_map: dict, offload_folder: str, p_type: torch.dtype = torch.float16):
        self.model_name = model_name
        self.config = config
        self.accelerator = None
        self.model_load_time = 0

        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.pad_token is None else self.tokenizer.pad_token
        
        model_kwargs = {
            "device_map": device_map,
            "offload_folder": offload_folder
        }

        if self.config.quantize_4bit:
            logger.info("Loading model with 4-bit quantization...")
            model_kwargs.update({
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16 
            })
        else:
            model_kwargs["torch_dtype"] = p_type

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        if config.enable_offload and not self.config.quantize_4bit:
            self.accelerator = Accelerator()
            self.model = self.accelerator.prepare(self.model)
        
        end_time = time.time()
        self.model_load_time = end_time - start_time

        device_sizes = calc_mem_per_device(self.model)
        logger.info(f"Model weights size per device (GB): {device_sizes}")

    def run_accelerate(self, prompts: List[str], max_new_tokens: int = 50) -> dict:
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
        )
        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
        }
        
        if self.config.enable_kv_offload:
            generation_kwargs["cache_implementation"] = "offloaded"

        start_time = time.time()
        with torch.no_grad():
            model_to_generate = self.accelerator.unwrap_model(self.model) if self.accelerator else self.model
            generation_output = model_to_generate.generate(
                **inputs,
                **generation_kwargs
            )
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        input_token_len = inputs.input_ids.shape[1]
        generated_tokens = generation_output[:, input_token_len:]
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return {
            "generated_texts": generated_texts,
            "inference_time": inference_time
        }