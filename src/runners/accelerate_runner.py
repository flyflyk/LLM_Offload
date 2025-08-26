import torch
import logging
import time
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM
from accelerate import Accelerator
from src.utils.memory import calc_mem_per_device

logger = logging.getLogger(__name__)

class AccelerateRunner:
    def __init__(self, model_name: str, config: object, device_map: dict, p_type: torch.dtype = torch.float16):
        self.model_name = model_name
        self.config = config
        self.p_type = p_type
        self.device_map = device_map
        self.use_offload = getattr(config, 'ENABLE_OFFLOAD', False)
        self.accelerator = None
        self.streamer = None
        self.model_load_time = 0

        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token if self.tokenizer.pad_token is None else self.tokenizer.pad_token
        
        model_kwargs = {
            "torch_dtype": self.p_type,
            "device_map": self.device_map,
            "offload_folder": getattr(config, 'OFFLOAD_FOLDER', 'offload_dir')
        }
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        if self.use_offload:
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            self.model = self.accelerator.prepare(self.model)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        self.model = torch.compile(self.model, mode="reduce-overhead")
        
        end_time = time.time()
        self.model_load_time = end_time - start_time

        device_sizes = calc_mem_per_device(self.model)
        logger.info(f"Model weights size per device (GB): {device_sizes}")

    def run_accelerate(self, prompts: List[str], max_new_tokens: int = 50) -> dict:
        if not prompts or not all(prompts):
            raise ValueError("Prompt list cannot be empty or contain empty prompts.")

        target_device = self.model.device
        inputs = self.tokenizer(
            prompts,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).to(target_device)

        generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "do_sample": True,
            "temperature": 1.0,
            "top_k": 50,
        }
        
        if self.config.ENABLE_KV_OFFLOAD:
            generation_kwargs["cache_implementation"] = "offloaded_static"

        start_time = time.time()
        with torch.no_grad():
            generation_output = self.model.generate(
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
