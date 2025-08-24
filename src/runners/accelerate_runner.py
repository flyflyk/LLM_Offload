import torch
import logging
import time
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, AutoConfig
from accelerate import Accelerator
from src.utils.memory import get_model_device_mem, get_max_mem_dict

logger = logging.getLogger(__name__)

class AccelerateRunner:
    def __init__(self, model_name: str, config: object, device_map: dict, p_type: torch.dtype = torch.float16):
        self.model_name = model_name
        self.config = config
        self.p_type = p_type
        self.device_map = device_map
        self.use_accelerate = getattr(config, 'ENABLE_STREAMING', False)
        self.accelerator = None
        self.streamer = None
        self.model_load_time = 0

        logger.info(f"Loading model '{self.model_name}' with a predefined device map...")
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "torch_dtype": self.p_type,
            "device_map": self.device_map,
            "offload_folder": getattr(config, 'OFFLOAD_FOLDER', 'offload_dir')
        }
        
        if self.use_accelerate:
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)

        if self.use_accelerate:
            self.model = self.accelerator.prepare(self.model)
        
        end_time = time.time()
        self.model_load_time = end_time - start_time

        if getattr(config, 'ENABLE_STREAMING', False):
            # For benchmark mode, we don't want to stream to stdout
            if not getattr(config, 'IS_BENCHMARK', False):
                self.streamer = TextStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        logger.info(f"Model '{self.model_name}' loaded in {self.model_load_time:.4f} seconds.")
        if hasattr(self.model, 'hf_device_map'):
            logger.info(f"Model device map: {self.model.hf_device_map}")
            device_sizes = get_model_device_mem(self.model, self.model.hf_device_map)
            # Convert to GB
            for device, total_size in device_sizes.items():
                device_sizes[device] = f"{total_size / (1024**3):.4f} GB"

            if device_sizes:
                logger.info(f"Model weights size per device (GB): {device_sizes}")
        else:
            logger.info(f"Model loaded on device: {self.device}")

    def run_accelerate(self, prompts: List[str], max_new_tokens: int = 50) -> dict:
        if not prompts or not all(prompts):
            raise ValueError("Prompt list cannot be empty or contain empty prompts.")

        logger.info(f"Running accelerate for batch of {len(prompts)} prompts.")

        use_streamer_for_this_run = self.streamer and len(prompts) == 1

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
        
        if use_streamer_for_this_run:
            generation_kwargs["streamer"] = self.streamer

        start_time = time.time()
        with torch.no_grad():
            generation_output = self.model.generate(
                **inputs,
                **generation_kwargs
            )
        end_time = time.time()
        
        inference_time = end_time - start_time
        
        if use_streamer_for_this_run:
            generated_texts = [""] * len(prompts) 
        else:
            input_token_len = inputs.input_ids.shape[1]
            generated_tokens = generation_output[:, input_token_len:]
            generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return {
            "generated_texts": generated_texts,
            "inference_time": inference_time
        }
