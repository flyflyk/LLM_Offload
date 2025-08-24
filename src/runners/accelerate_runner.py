import torch
import logging
import time
from typing import List
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from accelerate import Accelerator

logger = logging.getLogger(__name__)

class AccelerateRunner:
    def __init__(self, model_name: str, config: object, p_type: torch.dtype = torch.float16):
        self.model_name = model_name
        self.config = config
        self.p_type = p_type
        self.use_accelerate = getattr(config, 'ENABLE_STREAMING', False)
        self.accelerator = None
        self.streamer = None
        self.model_load_time = 0

        logger.info(f"Loading model '{self.model_name}'...")
        start_time = time.time()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {"torch_dtype": self.p_type}
        
        if self.use_accelerate:
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
            
            model_kwargs["device_map"] = "auto"
            model_kwargs["offload_folder"] = getattr(config, 'OFFLOAD_FOLDER', 'offload_dir')
            
            max_ram_gb = getattr(config, 'OFFLOAD_FOLDER_MAX_CPU_OFFLOAD_RAM_GB', 0)
            if max_ram_gb > 0:
                model_kwargs["max_memory"] = { "cpu": f"{max_ram_gb}GB"}
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_kwargs["device_map"] = "auto"

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
            device_sizes = {}
            for layer_name, device in self.model.hf_device_map.items():
                try:
                    module = self.model.get_submodule(layer_name)
                    total_size = sum(p.numel() * p.element_size() for p in module.parameters())
                    if device not in device_sizes:
                        device_sizes[device] = 0
                    device_sizes[device] += total_size
                except (AttributeError, TypeError):
                    pass  # Ignore entries that are not modules or have no parameters

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
