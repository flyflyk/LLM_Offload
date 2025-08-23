import os
import time
import logging
import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer

from flexllmgen.flex_opt import OptLM, Policy
from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice, TorchTensor, DeviceType
from flexllmgen.utils import ExecutionEnv, ValueHolder

logger = logging.getLogger(__name__)

class FlexRunner:
    def __init__(self, model_name: str, policy: Policy, offload_dir: str = "./flexgen_offload", cache_dir: str = "./flexgen_cache", device: str = "cuda"):
        self.model_name = model_name
        self.policy = policy
        self.device = TorchDevice(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = None
        self.env = None
        self.model_load_time = 0

        logger.info(f"[FlexGen] Loading model '{model_name}' with a custom policy...")

        start_time = time.time()

        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(offload_dir, exist_ok=True)
        gpu = self.device
        cpu = TorchDevice("cpu")
        disk = TorchDisk(offload_dir, num_copy_threads=1)
        self.env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

        self.model = OptLM(self.model_name, self.env, cache_dir, self.policy)

        end_time = time.time()
        self.model_load_time = end_time - start_time

        logger.info(f"[FlexGen] Model loaded in {self.model_load_time:.4f} seconds.")
        self.log_model_size()

    def log_model_size(self):
        device_sizes = {}

        def get_tensor_size(tensor):
            if tensor is None:
                return

            if isinstance(tensor, (list, tuple)):
                for t in tensor:
                    get_tensor_size(t)
            elif isinstance(tensor, ValueHolder):
                get_tensor_size(tensor.val)
            elif isinstance(tensor, TorchTensor):
                if tensor.device.device_type == DeviceType.MIXED:
                    for t in tensor.data[0]:
                        get_tensor_size(t)
                else:
                    device_name = tensor.device.name
                    if device_name not in device_sizes:
                        device_sizes[device_name] = 0
                    device_sizes[device_name] += tensor.bytes

        for layer_weight_holder in self.model.weight_home:
            get_tensor_size(layer_weight_holder.val)

        # Convert to GB
        for device, total_size in device_sizes.items():
            device_sizes[device] = f"{total_size / (1024**3):.4f} GB"

        if device_sizes:
            logger.info(f"[FlexGen] Model weights size per device (GB): {device_sizes}")

    def run(self, prompts: List[str], input_len: int, max_new_tokens: int) -> dict:
        if not prompts or not all(prompts):
            raise ValueError("Prompt list cannot be empty or contain empty prompts.")
        
        logger.info(f"[FlexGen] Running inference on {len(prompts)} prompts (input_len: {input_len}, gen_len: {max_new_tokens}).")

        # Tokenize and create a batch
        tokenized_prompts = self.tokenizer(prompts, padding="max_length", max_length=input_len, truncation=True, return_tensors="np").input_ids
        input_ids_batch = np.tile(tokenized_prompts, (len(prompts), 1))

        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(input_ids_batch, max_new_tokens=max_new_tokens)
        end_time = time.time()
        
        inference_time = end_time - start_time

        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            "generated_texts": generated_texts,
            "inference_time": inference_time,
            "load_time": self.model_load_time
        }

    def cleanup(self):
        """Cleans up resources, especially for offloading."""
        if self.env:
            self.env.close_copy_threads()
        logger.info("[FlexGen] Resources cleaned up.")