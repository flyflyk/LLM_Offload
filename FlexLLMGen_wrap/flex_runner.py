import os
import time
import logging
import torch
import numpy as np
from typing import List
from transformers import AutoTokenizer

from flexllmgen.flex_opt import OptLM, Policy, CompressionConfig
from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
from flexllmgen.utils import ExecutionEnv

logger = logging.getLogger(__name__)

class FlexRunner:
    def __init__(self, model_name: str, prompt: str, device: str = "cuda"):
        self.model_name = model_name
        self.prompt = prompt
        self.device = TorchDevice(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = None
        self.model_load_time = 0

        logger.info(f"[FlexLLMGen] Loading model '{model_name}'...")

        start_time = time.time()

        # 建立環境與路徑
        cache_path = "./flexllmgen_cache"
        offload_path = "./flexllmgen_offload"
        os.makedirs(cache_path, exist_ok=True)
        os.makedirs(offload_path, exist_ok=True)

        cpu = TorchDevice("cpu")
        disk = TorchDisk(offload_path)
        env = ExecutionEnv(gpu=self.device, cpu=cpu, disk=disk, mixed=TorchMixedDevice([self.device, cpu, disk]))

        # 建立權重與快取壓縮策略
        weight_comp_config = CompressionConfig(num_bits=16, group_size=256, group_dim=1, symmetric=False)
        cache_comp_config = CompressionConfig(num_bits=16, group_size=256, group_dim=2, symmetric=False)

        policy = Policy(
            gpu_batch_size=1,
            num_gpu_batches=1,
            w_gpu_percent=100, w_cpu_percent=0,
            cache_gpu_percent=100, cache_cpu_percent=0,
            act_gpu_percent=100, act_cpu_percent=0,
            overlap=True,
            sep_layer=True,
            pin_weight=True,
            cpu_cache_compute=False,
            attn_sparsity=1.0,
            compress_weight=False,
            comp_weight_config=weight_comp_config,
            compress_cache=False,
            comp_cache_config=cache_comp_config,
        )

        self.model = OptLM(model_name, env, cache_path, policy)

        end_time = time.time()
        self.model_load_time = end_time - start_time

        logger.info(f"[FlexLLMGen] Model loaded in {self.model_load_time:.4f} seconds.")
    
    
    def run(self, prompts: List[str], max_new_tokens: int = 50) -> dict:
        if not prompts or not all(prompts):
            raise ValueError("Prompt list cannot be empty or contain empty prompts.")
        
        logger.info(f"[FlexLLMGen] Running inference on {len(prompts)} prompts.")

        # Tokenize all prompts to numpy
        input_ids = self.tokenizer(prompts, padding="max_length", max_length=512, truncation=True, return_tensors="np").input_ids

        # Stack to create a batch (np array shape: [batch_size, seq_len])
        start_time = time.time()
        with torch.no_grad():
            outputs = self.model.generate(input_ids, max_new_tokens=max_new_tokens)
        end_time = time.time()
        
        inference_time = end_time - start_time

        # Decode outputs
        generated_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            "generated_texts": generated_texts,
            "inference_time": inference_time
        }
