import os
import time
import sys
import logging
import torch
import argparse
import numpy as np
from typing import List
from transformers import AutoTokenizer

from flexllmgen.flex_opt import OptLM, Policy
from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice, TorchTensor, DeviceType
from flexllmgen.utils import ExecutionEnv, ValueHolder
from src.auto_policy.profiler import get_hardware_profile
from src.auto_policy.cost_model import CostModel, get_model_info
from src.auto_policy.optimizer import find_best_policy

# Add the FlexLLMGen submodule to the Python path
flexllmgen_path = os.path.abspath("./FlexLLMGen")
if flexllmgen_path not in sys.path:
    sys.path.insert(0, flexllmgen_path)
from flexllmgen.flex_opt import Policy, CompressionConfig

logger = logging.getLogger(__name__)

def _check_vram(args, get_model_info):
    """Checks if the model weights can fit into VRAM."""
    print("--- Performing VRAM Pre-check for All-GPU Policy ---")
    model_info = get_model_info(args.model, 1)
    model_size_gb = model_info.weight_size_gb
    free_vram_bytes, _ = torch.cuda.mem_get_info(0)
    free_vram_gb = free_vram_bytes / (1024**3)

    print(f"Estimated Model Size: {model_size_gb:.2f} GB")
    print(f"Available VRAM: {free_vram_gb:.2f} GB")

    if model_size_gb > free_vram_gb * 0.95:
        print("Model is too large to fit entirely in VRAM.")
        return False
    
    print("Model should fit in VRAM.")
    return True

class FlexRunner:
    def __init__(self, model_name: str, use_autoflex: bool, args: argparse.Namespace, offload_dir: str = "./flexgen_offload", cache_dir: str = "./flexgen_cache", device: str = "cuda:0"):
        self.model_name = model_name
        self.policy = self.create_policy(args, use_autoflex)
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
        input_ids_batch = tokenized_prompts

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
        if self.env:
            self.env.close_copy_threads()
        logger.info("[FlexGen] Resources cleaned up.")
    
    def create_policy(self, args: argparse.Namespace, use_autoflex: bool):
        policy = None
        if use_autoflex:
            logger.info("Finding optimal policy for AutoFlex...")
            hardware_profile = get_hardware_profile(force_rerun=args.force_rerun_profiler)
            cost_model = CostModel(hardware_profile)
            model_info = get_model_info(args.model, args.batch_size)
            policy = find_best_policy(cost_model, model_info, args.input_len, args.gen_len, args.batch_size)
            if not policy:
                logger.error("Could not find an optimal policy for AutoFlex. Exiting.")
                return None
            logger.info(f"Optimal Policy Found: W: {policy.w_gpu_percent}/{policy.w_cpu_percent}, C: {policy.cache_gpu_percent}/{policy.cache_cpu_percent}")
        else:
            from src.flexgen import config as flex_config
            
            # Validate policy percentages
            if flex_config.W_GPU_PERCENT + flex_config.W_CPU_PERCENT > 100:
                raise ValueError("Weight GPU and CPU percentages cannot sum to more than 100.")
            if flex_config.CACHE_GPU_PERCENT + flex_config.CACHE_CPU_PERCENT > 100:
                raise ValueError("Cache GPU and CPU percentages cannot sum to more than 100.")

            # Special case: If policy is 100% GPU, perform VRAM check
            if flex_config.W_GPU_PERCENT == 100 and flex_config.W_CPU_PERCENT == 0:
                if not _check_vram(args, get_model_info):
                    logger.error("Not enough VRAM for a 100% GPU policy. "
                                 "Consider reducing W_GPU_PERCENT in config or use '--mode autoflex'. Exiting.")
                    sys.exit(1)

            logger.info(f"Using policy from config: "
                        f"Weights(GPU/CPU): {flex_config.W_GPU_PERCENT}/{flex_config.W_CPU_PERCENT}, "
                        f"Cache(GPU/CPU): {flex_config.CACHE_GPU_PERCENT}/{flex_config.CACHE_CPU_PERCENT}")

            policy = Policy(
                gpu_batch_size=args.batch_size, num_gpu_batches=1,
                w_gpu_percent=flex_config.W_GPU_PERCENT,
                w_cpu_percent=flex_config.W_CPU_PERCENT,
                cache_gpu_percent=flex_config.CACHE_GPU_PERCENT,
                cache_cpu_percent=flex_config.CACHE_CPU_PERCENT,
                act_gpu_percent=flex_config.ACT_GPU_PERCENT,
                act_cpu_percent=flex_config.ACT_CPU_PERCENT,
                overlap=True, sep_layer=True, pin_weight=True,
                cpu_cache_compute=False, attn_sparsity=1.0,
                compress_weight=False, comp_weight_config=CompressionConfig(num_bits=16, group_size=256, group_dim=1, symmetric=False),
                compress_cache=False, comp_cache_config=CompressionConfig(num_bits=16, group_size=256, group_dim=2, symmetric=False),
            )

        return policy