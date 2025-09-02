import os
import time
import sys
import logging
import torch
import argparse
import json
from collections import Counter
from typing import List
from transformers import AutoTokenizer
from types import SimpleNamespace

from flexllmgen.flex_opt import OptLM, Policy, SelfAttention, InputEmbed, MLP, OutputEmbed, ValueHolder
from src.custom_flex.custom_flex_opt import CustomOptLM
from flexllmgen.pytorch_backend import TorchDevice, TorchDisk, TorchMixedDevice
from flexllmgen.utils import ExecutionEnv
from src.auto_policy.profiler import get_hardware_profile
from src.auto_policy.cost_model import CostModel, get_model_info
from src.auto_policy.optimizer import find_best_policy
from src.utils.memory import calc_mem_per_device     

# Add the FlexLLMGen submodule to the Python path
flexllmgen_path = os.path.abspath("./FlexLLMGen")
if flexllmgen_path not in sys.path:
    sys.path.insert(0, flexllmgen_path)
from flexllmgen.flex_opt import Policy, CompressionConfig

logger = logging.getLogger(__name__)

class FlexRunner:
    def __init__(self, model_name: str, use_autoflex: bool, common_args: argparse.Namespace, config: SimpleNamespace, force_rerun: bool = False):
        self.model_name = model_name
        self.config = config
        self.policy = self.create_policy(common_args, use_autoflex)
        self.force_rerun = force_rerun
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = None
        self.env = None
        self.model_load_time = 0

        logger.info(f"[FlexGen] Loading model '{model_name}' with a custom policy...")

        start_time = time.time()

        cache_dir = os.path.expanduser(config.cache_path)
        offload_dir = os.path.expanduser(common_args.offload_dir)
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(offload_dir, exist_ok=True)
        
        gpu = TorchDevice("cuda:0")
        cpu = TorchDevice("cpu")
        disk = TorchDisk(offload_dir, num_copy_threads=1)
        self.env = ExecutionEnv(gpu=gpu, cpu=cpu, disk=disk, mixed=TorchMixedDevice([gpu, cpu, disk]))

        if self.config.use_custom:
            logger.info("Using custom FlexGen model...")
            self.model = CustomOptLM(self.model_name, self.env, cache_dir, self.policy)
        else:
            logger.info("Using original FlexGen model...")
            self.model = OptLM(self.model_name, self.env, cache_dir, self.policy)

        end_time = time.time()
        self.model_load_time = end_time - start_time
        
        # Log model info after init
        initial_model_info = self.get_model_info()
        logger.info("--- [FlexGen] Layer-to-Device Map ---")
        logger.info(json.dumps(initial_model_info['device_map'], indent=4))
        logger.info("-------------------------------------")
        if initial_model_info['device_sizes']:
             logger.info(f"[FlexGen] Memory Distribution Summary (GB): {initial_model_info['device_sizes']}")

    def get_policy_info(self):
        return {
            "Weights Policy(GPU/CPU/Disk)%": f"{self.policy.w_gpu_percent:.1f} / {self.policy.w_cpu_percent:.1f} / {(100 - self.policy.w_gpu_percent - self.policy.w_cpu_percent):.1f}",
        }

    def get_model_info(self):
        device_map = {}
        for i, layer in enumerate(self.model.layers):
            all_devices = []
            def get_devices(value_holder):
                items = value_holder.val
                if not isinstance(items, (list, tuple)):
                    items = [items]

                for item in items:
                    if hasattr(item, 'device'):
                        all_devices.append(item.device.name)
                    elif isinstance(item, ValueHolder):
                        get_devices(item)

            get_devices(self.model.weight_home[i])
            if not all_devices:
                device_str = "N/A"
            else:
                device_counts = Counter(all_devices)
                if len(device_counts) == 1:
                    device_str = list(device_counts.keys())[0]
                else:
                    device_str = ", ".join([f"{d}: {c}" for d, c in sorted(device_counts.items())])

            if isinstance(layer, InputEmbed):
                layer_key = "embed_tokens"
            elif isinstance(layer, SelfAttention):
                layer_key = f"decoder.layer.{layer.layer_id}.self_attn"
            elif isinstance(layer, MLP):
                layer_key = f"decoder.layer.{layer.layer_id}.mlp"
            elif isinstance(layer, OutputEmbed):
                layer_key = "lm_head"
            else:
                if hasattr(layer, 'layer_id'):
                    layer_key = f"decoder.layer.{layer.layer_id}"
                else:
                    layer_key = f"layer.{i}"

            device_map[layer_key] = device_str
        
        device_sizes = calc_mem_per_device(self.model)
        
        cache_size = 0
        hidden_size = 0
        if hasattr(self, 'input_len') and hasattr(self, 'gen_len'):
            num_prompts = self.policy.gpu_batch_size * self.policy.num_gpu_batches
            cache_size = self.model.config.cache_bytes(num_prompts, self.input_len + self.gen_len)
            hidden_size = self.model.config.hidden_bytes(num_prompts, self.input_len + self.gen_len)

        return {
            "device_map": device_map,
            "device_sizes": device_sizes,
            "cache_size_gb": cache_size / (1024**3),
            "hidden_size_gb": hidden_size / (1024**3),
        }

    def run(self, prompts: List[str], input_len: int, max_new_tokens: int) -> dict:
        if not prompts or not all(prompts):
            raise ValueError("Prompt list cannot be empty or contain empty prompts.")
        
        self.input_len = input_len
        self.gen_len = max_new_tokens
        
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
    
    def create_policy(self, common_args: argparse.Namespace, use_autoflex: bool):
        policy = None
        if use_autoflex:
            logger.info("Finding optimal policy for AutoFlex...")
            hardware_profile = get_hardware_profile(force_rerun=self.force_rerun)
            cost_model = CostModel(hardware_profile)
            model_info = get_model_info(common_args.model, common_args.batch_size)
            policy = find_best_policy(cost_model, model_info, common_args.input_len, common_args.gen_len, common_args.batch_size)
            if not policy:
                logger.error("Could not find an optimal policy for AutoFlex. Exiting.")
                return None
            logger.info(f"Optimal Policy Found: W: {policy.w_gpu_percent}/{policy.w_cpu_percent}, C: {policy.cache_gpu_percent}/{policy.cache_cpu_percent}")
        else:
            # Validate policy percentages
            if self.config.w_gpu_percent + self.config.w_cpu_percent > 100:
                raise ValueError("Weight GPU and CPU percentages cannot sum to more than 100.")
            if self.config.cache_gpu_percent + self.config.cache_cpu_percent > 100:
                raise ValueError("Cache GPU and CPU percentages cannot sum to more than 100.")

            logger.info(f"Using policy from config: "
                        f"Weights(GPU/CPU): {self.config.w_gpu_percent}/{self.config.w_cpu_percent}, "
                        f"KV Cache(GPU/CPU): {self.config.cache_gpu_percent}/{self.config.cache_cpu_percent}, "
                        f"Activations(GPU/CPU): {self.config.act_gpu_percent}/{self.config.act_cpu_percent}, "
                        f"Pinned-Memory-for-Weights: {self.config.PIN_WEIGHT}")

            policy = Policy(
                gpu_batch_size=common_args.batch_size, num_gpu_batches=1,
                w_gpu_percent=self.config.w_gpu_percent,
                w_cpu_percent=self.config.w_cpu_percent,
                cache_gpu_percent=self.config.cache_gpu_percent,
                cache_cpu_percent=self.config.cache_cpu_percent,
                act_gpu_percent=self.config.act_gpu_percent,
                act_cpu_percent=self.config.act_cpu_percent,
                overlap=True, sep_layer=True, pin_weight=self.config.PIN_WEIGHT,
                cpu_cache_compute=False, attn_sparsity=1.0,
                compress_weight=False, comp_weight_config=CompressionConfig(num_bits=16, group_size=256, group_dim=1, symmetric=False),
                compress_cache=False, comp_cache_config=CompressionConfig(num_bits=16, group_size=256, group_dim=2, symmetric=False),
            )

        return policy


    def get_policy_info(self):
        return {
            "Weights Policy(GPU/CPU/Disk)%": f"{self.policy.w_gpu_percent:.1f} / {self.policy.w_cpu_percent:.1f} / {(100 - self.policy.w_gpu_percent - self.policy.w_cpu_percent):.1f}",
        }

    def get_model_info(self):
        device_map = {}
        for i, layer in enumerate(self.model.layers):
            all_devices = []
            def get_devices(value_holder):
                items = value_holder.val
                if not isinstance(items, (list, tuple)):
                    items = [items]

                for item in items:
                    if hasattr(item, 'device'):
                        all_devices.append(item.device.name)
                    elif isinstance(item, ValueHolder):
                        get_devices(item)

            get_devices(self.model.weight_home[i])
            if not all_devices:
                device_str = "N/A"
            else:
                device_counts = Counter(all_devices)
                if len(device_counts) == 1:
                    device_str = list(device_counts.keys())[0]
                else:
                    device_str = ", ".join([f"{d}: {c}" for d, c in sorted(device_counts.items())])

            if isinstance(layer, InputEmbed):
                layer_key = "embed_tokens"
            elif isinstance(layer, SelfAttention):
                layer_key = f"decoder.layer.{layer.layer_id}.self_attn"
            elif isinstance(layer, MLP):
                layer_key = f"decoder.layer.{layer.layer_id}.mlp"
            elif isinstance(layer, OutputEmbed):
                layer_key = "lm_head"
            else:
                if hasattr(layer, 'layer_id'):
                    layer_key = f"decoder.layer.{layer.layer_id}"
                else:
                    layer_key = f"layer.{i}"

            device_map[layer_key] = device_str
        
        device_sizes = calc_mem_per_device(self.model)
        
        cache_size = 0
        hidden_size = 0
        if hasattr(self, 'input_len') and hasattr(self, 'gen_len'):
            num_prompts = self.policy.gpu_batch_size * self.policy.num_gpu_batches
            cache_size = self.model.config.cache_bytes(num_prompts, self.input_len + self.gen_len)
            hidden_size = self.model.config.hidden_bytes(num_prompts, self.input_len + self.gen_len)

        return {
            "device_map": device_map,
            "device_sizes": device_sizes,
            "cache_size_gb": cache_size / (1024**3),
            "hidden_size_gb": hidden_size / (1024**3),
        }

    def run(self, prompts: List[str], input_len: int, max_new_tokens: int) -> dict:
        if not prompts or not all(prompts):
            raise ValueError("Prompt list cannot be empty or contain empty prompts.")
        
        self.input_len = input_len
        self.gen_len = max_new_tokens
        
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
            # Validate policy percentages
            if self.config.w_gpu_percent + self.config.w_cpu_percent > 100:
                raise ValueError("Weight GPU and CPU percentages cannot sum to more than 100.")
            if self.config.cache_gpu_percent + self.config.cache_cpu_percent > 100:
                raise ValueError("Cache GPU and CPU percentages cannot sum to more than 100.")

            logger.info(f"Using policy from config: "
                        f"Weights(GPU/CPU): {self.config.w_gpu_percent}/{self.config.w_cpu_percent}, "
                        f"KV Cache(GPU/CPU): {self.config.cache_gpu_percent}/{self.config.cache_cpu_percent}, "
                        f"Activations(GPU/CPU): {self.config.act_gpu_percent}/{self.config.act_cpu_percent}, "
                        f"Pinned-Memory-for-Weights: {self.config.pin_weight}")

            policy = Policy(
                gpu_batch_size=args.batch_size, num_gpu_batches=1,
                w_gpu_percent=self.config.w_gpu_percent,
                w_cpu_percent=self.config.w_cpu_percent,
                cache_gpu_percent=self.config.cache_gpu_percent,
                cache_cpu_percent=self.config.cache_cpu_percent,
                act_gpu_percent=self.config.act_gpu_percent,
                act_cpu_percent=self.config.act_cpu_percent,
                overlap=True, sep_layer=True, pin_weight=self.config.pin_weight,
                cpu_cache_compute=False, attn_sparsity=1.0,
                compress_weight=False, comp_weight_config=CompressionConfig(num_bits=16, group_size=256, group_dim=1, symmetric=False),
                compress_cache=False, comp_cache_config=CompressionConfig(num_bits=16, group_size=256, group_dim=2, symmetric=False),
            )

        return policy