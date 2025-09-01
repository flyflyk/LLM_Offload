import numpy as np
import os
import torch

from FlexLLMGen.flexllmgen.flex_opt import (
    OptLM, InputEmbed, OutputEmbed, SelfAttention, MLP, TransformerLayer,
    get_opt_config, DUMMY_WEIGHT
)
from FlexLLMGen.flexllmgen.pytorch_backend import (
    torch_dtype_to_np_dtype
)
from FlexLLMGen.flexllmgen.utils import ValueHolder

def init_weight_list(weight_specs, policy, env):
    devices = [env.gpu, env.cpu, env.disk]
    total_weight_size = sum(np.prod(spec[0]) * np.dtype(spec[1]).itemsize for spec in weight_specs)
    
    dev_capacities = [
        total_weight_size * (policy.w_gpu_percent / 100.0),
        total_weight_size * (policy.w_cpu_percent / 100.0),
        float('inf')
    ]
    
    dev_used = [0, 0, 0]
    
    ret = []
    for spec in weight_specs:
        shape, dtype, filename = spec
        size_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        
        home = None
        for i in range(len(devices)):
            if dev_used[i] + size_bytes <= dev_capacities[i]:
                home = devices[i]
                dev_used[i] += size_bytes
                break
        
        if home is None:
            # Fallback to disk if no space, though this shouldn't be hit with inf disk
            home = env.disk

        if len(shape) < 2:
            pin_memory = True
            compress = False
        else:
            pin_memory = policy.pin_weight
            compress = policy.compress_weight

        if not compress:
            weight = home.allocate(shape, dtype, pin_memory=pin_memory)
            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(filename)
            else:
                weight.load_from_np(np.ones(shape, dtype))
        else:
            weight = home.compressed_device.allocate(
                shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)
            if DUMMY_WEIGHT not in filename:
                weight.load_from_np_file(filename)
            else:
                for i in range(2):
                    x = weight.data[i]
                    x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))
        
        ret.append(weight)
        
    return ret

class CustomInputEmbed(InputEmbed):
    def init_weight(self, weight_home, path):
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.max_seq_len, self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
            ((s + 2, h), dtype, path + "decoder.embed_positions.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

class CustomOutputEmbed(OutputEmbed):
    def init_weight(self, weight_home, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.dtype)
        path = os.path.join(path, "")
        weight_specs = [
            ((h,), dtype, path + "decoder.layer_norm.weight"),
            ((h,), dtype, path + "decoder.layer_norm.bias"),
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

class CustomSelfAttention(SelfAttention):
    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        weight_specs = [
            ((h, h), dtype, path + ".q_proj.weight"),
            ((h,), dtype, path + ".q_proj.bias"),
            ((h, h), dtype, path + ".k_proj.weight"),
            ((h,), dtype, path + ".k_proj.bias"),
            ((h, h), dtype, path + ".v_proj.weight"),
            ((h,), dtype, path + ".v_proj.bias"),
            ((h, h), dtype, path + ".out_proj.weight"),
            ((h,), dtype, path + ".out_proj.bias"),
            ((h,), dtype, path + "_layer_norm.weight"),
            ((h,), dtype, path + "_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

class CustomMLP(MLP):
    def init_weight(self, weight_home, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        weight_specs = [
            ((4 * h, h), dtype, path + "fc1.weight"),
            ((4 * h,), dtype, path + "fc1.bias"),
            ((h, 4 * h), dtype, path + "fc2.weight"),
            ((h,), dtype, path + "fc2.bias"),
            ((h,), dtype, path + "final_layer_norm.weight"),
            ((h,), dtype, path + "final_layer_norm.bias"),
        ]
        weights = init_weight_list(weight_specs, self.policy, self.env)
        weight_home.store(weights)

class CustomTransformerLayer(TransformerLayer):
    def __init__(self, config, env, policy, i):
        # Override to use custom SelfAttention and MLP
        self.attention = CustomSelfAttention(config, env, policy, i)
        self.mlp = CustomMLP(config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute

class CustomOptLM(OptLM):
    def __init__(self, config, env, path, policy):
        if isinstance(config, str):
            config = get_opt_config(config)
        self.config = config
        self.env = env
        self.path = path
        self.policy = policy
        self.num_gpu_batches = policy.num_gpu_batches

        layers = []
        layers.append(CustomInputEmbed(self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(CustomSelfAttention(self.config, self.env, self.policy, i))
                layers.append(CustomMLP(self.config, self.env, self.policy, i))
            else:
                layers.append(CustomTransformerLayer(self.config, self.env, self.policy, i))
        layers.append(CustomOutputEmbed(self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        if self.policy.act_gpu_percent == 100:
            self.act_home = self.env.gpu
        elif self.policy.act_cpu_percent == 100:
            self.act_home = self.env.cpu
        elif self.policy.act_disk_percent == 100:
            self.act_home = self.env.disk
        else:
            raise NotImplementedError()
        
        self.load_weight_stream = torch.cuda.Stream()
        self.load_cache_stream = torch.cuda.Stream()
        self.store_cache_stream = torch.cuda.Stream()

        num_layers, num_gpu_batches = self.num_layers, self.policy.num_gpu_batches
        self.cache_home = [[ValueHolder() for _ in range(num_gpu_batches)] for _ in range(num_layers)]
        self.cache_read_buf = [[ValueHolder() for _ in range(num_gpu_batches)] for _ in range(num_layers)]
        self.cache_write_buf = [[ValueHolder() for _ in range(num_gpu_batches)] for _ in range(num_layers)]
        self.weight_read_buf = [ValueHolder() for _ in range(num_layers)]
        self.attention_mask = [ValueHolder() for _ in range(num_gpu_batches)]

        self.task = None
        self.init_all_weights()
