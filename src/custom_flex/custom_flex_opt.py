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

def init_weight_list(weight_specs, policy, env, model):
    gpu_capacity, cpu_capacity, disk_capacity = model.dev_capacities
    gpu_limit = gpu_capacity
    cpu_limit = gpu_capacity + cpu_capacity

    ret = []
    for spec in weight_specs:
        shape, dtype, filename = spec
        size_bytes = np.prod(shape) * np.dtype(dtype).itemsize
        if model.dev_used[0] < gpu_limit:
            home = env.gpu
        elif model.dev_used[0] < cpu_limit:
            home = env.cpu
        else:
            home = env.disk

        # Increment the total processed size.
        model.dev_used[0] += size_bytes

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
    def __init__(self, model, config, env, policy):
        super().__init__(config, env, policy)
        self.model = model

    def get_weight_specs(self, path):
        v, h, s, dtype = (self.config.vocab_size, self.config.input_dim,
            self.config.max_seq_len, self.config.dtype)
        path = os.path.join(path, "")
        return [
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
            ((s + 2, h), dtype, path + "decoder.embed_positions.weight"),
        ]

    def init_weight(self, weight_home, path):
        weights = init_weight_list(self.get_weight_specs(path), self.policy, self.env, self.model)
        weight_home.store(weights)

class CustomOutputEmbed(OutputEmbed):
    def __init__(self, model, config, env, policy):
        super().__init__(config, env, policy)
        self.model = model

    def get_weight_specs(self, path):
        v, h, dtype = (self.config.vocab_size, self.config.input_dim, self.config.dtype)
        path = os.path.join(path, "")
        return [
            ((h,), dtype, path + "decoder.layer_norm.weight"),
            ((h,), dtype, path + "decoder.layer_norm.bias"),
            ((v, h), dtype, path + "decoder.embed_tokens.weight"),
        ]

    def init_weight(self, weight_home, path):
        weights = init_weight_list(self.get_weight_specs(path), self.policy, self.env, self.model)
        weight_home.store(weights)

class CustomSelfAttention(SelfAttention):
    def __init__(self, model, config, env, policy, layer_id):
        super().__init__(config, env, policy, layer_id)
        self.model = model

    def get_weight_specs(self, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}.self_attn"))
        return [
            ((h, h), dtype, path + ".q_proj.weight"), ((h,), dtype, path + ".q_proj.bias"),
            ((h, h), dtype, path + ".k_proj.weight"), ((h,), dtype, path + ".k_proj.bias"),
            ((h, h), dtype, path + ".v_proj.weight"), ((h,), dtype, path + ".v_proj.bias"),
            ((h, h), dtype, path + ".out_proj.weight"), ((h,), dtype, path + ".out_proj.bias"),
            ((h,), dtype, path + "_layer_norm.weight"), ((h,), dtype, path + "_layer_norm.bias"),
        ]

    def init_weight(self, weight_home, path):
        weights = init_weight_list(self.get_weight_specs(path), self.policy, self.env, self.model)
        weight_home.store(weights)

class CustomMLP(MLP):
    def __init__(self, model, config, env, policy, layer_id):
        super().__init__(config, env, policy, layer_id)
        self.model = model

    def get_weight_specs(self, path):
        h, dtype = (self.config.input_dim, self.config.dtype)
        path = os.path.join(os.path.join(path, f"decoder.layers.{self.layer_id}."))
        return [
            ((4 * h, h), dtype, path + "fc1.weight"), ((4 * h,), dtype, path + "fc1.bias"),
            ((h, 4 * h), dtype, path + "fc2.weight"), ((h,), dtype, path + "fc2.bias"),
            ((h,), dtype, path + "final_layer_norm.weight"), ((h,), dtype, path + "final_layer_norm.bias"),
        ]

    def init_weight(self, weight_home, path):
        weights = init_weight_list(self.get_weight_specs(path), self.policy, self.env, self.model)
        weight_home.store(weights)

class CustomTransformerLayer(TransformerLayer):
    def __init__(self, model, config, env, policy, i):
        self.attention = CustomSelfAttention(model, config, env, policy, i)
        self.mlp = CustomMLP(model, config, env, policy, i)
        self.policy = policy
        self.compute = self.attention.compute
        self.model = model

    def get_weight_specs(self, path):
        specs = self.attention.get_weight_specs(path)
        specs.extend(self.mlp.get_weight_specs(path))
        return specs

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
        layers.append(CustomInputEmbed(self, self.config, self.env, self.policy))
        for i in range(self.config.num_hidden_layers):
            if policy.sep_layer:
                layers.append(CustomSelfAttention(self, self.config, self.env, self.policy, i))
                layers.append(CustomMLP(self, self.config, self.env, self.policy, i))
            else:
                layers.append(CustomTransformerLayer(self, self.config, self.env, self.policy, i))
        layers.append(CustomOutputEmbed(self, self.config, self.env, self.policy))
        self.layers = layers
        self.num_layers = len(layers)

        self._pre_calc_capacity()

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

    def _pre_calc_capacity(self):
        expanded_path = os.path.abspath(os.path.expanduser(
            os.path.join(self.path, f"{self.config.name}-np")))

        all_weight_specs = []
        for layer in self.layers:
            all_weight_specs.extend(layer.get_weight_specs(expanded_path))

        total_model_size_bytes = sum(np.prod(spec[0]) * np.dtype(spec[1]).itemsize for spec in all_weight_specs)

        self.dev_capacities = [
            total_model_size_bytes * (self.policy.w_gpu_percent / 100.0),
            total_model_size_bytes * (self.policy.w_cpu_percent / 100.0),
            total_model_size_bytes * ((100 - (self.policy.w_gpu_percent + self.policy.w_cpu_percent)) / 100.0),
        ]
        # Use a single counter for total processed size for partitioning
        self.dev_used = [0]