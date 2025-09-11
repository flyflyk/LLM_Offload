import json
import os
import time
import torch
import psutil
import dataclasses
import logging

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class HardwareProfile:
    gpu_mem: int
    cpu_mem: int
    cpu_gpu_bandwidth: float
    disk_cpu_bandwidth: float
    peak_gpu_tflops: float

def _profile_bandwidth(src_device: str, dst_device: str, size_mb: int = 256) -> float:
    tensor = torch.randn(size_mb * 1024 * 1024 // 4, dtype=torch.float32, device=src_device)
    # Warmup transfers
    for _ in range(3):
        tensor.to(dst_device)
    
    if 'cuda' in [src_device, dst_device]:
        torch.cuda.synchronize()
    
    start_time = time.time()
    tensor.to(dst_device)
    if 'cuda' in [src_device, dst_device]:
        torch.cuda.synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    return (size_mb * 1024 * 1024) / duration if duration > 0 else 0

def _profile_compute(device: str, model_dim: int = 4096) -> float:
    if device == 'cpu' and model_dim > 2048:
        model_dim = 2048 # Use a smaller matrix for CPU to avoid excessive profiling time

    N, K, M = model_dim, model_dim, model_dim
    a = torch.randn(N, K, device=device, dtype=torch.float16)
    b = torch.randn(K, M, device=device, dtype=torch.float16)
    
    # Warmup computations
    for _ in range(3):
        torch.matmul(a, b)
    
    if 'cuda' in device:
        torch.cuda.synchronize()
    
    start_time = time.time()
    torch.matmul(a, b)
    if 'cuda' in device:
        torch.cuda.synchronize()
    end_time = time.time()
    
    duration = end_time - start_time
    flops = 2 * N * K * M
    tflops = (flops / duration) / 1e12 if duration > 0 else 0
    return tflops

def get_hardware_profile(profile_path: str = "hardware_profile.json", force_rerun: bool = False) -> HardwareProfile:
    cache_dir = os.path.dirname(profile_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if not force_rerun and os.path.exists(profile_path) and os.path.getsize(profile_path) > 0:
        logger.info(f"Loading cached hardware profile from {profile_path}...")
        with open(profile_path, 'r') as f:
            profile_dict = json.load(f)
            return HardwareProfile(**profile_dict)

    logger.info("Running hardware profiling... (This may take a moment)")
    
    # Profile memory
    gpu_mem = torch.cuda.get_device_properties(0).total_memory
    cpu_mem = psutil.virtual_memory().total

    # Profile bandwidth (Bytes/s)
    cpu_gpu_bw = _profile_bandwidth('cpu', 'cuda:0')
    # Simulate disk bandwidth as a fraction of CPU memory bandwidth
    disk_cpu_bw = _profile_bandwidth('cpu', 'cpu') * 0.1 

    # Profile compute (TFLOPS)
    gpu_tflops = _profile_compute('cuda:0')

    profile = HardwareProfile(
        gpu_mem=gpu_mem,
        cpu_mem=cpu_mem,
        cpu_gpu_bandwidth=cpu_gpu_bw,
        disk_cpu_bandwidth=disk_cpu_bw,
        peak_gpu_tflops=gpu_tflops,
    )
    with open(profile_path, 'w') as f:
        json.dump(dataclasses.asdict(profile), f, indent=4)
        
    return profile

if __name__ == '__main__':
    profile = get_hardware_profile(force_rerun=True)
    print("Hardware Profile:")
    print(json.dumps(dataclasses.asdict(profile), indent=4))