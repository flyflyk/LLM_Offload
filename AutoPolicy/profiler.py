import json
import os
import time
import torch

def _profile_bandwidth_mb_s(src_device: str, dst_device: str, size_mb: int = 128) -> float:
    """
    Measures the bandwidth between two torch devices in MB/s.
    """
    tensor = torch.randn(size_mb * 1024 * 1024 // 4, dtype=torch.float32, device=src_device)
    # Warmup transfers to ensure accurate measurement.
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
    return size_mb / duration if duration > 0 else 0

def _profile_compute_tflops(device: str, model_dim: int = 4096) -> float:
    """
    Measures the compute performance of a device in TFLOPS.
    """
    N, K, M = model_dim, model_dim, model_dim
    a = torch.randn(N, K, device=device, dtype=torch.float16)
    b = torch.randn(K, M, device=device, dtype=torch.float16)
    
    # Warmup computations.
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

def get_hardware_profile(profile_path: str = "AutoPolicy/hardware_profile.json", force_rerun: bool = False) -> dict:
    """
    Gets hardware profile, from a cached file if available, otherwise runs profiling.
    """
    if not force_rerun and os.path.exists(profile_path):
        print("Loading cached hardware profile...")
        with open(profile_path, 'r') as f:
            return json.load(f)

    print("Running hardware profiling... (This may take a moment)")
    
    profile = {}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available for profiling.")
            
        profile = {
            # Bandwidth in GB/s
            "gpu_cpu_bandwidth": _profile_bandwidth_mb_s('cuda:0', 'cpu', 256) / 1024,
            "cpu_gpu_bandwidth": _profile_bandwidth_mb_s('cpu', 'cuda:0', 256) / 1024,
            # Simulate Disk bandwidth as 1/10th of CPU-CPU transfer for demonstration.
            "disk_cpu_bandwidth": _profile_bandwidth_mb_s('cpu', 'cpu', 256) / 1024 * 0.1,
            # Compute in TFLOPS (FP16)
            "gpu_tflops": _profile_compute_tflops('cuda:0'),
            "cpu_tflops": _profile_compute_tflops('cpu'),
        }
    except (RuntimeError, FileNotFoundError) as e:
        # Fallback to placeholder values if profiling fails (e.g., no GPU).
        print(f"Warning: Could not perform real profiling ({e}). Using default placeholder values.")
        profile = {
            "gpu_cpu_bandwidth": 12.0,  # GB/s
            "cpu_gpu_bandwidth": 12.0,  # GB/s
            "disk_cpu_bandwidth": 1.0,  # GB/s
            "gpu_tflops": 150.0,        # TFLOPS (A100-like)
            "cpu_tflops": 2.0,          # TFLOPS
        }

    print("Profiling complete.")
    # Cache the results.
    with open(profile_path, 'w') as f:
        json.dump(profile, f, indent=4)
        
    return profile

if __name__ == '__main__':
    """Run profiler directly to generate a hardware profile cache."""
    profile = get_hardware_profile(force_rerun=True)
    print("Hardware Profile:")
    print(json.dumps(profile, indent=4))