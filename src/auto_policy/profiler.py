import json
import os
import time
import torch
import psutil
import dataclasses
import logging
import numpy as np
import tempfile

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class HardwareProfile:
    gpu_mem: int
    cpu_mem: int
    cpu_gpu_bandwidth: float
    disk_cpu_bandwidth: float
    tflops_slope: float
    tflops_bias: float

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

def _profile_disk_bandwidth(size_mb: int = 128, tmp_dir: str = None) -> float:
    if tmp_dir and not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False) as tmp_file:
        tmp_file_path = tmp_file.name

    data = np.ones(size_mb * 1024 * 1024, dtype=np.uint8)
    data_bytes = data.nbytes

    try:
        # Profile Write
        logger.info(f"  - Profiling disk write bandwidth with a {size_mb}MB temp file...")
        start_time = time.time()
        with open(tmp_file_path, "wb") as f:
            f.write(data)
        end_time = time.time()
        write_duration = end_time - start_time
        write_bw = data_bytes / write_duration if write_duration > 0 else 0
        logger.info(f"  - Disk Write Bandwidth: {write_bw / 1e9:.2f} GB/s")

        # Profile Read
        logger.info(f"  - Profiling disk read bandwidth with a {size_mb}MB temp file...")
        start_time = time.time()
        with open(tmp_file_path, "rb") as f:
            _ = f.read()
        end_time = time.time()
        read_duration = end_time - start_time
        read_bw = data_bytes / read_duration if read_duration > 0 else 0
        logger.info(f"  - Disk Read Bandwidth: {read_bw / 1e9:.2f} GB/s")

        # Return the average bandwidth
        avg_bw = (write_bw + read_bw) / 2
        logger.info(f"  - Average Disk-CPU Bandwidth: {avg_bw / 1e9:.2f} GB/s")
        return avg_bw

    finally:
        # Ensure the temporary file is deleted
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def _profile_compute_model(device: str) -> tuple[float, float]:
    logger.info("Profiling compute performance across different batch sizes...")
    H = 4096 
    
    # Profile TFLOPs for a range of batch sizes
    batch_sizes_to_test = [1, 2, 4, 8, 16, 32, 64]
    tflops_results = []

    for bs in batch_sizes_to_test:
        # Simulate a typical GEMM in a transformer MLP: (B, H) @ (H, 4*H)
        N, K, M = bs, H, 4 * H
        
        try:
            a = torch.randn(N, K, device=device, dtype=torch.float16)
            b = torch.randn(K, M, device=device, dtype=torch.float16)
            
            # Warmup
            for _ in range(5):
                torch.matmul(a, b)
            if 'cuda' in device:
                torch.cuda.synchronize()
            
            start_time = time.time()
            iters = 20
            for _ in range(iters):
                torch.matmul(a, b)
            if 'cuda' in device:
                torch.cuda.synchronize()
            end_time = time.time()
            
            duration = (end_time - start_time) / iters
            flops = 2 * N * K * M
            tflops = (flops / duration) / 1e12 if duration > 0 else 0
            tflops_results.append(tflops)
            logger.info(f"  - Batch Size {bs:3d}: {tflops:.2f} TFLOPs")

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"  - OOM at batch size {bs}. Stopping compute profiling here.")
            break

    if len(batch_sizes_to_test) != len(tflops_results):
        # This happens if OOM occurred. We only use the successful runs.
        batch_sizes_to_test = batch_sizes_to_test[:len(tflops_results)]

    if len(tflops_results) < 2:
        logger.warning("Could not collect enough data points to fit a compute model. Using a constant TFLOPs value.")
        slope = 0.0
        intercept = np.mean(tflops_results) if tflops_results else 1.0
    else:
        # Fit a linear model: TFLOPs = slope * batch_size + intercept
        slope, intercept = np.polyfit(batch_sizes_to_test, tflops_results, 1)
        logger.info(f"Fitted compute model: effective_tflops = {slope:.4f} * batch_size + {intercept:.2f}")

    return slope, intercept

def get_hardware_profile(profile_path: str = "hardware_profile.json", force_rerun: bool = False) -> HardwareProfile:
    cache_dir = os.path.dirname(profile_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if not force_rerun and os.path.exists(profile_path) and os.path.getsize(profile_path) > 0:
        logger.info(f"Loading cached hardware profile from {profile_path}...")
        with open(profile_path, 'r') as f:
            profile_dict = json.load(f)
            # Check for new fields, if not present, re-run profiling
            if 'tflops_slope' in profile_dict and 'tflops_bias' in profile_dict:
                return HardwareProfile(**profile_dict)
            else:
                logger.info("Cached profile is outdated. Re-running profiling.")
    logger.info("Running hardware profiling... (This may take a moment)")
    
    # Profile memory
    _, gpu_mem = torch.cuda.mem_get_info(0)
    cpu_mem = psutil.virtual_memory().available

    # Profile bandwidth (Bytes/s)
    cpu_gpu_bw = _profile_bandwidth('cpu', 'cuda:0')
    disk_cpu_bw = _profile_disk_bandwidth()

    # Profile compute model (TFLOPs)
    tflops_slope, tflops_bias = _profile_compute_model('cuda:0')

    profile = HardwareProfile(
        gpu_mem=gpu_mem,
        cpu_mem=cpu_mem,
        cpu_gpu_bandwidth=cpu_gpu_bw,
        disk_cpu_bandwidth=disk_cpu_bw,
        tflops_slope=tflops_slope,
        tflops_bias=tflops_bias,
    )
    with open(profile_path, 'w') as f:
        json.dump(dataclasses.asdict(profile), f, indent=4)
        
    return profile

if __name__ == '__main__':
    profile = get_hardware_profile(force_rerun=True)
    print("Hardware Profile:")
    print(json.dumps(dataclasses.asdict(profile), indent=4))
