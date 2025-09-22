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
    cpu_gpu_write_bandwidth: float
    cpu_gpu_read_bandwidth: float
    disk_cpu_write_bandwidth: float
    disk_cpu_read_bandwidth: float
    gpu_tflops_a: float
    gpu_tflops_b: float

    def get_gpu_tflops(self, batch_size):
        if batch_size <= 0:
            return 0
        return self.gpu_tflops_a + self.gpu_tflops_b * np.log(batch_size)

def _profile_bandwidth(device1: str, device2: str, size_mb: int = 256) -> float:
    logger.info(f"Profiling bandwidth between {device1} and {device2}...")
    data_size_bytes = size_mb * 1024 * 1024
    
    # Measure device1 -> device2
    tensor1 = torch.randn(data_size_bytes // 4, dtype=torch.float32, device=device1)
    # Warmup
    for _ in range(3):
        tensor1.to(device2)
    if 'cuda' in [device1, device2]:
        torch.cuda.synchronize()
    
    start_time = time.time()
    tensor1.to(device2)
    if 'cuda' in [device1, device2]:
        torch.cuda.synchronize()
    end_time = time.time()
    duration1 = end_time - start_time
    bw1 = data_size_bytes / duration1 if duration1 > 0 else 0
    logger.info(f"{device1} -> {device2}: {bw1 / 1e9:.2f} GB/s")
    del tensor1

    # Measure device2 -> device1
    tensor2 = torch.randn(data_size_bytes // 4, dtype=torch.float32, device=device2)
    # Warmup
    for _ in range(3):
        tensor2.to(device1)
    if 'cuda' in [device1, device2]:
        torch.cuda.synchronize()

    start_time = time.time()
    tensor2.to(device1)
    if 'cuda' in [device1, device2]:
        torch.cuda.synchronize()
    end_time = time.time()
    duration2 = end_time - start_time
    bw2 = data_size_bytes / duration2 if duration2 > 0 else 0
    logger.info(f"{device2} -> {device1}: {bw2 / 1e9:.2f} GB/s")
    del tensor2

    if 'cuda' in [device1, device2]:
        torch.cuda.empty_cache()

    return bw1, bw2

def _profile_disk_bandwidth(size_mb: int = 128, tmp_dir: str = None) -> tuple[float, float]:
    if tmp_dir and not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    with tempfile.NamedTemporaryFile(dir=tmp_dir, delete=False) as tmp_file:
        tmp_file_path = tmp_file.name
    data = np.ones(size_mb * 1024 * 1024, dtype=np.uint8)
    data_bytes = data.nbytes
    try:
        # Profile Write
        logger.info(f"Profiling disk write bandwidth with a {size_mb}MB temp file...")
        start_time = time.time()
        with open(tmp_file_path, "wb") as f:
            f.write(data)
        end_time = time.time()
        write_duration = end_time - start_time
        write_bw = data_bytes / write_duration if write_duration > 0 else 0
        logger.info(f"Disk Write: {write_bw / 1e9:.2f} GB/s")

        # Profile Read
        logger.info(f"Profiling disk read bandwidth with a {size_mb}MB temp file...")
        start_time = time.time()
        with open(tmp_file_path, "rb") as f:
            _ = f.read()
        end_time = time.time()
        read_duration = end_time - start_time
        read_bw = data_bytes / read_duration if read_duration > 0 else 0
        logger.info(f"Disk Read: {read_bw / 1e9:.2f} GB/s")

        return write_bw, read_bw

    finally:
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)

def _profile_compute_model(device: str, dtype: torch.dtype = torch.float16) -> tuple[float, float]:
    logger.info(f"Profiling compute performance on {device} across different batch sizes...")
    H = 4096
    batch_sizes = list(range(4, 1025, 4))
    tflops_results = []

    for bs in batch_sizes:
        N, K, M = bs, H, 4 * H
        try:
            a = torch.randn(N, K, device=device, dtype=dtype)
            b = torch.randn(K, M, device=device, dtype=dtype)
            
            # Warmup
            for _ in range(5):
                torch.matmul(a, b)
            if "cuda" in device:
                torch.cuda.synchronize()
            
            start_time = time.time()
            iters = 20
            for _ in range(iters):
                torch.matmul(a, b)
            if "cuda" in device:
                torch.cuda.synchronize()
            end_time = time.time()
            
            duration = (end_time - start_time) / iters
            flop = 2 * N * K * M
            tflops = (flop / duration) / 1e12 if duration > 0 else 0
            tflops_results.append(tflops)

        except Exception as e:
            logger.warning(f"Failed at batch size {bs} on {device} due to: {e}. Stopping profiling here.")
            break

    if len(batch_sizes) > len(tflops_results):
        batch_sizes = batch_sizes[:len(tflops_results)]

    if len(tflops_results) < 2:
        logger.warning(f"Could not collect enough data points on {device} to fit a model. Using a constant TFLOPs value.")
        tflops_a = np.mean(tflops_results) if tflops_results else 1.0
        tflops_b = 0.0
    else:
        tflops_results = np.maximum.accumulate(tflops_results)
        x = np.log(batch_sizes)
        y = tflops_results
        A = np.vstack([np.ones(len(x)), x]).T
        tflops_a, tflops_b = np.linalg.lstsq(A, y, rcond=None)[0]
        logger.info(f"Fitted compute model for {device}: effective_tflops = {tflops_a:.2f} + {tflops_b:.4f} * ln(batch_size)")

    return tflops_a, tflops_b

def get_hardware_profile(profile_path: str = "hardware_profile.json", force_rerun: bool = False) -> HardwareProfile:
    cache_dir = os.path.dirname(profile_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if not force_rerun and os.path.exists(profile_path) and os.path.getsize(profile_path) > 0:
        logger.info(f"Loading cached hardware profile from {profile_path}...")
        with open(profile_path, 'r') as f:
            profile_dict = json.load(f)
            if 'gpu_tflops_a' in profile_dict and 'cpu_tflops_a' in profile_dict:
                return HardwareProfile(**profile_dict)
            else:
                logger.info("Cached profile is outdated. Re-running profiling.")

    logger.info("Running hardware profiling... (This may take a moment)")
    
    # Profile memory
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.cuda.empty_cache()
        initial_reserved = torch.cuda.memory_reserved()

        # Allocate a dummy tensor
        _ = torch.tensor([1.0]).cuda()
        torch.cuda.synchronize()

        # Measure reserved memory
        overhead = torch.cuda.memory_reserved()
        context_and_allocator_overhead = overhead - initial_reserved
        total_gpu_mem = torch.cuda.get_device_properties(0).total_memory
        gpu_mem = total_gpu_mem - context_and_allocator_overhead
        
        logger.info(f"Total GPU Memory: {total_gpu_mem / 1024**3:.2f} GB")
        logger.info(f"CUDA Context + Allocator Overhead: {context_and_allocator_overhead / 1024**2:.2f} MB")
        logger.info(f"Usable GPU Memory for Optimizer: {gpu_mem / 1024**3:.2f} GB")
        
        # Clean up
        del _
        torch.cuda.empty_cache()
    else:
        gpu_mem = 0

    cpu_mem = psutil.virtual_memory().available * 0.9
    logger.info(f"Availabe CPU Memory: {cpu_mem / 1024**3:.2f} GB")

    # Profile bandwidth (Bytes/s)
    logger.info("Profiling CPU<->GPU bandwidth...")
    cpu_gpu_write_bw, cpu_gpu_read_bw = _profile_bandwidth('cpu', 'cuda:0')
    logger.info(f"CPU -> GPU Bandwidth: {cpu_gpu_write_bw / 1e9:.2f} GB/s")
    logger.info(f"GPU -> CPU Bandwidth: {cpu_gpu_read_bw / 1e9:.2f} GB/s")
    logger.info("Profiling Disk<->CPU bandwidth...")
    disk_cpu_write_bw, disk_cpu_read_bw = _profile_disk_bandwidth()
    logger.info(f"Disk Write Bandwidth: {disk_cpu_write_bw / 1e9:.2f} GB/s")
    logger.info(f"Disk Read Bandwidth: {disk_cpu_read_bw / 1e9:.2f} GB/s")

    # Profile compute models
    gpu_tflops_a, gpu_tflops_b = _profile_compute_model('cuda:0', dtype=torch.float16)

    profile = HardwareProfile(
        gpu_mem=gpu_mem,
        cpu_mem=cpu_mem,
        cpu_gpu_write_bandwidth=cpu_gpu_write_bw,
        cpu_gpu_read_bandwidth=cpu_gpu_read_bw,
        disk_cpu_write_bandwidth=disk_cpu_write_bw,
        disk_cpu_read_bandwidth=disk_cpu_read_bw,
        gpu_tflops_a=gpu_tflops_a,
        gpu_tflops_b=gpu_tflops_b,
    )
    with open(profile_path, 'w') as f:
        json.dump(dataclasses.asdict(profile), f, indent=4)
        
    return profile

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    profile = get_hardware_profile(force_rerun=True)
    print("Hardware Profile:")
    print(json.dumps(dataclasses.asdict(profile), indent=4))