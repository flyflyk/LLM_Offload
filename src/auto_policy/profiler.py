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
    tflops_a: float
    tflops_b: float

    def get_eff_tflops(self, batch_size):
        if batch_size <= 0:
            return 0
        return self.tflops_a + self.tflops_b * np.log(batch_size)

    def get_flops(self, batch_size):
        return self.get_eff_tflops(batch_size) * 1e12

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

def _profile_disk_bandwidth(size_mb: int = 128, tmp_dir: str = None) -> float:
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

def _profile_compute_model(device: str) -> tuple[float, float, list, list]:
    logger.info("Profiling compute performance across different batch sizes...")
    H = 4096 
    batch_sizes = list(range(4, 1025, 4))
    tflops_results = []

    for bs in batch_sizes:
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
            flop = 2 * N * K * M
            tflops = (flop / duration) / 1e12 if duration > 0 else 0
            tflops_results.append(tflops)

        except torch.cuda.OutOfMemoryError:
            logger.warning(f"OOM at batch size {bs}. Stopping compute profiling here.")
            break

    # Trim the batch size list to match the results if OOM occurred
    if len(batch_sizes) != len(tflops_results):
        batch_sizes = batch_sizes[:len(tflops_results)]

    if len(tflops_results) < 2:
        logger.warning("Could not collect enough data points to fit a compute model. Using a constant TFLOPs value.")
        tflops_a = np.mean(tflops_results) if tflops_results else 1.0
        tflops_b = 0.0
    else:
        # Ensure the collected data is monotonically increasing
        tflops_results = np.maximum.accumulate(tflops_results)
        
        # Log model: TFLOPs = a + b * ln(batch_size)
        x = np.log(batch_sizes)
        y = tflops_results
        A = np.vstack([np.ones(len(x)), x]).T
        tflops_a, tflops_b = np.linalg.lstsq(A, y, rcond=None)[0]
        logger.info(f"Fitted compute model: effective_tflops = {tflops_a:.2f} + {tflops_b:.4f} * ln(batch_size)")

    return tflops_a, tflops_b, batch_sizes, tflops_results

def get_hardware_profile(profile_path: str = "hardware_profile.json", force_rerun: bool = False) -> HardwareProfile:
    cache_dir = os.path.dirname(profile_path)
    if cache_dir and not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if not force_rerun and os.path.exists(profile_path) and os.path.getsize(profile_path) > 0:
        logger.info(f"Loading cached hardware profile from {profile_path}...")
        with open(profile_path, 'r') as f:
            profile_dict = json.load(f)
            # Check for new fields, if not present, re-run profiling
            if 'tflops_a' in profile_dict and 'tflops_b' in profile_dict:
                return HardwareProfile(**profile_dict)
            else:
                logger.info("Cached profile is outdated. Re-running profiling.")

    logger.info("Running hardware profiling... (This may take a moment)")
    
    # Profile memory
    _, gpu_mem = torch.cuda.mem_get_info(0)
    cpu_mem = psutil.virtual_memory().available

    # Profile bandwidth (Bytes/s)
    logger.info("Profiling CPU<->GPU bandwidth...")
    cpu_gpu_write_bw, cpu_gpu_read_bw = _profile_bandwidth('cpu', 'cuda:0')
    logger.info(f"CPU -> GPU Bandwidth: {cpu_gpu_write_bw / 1e9:.2f} GB/s")
    logger.info(f"GPU -> CPU Bandwidth: {cpu_gpu_read_bw / 1e9:.2f} GB/s")
    logger.info("Profiling Disk<->CPU bandwidth...")
    disk_cpu_write_bw, disk_cpu_read_bw = _profile_disk_bandwidth()
    logger.info(f"Disk Write Bandwidth: {disk_cpu_write_bw / 1e9:.2f} GB/s")
    logger.info(f"Disk Read Bandwidth: {disk_cpu_read_bw / 1e9:.2f} GB/s")

    # Profile compute model (TFLOPS vs. Batch Size)
    tflops_a, tflops_b, _, _ = _profile_compute_model('cuda:0')

    profile = HardwareProfile(
        gpu_mem=gpu_mem,
        cpu_mem=cpu_mem,
        cpu_gpu_write_bandwidth=cpu_gpu_write_bw,
        cpu_gpu_read_bandwidth=cpu_gpu_read_bw,
        disk_cpu_write_bandwidth=disk_cpu_write_bw,
        disk_cpu_read_bandwidth=disk_cpu_read_bw,
        tflops_a=tflops_a,
        tflops_b=tflops_b,
    )
    with open(profile_path, 'w') as f:
        json.dump(dataclasses.asdict(profile), f, indent=4)
        
    return profile

if __name__ == '__main__':
    profile = get_hardware_profile(force_rerun=True)
    print("Hardware Profile:")
    print(json.dumps(dataclasses.asdict(profile), indent=4))
