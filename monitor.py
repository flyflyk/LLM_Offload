import subprocess
import threading
import time
import psutil
import pynvml
import sys

peak_cpu_memory = 0
peak_gpu_memory = 0
monitoring = True

def monitor_memory(process):
    global peak_cpu_memory, peak_gpu_memory, monitoring
    
    handle = None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0) # Monitor GPU 0
    except pynvml.NVMLError as e:
        print(f"Warning: Failed to initialize NVML for GPU monitoring: {e}", file=sys.stderr)
        print("Will monitor CPU memory only.", file=sys.stderr)

    while monitoring:
        try:
            # Get CPU memory (RSS)
            current_cpu_mem = process.memory_info().rss
            if current_cpu_mem > peak_cpu_memory:
                peak_cpu_memory = current_cpu_mem

            # Get GPU memory
            if handle:
                gpu_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                current_gpu_mem = gpu_info.used
                if current_gpu_mem > peak_gpu_memory:
                    peak_gpu_memory = current_gpu_mem

        except (psutil.NoSuchProcess, psutil.AccessDenied, pynvml.NVMLError):
            # Process ended or access error, stop monitoring
            break
        
        time.sleep(0.1)

    if handle:
        pynvml.nvmlShutdown()

def main():
    global monitoring

    command = ["python"] + sys.argv[1:]
    print(f"--- [Monitoring] Executing command: {' '.join(command)} ---")
    print("-" * 60)

    try:
        # Start the subprocess
        with psutil.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', errors='replace') as process:
            monitor_thread = threading.Thread(target=monitor_memory, args=(process,))
            monitor_thread.start()

            # Stream subprocess stdout
            for line in iter(process.stdout.readline, ''):
                sys.stdout.write(line)
            
            # Print stderr after completion
            stderr_output = process.stderr.read()
            if stderr_output:
                sys.stderr.write("\n--- Subprocess Stderr ---")
                sys.stderr.write(stderr_output)
                sys.stderr.write("-------------------------")
    finally:
        monitoring = False
        if 'monitor_thread' in locals() and monitor_thread.is_alive():
            monitor_thread.join()

    # --- Final Report ---
    print("-" * 60)
    print("--- Memory Usage Report ---")
    print(f"  Peak CPU Memory: {peak_cpu_memory / 1024 / 1024 / 1024:.2f} GB")
    print(f"  Peak GPU Memory: {peak_gpu_memory / 1024 / 1024 / 1024:.2f} GB")
    print("---------------------------")


if __name__ == "__main__":
    main()
