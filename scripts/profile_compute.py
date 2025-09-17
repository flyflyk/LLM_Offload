import pandas as pd
import os
import sys
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.auto_policy.profiler import get_hardware_profile

def main():
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Run the full hardware profile to get the fitted model for TFLOPs calculation.
        # force_rerun=True ensures we get fresh measurements.
        print("Running hardware profiling to determine TFLOPs model...")
        hardware_profile = get_hardware_profile(force_rerun=True)
        print("Profiling complete.")

    except Exception as e:
        print(f"An error occurred during hardware profiling: {e}")
        return

    # A list of batch sizes to get TFLOPs for
    batch_sizes = [1, 2, 4, 8, 16, 32]
    results = []

    print("\nCalculating TFLOPs for specific batch sizes...")
    for bs in batch_sizes:
        # Use the method from the profile to get the TFLOPs for the specific batch size
        tflops = hardware_profile.get_gpu_tflops(bs)
        print(f"Batch Size: {bs:2d}, TFLOPs: {tflops:.2f}")
        results.append({"Batch Size": bs, "TFLOPs": tflops})

    # Create a pandas DataFrame
    df = pd.DataFrame(results)

    # Define output path and ensure the directory exists
    output_dir = 'logs'
    # This script is in scripts/, so we go up one level to the project root.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, output_dir, 'compute_tflops.xlsx')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to Excel
    df.to_excel(output_path, index=False, sheet_name='TFLOPs_Profile')

    print(f"\nResults saved to {os.path.relpath(output_path)}")

if __name__ == "__main__":
    main()
