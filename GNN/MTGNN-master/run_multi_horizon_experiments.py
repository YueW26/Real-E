#!/usr/bin/env python3
"""
Script to run multi-horizon experiments for MTGNN on different datasets.
This script demonstrates the proper way to test different horizons with 
corresponding data generation for each horizon.
"""

import subprocess
import sys
import os

def run_experiment(dataset, epochs=50, runs=1):
    """
    Run multi-horizon experiment for a specific dataset
    """
    print(f"\n{'='*80}")
    print(f"Running Multi-Horizon Experiment for {dataset}")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable, 'train_multi_horizon.py',
        '--dataset', dataset,
        '--run_multiple_horizons', 'True',
        '--epochs', str(epochs),
        '--runs', str(runs),
        '--device', 'cuda:0'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {dataset} experiment completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {dataset} experiment failed: {e}")
        return False

def run_single_horizon_experiment(dataset, seq_in=12, seq_out=12, epochs=50, runs=1):
    """
    Run single horizon experiment for comparison
    """
    print(f"\n{'='*80}")
    print(f"Running Single Horizon Experiment for {dataset} (seq_in={seq_in}, seq_out={seq_out})")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable, 'train_multi_horizon.py',
        '--dataset', dataset,
        '--run_multiple_horizons', 'False',
        '--seq_in_len', str(seq_in),
        '--seq_out_len', str(seq_out),
        '--epochs', str(epochs),
        '--runs', str(runs),
        '--device', 'cuda:0'
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✅ {dataset} single horizon experiment completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {dataset} single horizon experiment failed: {e}")
        return False

def main():
    """
    Main function to run experiments
    """
    print("🚀 MTGNN Multi-Horizon Experiment Runner")
    print("This script will run experiments with proper horizon-specific data generation")
    
    # Configuration
    epochs = 20  # Use fewer epochs for demonstration
    runs = 1     # Use single run for demonstration
    
    # Datasets to test
    datasets = ['GERMANY', 'FRANCE']  # Can add 'METR-LA' if needed
    
    print(f"\nConfiguration:")
    print(f"  Epochs per experiment: {epochs}")
    print(f"  Runs per horizon: {runs}")
    print(f"  Datasets: {datasets}")
    
    # Run multi-horizon experiments
    print(f"\n{'='*80}")
    print("RUNNING MULTI-HORIZON EXPERIMENTS")
    print(f"{'='*80}")
    
    for dataset in datasets:
        success = run_experiment(dataset, epochs=epochs, runs=runs)
        if not success:
            print(f"Skipping {dataset} due to failure")
            continue
    
    # Example: Run single horizon experiments for comparison
    print(f"\n{'='*80}")
    print("RUNNING SINGLE HORIZON EXPERIMENTS FOR COMPARISON")
    print(f"{'='*80}")
    
    # Germany single horizon
    run_single_horizon_experiment('GERMANY', seq_in=12, seq_out=12, epochs=epochs, runs=runs)
    
    # France single horizon  
    run_single_horizon_experiment('FRANCE', seq_in=48, seq_out=48, epochs=epochs, runs=runs)
    
    print(f"\n{'='*80}")
    print("ALL EXPERIMENTS COMPLETED!")
    print(f"{'='*80}")
    print("\nKey differences in the new implementation:")
    print("1. ✅ Each horizon now uses proper data generation with matching seq_out_len")
    print("2. ✅ Results are computed for the actual target horizon, not just the final timestep")
    print("3. ✅ Support for France and Germany datasets with custom scalers") 
    print("4. ✅ Automatic data regeneration when horizon configuration changes")
    print("5. ✅ Clear separation between multi-horizon and single-horizon modes")

if __name__ == "__main__":
    main() 