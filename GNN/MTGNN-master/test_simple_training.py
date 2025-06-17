#!/usr/bin/env python3
"""
Simple test script for the updated train_multi_horizon.py
Test different seq_in_len and seq_out_len configurations
"""

import subprocess
import sys

def run_experiment(dataset, seq_in_len, seq_out_len, epochs=10, device='cuda:0'):
    """Run experiment with specific seq_in_len and seq_out_len"""
    print(f"\n{'='*60}")
    print(f"Testing {dataset} with seq_in_len={seq_in_len}, seq_out_len={seq_out_len}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, 'train_multi_horizon.py',
        '--dataset', dataset,
        '--seq_in_len', str(seq_in_len),
        '--seq_out_len', str(seq_out_len),
        '--epochs', str(epochs),
        '--device', device,
        '--runs', '1'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Experiment completed successfully!")
        
        # Extract and print the final result
        lines = result.stdout.split('\n')
        for line in lines:
            if 'seq_in_len' in line and 'MAE=' in line:
                print(f"Result: {line}")
                break
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Experiment failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Run test experiments"""
    print("Testing simplified training script")
    
    # Test configurations
    test_configs = [
        # (dataset, seq_in_len, seq_out_len)
        ('GERMANY', 6, 3),
        ('GERMANY', 12, 6),
        ('FRANCE', 24, 12),
        ('FRANCE', 48, 24),
    ]
    
    successful_tests = 0
    total_tests = len(test_configs)
    
    for dataset, seq_in, seq_out in test_configs:
        success = run_experiment(dataset, seq_in, seq_out, epochs=5)  # Short test
        if success:
            successful_tests += 1
    
    print(f"\n{'='*80}")
    print(f"Test Summary: {successful_tests}/{total_tests} tests passed")
    print(f"{'='*80}")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed! The simplified training script is working correctly.")
        print("\nYou can now run experiments like:")
        print("python train_multi_horizon.py --dataset GERMANY --seq_in_len 6 --seq_out_len 6 --epochs 50")
        print("python train_multi_horizon.py --dataset FRANCE --seq_in_len 24 --seq_out_len 12 --epochs 50")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == '__main__':
    main() 