#!/usr/bin/env python3
"""
Test script for the run_multiple functionality
This will test different sequence lengths (6,12,48,96) where seq_in_len = seq_out_len
"""

import subprocess
import sys

def test_run_multiple(dataset, epochs=10, runs=1, device='cuda:0'):
    """Test run_multiple functionality with different sequence lengths"""
    print(f"\n{'='*80}")
    print(f"Testing run_multiple functionality on {dataset}")
    print(f"This will test sequence lengths: 6, 12, 48, 96")
    print(f"{'='*80}")
    
    cmd = [
        sys.executable, 'train_multi_horizon.py',
        '--dataset', dataset,
        '--run_multiple', 'True',
        '--epochs', str(epochs),
        '--runs', str(runs),
        '--device', device
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ run_multiple test completed successfully!")
        
        # Extract and print the final results
        lines = result.stdout.split('\n')
        print("\n🎯 Final Results:")
        print("="*60)
        
        final_results_started = False
        for line in lines:
            if 'FINAL RESULTS FOR ALL SEQUENCE LENGTHS' in line:
                final_results_started = True
                continue
            if final_results_started and 'Seq Length' in line and 'MAE=' in line:
                print(line)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def test_single_run_for_comparison(dataset, seq_in=12, seq_out=12, epochs=10, device='cuda:0'):
    """Test single run for comparison"""
    print(f"\n{'='*60}")
    print(f"Testing single run mode for comparison")
    print(f"Dataset: {dataset}, seq_in_len: {seq_in}, seq_out_len: {seq_out}")
    print(f"{'='*60}")
    
    cmd = [
        sys.executable, 'train_multi_horizon.py',
        '--dataset', dataset,
        '--seq_in_len', str(seq_in),
        '--seq_out_len', str(seq_out),
        '--epochs', str(epochs),
        '--runs', '1',
        '--device', device
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Single run test completed successfully!")
        
        # Extract and print the result
        lines = result.stdout.split('\n')
        for line in lines:
            if 'seq_in_len' in line and 'MAE=' in line:
                print(f"Result: {line}")
                break
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def main():
    """Run test experiments"""
    print("🧪 Testing run_multiple functionality")
    
    # Test datasets
    test_datasets = ['GERMANY']  # Start with one dataset for quick test
    
    successful_tests = 0
    total_tests = 0
    
    for dataset in test_datasets:
        print(f"\n{'='*100}")
        print(f"Testing {dataset} dataset")
        print(f"{'='*100}")
        
        # Test run_multiple functionality
        total_tests += 1
        success = test_run_multiple(dataset, epochs=5, runs=1)  # Short test
        if success:
            successful_tests += 1
        
        # Test single run for comparison
        total_tests += 1
        success = test_single_run_for_comparison(dataset, seq_in=12, seq_out=12, epochs=5)
        if success:
            successful_tests += 1
    
    print(f"\n{'='*100}")
    print(f"Test Summary: {successful_tests}/{total_tests} tests passed")
    print(f"{'='*100}")
    
    if successful_tests == total_tests:
        print("🎉 All tests passed! The run_multiple functionality is working correctly.")
        print("\n📖 Usage instructions:")
        print("For multiple sequence lengths:")
        print("python train_multi_horizon.py --dataset GERMANY --run_multiple True --epochs 50")
        print("\nFor single configuration:")
        print("python train_multi_horizon.py --dataset GERMANY --seq_in_len 12 --seq_out_len 12 --epochs 50")
        
        print("\n🎯 Expected output format for run_multiple:")
        print("Seq Length  6, Pred Length  6: MAE=0.0096, MAPE=0.0691, RMSE=0.0198")
        print("Seq Length 12, Pred Length 12: MAE=0.0124, MAPE=0.0917, RMSE=0.0246")
        print("Seq Length 48, Pred Length 48: MAE=0.0232, MAPE=0.1700, RMSE=0.0437")
        print("Seq Length 96, Pred Length 96: MAE=0.0249, MAPE=0.1813, RMSE=0.0459")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == '__main__':
    main() 