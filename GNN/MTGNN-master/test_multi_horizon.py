#!/usr/bin/env python3
"""
Quick test script to verify multi-horizon functionality
"""

import subprocess
import sys

def test_single_horizon():
    """Test single horizon mode"""
    print("🧪 Testing Single Horizon Mode...")
    
    cmd = [
        sys.executable, 'train_multi_horizon.py',
        '--dataset', 'GERMANY',
        '--run_multiple_horizons', 'False',
        '--seq_in_len', '12',
        '--seq_out_len', '3',
        '--epochs', '2',  # Very few epochs for quick test
        '--runs', '1',
        '--device', 'cuda:0'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Single horizon test passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Single horizon test failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def test_multi_horizon():
    """Test multi-horizon mode"""
    print("🧪 Testing Multi-Horizon Mode...")
    
    cmd = [
        sys.executable, 'train_multi_horizon.py',
        '--dataset', 'GERMANY', 
        '--run_multiple_horizons', 'True',
        '--epochs', '2',  # Very few epochs for quick test
        '--runs', '1',
        '--device', 'cuda:0'
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ Multi-horizon test passed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Multi-horizon test failed: {e}")
        print(f"stdout: {e.stdout}")
        print(f"stderr: {e.stderr}")
        return False

def main():
    """Run all tests"""
    print("🚀 MTGNN Multi-Horizon Functionality Test")
    print("This will run quick tests with minimal epochs to verify functionality")
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Single horizon
    if test_single_horizon():
        success_count += 1
    
    # Test 2: Multi-horizon  
    if test_multi_horizon():
        success_count += 1
    
    print(f"\n📊 Test Results: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! The multi-horizon implementation is working correctly.")
        print("\nNext steps:")
        print("1. Run with more epochs for real experiments")
        print("2. Try with France dataset")
        print("3. Use run_multi_horizon_experiments.py for full experiments")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")

if __name__ == "__main__":
    main() 