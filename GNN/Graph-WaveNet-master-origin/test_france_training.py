#!/usr/bin/env python
"""
Simple test script to verify France dataset training with equal seq_length and pred_length
"""

import subprocess
import sys
import os

def test_france_training_with_equal_lengths():
    """
    Test France dataset training with seq_length = pred_length
    """
    print("🧪 测试France数据集训练（seq_length = pred_length）...")
    
    # Test with seq_length = pred_length = 6
    seq_length = 6
    pred_length = 6
    
    print(f"📊 测试配置: seq_length={seq_length}, pred_length={pred_length}")
    
    # First, generate the data
    print("🔄 生成数据...")
    data_cmd = [
        sys.executable, 'process_france_with_dataloader.py',
        '--step', 'process',
        '--seq_length', str(seq_length),
        '--pred_length', str(pred_length)
    ]
    
    try:
        result = subprocess.run(data_cmd, check=True, capture_output=True, text=True)
        print("✅ 数据生成成功")
    except subprocess.CalledProcessError as e:
        print(f"❌ 数据生成失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False
    
    # Then, test training with 1 epoch
    print("🏋️ 测试训练（1个epoch）...")
    train_cmd = [
        sys.executable, 'train.py',
        '--data', 'data/FRANCE',
        '--gcn_bool',
        '--adjtype', 'doubletransition',
        '--addaptadj',
        '--randomadj',
        '--epochs', '1',
        '--seq_length', str(seq_length),
        '--pred_length', str(pred_length)
        # Note: --run_multiple_experiments is False by default (action='store_true')
    ]
    
    try:
        result = subprocess.run(train_cmd, check=True, capture_output=True, text=True)
        print("✅ 训练成功完成")
        
        # Print last few lines of output
        output_lines = result.stdout.strip().split('\n')
        print("\n📊 训练结果摘要:")
        for line in output_lines[-10:]:
            if any(keyword in line for keyword in ['Test MAE', 'Test MAPE', 'Test RMSE', 'average']):
                print(f"  {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ 训练失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

if __name__ == "__main__":
    print("🇫🇷 France数据集训练测试开始...")
    
    # Change to the correct directory
    os.chdir('/home/robot/GCN/5.26/Graph-WaveNet-master-origin')
    
    success = test_france_training_with_equal_lengths()
    
    if success:
        print("\n🎉 测试成功！France数据集可以正确处理相等的seq_length和pred_length")
    else:
        print("\n❌ 测试失败！需要进一步检查问题") 