#!/usr/bin/env python3
"""
测试 PyTorch 升级后的兼容性
"""

import torch
import numpy as np
from models.base_model import Model
from models.handler import train, test
import pandas as pd
import os
import argparse

def test_model_creation():
    """测试模型创建"""
    print("测试模型创建...")
    model = Model(units=10, stack_cnt=2, time_step=12, multi_layer=5, horizon=3, device='cpu')
    print(f"✓ 模型创建成功! 参数数量: {sum(p.numel() for p in model.parameters())}")
    return model

def test_forward_pass():
    """测试前向传播"""
    print("测试前向传播...")
    model = Model(units=5, stack_cnt=2, time_step=12, multi_layer=3, horizon=3, device='cpu')
    
    # 创建测试数据
    batch_size = 2
    time_step = 12
    node_cnt = 5
    test_input = torch.randn(batch_size, time_step, node_cnt)
    
    with torch.no_grad():
        forecast, attention = model(test_input)
    
    print(f"✓ 前向传播成功!")
    print(f"  输入形状: {test_input.shape}")
    print(f"  预测输出形状: {forecast.shape}")
    print(f"  注意力矩阵形状: {attention.shape}")
    
    return forecast, attention

def test_fft_functions():
    """测试新的 FFT 函数"""
    print("测试 FFT 函数...")
    
    # 创建测试数据
    x = torch.randn(2, 4, 3, 5)
    
    # 测试 FFT
    ffted = torch.fft.fft(x, dim=-1)
    real_part = ffted.real
    imag_part = ffted.imag
    
    # 测试 IFFT
    complex_tensor = torch.complex(real_part, imag_part)
    iffted = torch.fft.ifft(complex_tensor, dim=-1).real
    
    # 检查是否近似相等（由于浮点精度）
    diff = torch.abs(x - iffted).max()
    print(f"✓ FFT/IFFT 测试成功! 最大误差: {diff.item():.2e}")
    
    return diff < 1e-5

def main():
    print("="*50)
    print("PyTorch 2.4.1 升级兼容性测试")
    print("="*50)
    
    # 显示版本信息
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"NumPy 版本: {np.__version__}")
    print("-"*50)
    
    try:
        # 测试模型创建
        model = test_model_creation()
        print()
        
        # 测试前向传播
        forecast, attention = test_forward_pass()
        print()
        
        # 测试 FFT 函数
        fft_success = test_fft_functions()
        print()
        
        if fft_success:
            print("🎉 所有测试通过! PyTorch 升级成功!")
        else:
            print("❌ FFT 测试失败")
            
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 