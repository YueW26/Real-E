#!/usr/bin/env python3
"""
测试 wandb 功能
"""

import subprocess
import sys

def test_wandb_functionality():
    """测试 wandb 功能"""
    print("="*50)
    print("测试 StemGNN wandb 功能")
    print("="*50)
    
    # 检查 wandb 是否安装
    try:
        import wandb
        print("✓ wandb 已安装")
    except ImportError:
        print("❌ wandb 未安装，请运行: pip install wandb")
        return False
    
    # 测试 1: 不使用 wandb 的单次运行
    print("\n1. 测试单次运行（不使用 wandb）")
    cmd1 = [
        "python", "main.py",
        "--dataset", "ECG_data",
        "--epoch", "2",
        "--batch_size", "16",
        "--device", "cpu"
    ]
    
    try:
        result = subprocess.run(cmd1, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print("✓ 单次运行测试成功")
        else:
            print(f"❌ 单次运行测试失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ 单次运行测试超时")
    except Exception as e:
        print(f"❌ 单次运行测试出错: {e}")
    
    # 测试 2: 使用 wandb 的单次运行（离线模式）
    print("\n2. 测试 wandb 单次运行（离线模式）")
    cmd2 = [
        "python", "main.py",
        "--dataset", "ECG_data",
        "--epoch", "2",
        "--batch_size", "16",
        "--device", "cpu",
        "--wandb",
        "--wandb_project", "StemGNN_Test",
        "--experiment_name", "test_single_run"
    ]
    
    # 设置环境变量为离线模式
    import os
    env = os.environ.copy()
    env['WANDB_MODE'] = 'offline'
    
    try:
        result = subprocess.run(cmd2, capture_output=True, text=True, timeout=300, env=env)
        if result.returncode == 0:
            print("✓ wandb 单次运行测试成功（离线模式）")
        else:
            print(f"❌ wandb 单次运行测试失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ wandb 单次运行测试超时")
    except Exception as e:
        print(f"❌ wandb 单次运行测试出错: {e}")
    
    # 测试 3: 多次运行
    print("\n3. 测试多次运行（2次，离线模式）")
    cmd3 = [
        "python", "main.py",
        "--dataset", "ECG_data",
        "--epoch", "2",
        "--runs", "2",
        "--batch_size", "16",
        "--device", "cpu",
        "--wandb",
        "--wandb_project", "StemGNN_Test",
        "--experiment_name", "test_multi_run"
    ]
    
    try:
        result = subprocess.run(cmd3, capture_output=True, text=True, timeout=600, env=env)
        if result.returncode == 0:
            print("✓ 多次运行测试成功（离线模式）")
            # 检查是否生成了汇总文件
            summary_file = "output/test_multi_run/experiment_summary.json"
            if os.path.exists(summary_file):
                print("✓ 实验汇总文件生成成功")
            else:
                print("⚠ 实验汇总文件未找到")
        else:
            print(f"❌ 多次运行测试失败: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ 多次运行测试超时")
    except Exception as e:
        print(f"❌ 多次运行测试出错: {e}")
    
    print("\n" + "="*50)
    print("测试完成!")
    print("\n使用说明:")
    print("• 首次使用 wandb 需要登录: wandb login")
    print("• 使用 --wandb 启用实验跟踪")
    print("• 使用 --runs N 进行多次运行")
    print("• 使用 --wandb_project 设置项目名称")
    print("• 设置 WANDB_MODE=offline 可离线使用")

if __name__ == "__main__":
    test_wandb_functionality() 