#!/usr/bin/env python3
"""
实验运行脚本 - 演示多次运行和 wandb 集成
"""

import subprocess
import sys
import os

def run_experiment(args_list):
    """运行实验"""
    cmd = ["python", "main.py"] + args_list
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"运行出错: {e}")
        return False

def main():
    # 检查是否安装了 wandb
    try:
        import wandb
        wandb_available = True
        print("✓ wandb 可用")
    except ImportError:
        wandb_available = False
        print("⚠ wandb 未安装，将不使用 wandb 记录")
    
    print("="*60)
    print("StemGNN 实验运行示例")
    print("="*60)
    
    # 示例 1: 单次运行，不使用 wandb
    print("\n1. 单次运行示例（不使用 wandb）:")
    args1 = [
        "--dataset", "ECG_data",
        "--epoch", "5",
        "--batch_size", "16",
        "--device", "cpu",
        "--experiment_name", "single_run_demo"
    ]
    
    success = run_experiment(args1)
    if success:
        print("✓ 单次运行完成")
    else:
        print("✗ 单次运行失败")
    
    # 示例 2: 多次运行，不使用 wandb
    print("\n2. 多次运行示例（3次运行，不使用 wandb）:")
    args2 = [
        "--dataset", "ECG_data",
        "--epoch", "3",
        "--runs", "3",
        "--batch_size", "16",
        "--device", "cpu",
        "--experiment_name", "multi_run_demo"
    ]
    
    success = run_experiment(args2)
    if success:
        print("✓ 多次运行完成")
    else:
        print("✗ 多次运行失败")
    
    # 示例 3: 使用 wandb（如果可用）
    if wandb_available:
        print("\n3. 使用 wandb 的多次运行示例:")
        args3 = [
            "--dataset", "ECG_data",
            "--epoch", "3",
            "--runs", "2",
            "--batch_size", "16",
            "--device", "cpu",
            "--wandb",
            "--wandb_project", "StemGNN_Demo",
            "--experiment_name", "wandb_demo"
        ]
        
        print("注意: 首次使用 wandb 可能需要登录")
        success = run_experiment(args3)
        if success:
            print("✓ wandb 实验完成")
        else:
            print("✗ wandb 实验失败")
    
    print("\n实验演示完成！")
    print("\n使用说明:")
    print("- 使用 --runs N 来指定运行次数")
    print("- 使用 --wandb 来启用 wandb 记录")
    print("- 使用 --experiment_name 来设置实验名称")
    print("- 结果保存在 output/实验名称/ 目录下")

if __name__ == "__main__":
    main() 