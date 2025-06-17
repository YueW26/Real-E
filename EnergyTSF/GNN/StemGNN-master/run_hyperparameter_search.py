#!/usr/bin/env python3
"""
超参数搜索演示脚本

这个脚本展示如何使用 main.py 中新增的超参数搜索功能。
它会测试不同的 window_size 和 horizon 组合，并返回最佳结果。

使用方法:
python run_hyperparameter_search.py --dataset ECG_data --epoch 10
python run_hyperparameter_search.py --dataset France_processed_0 --epoch 20 --wandb --wandb_project "France_HyperSearch"
"""

import subprocess
import sys
import os

def run_hyperparameter_search(dataset="ECG_data", epochs=10, use_wandb=False, wandb_project=None):
    """
    运行超参数搜索
    
    Args:
        dataset: 数据集名称
        epochs: 训练轮数
        use_wandb: 是否使用wandb记录
        wandb_project: wandb项目名称
    """
    
    print("="*60)
    print("StemGNN 超参数搜索")
    print("="*60)
    print(f"数据集: {dataset}")
    print(f"训练轮数: {epochs}")
    print(f"搜索空间: window_size 和 horizon 从 [6, 12, 48, 96] 中选择")
    print(f"约束条件: window_size >= horizon")
    print("="*60)
    
    # 构建命令
    cmd = [
        "python", "main.py",
        "--hyperparameter_search",
        "--dataset", dataset,
        "--epoch", str(epochs)
    ]
    
    # 添加wandb相关参数
    if use_wandb:
        cmd.extend(["--wandb"])
        if wandb_project:
            cmd.extend(["--wandb_project", wandb_project])
    
    print(f"执行命令: {' '.join(cmd)}")
    print()
    
    # 运行命令
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("\n✓ 超参数搜索完成!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 超参数搜索失败: {e}")
        return False
    except KeyboardInterrupt:
        print("\n⚠ 用户中断了搜索过程")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="StemGNN 超参数搜索演示")
    parser.add_argument("--dataset", type=str, default="ECG_data", 
                       choices=['ECG_data', 'ECG_data_0', 'PeMS07', 'France_processed_0', 'Germany_processed_0'],
                       help="选择数据集")
    parser.add_argument("--epoch", type=int, default=10, help="训练轮数")
    parser.add_argument("--wandb", action="store_true", help="使用wandb记录实验")
    parser.add_argument("--wandb_project", type=str, default="StemGNN_HyperSearch", help="wandb项目名称")
    
    args = parser.parse_args()
    
    # 检查main.py是否存在
    if not os.path.exists("main.py"):
        print("❌ 找不到 main.py 文件，请确保在正确的目录下运行此脚本")
        sys.exit(1)
    
    # 运行超参数搜索
    success = run_hyperparameter_search(
        dataset=args.dataset,
        epochs=args.epoch,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project if args.wandb else None
    )
    
    if success:
        print("\n" + "="*60)
        print("搜索结果文件:")
        print("="*60)
        
        # 显示结果文件位置
        results_json = f"output/hyperparameter_search_results_{args.dataset}.json"
        results_txt = f"output/hyperparameter_search_summary_{args.dataset}.txt"
        
        if os.path.exists(results_json):
            print(f"✓ 详细结果 (JSON): {results_json}")
        if os.path.exists(results_txt):
            print(f"✓ 汇总结果 (TXT): {results_txt}")
            print("\n格式化结果预览:")
            print("-" * 40)
            try:
                with open(results_txt, 'r') as f:
                    content = f.read()
                    print(content)
            except Exception as e:
                print(f"无法读取结果文件: {e}")
    else:
        print("\n❌ 超参数搜索失败，请检查错误信息")
        sys.exit(1)

if __name__ == "__main__":
    main() 