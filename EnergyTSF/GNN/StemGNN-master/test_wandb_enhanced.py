#!/usr/bin/env python3
"""
测试增强的wandb记录功能
"""

import subprocess
import sys
import os

def test_enhanced_wandb_logging():
    """测试增强的wandb记录功能"""
    print("="*60)
    print("测试增强的wandb记录功能")
    print("="*60)
    
    # 设置环境变量为离线模式
    env = os.environ.copy()
    env['WANDB_MODE'] = 'offline'
    env['WANDB_SILENT'] = 'true'
    
    # 测试德国数据集的单次运行
    print("\n1. 测试德国数据集单次运行（增强wandb记录）")
    print("-" * 50)
    
    cmd = [
        "python", "main.py",
        "--dataset", "Germany_processed_0",
        "--epoch", "2",
        "--batch_size", "16",
        "--device", "cpu",
        "--wandb",
        "--wandb_project", "StemGNN_Enhanced_Test",
        "--experiment_name", "test_enhanced_wandb"
    ]
    
    try:
        print(f"执行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
        
        if result.returncode == 0:
            print("✓ 增强wandb记录测试成功!")
            
            # 检查CSV文件是否生成
            csv_file = "output/results_Germany_processed_0_single_run.csv"
            if os.path.exists(csv_file):
                print(f"✓ CSV结果文件已生成: {csv_file}")
                
                # 读取并显示CSV文件的前几行
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                    print(f"CSV文件内容 ({len(lines)} 行):")
                    for i, line in enumerate(lines[:3]):  # 显示前3行
                        print(f"  {i+1}: {line.strip()}")
            else:
                print("❌ CSV结果文件未生成")
                
        else:
            print("❌ 增强wandb记录测试失败")
            print(f"错误输出: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
    except Exception as e:
        print(f"❌ 测试出错: {e}")
    
    print("\n2. 测试多次运行（增强wandb记录）")
    print("-" * 50)
    
    cmd_multi = [
        "python", "main.py",
        "--dataset", "Germany_processed_0",
        "--epoch", "1",
        "--runs", "2",
        "--batch_size", "16",
        "--device", "cpu",
        "--wandb",
        "--wandb_project", "StemGNN_Enhanced_Test",
        "--experiment_name", "test_enhanced_wandb_multi"
    ]
    
    try:
        print(f"执行命令: {' '.join(cmd_multi)}")
        result = subprocess.run(cmd_multi, capture_output=True, text=True, timeout=600, env=env)
        
        if result.returncode == 0:
            print("✓ 多次运行增强wandb记录测试成功!")
            
            # 检查多次运行的CSV文件
            csv_file_multi = "output/results_Germany_processed_0_2runs.csv"
            if os.path.exists(csv_file_multi):
                print(f"✓ 多次运行CSV结果文件已生成: {csv_file_multi}")
            else:
                print("❌ 多次运行CSV结果文件未生成")
                
        else:
            print("❌ 多次运行增强wandb记录测试失败")
            print(f"错误输出: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ 多次运行测试超时")
    except Exception as e:
        print(f"❌ 多次运行测试出错: {e}")

def main():
    """主函数"""
    print("StemGNN 增强wandb记录功能测试")
    
    # 测试增强的wandb记录
    test_enhanced_wandb_logging()
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print("\n📊 新增的wandb记录指标包括:")
    print("  • 训练阶段: train_loss, learning_rate, epoch_time, best_val_mae")
    print("  • 验证阶段: val_mape, val_mae, val_rmse (norm & raw)")
    print("  • 测试阶段: test_mape, test_mae, test_rmse (norm & raw)")
    print("  • 测试详情: test_duration, test_samples, node-level统计")
    print("  • 实验信息: dataset_name, experiment_name, total_params")
    print("  • 运行统计: run_id, train_time, eval_time")
    
    print("\n📄 CSV文件保存内容:")
    print("  • 基本信息: run_id, dataset, timestamp")
    print("  • 性能指标: final_test_mape, final_test_mae, final_test_rmse")
    print("  • 时间统计: total_train_time_min, total_eval_time_min")
    print("  • 模型信息: best_epoch, total_params")
    print("  • 配置信息: batch_size, learning_rate, device, window_size, horizon")

if __name__ == "__main__":
    main() 