#!/usr/bin/env python3
"""
检查 wandb 记录的指标
"""

import subprocess
import sys
import os
import tempfile

def check_wandb_metrics():
    """检查 wandb 记录的指标"""
    print("="*60)
    print("检查 StemGNN wandb 记录的指标")
    print("="*60)
    
    # 检查 wandb 是否安装
    try:
        import wandb
        print("✓ wandb 已安装")
    except ImportError:
        print("❌ wandb 未安装，请运行: pip install wandb")
        return False
    
    # 设置离线模式，避免需要登录
    env = os.environ.copy()
    env['WANDB_MODE'] = 'offline'
    env['WANDB_SILENT'] = 'true'
    
    print("\n开始测试 wandb 指标记录...")
    print("运行命令: python main.py --wandb --dataset ECG_data --epoch 3 --batch_size 16 --device cpu")
    
    # 运行命令
    cmd = [
        "python", "main.py",
        "--wandb",
        "--dataset", "ECG_data", 
        "--epoch", "3",
        "--batch_size", "16",
        "--device", "cpu",
        "--wandb_project", "StemGNN_Metrics_Test",
        "--experiment_name", "metrics_check"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
        
        if result.returncode == 0:
            print("✓ 程序运行成功")
            
            # 分析输出，查看记录的指标
            output_lines = result.stdout.split('\n')
            
            # 检查是否有关键指标出现在输出中
            metrics_found = {
                'MAPE': False,
                'MAE': False, 
                'RMSE': False,
                'train_loss': False,
                'wandb初始化': False
            }
            
            for line in output_lines:
                if 'MAPE' in line:
                    metrics_found['MAPE'] = True
                    print(f"  找到MAPE指标: {line.strip()}")
                if 'MAE' in line:
                    metrics_found['MAE'] = True
                    print(f"  找到MAE指标: {line.strip()}")
                if 'RMSE' in line:
                    metrics_found['RMSE'] = True
                    print(f"  找到RMSE指标: {line.strip()}")
                if 'train_total_loss' in line:
                    metrics_found['train_loss'] = True
                    print(f"  找到训练损失: {line.strip()}")
                if 'Wandb 初始化成功' in line:
                    metrics_found['wandb初始化'] = True
                    print(f"  ✓ {line.strip()}")
            
            print(f"\n指标记录检查结果:")
            print("-" * 40)
            for metric, found in metrics_found.items():
                status = "✓" if found else "❌"
                print(f"{status} {metric}: {'已记录' if found else '未找到'}")
            
            # 检查wandb文件是否生成
            wandb_dirs = [d for d in os.listdir('.') if d.startswith('wandb')]
            if wandb_dirs:
                print(f"\n✓ 找到 wandb 记录目录: {wandb_dirs}")
            else:
                print("\n⚠ 未找到 wandb 记录目录")
                
        else:
            print(f"❌ 程序运行失败")
            print(f"错误输出: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("❌ 程序运行超时")
    except Exception as e:
        print(f"❌ 运行出错: {e}")
    
    print("\n" + "="*60)
    print("期望记录的指标列表:")
    print("="*60)
    
    expected_metrics = [
        "训练过程指标:",
        "  - train_loss: 每个epoch的训练损失",
        "  - learning_rate: 学习率变化", 
        "  - epoch_time: 每个epoch的训练时间",
        "  - total_params: 模型参数总数",
        "",
        "验证过程指标:",
        "  - val_mape: 验证集MAPE",
        "  - val_mae: 验证集MAE",
        "  - val_rmse: 验证集RMSE", 
        "  - val_mape_norm: 归一化验证集MAPE",
        "  - val_mae_norm: 归一化验证集MAE",
        "  - val_rmse_norm: 归一化验证集RMSE",
        "  - best_val_mae: 最佳验证MAE",
        "  - best_epoch: 最佳模型对应的epoch",
        "",
        "测试结果指标:",
        "  - test_mape: 测试集MAPE",
        "  - test_mae: 测试集MAE", 
        "  - test_rmse: 测试集RMSE",
        "  - final_test_*: 最终测试结果",
        "",
        "多次运行汇总指标:",
        "  - summary_*_mean: 多次运行指标的均值",
        "  - summary_*_std: 多次运行指标的标准差",
        "  - successful_runs: 成功运行次数",
        "  - total_runs: 总运行次数"
    ]
    
    for metric in expected_metrics:
        print(metric)
    
    print(f"\n使用说明:")
    print("• 查看 wandb 记录: 访问 https://wandb.ai")
    print("• 离线同步: wandb sync wandb/offline-*")
    print("• 详细使用说明见: WANDB_USAGE.md")

if __name__ == "__main__":
    check_wandb_metrics() 