#!/usr/bin/env python3
"""
测试所有数据集的加载和基本运行
"""

import subprocess
import sys
import os
from utils.data_utils import print_dataset_info, load_and_clean_dataset

def test_dataset_loading():
    """测试所有数据集的加载"""
    print("="*60)
    print("测试所有数据集的加载")
    print("="*60)
    
    available_datasets = [
        'ECG_data', 'ECG_data_0', 'PeMS07', 
        'France_processed_0', 'V_France_processed_0',
        'Germany_processed_0', 'V_Germany_processed_0'
    ]
    
    loading_results = {}
    
    for dataset in available_datasets:
        print(f"\n测试数据集: {dataset}")
        print("-" * 40)
        
        data_file = os.path.join('dataset', dataset + '.csv')
        
        if not os.path.exists(data_file):
            print(f"❌ 文件不存在: {data_file}")
            loading_results[dataset] = 'file_not_found'
            continue
        
        try:
            data = load_and_clean_dataset(data_file)
            print(f"✓ 成功加载 {dataset}")
            loading_results[dataset] = 'success'
            
        except Exception as e:
            print(f"❌ 加载失败 {dataset}: {e}")
            loading_results[dataset] = f'error: {str(e)}'
    
    # 显示汇总结果
    print("\n" + "="*60)
    print("数据集加载测试汇总")
    print("="*60)
    
    successful = 0
    failed = 0
    
    for dataset, result in loading_results.items():
        status = "✓" if result == 'success' else "❌"
        print(f"{status} {dataset}: {result}")
        if result == 'success':
            successful += 1
        else:
            failed += 1
    
    print(f"\n成功: {successful}/{len(available_datasets)} 个数据集")
    print(f"失败: {failed}/{len(available_datasets)} 个数据集")
    
    return loading_results

def test_quick_training():
    """对成功加载的数据集进行快速训练测试"""
    print("\n" + "="*60)
    print("快速训练测试")
    print("="*60)
    
    # 先测试加载
    loading_results = test_dataset_loading()
    successful_datasets = [ds for ds, result in loading_results.items() if result == 'success']
    
    if not successful_datasets:
        print("❌ 没有可用的数据集进行训练测试")
        return
    
    # 选择几个代表性数据集进行快速训练测试
    test_datasets = []
    if 'ECG_data' in successful_datasets:
        test_datasets.append('ECG_data')
    if 'France_processed_0' in successful_datasets:
        test_datasets.append('France_processed_0')
    if 'Germany_processed_0' in successful_datasets:
        test_datasets.append('Germany_processed_0')
    
    if not test_datasets:
        test_datasets = successful_datasets[:2]  # 至少测试前两个
    
    print(f"\n将测试以下数据集的训练: {test_datasets}")
    
    env = os.environ.copy()
    env['WANDB_MODE'] = 'offline'
    env['WANDB_SILENT'] = 'true'
    
    training_results = {}
    
    for dataset in test_datasets:
        print(f"\n测试训练 {dataset}...")
        print("-" * 40)
        
        cmd = [
            "python", "main.py",
            "--dataset", dataset,
            "--epoch", "2",
            "--batch_size", "16", 
            "--device", "cpu",
            "--wandb",
            "--wandb_project", f"StemGNN_Test_{dataset}",
            "--experiment_name", f"test_{dataset}"
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, env=env)
            
            if result.returncode == 0:
                print(f"✓ {dataset} 训练测试成功")
                training_results[dataset] = 'success'
            else:
                print(f"❌ {dataset} 训练测试失败")
                print(f"错误输出: {result.stderr[:200]}...")
                training_results[dataset] = 'failed'
                
        except subprocess.TimeoutExpired:
            print(f"❌ {dataset} 训练测试超时")
            training_results[dataset] = 'timeout'
        except Exception as e:
            print(f"❌ {dataset} 训练测试出错: {e}")
            training_results[dataset] = 'error'
    
    # 显示训练测试汇总
    print(f"\n训练测试汇总:")
    print("-" * 40)
    for dataset, result in training_results.items():
        status = "✓" if result == 'success' else "❌"
        print(f"{status} {dataset}: {result}")

def main():
    """主函数"""
    print("StemGNN 数据集兼容性测试")
    
    # 显示数据集信息
    print_dataset_info()
    
    # 测试数据集加载
    loading_results = test_dataset_loading()
    
    # 进行快速训练测试
    successful_count = sum(1 for result in loading_results.values() if result == 'success')
    if successful_count > 0:
        print(f"\n发现 {successful_count} 个可用数据集，开始训练测试...")
        test_quick_training()
    else:
        print("\n❌ 没有可用的数据集，跳过训练测试")
    
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)

if __name__ == "__main__":
    main() 