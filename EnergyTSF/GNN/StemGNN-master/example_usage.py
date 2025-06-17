#!/usr/bin/env python3
"""
使用示例：展示如何使用修改后的data_utils.py

这个示例展示了两种使用方式：
1. 新的load_and_process_dataset函数 - 推荐使用
2. 向后兼容的load_and_clean_dataset函数 - 用于旧代码
"""

import os
import sys
sys.path.append('utils')

from utils.data_utils import load_and_process_dataset, load_and_clean_dataset, print_dataset_info

def example_new_api():
    """使用新的API加载数据"""
    print("="*60)
    print("示例1: 使用新的load_and_process_dataset函数")
    print("="*60)
    
    try:
        # 加载ECG数据集
        result = load_and_process_dataset(
            root_path='./dataset',
            data_file='ECG_data.csv',
            target_column=None,  # 自动检测
            features='M',        # 多变量预测
            seq_len=96,         # 输入序列长度
            label_len=48,       # 标签序列长度
            pred_len=24,        # 预测序列长度
            scale_to_01=True,   # 缩放到0-1范围
            batch_size=32,      # 批次大小
            freq='h',           # 频率
            timeenc=0           # 时间编码方式
        )
        
        print(f"✓ 数据加载成功!")
        print(f"  数据集名称: {result['dataset_name']}")
        print(f"  目标列: {result['target_column']}")
        print(f"  特征维度: {result['feature_dim']}")
        print(f"  时间特征维度: {result['time_feature_dim']}")
        print(f"  总样本数: {result['total_samples']}")
        
        # 获取一个批次的数据
        train_loader = result['train_loader']
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(train_loader))
        
        print(f"  批次数据形状:")
        print(f"    输入序列 (seq_x): {batch_x.shape}")
        print(f"    输出序列 (seq_y): {batch_y.shape}")
        print(f"    输入时间特征: {batch_x_mark.shape}")
        print(f"    输出时间特征: {batch_y_mark.shape}")
        print(f"  数据范围: [{batch_x.min():.6f}, {batch_x.max():.6f}]")
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")

def example_compatibility_api():
    """使用向后兼容的API"""
    print("\n" + "="*60)
    print("示例2: 使用向后兼容的load_and_clean_dataset函数")
    print("="*60)
    
    try:
        # 使用兼容函数加载数据
        data = load_and_clean_dataset('ECG_data.csv')
        
        print(f"✓ 兼容模式加载成功!")
        print(f"  数据形状: {data.shape}")
        print(f"  数据类型: {type(data)}")
        print(f"  数据范围: [{data.min():.6f}, {data.max():.6f}]")
        print(f"  特征数量: {data.shape[1]}")
        
    except Exception as e:
        print(f"❌ 兼容模式加载失败: {e}")

def show_available_datasets():
    """显示可用的数据集信息"""
    print("\n" + "="*60)
    print("可用数据集信息")
    print("="*60)
    
    print_dataset_info()

def main():
    """主函数"""
    print("StemGNN 数据加载工具使用示例")
    print("="*60)
    
    # 显示可用数据集
    show_available_datasets()
    
    # 检查数据集目录是否存在
    if not os.path.exists('./dataset'):
        print("\n❌ 数据集目录 './dataset' 不存在")
        print("请确保数据集文件位于 './dataset' 目录中")
        return
    
    # 检查是否有数据文件
    dataset_files = [f for f in os.listdir('./dataset') if f.endswith('.csv')]
    if not dataset_files:
        print("\n❌ 数据集目录中没有找到CSV文件")
        return
    
    print(f"\n发现 {len(dataset_files)} 个数据文件:")
    for f in dataset_files[:5]:  # 只显示前5个
        print(f"  - {f}")
    if len(dataset_files) > 5:
        print(f"  ... 还有 {len(dataset_files) - 5} 个文件")
    
    # 运行示例
    example_new_api()
    example_compatibility_api()
    
    print("\n" + "="*60)
    print("使用提示:")
    print("="*60)
    print("1. 新项目推荐使用 load_and_process_dataset() 函数")
    print("2. 旧项目可以使用 load_and_clean_dataset() 保持兼容性")
    print("3. 数据会自动缩放到 [0, 1] 范围")
    print("4. 支持自动目标列检测和数据预处理")
    print("5. 使用 Dataset_Opennem 类确保与原始代码兼容")

if __name__ == "__main__":
    main() 