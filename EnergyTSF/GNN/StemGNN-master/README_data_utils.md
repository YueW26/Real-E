# StemGNN 数据加载工具 - 修改说明

## 概述

本文档描述了对 `utils/data_utils.py` 的重要修改，这些修改旨在：

1. **使用 `Dataset_Opennem` 类** - 确保与原始StemGNN代码完全兼容
2. **数据范围在0-1之间** - 应用MinMax缩放，确保数据在[0,1]范围内
3. **保持向后兼容** - 提供兼容函数，不破坏现有代码
4. **不修改 `dataloader_joella.py`** - 通过继承和wrapper解决pandas语法问题

## 主要功能

### ✅ 已实现的功能

1. **完全兼容的数据加载**
   - 使用 `CompatibleDataset_Opennem` 类继承 `Dataset_Opennem`
   - 修复了pandas新版本的语法问题
   - 保持原有的数据处理逻辑

2. **0-1数据缩放**
   - 自动应用MinMax缩放
   - 确保所有数据在[0, 1]范围内
   - 支持浮点精度容错

3. **智能数据预处理**
   - 自动检测和处理日期列
   - 智能目标列检测
   - 缺失值处理
   - 数据类型转换

4. **向后兼容性**
   - `load_and_clean_dataset()` 函数保持旧API
   - 新的 `load_and_process_dataset()` 函数提供更多功能

## 使用方法

### 方法1: 新的API（推荐）

```python
from utils.data_utils import load_and_process_dataset

# 加载数据集
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

# 获取数据加载器
train_loader = result['train_loader']
val_loader = result['val_loader']
test_loader = result['test_loader']

# 获取数据信息
print(f"特征维度: {result['feature_dim']}")
print(f"总样本数: {result['total_samples']}")
```

### 方法2: 向后兼容API

```python
from utils.data_utils import load_and_clean_dataset

# 简单加载数据（返回numpy数组）
data = load_and_clean_dataset('ECG_data.csv')
print(f"数据形状: {data.shape}")
print(f"数据范围: [{data.min():.6f}, {data.max():.6f}]")
```

### 方法3: 查看可用数据集

```python
from utils.data_utils import print_dataset_info

# 显示所有可用数据集的信息
print_dataset_info()
```

## 返回值说明

### `load_and_process_dataset()` 返回字典

```python
{
    'train_loader': DataLoader,      # 训练数据加载器
    'val_loader': DataLoader,        # 验证数据加载器  
    'test_loader': DataLoader,       # 测试数据加载器
    'train_dataset': Dataset,        # 训练数据集
    'val_dataset': Dataset,          # 验证数据集
    'test_dataset': Dataset,         # 测试数据集
    'dataset_name': str,             # 数据集名称
    'target_column': str,            # 目标列名
    'feature_dim': int,              # 特征维度
    'time_feature_dim': int,         # 时间特征维度
    'total_samples': int,            # 总样本数
    'scaler': MinMaxScaler,          # 缩放器对象
    'data_shape': tuple,             # 数据形状
    'original_columns': list         # 原始列名
}
```

## 支持的数据集

系统自动识别以下数据集类型：

1. **ECG_data** - ECG时间序列数据 (140特征)
2. **PeMS07** - 交通流量数据 (可变特征)
3. **France_processed_0** - 法国电力消费数据 (10特征)
4. **Germany_processed_0** - 德国电力生产数据 (16特征)
5. **V_Germany_processed_0** - 德国电力数据处理版本

## 数据预处理特性

### 自动日期处理
- 自动检测日期列
- 如果没有日期列，创建虚拟日期索引
- 支持多种日期格式

### 智能目标列检测
- 如果未指定目标列，自动使用最后一列
- 支持列名模糊匹配
- 智能处理数值列识别

### 缺失值处理
- 缺失率 > 50%：用0填充
- 缺失率 10-50%：线性插值
- 缺失率 < 10%：前向/后向填充

## 技术实现

### CompatibleDataset_Opennem 类

```python
class CompatibleDataset_Opennem(Dataset_Opennem):
    """兼容的Dataset_Opennem类，修复pandas语法问题"""
    
    def __read_data__(self):
        # 重写数据读取方法
        # 修复 df.drop(['date'], 1) -> df.drop(columns=['date'])
        # 安全处理目标列移除
        # 保持原有数据处理逻辑
```

### 0-1缩放实现

```python
def apply_01_scaling(train_dataset, val_dataset, test_dataset):
    """应用0-1缩放到数据集"""
    # 使用训练集拟合MinMaxScaler
    # 应用到所有数据集
    # 确保数据范围在[0, 1]
```

## 测试验证

运行 `example_usage.py` 查看完整使用示例：

```bash
python example_usage.py
```

### 测试结果确认

✅ **数据范围**: [0.000000, 1.000000]  
✅ **Dataset_Opennem兼容**: 正常工作  
✅ **向后兼容**: 旧API正常工作  
✅ **pandas语法**: 已修复  
✅ **数据预处理**: 自动处理各种数据格式  

## 文件结构

```
utils/
├── data_utils.py           # 主要修改文件
├── dataloader_joella.py    # 原始文件（未修改）
└── ...

example_usage.py           # 使用示例
README_data_utils.md       # 本文档
```

## 注意事项

1. **数据文件位置**: 确保数据文件在 `./dataset/` 目录中
2. **内存使用**: 大数据集可能需要调整batch_size
3. **GPU支持**: 数据加载器自动支持GPU训练
4. **缩放器保存**: 可以通过返回的scaler对象进行逆变换

## 故障排除

### 常见问题

1. **"list.remove(x): x not in list"**
   - 已修复：安全的目标列移除逻辑

2. **"DataFrame.drop() got too many positional arguments"**
   - 已修复：使用新的pandas语法

3. **数据范围不在[0,1]**
   - 已修复：正确的MinMax缩放实现

4. **目标列不存在**
   - 已修复：智能目标列检测和匹配

### 调试建议

1. 使用 `print_dataset_info()` 查看可用数据集
2. 检查数据文件格式和列名
3. 确认数据文件路径正确
4. 查看详细错误信息进行诊断

## 更新历史

- **v1.0**: 初始实现，基本功能
- **v1.1**: 修复pandas语法问题
- **v1.2**: 添加0-1缩放功能
- **v1.3**: 完善向后兼容性
- **v1.4**: 最终版本，所有功能正常工作

---

**作者**: AI Assistant  
**最后更新**: 2024年  
**状态**: ✅ 完成并测试通过 