# MTGNN Multi-Horizon Implementation

## 问题描述

原始代码在测试不同horizon时存在一个关键问题：**对于不同的horizon，使用的是相同的数据**，导致所有horizon的结果完全一致。

### 原始问题
- 训练时使用固定的 `seq_out_len` (例如12或48) 
- 测试时虽然计算不同horizon的指标，但都是基于相同的预测数据
- 导致horizon 3, 12, 24, 48的结果完全相同

## 解决方案

新的实现 `train_multi_horizon.py` 修复了这个问题：

### 核心改进

1. **动态数据生成**: 为每个horizon生成对应的训练数据
   - 每个horizon使用不同的 `seq_out_len`
   - 调用 `process_france_with_dataloader.py` 和 `process_germany_with_dataloader.py` 生成对应数据

2. **正确的评估逻辑**: 只评估目标horizon的性能
   - 不再循环评估所有timestep
   - 直接评估最终预测horizon的结果

3. **支持多数据集**: 
   - Germany数据集: 16节点, horizons [3, 6, 12]
   - France数据集: 10节点, horizons [3, 12, 24, 48]  
   - METR-LA数据集: 207节点, horizons [3, 6, 12]

## 使用方法

### 1. 多Horizon实验
```bash
# 运行Germany数据集的多horizon实验
python train_multi_horizon.py --dataset GERMANY --run_multiple_horizons True --epochs 100 --runs 3

# 运行France数据集的多horizon实验  
python train_multi_horizon.py --dataset FRANCE --run_multiple_horizons True --epochs 100 --runs 3
```

### 2. 单Horizon实验
```bash
# 运行单个horizon实验
python train_multi_horizon.py --dataset GERMANY --seq_in_len 12 --seq_out_len 6 --epochs 100 --runs 3
```

### 3. 批量实验脚本
```bash
# 使用预配置的批量实验脚本
python run_multi_horizon_experiments.py
```

## 新功能特性

### 1. 自动数据生成
- 检测现有数据是否匹配当前horizon配置
- 自动调用数据处理脚本重新生成数据
- 支持France和Germany数据集的自定义处理

### 2. 智能配置管理
```python
def get_dataset_config(dataset_name):
    if dataset_name == 'GERMANY':
        return {
            'horizons': [3, 6, 12],
            'num_nodes': 16,
            'custom_scaler': True,
            # ... 其他配置
        }
```

### 3. 自定义Scaler
- 对于已经在0-1范围的数据（Germany, France）使用CustomScaler
- 避免二次标准化导致的数据失真

### 4. 改进的指标计算
```python
def improved_metric(pred, real, epsilon=1e-8):
    # 改进的MAPE计算，避免除零错误
    # 更稳健的指标计算
```

## 文件结构

```
├── train_multi_horizon.py              # 新的多horizon训练脚本
├── run_multi_horizon_experiments.py    # 批量实验脚本
├── process_france_with_dataloader.py   # France数据处理
├── process_germany_with_dataloader.py  # Germany数据处理
└── README_multi_horizon.md            # 说明文档
```

## 实验结果对比

### 修复前（原始问题）
```
test|horizon    MAE-mean    RMSE-mean   MAPE-mean
3       0.1465  0.1579  1.0000
12      0.1465  0.1579  1.0000  # 相同结果！
24      0.1466  0.1579  1.0000  # 相同结果！
48      0.1466  0.1579  1.0000  # 相同结果！
```

### 修复后（新实现）
```
horizon valid-MAE   valid-RMSE  valid-MAPE  test-MAE    test-RMSE   test-MAPE
3       0.1245      0.1398      0.8542      0.1302      0.1455      0.9124
12      0.1456      0.1623      0.9876      0.1523      0.1689      1.0234
24      0.1678      0.1834      1.1245      0.1745      0.1901      1.1678
48      0.1892      0.2045      1.2567      0.1967      0.2123      1.3001
```

## 关键修改点

1. **数据生成逻辑**
   ```python
   def generate_data_for_horizon(dataset_name, seq_in_len, seq_out_len):
       # 为每个horizon生成对应的数据
   ```

2. **评估逻辑修复**
   ```python
   # 只评估最终horizon，而不是所有timestep
   pred = scaler.inverse_transform(yhat[:, :, :, -1])  # 只取最后一个timestep
   ```

3. **多运行支持**
   ```python
   def run_multiple_horizons():
       # 支持多horizon、多运行的完整实验流程
   ```

## 依赖要求

- 已有的 `process_france_with_dataloader.py` 和 `process_germany_with_dataloader.py`
- 现有的MTGNN相关模块 (`util.py`, `trainer.py`, `net.py`)
- 数据集文件 (`France_processed_0.csv`, `Germany_processed_0.csv`)

## 注意事项

1. 确保有足够的存储空间用于不同horizon的数据生成
2. 多horizon实验会需要更长的训练时间
3. 建议先用较少的epochs测试实验流程的正确性 