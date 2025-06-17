# StemGNN 数据集支持文档

## 概述

现在 StemGNN 支持多种不同格式的数据集，包含自动数据预处理功能，可以处理：
- 纯数值CSV文件
- 包含头行和日期列的CSV文件
- 自动检测和清理非数值数据

## 支持的数据集

### 1. ECG 数据集
- **ECG_data.csv** - ECG时间序列数据
- **ECG_data_0.csv** - ECG时间序列数据（版本0）
- **特征数**: 140
- **格式**: 纯数值
- **使用示例**:
```bash
python main.py --dataset ECG_data --wandb
```

### 2. 交通流量数据
- **PeMS07.csv** - PeMS交通流量数据
- **特征数**: 变动
- **格式**: 纯数值
- **使用示例**:
```bash
python main.py --dataset PeMS07 --wandb --wandb_project "StemGNN_Traffic"
```

### 3. 法国电力数据
- **France_processed_0.csv** - 法国电力消费数据
- **V_France_processed_0.csv** - 法国电力消费数据（处理版本）
- **特征数**: 10
- **格式**: 纯数值
- **使用示例**:
```bash
python main.py --dataset France_processed_0 --wandb --wandb_project "StemGNN_France"
python main.py --dataset V_France_processed_0 --wandb
```

### 4. 德国电力数据
- **Germany_processed_0.csv** - 德国电力生产数据（含日期和头行）
- **V_Germany_processed_0.csv** - 德国电力生产数据（处理版本）
- **特征数**: 16
- **格式**: 自动处理日期列和头行
- **使用示例**:
```bash
python main.py --dataset Germany_processed_0 --wandb --wandb_project "StemGNN_Germany"
python main.py --dataset V_Germany_processed_0 --wandb
```

## 新功能特性

### 自动数据预处理
- 自动检测和移除日期列
- 自动处理CSV头行
- 自动转换数据类型为数值
- 自动填充缺失值
- 数据格式验证

### 数据集信息查看
```bash
python main.py --show_datasets
```

### 兼容性测试
```bash
python test_all_datasets.py
```

## 使用方法

### 基本使用
```bash
# 选择数据集运行
python main.py --dataset [数据集名称]

# 启用 wandb 跟踪
python main.py --dataset [数据集名称] --wandb

# 设置项目名称
python main.py --dataset [数据集名称] --wandb --wandb_project "项目名称"
```

### 多次运行实验
```bash
# 进行5次独立运行
python main.py --dataset Germany_processed_0 --runs 5 --wandb
```

### GPU训练
```bash
# 使用GPU训练
python main.py --dataset Germany_processed_0 --device cuda:0 --batch_size 64
```

## 完整示例命令

### ECG数据实验
```bash
python main.py --dataset ECG_data --wandb --wandb_project "StemGNN_ECG" \
               --epoch 50 --batch_size 32 --device cuda:0
```

### 法国电力数据实验
```bash
python main.py --dataset France_processed_0 --wandb --wandb_project "StemGNN_France" \
               --epoch 100 --lr 0.0001 --runs 3
```

### 德国电力数据实验
```bash
python main.py --dataset Germany_processed_0 --wandb --wandb_project "StemGNN_Germany" \
               --epoch 100 --batch_size 64 --device cuda:0 --runs 5
```

### 交通数据实验
```bash
python main.py --dataset PeMS07 --wandb --wandb_project "StemGNN_Traffic" \
               --epoch 50 --window_size 12 --horizon 3
```

## 数据预处理细节

### 处理的问题类型
1. **日期列**: 自动检测包含'date'或'time'的列并移除
2. **字符串列**: 检测不能转换为数值的列并移除
3. **缺失值**: 使用前向填充和后向填充处理NaN值
4. **数据类型**: 强制转换所有列为float64类型

### 支持的CSV格式
- ✅ 纯数值CSV（如ECG_data.csv）
- ✅ 带头行的CSV（如Germany_processed_0.csv）
- ✅ 包含日期列的CSV
- ✅ 混合数据类型的CSV

## 错误处理

### 常见错误和解决方案

#### 1. 数据集文件不存在
```
❌ 数据集文件不存在: dataset/xxx.csv
```
**解决**: 检查文件名拼写，使用 `--show_datasets` 查看可用数据集

#### 2. 数据类型错误
```
TypeError: unsupported operand type(s) for /: 'str' and 'int'
```
**解决**: 现在已自动处理，数据预处理会自动清理非数值数据

#### 3. 内存不足
**解决**: 减少batch_size或使用数据采样

## 性能建议

### 不同数据集的推荐配置

| 数据集 | 推荐batch_size | 推荐device | 备注 |
|--------|----------------|------------|------|
| ECG_data | 32-64 | GPU | 特征数多，建议GPU |
| France_processed_0 | 64-128 | CPU/GPU | 数据量适中 |
| Germany_processed_0 | 32-64 | GPU | 数据量大，建议GPU |
| PeMS07 | 32-64 | GPU | 数据量大 |

### 推荐参数组合
```bash
# 快速测试
--epoch 10 --batch_size 16 --device cpu

# 标准训练
--epoch 50 --batch_size 32 --device cuda:0

# 深度训练
--epoch 100 --batch_size 64 --device cuda:0 --runs 5
```

## 故障排除

如果遇到问题，请按以下步骤检查：

1. **验证数据集可用性**:
   ```bash
   python main.py --show_datasets
   ```

2. **测试数据加载**:
   ```bash
   python test_all_datasets.py
   ```

3. **快速验证**:
   ```bash
   python main.py --dataset [数据集名称] --epoch 2 --batch_size 16 --device cpu
   ```

4. **查看详细错误**:
   ```bash
   python main.py --dataset [数据集名称] --epoch 1 --batch_size 8
   ```

现在所有数据集都已完全支持，包括之前有问题的 Germany_processed_0 数据集！ 