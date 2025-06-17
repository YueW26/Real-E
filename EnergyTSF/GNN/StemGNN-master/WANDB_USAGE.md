# StemGNN Wandb 使用指南

## 概述

已为 StemGNN 添加了 Wandb (Weights & Biases) 实验跟踪功能，支持：
- 自动记录训练过程中的损失和指标
- 记录模型超参数和配置
- 支持多次运行的统计分析
- 可视化训练曲线和结果对比

## 安装 Wandb

```bash
pip install wandb
```

## 首次使用

首次使用需要登录 wandb：

```bash
wandb login
```

或者设置离线模式：

```bash
export WANDB_MODE=offline
```

## 基本使用

### 1. 启用 Wandb 记录

在运行命令中添加 `--wandb` 参数：

```bash
python main.py --wandb --dataset ECG_data --epoch 50
```

### 2. 设置项目名称

使用 `--wandb_project` 指定项目名称：

```bash
python main.py --wandb --wandb_project "My_StemGNN_Project" --dataset ECG_data
```

### 3. 设置实验名称

使用 `--experiment_name` 指定实验名称：

```bash
python main.py --wandb --experiment_name "baseline_experiment" --dataset ECG_data
```

## 高级功能

### 多次运行实验

进行多次独立运行以获得统计结果：

```bash
python main.py --wandb --runs 5 --dataset ECG_data --epoch 50
```

这将：
- 执行 5 次独立运行（不同随机种子）
- 记录每次运行的结果
- 计算并记录统计信息（均值、标准差）
- 生成实验汇总文件

### 完整示例

```bash
# 多次运行实验，使用 wandb 记录
python main.py \
  --wandb \
  --wandb_project "StemGNN_ECG_Study" \
  --experiment_name "baseline_5runs" \
  --dataset ECG_data \
  --runs 5 \
  --epoch 50 \
  --batch_size 32 \
  --lr 1e-4 \
  --device cuda:0
```

## 记录的指标

### 训练过程
- `train_loss`: 每个 epoch 的训练损失
- `learning_rate`: 学习率变化
- `epoch_time`: 每个 epoch 的训练时间
- `total_params`: 模型参数总数

### 验证过程
- `val_mape`: 验证集 MAPE
- `val_mae`: 验证集 MAE  
- `val_rmse`: 验证集 RMSE
- `val_mape_norm`: 归一化验证集 MAPE
- `val_mae_norm`: 归一化验证集 MAE
- `val_rmse_norm`: 归一化验证集 RMSE
- `best_val_mae`: 最佳验证 MAE
- `best_epoch`: 最佳模型对应的 epoch

### 测试结果
- `test_mape`: 测试集 MAPE
- `test_mae`: 测试集 MAE
- `test_rmse`: 测试集 RMSE
- `final_test_*`: 最终测试结果

### 多次运行汇总
- `summary_*_mean`: 多次运行指标的均值
- `summary_*_std`: 多次运行指标的标准差
- `successful_runs`: 成功运行次数
- `total_runs`: 总运行次数

## 结果文件

### 单次运行
结果保存在：`output/{dataset}/`

### 多次运行
结果保存在：`output/{experiment_name}/`
- `config.json`: 实验配置
- `experiment_summary.json`: 多次运行汇总统计
- `run_{i}/`: 每次运行的详细结果

## 离线模式

如果无法联网，可以使用离线模式：

```bash
export WANDB_MODE=offline
python main.py --wandb --dataset ECG_data
```

离线记录的数据可以后续同步：

```bash
wandb sync wandb/offline-*
```

## 测试功能

运行测试脚本验证 wandb 功能：

```bash
python test_wandb.py
```

## 常见问题

### Q: 如何查看 wandb 记录的实验？
A: 访问 https://wandb.ai 或在本地运行 `wandb server`

### Q: 如何禁用 wandb？
A: 不添加 `--wandb` 参数即可

### Q: 如何更改 wandb 配置？
A: 修改 `~/.netrc` 文件或使用环境变量

### Q: 多次运行时每次结果都不同吗？
A: 是的，每次运行使用不同的随机种子以确保独立性

## 示例命令

```bash
# 基础实验
python main.py --wandb --dataset ECG_data --epoch 20

# 参数调优实验  
python main.py --wandb --experiment_name "lr_0.001" --lr 0.001 --dataset ECG_data

# 多次运行获得统计结果
python main.py --wandb --runs 10 --dataset ECG_data --epoch 50

# GPU 训练
python main.py --wandb --device cuda:0 --batch_size 64 --dataset ECG_data

# 完整的消融实验
python main.py --wandb --wandb_project "StemGNN_Ablation" \
               --experiment_name "dropout_0.3" --dropout_rate 0.3 \
               --runs 5 --dataset ECG_data --epoch 100
``` 