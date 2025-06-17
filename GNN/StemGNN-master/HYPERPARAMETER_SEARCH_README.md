# StemGNN 超参数搜索功能

## 功能说明

### 参数解释

**window_size (序列长度)**：
- 模型输入的历史时间步长度
- 表示模型会看多少个过去的时间点来进行预测
- 例如：`window_size=12` 表示使用过去12个时间步的数据

**horizon (预测长度)**：
- 模型要预测的未来时间步长度
- 表示模型要预测未来多少个时间点
- 例如：`horizon=6` 表示预测未来6个时间步

**约束条件**：`window_size >= horizon`

## 使用方法

### 1. 直接使用 main.py

```bash
# 基本用法
python main.py --hyperparameter_search --dataset ECG_data --epoch 10

# 使用wandb记录实验
python main.py --hyperparameter_search --dataset ECG_data --epoch 20 --wandb --wandb_project "StemGNN_HyperSearch"

# 使用其他数据集
python main.py --hyperparameter_search --dataset France_processed_0 --epoch 15
```

### 2. 使用演示脚本

```bash
# 基本用法
python run_hyperparameter_search.py --dataset ECG_data --epoch 10

# 使用wandb记录
python run_hyperparameter_search.py --dataset France_processed_0 --epoch 20 --wandb --wandb_project "France_HyperSearch"
```

## 搜索空间

- **可选值**: [6, 12, 48, 96]
- **有效组合**: 
  - window_size=6: horizon可以是[6]
  - window_size=12: horizon可以是[6, 12]
  - window_size=48: horizon可以是[6, 12, 48]
  - window_size=96: horizon可以是[6, 12, 48, 96]

总共 **10种组合** 需要测试。

## 输出格式

### 控制台输出示例
```
Window Size 6 的结果:
  Seq Length  6, Pred Length  6: MAE=0.0095, RMSE=0.0197

Window Size 12 的结果:
  Seq Length 12, Pred Length  6: MAE=0.0085, RMSE=0.0187
  Seq Length 12, Pred Length 12: MAE=0.0119, RMSE=0.0229

Window Size 48 的结果:
  Seq Length 48, Pred Length  6: MAE=0.0075, RMSE=0.0177
  Seq Length 48, Pred Length 12: MAE=0.0089, RMSE=0.0199
  Seq Length 48, Pred Length 48: MAE=0.0444, RMSE=0.0788

Window Size 96 的结果:
  Seq Length 96, Pred Length  6: MAE=0.0070, RMSE=0.0167
  Seq Length 96, Pred Length 12: MAE=0.0085, RMSE=0.0189
  Seq Length 96, Pred Length 48: MAE=0.0389, RMSE=0.0688
  Seq Length 96, Pred Length 96: MAE=0.0955, RMSE=0.1249

🏆 最佳结果:
Seq Length 96, Pred Length  6: MAE=0.0070, RMSE=0.0167
```

### 结果文件

1. **详细结果 (JSON)**：`output/hyperparameter_search_results_{dataset}.json`
   - 包含所有组合的完整指标信息
   - 包含训练时间、测试时间等详细信息

2. **格式化结果 (TXT)**：`output/hyperparameter_search_summary_{dataset}.txt`
   - 按您要求的格式保存结果
   - 便于直接查看和复制

## 实验管理

### 目录结构
```
output/
├── hypersearch/
│   └── {dataset}/
│       ├── ws6_hz6/          # window_size=6, horizon=6
│       ├── ws12_hz6/         # window_size=12, horizon=6
│       ├── ws12_hz12/        # window_size=12, horizon=12
│       └── ...
├── hyperparameter_search_results_{dataset}.json
└── hyperparameter_search_summary_{dataset}.txt
```

### Wandb 记录
如果启用了 wandb，会记录：
- 每个参数组合的结果
- 最佳参数组合
- 实验进度和时间信息

## 注意事项

1. **训练时间**：完整搜索需要训练10个模型，建议先用较少的epoch数测试
2. **内存使用**：较大的window_size会占用更多内存
3. **早停策略**：可以通过 `--early_stop True` 减少训练时间
4. **批次大小**：如果内存不足，可以减小 `--batch_size`

## 快速测试

```bash
# 快速测试（仅训练1个epoch）
python main.py --hyperparameter_search --dataset ECG_data --epoch 1

# 中等测试（训练5个epoch）
python main.py --hyperparameter_search --dataset ECG_data --epoch 5

# 完整实验（训练50个epoch）
python main.py --hyperparameter_search --dataset ECG_data --epoch 50
``` 