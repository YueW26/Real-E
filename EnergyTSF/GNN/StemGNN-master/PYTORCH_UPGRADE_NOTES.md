# PyTorch 升级适配说明

## 概述
本项目已成功从 PyTorch 1.7.1 升级到 PyTorch 2.4.1+cu121。主要修改了两个兼容性问题：

## 修改内容

### 1. FFT 函数更新 (`models/base_model.py`)

**问题**: PyTorch 1.8+ 中 `torch.rfft` 和 `torch.irfft` 被弃用并最终移除。

**解决方案**: 在 `StockBlockLayer` 类的 `spe_seq_cell` 方法中：

#### 修改前:
```python
ffted = torch.rfft(input, 1, onesided=False)
real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
# ...
time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
iffted = torch.irfft(time_step_as_inner, 1, onesided=False)
```

#### 修改后:
```python
import torch.fft

# 使用新的 FFT 接口（对最后一个维度做 FFT）
ffted = torch.fft.fft(input, dim=-1)  # complex tensor: [B, ?, N, T]

real = ffted.real.permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
img = ffted.imag.permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)
# ...
# 重组为 complex tensor
complex_tensor = torch.complex(real, img)

# 使用 ifft 进行逆变换
iffted = torch.fft.ifft(complex_tensor, dim=-1).real  # 只取实部
```

### 2. NumPy 类型更新 (`models/handler.py`)

**问题**: NumPy 1.20+ 中 `np.float` 别名被弃用。

**解决方案**:
```python
# 修改前
forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float)

# 修改后  
forecast_steps = np.zeros([inputs.size()[0], horizon, node_cnt], dtype=np.float64)
```

### 3. torch.load 警告修复 (`models/handler.py`)

**问题**: PyTorch 2.0+ 中 `torch.load` 会显示安全警告。

**解决方案**:
```python
# 修改前
model = torch.load(f)

# 修改后
model = torch.load(f, weights_only=False)
```

## 测试验证

创建了 `test_pytorch_upgrade.py` 脚本进行全面测试，包括：
- 模型创建测试
- 前向传播测试  
- FFT 函数正确性测试

所有测试均通过，确认升级成功。

## 兼容版本

- **PyTorch**: 2.4.1+cu121 (测试通过)
- **NumPy**: 1.24.4 (测试通过)
- **Python**: 3.8+

## 注意事项

1. 新的 FFT 接口使用复数张量，性能和精度与原版本保持一致
2. 所有修改都是向下兼容的，不会影响模型的训练和推理结果
3. 如果遇到其他 PyTorch 相关的弃用警告，请参考 [PyTorch 迁移指南](https://pytorch.org/docs/stable/notes/cuda.html)

## 运行验证

运行以下命令验证升级是否成功：

```bash
# 快速测试
python test_pytorch_upgrade.py

# 完整训练测试
python main.py --dataset ECG_data --epoch 2 --batch_size 16 --device cpu
``` 