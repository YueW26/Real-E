
# ⚡ \data: The Largest Real-World Multivariate Electricity Forecasting Benchmark

\data is a comprehensive, high-resolution benchmark dataset for **multivariate time series forecasting in electricity systems**. It spans 10 years across 39 European countries, including over 74 electricity stations and 20+ energy categories. With rich metadata, non-stationary dynamics, and high temporal granularity, \data provides a rigorous foundation for developing and evaluating robust forecasting models.

---

## 📚 Background and Motivation

Electricity forecasting is essential for:
- Ensuring grid stability
- Integrating renewable energy
- Managing market operations and planning

While many recent models (especially Transformer and GNN-based ones) show promising results on synthetic or narrow-scope datasets, these models often **fail to generalize** in real-world scenarios due to:
- Limited spatial/temporal diversity in benchmarks
- Lack of multi-energy interaction modeling
- Ignoring dynamic correlation shifts

\data addresses these challenges by providing **the most comprehensive real-world electricity time series benchmark to date**.

---

## 🧱 Dataset Overview

| Property                  | Description |
|---------------------------|-------------|
| 📍 Geography              | 39 European countries (10.18M km²) |
| 🕒 Duration               | 2014–2024 (10 years) |
| ⏱ Resolution             | 15 minutes to 1 hour |
| ⚡ Energy Types           | Solar, Wind, Hydro, Thermal, Nuclear, Coal, etc. |
| 📑 Data Sources           | ENTSO-E Transparency Platform |
| 🧭 Metadata               | Geolocation, Bidding Zone, Voltage Level, Plant Type |
| 🧵 Modalities             | Generation, Load, Market Pricing, Transmission, Balancing |

Each power station is provided as a separate multivariate CSV file with aligned metadata.

---


# BZN Dataset

This repository provides access to the BZN dataset, available for download and use in research projects.

## Dataset Overview

The BZN dataset contains valuable data designed for graph convolution network for multivariant time series forecasting. This dataset is hosted on Zenodo and can be easily accessed using the link below.

## How to Access the Dataset

The dataset can be tracked and downloaded using the following Zenodo link:

- [Zenodo: BZN Dataset](https://zenodo.org/records/13835030?preview=1&token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjE1MTY1NGY4LTA1OGUtNDRhMC1hZTI2LWI3MGMyZTI0MGY0MSIsImRhdGEiOnt9LCJyYW5kb20iOiI1NTQwYmU1YTM1N2M0ZTdkNzljYzZjMjdhYmM0MmViZSJ9.epUG3JX3xZafWQ8VHlYaUFCLUrsCmQ2zIamGuFI935TfHuzmDybQx3koqiYn-rqGl9IjsNx4-Qvc8Nwi6is6og)

## Direct Download

You can download the dataset directly using the following command:

```
bash
wget "https://zenodo.org/records/13952409/files/BZN.zip?download=1" -O BZN.zip
```

## Preprocessing 
In this repository we leverage the preprocessed datasets, including germany, ..... could be found under /mnt/datasets.


## 📈 Metrics and Analysis

To measure structural complexity and correlation dynamics, we introduce:

### 🔁 Temporal Graph Volatility (TGV)
Measures magnitude of correlation changes over time via Frobenius norm.

### 🧠 Graph Spectral Divergence (GSD)
Measures shifts in Laplacian eigenvalues to detect structural volatility.

| Dataset     | TGV   | GSD   |
|-------------|-------|-------|
| Electricity | 0.5391 | 1.4394 |
| ETTm1       | 0.7534 | 1.9210 |
| Solar       | 0.6245 | 1.8027 |
| **\data**  | **0.9046** | **2.2086** |

---

## 🧪 Benchmarking Overview

We benchmark 20 models from different families:

### 🔢 Classical Methods
- ARIMA
- S-ARIMA
- VAR

### 🧠 MLP-based
- DLinear
- N-Beats
- TimeMixer

### 🔁 RNN/CNN
- LSTM, TCN, DeepGLO, SFM

### 🧠 Transformer-based
- Informer
- Autoformer
- FEDformer
- Reformer

### 🌐 Graph Neural Networks
- Spectral GNNs: FourierGNN, LSGCN, StemGNN
- Spatial GNNs: MTGNN, TPGNN, WaveNet

### 📊 Results Summary

- **Transformers perform well on small benchmarks**, but degrade >50% in MAE on \data.
- **Spatial GNNs outperform all models on \data**, thanks to dynamic correlation modeling.
- **FourierGNN and WaveNet** showed the best robustness across dynamic regimes.

See full table in the paper for detailed MAE/RMSE rankings.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/ChenS676/Time-Series-Library.git
cd Time-Series-Library
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ⚙️ Training Examples

### 🧭 Spatial Graph Neural Network (GNN)

```bash
python train.py --data data/FRANCE --gcn_bool --adjtype doubletransition --addaptadj --randomadj --epochs 50
```

- `--gcn_bool`: Use GCN layers
- `--adjtype`: Type of adjacency matrix ("doubletransition" recommended)
- `--addaptadj`: Learnable adjacency
- `--randomadj`: Random init of graph structure

---

### 🔁 Reformer (Transformer-based)

```bash
python /EnergyTSF/run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model_id Reformer_test \
  --model Reformer \
  --data Opennem \
  --data_path Germany_processed_0.csv \
  --features M \
  --seq_len 12 --label_len 12 --pred_len 12 \
  --enc_in 16 --dec_in 16 --c_out 16 \
  --des 'debug_run' --itr 1
```

- `seq_len`, `label_len`, `pred_len`: Input-output time window
- `enc_in`, `dec_in`, `c_out`: Number of input/output features
- `itr`: Repeat experiment for robustness

---

## 💡 Research Highlights

- **New Correlation Metrics** reveal temporal and structural shifts in energy systems.
- **Spatial GNNs** adapt better to non-stationary, policy-driven, and seasonal regimes.
- **Transformer models** struggle with heterogeneous, multi-energy, multi-national inputs.
- We release all data, preprocessing scripts, and model configs for reproducibility.

---

## 📎 Citation


---

## 📬 Contact

For questions or contributions, please raise an issue or contact the maintainer.

