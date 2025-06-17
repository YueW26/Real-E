import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 
csv_path = "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/hyperparameter_tuning_summary.csv"
df = pd.read_csv(csv_path)

# 
output_dir = os.path.dirname(csv_path)
metrics = [
    'mean_change_rate',
    'variance_change_rate',
    'graph_entropy',
    'graph_change_entropy',
    'mean_ve_distance_change',
    'mean_sparsity',
    'mean_spectral_diff'
]
datasets = ['electricity', 'ETTm1', 'solar', 'new']
params = ['window_size', 'threshold', 'downsample', 'max_steps']

# 
for metric in metrics:
    fig, axes = plt.subplots(len(datasets), len(params), figsize=(20, 12))
    fig.suptitle(f"Delta Metric Analysis: {metric}", fontsize=16)

    for i, dataset in enumerate(datasets):
        for j, param in enumerate(params):
            ax = axes[i, j]

            # 
            if dataset == 'new':
                y = df[metric]
            else:
                delta_col = f"{dataset}_delta_{metric}"
                if delta_col not in df.columns:
                    ax.set_visible(False)
                    continue
                y = df[delta_col]

            x = df[param]
            ax.scatter(x, y, alpha=0.6, label="Data")

            # 
            if len(x.unique()) > 1:
                try:
                    coef = np.polyfit(x, y, 1)
                    poly = np.poly1d(coef)
                    x_fit = np.linspace(x.min(), x.max(), 100)
                    y_fit = poly(x_fit)
                    ax.plot(x_fit, y_fit, color='red', linestyle='--', label="Fit")
                except Exception as e:
                    print(f"⚠️ Fit failed for {metric}, {dataset}, {param}: {e}")

            # 
            ax.set_title(f"{dataset} - {param}")
            ax.set_xlabel(param)
            ax.set_ylabel(f"{'Δ ' if dataset != 'new' else ''}{metric}")
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf_file = os.path.join(output_dir, f"{metric}_delta_analysis.pdf")
    plt.savefig(pdf_file)
    plt.close()
    print(f"✅ Saved: {pdf_file}")