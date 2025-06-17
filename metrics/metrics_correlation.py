import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm, eigvals
import itertools
import matplotlib.pyplot as plt

def compute_dynamic_graph_metrics(data, window_size=24, threshold=0.1, downsample=10, max_steps=2000):
    data = data.iloc[::downsample, :].reset_index(drop=True).iloc[:max_steps]
    adj_matrices = []

    for t in range(len(data) - window_size + 1):
        window_data = data.iloc[t:t + window_size].T
        sim_matrix = cosine_similarity(window_data)
        adj_matrices.append(sim_matrix)

    edge_changes = []
    ve_distance_changes = []
    sparsities = []
    spectral_diffs = []

    for i in range(1, len(adj_matrices)):
        A_t, A_prev = adj_matrices[i], adj_matrices[i - 1]
        diff = np.abs(A_t - A_prev)
        edge_changes.append((diff > threshold).sum() / diff.size) ###### 1; 2
        ve_distance_changes.append(norm(A_t - A_prev)) ###### 5
        sparsities.append(np.sum(A_t < threshold) / A_t.size) ###### 6

        D_t, D_prev = np.diag(A_t.sum(axis=1)), np.diag(A_prev.sum(axis=1))
        L_t, L_prev = D_t - A_t, D_prev - A_prev
        eig_diff = norm(np.sort(eigvals(L_t)) - np.sort(eigvals(L_prev)))
        spectral_diffs.append(eig_diff) ###### 7

    # Static structure entropy (long-term edge presence)
    pij = np.mean(np.array(adj_matrices) > threshold, axis=0)
    corr_entropy = -np.nansum(pij * np.log(pij + 1e-8)) ###### 3

    # Structure change entropy (short-term edge volatility)
    diff_flags = []
    for i in range(1, len(adj_matrices)):
        delta = (np.abs(adj_matrices[i] - adj_matrices[i - 1]) > threshold).astype(int)
        diff_flags.append(delta)
    diff_flags = np.array(diff_flags)
    qij = np.mean(diff_flags, axis=0)
    change_entropy = -np.nansum(qij * np.log(qij + 1e-8)) ###### 4

    return {
        'mean_change_rate': np.mean(edge_changes),
        'variance_change_rate': np.var(edge_changes),
        'graph_entropy': corr_entropy,
        'graph_change_entropy': change_entropy,
        'mean_ve_distance_change': np.mean(ve_distance_changes),
        'mean_sparsity': np.mean(sparsities),
        'mean_spectral_diff': np.mean(spectral_diffs),
    }

# === Dataset paths ===
path_mydata = "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/Germany_processed_0.csv"
old_paths = {
    "electricity": "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/electricity.csv",
    "ETTm1": "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ETTm1.csv",
    "solar": "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/solar.csv"
}

# === Load new data ===
my_data = pd.read_csv(path_mydata).select_dtypes(include=[np.number]).interpolate().dropna()

# === Hyperparameter grid ===
param_grid = {
    'window_size': [3, 6, 12, 24, 48],
    'threshold': [0.05, 0.1, 0.2],
    'downsample': [5, 10],
    'max_steps': [1000, 2000]
}

# === Run combinations ===
results = []
for ws, thres, ds, maxs in itertools.product(*param_grid.values()):
    print(f"Testing: window_size={ws}, threshold={thres}, downsample={ds}, max_steps={maxs}")
    my_metrics = compute_dynamic_graph_metrics(my_data, ws, thres, ds, maxs)
    deltas = {}

    for name, path in old_paths.items():
        old_data = pd.read_csv(path).select_dtypes(include=[np.number]).interpolate().dropna()
        old_data = old_data.iloc[:, :my_data.shape[1]]
        old_metrics = compute_dynamic_graph_metrics(old_data, ws, thres, ds, maxs)

        for key in my_metrics:
            delta = my_metrics[key] - old_metrics[key]
            deltas[f"{name}_delta_{key}"] = delta

    results.append({
        'window_size': ws,
        'threshold': thres,
        'downsample': ds,
        'max_steps': maxs,
        **my_metrics,
        **deltas
    })

# === Save results ===
df_results = pd.DataFrame(results)
csv_path = "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/hyperparameter_tuning_summary.csv"
df_results.to_csv(csv_path, index=False)
print(f"📁 Results saved to: {csv_path}")

# === Find best parameter set ===
better_all = df_results[(df_results.filter(like="delta_") > 0).all(axis=1)]

if not better_all.empty:
    print("✅ Best parameter combinations where new data dominates all old datasets:")
    print(better_all)
    best = better_all.iloc[0]
else:
    print("❌ No parameter set found where new data dominates all old datasets across all metrics.")
    best = df_results.iloc[df_results.filter(like="delta_").sum(axis=1).idxmax()]
    print("⚠️ Best found based on maximum total delta:")
    print(best)

# === Plot key metrics ===
metrics_to_plot = [
    'mean_change_rate',
    'variance_change_rate',
    'graph_entropy',
    'graph_change_entropy',
    'mean_ve_distance_change',
    'mean_sparsity',
    'mean_spectral_diff'
]

plt.figure(figsize=(14, 16))
for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(4, 2, i)
    plt.plot(df_results[metric], marker='o')
    plt.title(metric)
    plt.xlabel("Experiment #")
    plt.ylabel(metric)

plt.tight_layout()
pdf_path = "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/hyperparameter_metrics_plot.pdf"
plt.savefig(pdf_path)
print(f"📊 Plot saved to: {pdf_path}")

'''
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from numpy.linalg import norm, eigvals

def compute_dynamic_graph_metrics(data, window_size=24, threshold=0.1, downsample=10, max_steps=2000):
    # Downsample and limit rows
    data = data.iloc[::downsample, :].reset_index(drop=True).iloc[:max_steps]

    adj_matrices = []

    # Sliding window to compute cosine similarity matrices
    for t in range(len(data) - window_size + 1):
        window_data = data.iloc[t:t + window_size].T  # shape: (variables, window)
        sim_matrix = cosine_similarity(window_data)
        adj_matrices.append(sim_matrix)

    # Compute edge change rate between adjacent matrices
    edge_changes = []
    ve_distance_changes = []
    sparsities = []
    spectral_diffs = []

    for i in range(1, len(adj_matrices)):
        A_t = adj_matrices[i]
        A_prev = adj_matrices[i - 1]
        diff = np.abs(A_t - A_prev)
        change_rate = (diff > threshold).sum() / diff.size
        edge_changes.append(change_rate)

        # VE embedding change: approximated using adjacency matrix diff norm
        ve_distance = norm(A_t - A_prev)
        ve_distance_changes.append(ve_distance)

        # Sparsity of the adjacency matrix
        sparsity = np.sum(A_t < threshold) / A_t.size
        sparsities.append(sparsity)

        # Spectral difference (Laplacian eigenvalue distance)
        D_t = np.diag(A_t.sum(axis=1))
        D_prev = np.diag(A_prev.sum(axis=1))
        L_t = D_t - A_t
        L_prev = D_prev - A_prev
        eig_diff = norm(np.sort(eigvals(L_t)) - np.sort(eigvals(L_prev)))
        spectral_diffs.append(eig_diff)

    mean_change = np.mean(edge_changes)
    var_change = np.var(edge_changes)

    # Compute structural entropy
    edge_probs = np.mean(np.array(adj_matrices) > threshold, axis=0)
    entropy = -np.nansum(edge_probs * np.log(edge_probs + 1e-8))

    return {
        'mean_change_rate': mean_change, #1 
        'variance_change_rate': var_change, #2 
        'graph_entropy': entropy, #3 
        'mean_ve_distance_change': np.mean(ve_distance_changes), #4 
        'mean_sparsity': np.mean(sparsities), #5 
        'mean_spectral_diff': np.mean(spectral_diffs), #6
    }

# Load datasets
path_mydata = "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/Germany_processed_0.csv"
path_oldata = "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/electricity.csv"

# Read CSV files and keep numeric columns only
my_data = pd.read_csv(path_mydata).select_dtypes(include=[np.number]).interpolate().dropna()
old_data = pd.read_csv(path_oldata).select_dtypes(include=[np.number]).interpolate().dropna()

# Ensure same number of columns for fair comparison
min_columns = min(my_data.shape[1], old_data.shape[1])
my_data = my_data.iloc[:, :min_columns]
old_data = old_data.iloc[:, :min_columns]

# window_size=24, threshold=0.1, downsample=10, max_steps=2000 
import itertools

# Define the hyperparameter space
param_grid = {
    'window_size': [1, 2, 3, 4],
    'threshold': [0.05, 0.1, 0.2],
    'downsample': [5, 10],
    'max_steps': [1000, 2000]
}
ws_interval = 24 
# Unpack the keys and values
keys = param_grid.keys()
values = param_grid.values()

# Loop over all combinations
for ws, thres, ds, maxs in itertools.product(*values):
    # Use these hyperparameters in your process
    print(f"Running with: window_size={ws}, threshold={thres}, downsample={ds}, max_steps={maxs}")
    # Compute metrics
    my_metrics = compute_dynamic_graph_metrics(my_data, ws*ws_interval, thres, ds, maxs)
    old_metrics = compute_dynamic_graph_metrics(old_data)


# Display results
def print_metrics(name, metrics):
    print(f"== {name} ==")
    print(f"Mean Change Rate: {metrics['mean_change_rate']:.6f}")
    print(f"Variance of Change Rate: {metrics['variance_change_rate']:.6f}")
    print(f"Graph Structure Entropy: {metrics['graph_entropy']:.6f}")
    print(f"Mean VE Embedding Change (Adj diff norm): {metrics['mean_ve_distance_change']:.6f}")
    print(f"Mean Sparsity of Graphs: {metrics['mean_sparsity']:.6f}")
    print(f"Mean Spectral Difference: {metrics['mean_spectral_diff']:.6f}\\n")

import itertools
import matplotlib.pyplot as plt
import pandas as pd

# Define hyperparameter grid
param_grid = {
    'window_size': [24, 48],
    'threshold': [0.05, 0.1],
    'downsample': [5, 10],
    'max_steps': [1000, 2000]
}

keys = list(param_grid.keys())
values = list(param_grid.values())
results = []

# Iterate over all hyperparameter combinations
for ws, thres, ds, maxs in itertools.product(*values):
    print(f"Running with: window_size={ws}, threshold={thres}, downsample={ds}, max_steps={maxs}")
    
    # Compute metrics
    my_metrics = compute_dynamic_graph_metrics(my_data, ws * ws_interval, thres, ds, maxs)
    old_metrics = compute_dynamic_graph_metrics(old_data)
    
    results.append({
        'window_size': ws,
        'threshold': thres,
        'downsample': ds,
        'max_steps': maxs,
        'mean_change_rate': my_metrics['mean_change_rate'],
        'variance_change_rate': my_metrics['variance_change_rate'],
        'graph_entropy': my_metrics['graph_entropy'],
        'mean_ve_distance_change': my_metrics['mean_ve_distance_change'],
        'mean_sparsity': my_metrics['mean_sparsity'],
        'mean_spectral_diff': my_metrics['mean_spectral_diff']
    })

# Convert results to DataFrame
df = pd.DataFrame(results)

# ---- PLOT RESULTS ----
metrics_to_plot = [
    'mean_change_rate',
    'variance_change_rate',
    'graph_entropy',
    'mean_ve_distance_change',
    'mean_sparsity',
    'mean_spectral_diff'
]

# Create subplots
plt.figure(figsize=(12, 14))
for i, metric in enumerate(metrics_to_plot, 1):
    plt.subplot(3, 2, i)
    plt.plot(df[metric], marker='o')
    plt.title(metric)
    plt.xlabel("Experiment #")
    plt.ylabel(metric)

plt.tight_layout()
plt.savefig("hyperparameter_metrics_plot.pdf")
print("Plot saved to 'hyperparameter_metrics_plot.pdf'.")
output_path = "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/hyperparameter_metrics_plot.pdf"
plt.savefig(output_path)
print(f"Plot saved to: {output_path}")

# ---- DISPLAY METRICS FUNCTION ----
def print_metrics(name, metrics):
    print(f"== {name} ==")
    print(f"Mean Change Rate: {metrics['mean_change_rate']:.6f}")
    print(f"Variance of Change Rate: {metrics['variance_change_rate']:.6f}")
    print(f"Graph Structure Entropy: {metrics['graph_entropy']:.6f}")
    print(f"Mean VE Embedding Change (Adj diff norm): {metrics['mean_ve_distance_change']:.6f}")
    print(f"Mean Sparsity of Graphs: {metrics['mean_sparsity']:.6f}")
    print(f"Mean Spectral Difference: {metrics['mean_spectral_diff']:.6f}\n")
print_metrics("New Data (Germany)", my_metrics)
print_metrics("Old Data (Electricity)", old_metrics)





import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from numpy.linalg import norm, eigvals

def compute_dynamic_graph_metrics(data, window_size=24, threshold=0.1, downsample=10, max_steps=2000):
    # Downsample the data and truncate to max_steps
    data = data.iloc[::downsample, :].reset_index(drop=True).iloc[:max_steps]

    adj_matrices = []

    # Generate cosine similarity matrices using a sliding window
    for t in range(len(data) - window_size + 1):
        window_data = data.iloc[t:t + window_size].T  # shape: (variables, window)
        sim_matrix = cosine_similarity(window_data)
        adj_matrices.append(sim_matrix)

    # Initialize lists to store metrics over time
    edge_changes = []
    ve_distance_changes = []
    sparsities = []
    spectral_diffs = []

    # Compute dynamic graph metrics
    for i in range(1, len(adj_matrices)):
        A_t = adj_matrices[i]
        A_prev = adj_matrices[i - 1]

        # Edge change rate based on thresholded difference
        diff = np.abs(A_t - A_prev)
        change_rate = (diff > threshold).sum() / diff.size
        edge_changes.append(change_rate)

        # Virtual embedding (VE) distance: norm difference
        ve_distance = norm(A_t - A_prev)
        ve_distance_changes.append(ve_distance)

        # Sparsity: proportion of low-weight edges
        sparsity = np.sum(A_t < threshold) / A_t.size
        sparsities.append(sparsity)

        # Spectral difference using Laplacian eigenvalues
        D_t = np.diag(A_t.sum(axis=1))
        D_prev = np.diag(A_prev.sum(axis=1))
        L_t = D_t - A_t
        L_prev = D_prev - A_prev
        eig_diff = norm(np.sort(eigvals(L_t)) - np.sort(eigvals(L_prev)))
        spectral_diffs.append(eig_diff)

    # Average metrics
    mean_change = np.mean(edge_changes)
    var_change = np.var(edge_changes)

    # Structural entropy based on edge probabilities
    edge_probs = np.mean(np.array(adj_matrices) > threshold, axis=0)
    entropy = -np.nansum(edge_probs * np.log(edge_probs + 1e-8))

    return {
        'mean_change_rate': mean_change,
        'variance_change_rate': var_change,
        'graph_entropy': entropy,
        'mean_ve_distance_change': np.mean(ve_distance_changes),
        'mean_sparsity': np.mean(sparsities),
        'mean_spectral_diff': np.mean(spectral_diffs),
    }

# Dataset paths
path_mydata = "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/Germany_processed_0.csv"
path_oldata = "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/electricity.csv"
path_ettdat = "/mnt/webscistorage/cc7738/ws_joella/EnergyTSF/datasets/ETTm1.csv"

# Load and preprocess datasets (keep only numeric columns)
my_data = pd.read_csv(path_mydata).select_dtypes(include=[np.number]).interpolate().dropna()
old_data = pd.read_csv(path_oldata).select_dtypes(include=[np.number]).interpolate().dropna()
ettdat = pd.read_csv(path_ettdat).select_dtypes(include=[np.number]).interpolate().dropna()

# Ensure all datasets have the same number of variables (columns)
min_columns = min(my_data.shape[1], old_data.shape[1], ettdat.shape[1])
my_data = my_data.iloc[:, :min_columns]
old_data = old_data.iloc[:, :min_columns]
ettdat = ettdat.iloc[:, :min_columns]

# Compute dynamic graph metrics
my_metrics = compute_dynamic_graph_metrics(my_data)
old_metrics = compute_dynamic_graph_metrics(old_data)
ettm1_metrics = compute_dynamic_graph_metrics(ettdat)

# Function to display metrics
def print_metrics(name, metrics):
    print(f"== {name} ==")
    print(f"Mean Change Rate: {metrics['mean_change_rate']:.6f}")
    print(f"Variance of Change Rate: {metrics['variance_change_rate']:.6f}")
    print(f"Graph Structure Entropy: {metrics['graph_entropy']:.6f}")
    print(f"Mean VE Embedding Change (Adj diff norm): {metrics['mean_ve_distance_change']:.6f}")
    print(f"Mean Sparsity of Graphs: {metrics['mean_sparsity']:.6f}")
    print(f"Mean Spectral Difference: {metrics['mean_spectral_diff']:.6f}\n")

# Print results for all datasets
print_metrics("New Data (Germany)", my_metrics)
print_metrics("Old Data (Electricity)", old_metrics)
print_metrics("ETT Data (ETTm1)", ettm1_metrics)
'''