import numpy as np
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd

# Function to compute Pearson Correlation Coefficient (PCC) matrix
def compute_correlation_matrix(data):
    n = data.shape[1]  # Number of columns (energy types)
    correlation_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:  # Avoid self-correlation being counted
                corr, _ = pearsonr(data.iloc[:, i], data.iloc[:, j])
                correlation_matrix[i, j] = corr

    return correlation_matrix

# Function to compute Granger Causality matrix
def compute_granger_causality_matrix(data, max_lag=3, significance_level=0.05):
    n = data.shape[1]  # Number of columns (energy types)
    causality_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                test_result = grangercausalitytests(data[[data.columns[i], data.columns[j]]], max_lag, verbose=False)
                p_values = [round(test_result[lag][0]['ssr_chi2test'][1], 4) for lag in range(1, max_lag + 1)]
                
                if min(p_values) < significance_level:
                    causality_matrix[i, j] = 1  # Node j Granger-causes node i
                else:
                    causality_matrix[i, j] = 0

    return causality_matrix

# Load the dataset
data = pd.read_csv("datasets/V_Germany_processed_0.csv")

# Compute Pearson Correlation Matrix
correlation_matrix = compute_correlation_matrix(data)

# Compute Granger Causality Matrix
granger_matrix = compute_granger_causality_matrix(data, max_lag=3)

# Save or display matrices
print("Pearson Correlation Matrix:")
print(correlation_matrix)

print("Granger Causality Matrix:")
print(granger_matrix)

# Optionally save to file
np.savetxt("G_pearson_correlation_matrix.csv", correlation_matrix, delimiter=",")
np.savetxt("G_granger_causality_matrix.csv", granger_matrix, delimiter=",")


# python scripts/tpgnn/W.py




'''
import numpy as np
from scipy.stats import pearsonr
from statsmodels.tsa.stattools import grangercausalitytests
import tensorflow as tf
import pandas as pd


# Function to compute Pearson Correlation Coefficient (PCC) matrix
def compute_correlation_matrix(data):
    n = data.shape[1]  # Number of columns (energy types)
    correlation_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:  # Avoid self-correlation being counted
                corr, _ = pearsonr(data.iloc[:, i], data.iloc[:, j])
                correlation_matrix[i, j] = corr

    return correlation_matrix

# Function to compute Granger Causality matrix
def compute_granger_causality_matrix(data, max_lag=3, significance_level=0.05):
    n = data.shape[1]  # Number of columns (energy types)
    causality_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                test_result = grangercausalitytests(data[[data.columns[i], data.columns[j]]], max_lag, verbose=False)
                p_values = [round(test_result[lag][0]['ssr_chi2test'][1], 4) for lag in range(1, max_lag + 1)]
                
                if min(p_values) < significance_level:
                    causality_matrix[i, j] = 1  # Node j Granger-causes node i
                else:
                    causality_matrix[i, j] = 0

    return causality_matrix

# Function to compute Attention-based matrix
def compute_attention_matrix(data):
    n = data.shape[1]
    attention_matrix = np.zeros((n, n))

    # Simulating attention mechanism using a simple feedforward neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(n, activation='softmax', input_shape=(n,))
    ])

    for i in range(n):
        input_vector = data.iloc[:, i].values.reshape(1, -1)
        attention_weights = model.predict(input_vector)[0]
        attention_matrix[i, :] = attention_weights

    return attention_matrix

# Function to compute Dynamic Graph Matrix using a sliding window
def compute_dynamic_graph_matrix(data, window_size=10):
    n = data.shape[1]
    dynamic_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                windowed_corr = []
                for start in range(0, len(data) - window_size, window_size):
                    sub_series_i = data.iloc[start:start + window_size, i]
                    sub_series_j = data.iloc[start:start + window_size, j]
                    corr, _ = pearsonr(sub_series_i, sub_series_j)
                    windowed_corr.append(corr)
                dynamic_matrix[i, j] = np.mean(windowed_corr)  # Average over windows

    return dynamic_matrix

# Load the dataset
data = pd.read_csv("datasets/V_France_processed_0.csv")

# Compute Pearson Correlation Matrix
correlation_matrix = compute_correlation_matrix(data)

# Compute Granger Causality Matrix
granger_matrix = compute_granger_causality_matrix(data, max_lag=3)

# Compute Attention-based Matrix
attention_matrix = compute_attention_matrix(data)

# Compute Dynamic Graph Matrix
dynamic_matrix = compute_dynamic_graph_matrix(data, window_size=10)

# Save or display matrices
print("Pearson Correlation Matrix:")
print(correlation_matrix)

print("Granger Causality Matrix:")
print(granger_matrix)

print("Attention-based Matrix:")
print(attention_matrix)

print("Dynamic Graph Matrix:")
print(dynamic_matrix)

# Save to CSV files
np.savetxt("pearson_correlation_matrix.csv", correlation_matrix, delimiter=",")
np.savetxt("granger_causality_matrix.csv", granger_matrix, delimiter=",")
np.savetxt("attention_matrix.csv", attention_matrix, delimiter=",")
np.savetxt("dynamic_graph_matrix.csv", dynamic_matrix, delimiter=",")


# python scripts/tpgnn/W.py
'''