import pandas as pd
import numpy as np
import os
import argparse
import pickle

def prepare_france_data():
    """
    Process France dataset from CSV to HDF5 format for Graph WaveNet
    """
    # Read the CSV file
    print("Reading France dataset...")
    df = pd.read_csv('data/France_processed_0.csv', header=None)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Number of time steps: {df.shape[0]}")
    print(f"Number of nodes: {df.shape[1]}")
    
    # Create a time index (assuming hourly data)
    # You may need to adjust this based on your actual data frequency
    start_date = '2020-01-01 00:00:00'
    time_index = pd.date_range(start=start_date, periods=len(df), freq='H')
    
    # Set the time index
    df.index = time_index
    
    # Save as HDF5 format (similar to METR-LA format)
    output_file = 'data/france.h5'
    df.to_hdf(output_file, key='df', mode='w')
    print(f"France dataset saved to {output_file}")
    
    return df

def generate_france_adj_matrix():
    """
    Generate adjacency matrix for France dataset (10 nodes)
    """
    num_nodes = 10
    
    # Create sensor IDs
    sensor_ids = [str(i) for i in range(num_nodes)]
    
    # Create sensor ID to index mapping
    sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    
    # Create adjacency matrix
    # For demonstration, create a simple connected graph
    adj_mx = np.eye(num_nodes, dtype=np.float32)
    
    # Add connections between neighboring nodes
    np.random.seed(42)  # For reproducibility
    
    # Connect each node to 2-4 random neighbors
    for i in range(num_nodes):
        num_connections = np.random.randint(2, 5)  # 2-4 connections per node
        neighbors = np.random.choice(num_nodes, size=num_connections, replace=False)
        for neighbor in neighbors:
            if i != neighbor:
                # Add bidirectional connection with random weight
                weight = np.random.uniform(0.1, 1.0)
                adj_mx[i, neighbor] = weight
                adj_mx[neighbor, i] = weight
    
    # Ensure diagonal is 1 (self-connection)
    np.fill_diagonal(adj_mx, 1.0)
    
    # Save adjacency matrix
    output_dir = 'data/sensor_graph'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'adj_mx_france.pkl')
    
    with open(output_path, 'wb') as f:
        pickle.dump((sensor_ids, sensor_id_to_ind, adj_mx), f, protocol=2)
    
    print(f"France adjacency matrix saved to {output_path}")
    print(f"Matrix shape: {adj_mx.shape}")
    print(f"Number of sensors: {len(sensor_ids)}")
    
    return sensor_ids, sensor_id_to_ind, adj_mx

def generate_france_training_data():
    """
    Generate training data for France dataset
    """
    # Read the HDF5 file
    df = pd.read_hdf('data/france.h5', key='df')
    
    # Parameters
    seq_length_x = 12
    seq_length_y = 12
    y_start = 1
    
    # Generate training data using similar logic to the original script
    x_offsets = np.sort(np.concatenate((np.arange(-(seq_length_x - 1), 1, 1),)))
    y_offsets = np.sort(np.arange(y_start, (seq_length_y + 1), 1))
    
    # Generate graph seq2seq data
    x, y = generate_graph_seq2seq_io_data(
        df,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        add_time_in_day=True,
        add_day_in_week=False
    )
    
    print("x shape: ", x.shape, ", y shape: ", y.shape)
    
    # Create output directory
    output_dir = 'data/FRANCE'
    os.makedirs(output_dir, exist_ok=True)
    
    # Split data
    num_samples = x.shape[0]
    num_test = round(num_samples * 0.2)
    num_train = round(num_samples * 0.7)
    num_val = num_samples - num_test - num_train
    
    x_train, y_train = x[:num_train], y[:num_train]
    x_val, y_val = (
        x[num_train: num_train + num_val],
        y[num_train: num_train + num_val],
    )
    x_test, y_test = x[-num_test:], y[-num_test:]
    
    for cat in ["train", "val", "test"]:
        _x, _y = locals()["x_" + cat], locals()["y_" + cat]
        print(cat, "x: ", _x.shape, "y:", _y.shape)
        np.savez_compressed(
            os.path.join(output_dir, f"{cat}.npz"),
            x=_x,
            y=_y,
            x_offsets=x_offsets.reshape(list(x_offsets.shape) + [1]),
            y_offsets=y_offsets.reshape(list(y_offsets.shape) + [1]),
        )
    
    print(f"France training data saved to {output_dir}")

def generate_graph_seq2seq_io_data(
        df, x_offsets, y_offsets, add_time_in_day=True, add_day_in_week=False, scaler=None
):
    """
    Generate samples from dataframe
    """
    num_samples, num_nodes = df.shape
    data = np.expand_dims(df.values, axis=-1)
    feature_list = [data]
    
    if add_time_in_day:
        time_ind = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
        time_in_day = np.tile(time_ind, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(time_in_day)
        
    if add_day_in_week:
        dow = df.index.dayofweek
        dow_tiled = np.tile(dow, [1, num_nodes, 1]).transpose((2, 1, 0))
        feature_list.append(dow_tiled)

    data = np.concatenate(feature_list, axis=-1)
    x, y = [], []
    min_t = abs(min(x_offsets))
    max_t = abs(num_samples - abs(max(y_offsets)))  # Exclusive
    for t in range(min_t, max_t):  # t is the index of the last observation.
        x.append(data[t + x_offsets, ...])
        y.append(data[t + y_offsets, ...])
    x = np.stack(x, axis=0)
    y = np.stack(y, axis=0)
    return x, y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, default='all', 
                       choices=['all', 'convert', 'adj', 'training'],
                       help='Which step to run: all, convert, adj, or training')
    
    args = parser.parse_args()
    
    if args.step in ['all', 'convert']:
        print("Step 1: Converting CSV to HDF5...")
        prepare_france_data()
    
    if args.step in ['all', 'adj']:
        print("Step 2: Generating adjacency matrix...")
        generate_france_adj_matrix()
    
    if args.step in ['all', 'training']:
        print("Step 3: Generating training data...")
        generate_france_training_data()
    
    print("France data preparation completed!") 