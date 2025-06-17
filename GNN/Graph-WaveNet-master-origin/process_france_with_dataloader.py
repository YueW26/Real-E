import pandas as pd
import numpy as np
import os
import argparse
from datetime import datetime
import pickle

# Import the Dataset_Opennem class from dataloader.py
from dataloader import Dataset_Opennem

def prepare_france_csv_for_dataloader():
    """
    Prepare the new France CSV data (which already has timestamps) for Dataset_Opennem
    """
    print("🔄 准备新的France CSV数据（已包含时间戳）...")
    
    # Read the CSV file that already has timestamps and proper structure
    csv_file = 'data/France_processed_0.csv'
    df = pd.read_csv(csv_file)
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # Verify the time column
    print(f"时间列: {df['date'].head()}")
    print(f"时间范围: {df['date'].iloc[0]} 到 {df['date'].iloc[-1]}")
    
    # The data is already in the correct format for Dataset_Opennem
    # Just need to specify target column (use the last power generation column)
    feature_columns = [col for col in df.columns if col != 'date']
    target_col = feature_columns[-1]  # Use last column as target
    
    print(f"特征列数量: {len(feature_columns)}")
    print(f"目标列: {target_col}")
    
    # Save the data (it's already in the right format)
    output_file = 'data/france_for_dataloader.csv'
    df.to_csv(output_file, index=False)
    print(f"✅ 数据保存到: {output_file}")
    
    # Check data frequency
    time_diff = pd.to_datetime(df['date'].iloc[1]) - pd.to_datetime(df['date'].iloc[0])
    freq_detected = pd.infer_freq(pd.to_datetime(df['date'][:100]))
    print(f"检测到的数据频率: {freq_detected} (时间间隔: {time_diff})")
    
    return output_file, target_col, len(feature_columns)

def create_france_dataset_using_dataloader(data_file, target_col, seq_length=12, pred_length=12):
    """
    Use Dataset_Opennem to create France dataset with configurable sequence lengths
    """
    print(f"📊 使用Dataset_Opennem创建France数据集 (seq_length={seq_length}, pred_length={pred_length})...")
    
    # Create datasets using Dataset_Opennem
    # Parameters for time series (configurable based on input)
    size = [seq_length, 0, pred_length]  # [seq_len, label_len, pred_len]
    
    datasets = {}
    for flag in ['train', 'val', 'test']:
        print(f"创建 {flag} 数据集...")
        dataset = Dataset_Opennem(
            root_path='data',
            flag=flag,
            size=size,
            features='M',  # Use all features (multivariate)
            data_path='france_for_dataloader.csv',
            target=target_col,
            scale=True,
            timeenc=1,  # Use advanced time encoding from dataloader
            freq='h'  # Hourly frequency (detected from the data)
        )
        datasets[flag] = dataset
        print(f"  {flag} 数据集大小: {len(dataset)}")
        
        # Get a sample to check dimensions
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"  样本形状: X={sample[0].shape}, Y={sample[1].shape}, X_mark={sample[2].shape}, Y_mark={sample[3].shape}")
    
    return datasets

def convert_to_graph_wavenet_format(datasets):
    """
    Convert Dataset_Opennem output to Graph WaveNet format
    """
    print("🔄 转换为Graph WaveNet格式...")
    
    output_dir = 'data/FRANCE'
    os.makedirs(output_dir, exist_ok=True)
    
    # First pass: collect all data to find global min/max for scaling
    all_X = []
    all_Y = []
    
    for flag in ['train', 'val', 'test']:
        dataset = datasets[flag]
        if len(dataset) == 0:
            continue
            
        # Collect all samples for this split
        X_list, Y_list = [], []
        
        for i in range(len(dataset)):
            seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[i]
            X_list.append(seq_x)
            Y_list.append(seq_y)
        
        if X_list:
            X = np.stack(X_list, axis=0)
            Y = np.stack(Y_list, axis=0)
            all_X.append(X)
            all_Y.append(Y)
    
    # Find global min and max for scaling to 0-1 range
    if all_X and all_Y:
        global_X = np.concatenate(all_X, axis=0)
        global_Y = np.concatenate(all_Y, axis=0)
        all_data = np.concatenate([global_X.flatten(), global_Y.flatten()])
        
        data_min = all_data.min()
        data_max = all_data.max()
        data_range = data_max - data_min
        
        print(f"📊 全局数据统计:")
        print(f"  原始范围: {data_min:.4f} - {data_max:.4f}")
        print(f"  将缩放到: 0.0 - 1.0")
    
    # Second pass: process and save each split with scaling
    for flag in ['train', 'val', 'test']:
        dataset = datasets[flag]
        if len(dataset) == 0:
            continue
            
        print(f"处理 {flag} 数据集...")
        
        # Collect all samples
        X_list, Y_list = [], []
        X_mark_list, Y_mark_list = [], []
        
        for i in range(len(dataset)):
            seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[i]
            X_list.append(seq_x)
            Y_list.append(seq_y)
            X_mark_list.append(seq_x_mark)
            Y_mark_list.append(seq_y_mark)
        
        # Stack all samples
        X = np.stack(X_list, axis=0)  # Shape: (num_samples, seq_len, num_features)
        Y = np.stack(Y_list, axis=0)  # Shape: (num_samples, pred_len, num_features)
        X_mark = np.stack(X_mark_list, axis=0)  # Time features
        Y_mark = np.stack(Y_mark_list, axis=0)
        
        print(f"  原始形状: X={X.shape}, Y={Y.shape}, X_mark={X_mark.shape}")
        
        # Apply min-max scaling to get 0-1 range
        X_scaled = (X - data_min) / data_range
        Y_scaled = (Y - data_min) / data_range
        
        # Ensure values are strictly in [0, 1] range
        X_scaled = np.clip(X_scaled, 0.0, 1.0)
        Y_scaled = np.clip(Y_scaled, 0.0, 1.0)
        
        print(f"  缩放后范围: X=[{X_scaled.min():.4f}, {X_scaled.max():.4f}], Y=[{Y_scaled.min():.4f}, {Y_scaled.max():.4f}]")
        
        # For Graph WaveNet, we need to reshape to include node dimension
        # Treat each feature as a separate node
        num_samples, seq_len, num_features = X_scaled.shape
        _, pred_len, _ = Y_scaled.shape
        
        # Reshape X: (num_samples, seq_len, num_nodes, 1) for data + time features
        X_reshaped = X_scaled.reshape(num_samples, seq_len, num_features, 1)
        Y_reshaped = Y_scaled.reshape(num_samples, pred_len, num_features, 1)
        
        # Add time features as second channel
        # X_mark shape: (num_samples, seq_len, time_features)
        # We need to broadcast it to (num_samples, seq_len, num_nodes, time_features)
        time_features_expanded = np.broadcast_to(
            X_mark[:, :, np.newaxis, :], 
            (num_samples, seq_len, num_features, X_mark.shape[-1])
        )
        
        # Take only the first time feature and scale it to 0-1 range
        time_feature_raw = time_features_expanded[:, :, :, 0:1]
        
        # Scale time features to 0-1 range
        time_min = time_feature_raw.min()
        time_max = time_feature_raw.max()
        time_range = time_max - time_min
        if time_range > 0:
            time_feature_scaled = (time_feature_raw - time_min) / time_range
        else:
            time_feature_scaled = np.zeros_like(time_feature_raw)
        
        # Ensure time features are in [0, 1] range
        time_feature_scaled = np.clip(time_feature_scaled, 0.0, 1.0)
        
        # Concatenate data and time features
        X_final = np.concatenate([X_reshaped, time_feature_scaled], axis=-1)
        Y_final = np.concatenate([Y_reshaped, time_feature_scaled[:, -pred_len:, :, :]], axis=-1)
        
        print(f"  Graph WaveNet格式: X={X_final.shape}, Y={Y_final.shape}")
        
        # Create offsets (required by Graph WaveNet)
        x_offsets = np.arange(-seq_len + 1, 1)
        y_offsets = np.arange(1, pred_len + 1)
        
        # Save in Graph WaveNet format
        np.savez_compressed(
            os.path.join(output_dir, f"{flag}.npz"),
            x=X_final,
            y=Y_final,
            x_offsets=x_offsets.reshape(-1, 1),
            y_offsets=y_offsets.reshape(-1, 1),
        )
        
        print(f"  ✅ {flag} 数据保存完成: {X_final.shape}")
    
    print(f"✅ 所有数据保存到: {output_dir}")
    return num_features

def generate_adjacency_matrix_for_features(num_nodes):
    """
    Generate adjacency matrix for France dataset based on power generation relationships
    """
    print(f"🔗 为{num_nodes}个电力特征生成邻接矩阵...")
    
    # Power generation types have natural relationships
    # Renewable sources might be connected, fossil fuels connected, etc.
    adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    
    # Create a more realistic connectivity pattern for power generation
    # Based on the order of columns in the France dataset
    power_types = [
        'Biomass', 'Fossil_Gas', 'Fossil_Hard_coal', 'Fossil_Oil',
        'Hydro_Run_of_river', 'Hydro_Water_Reservoir', 'Nuclear',
        'Solar', 'Waste', 'Wind_Onshore'
    ]
    
    # Group related power sources
    renewables = [0, 4, 5, 7, 9]  # Biomass, Hydro Run-of-river, Hydro Water Reservoir, Solar, Wind Onshore
    fossils = [1, 2, 3]  # Fossil Gas, Hard coal, Oil
    hydro = [4, 5]  # Hydro types
    
    # Connect renewable sources
    for i in renewables:
        for j in renewables:
            if i != j:
                adj[i, j] = 1.0
    
    # Connect fossil fuel sources
    for i in fossils:
        for j in fossils:
            if i != j:
                adj[i, j] = 1.0
    
    # Connect hydro sources strongly
    for i in hydro:
        for j in hydro:
            if i != j:
                adj[i, j] = 1.0
    
    # Connect all adjacent power types (temporal/operational relationships)
    for i in range(num_nodes):
        if i > 0:
            adj[i, i-1] = 1.0
        if i < num_nodes - 1:
            adj[i, i+1] = 1.0
    
    # Self connections
    np.fill_diagonal(adj, 1.0)
    
    # Create sensor IDs based on actual power generation types
    sensor_ids = [f"france_{power_types[i] if i < len(power_types) else f'feature_{i}'}" for i in range(num_nodes)]
    sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    
    # Save adjacency matrix
    adj_dir = 'data/sensor_graph'
    os.makedirs(adj_dir, exist_ok=True)
    
    with open(os.path.join(adj_dir, 'adj_mx_france.pkl'), 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj.astype(np.float32)], f, protocol=2)
    
    print(f"✅ 邻接矩阵保存到: {adj_dir}/adj_mx_france.pkl")
    print(f"邻接矩阵形状: {adj.shape}")
    print(f"连接数: {(adj > 0).sum()}")
    print(f"电力类型: {sensor_ids}")
    
    return adj

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process real France power generation data using Dataset_Opennem')
    parser.add_argument('--step', type=str, default='all', 
                       choices=['all', 'prepare', 'process', 'adj'],
                       help='Which step to run: all, prepare, process, or adj')
    parser.add_argument('--seq_length', type=int, default=12,
                       help='Input sequence length')
    parser.add_argument('--pred_length', type=int, default=12,
                       help='Prediction sequence length')
    
    args = parser.parse_args()
    
    print("🇫🇷 真实France电力数据集处理开始（使用Dataset_Opennem）...")
    print(f"📊 序列长度配置: seq_length={args.seq_length}, pred_length={args.pred_length}")
    
    if args.step in ['all', 'prepare']:
        print("\n步骤 1: 准备CSV数据...")
        data_file, target_col, num_features = prepare_france_csv_for_dataloader()
    else:
        data_file = 'data/france_for_dataloader.csv'
        target_col = 'Wind Onshore  - Actual Aggregated [MW]'  # Default target
        num_features = 10  # Default value for France (10 features)
    
    if args.step in ['all', 'process']:
        print("\n步骤 2: 使用Dataset_Opennem处理数据...")
        datasets = create_france_dataset_using_dataloader(data_file, target_col, args.seq_length, args.pred_length)
        
        print("\n步骤 3: 转换为Graph WaveNet格式...")
        num_features = convert_to_graph_wavenet_format(datasets)
        
        print("\n🔍 验证生成的数据:")
        for split in ['train', 'val', 'test']:
            file_path = f'data/FRANCE/{split}.npz'
            if os.path.exists(file_path):
                data = np.load(file_path)
                print(f"{split.upper()} - X: {data['x'].shape}, Y: {data['y'].shape}")
    
    if args.step in ['all', 'adj']:
        print(f"\n步骤 4: 生成{num_features}个电力特征的邻接矩阵...")
        generate_adjacency_matrix_for_features(num_features)
    
    print("\n🎉 真实France电力数据集处理完成!")
    print("📊 数据特点:")
    print("  ✅ 使用真实时间戳 (2015-2024)")
    print("  ✅ 10种电力生成类型")
    print("  ✅ 小时级数据频率")
    print("  ✅ Dataset_Opennem专业处理")
    print("  ✅ 数据范围控制在 [0, 1]")
    print(f"  ✅ 序列长度: {args.seq_length} -> {args.pred_length}")
    print("\n可以使用以下命令训练:")
    print(f"python train.py --data data/FRANCE --gcn_bool --adjtype doubletransition --addaptadj --randomadj --epochs 50 --seq_length {args.seq_length} --pred_length {args.pred_length}")
    
