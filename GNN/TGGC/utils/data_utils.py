import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import DataLoader
from .dataloader_joella import Dataset_Opennem, graph_based_interpolation, time_features
import torch

class CompatibleDataset_Opennem(Dataset_Opennem):
    """
    兼容版本的Dataset_Opennem，解决pandas语法问题
    """
    
    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        
        # 确保date列存在并正确解析
        if 'date' in df_raw.columns:
            df_raw['date'] = pd.to_datetime(df_raw['date'])
        else:
            raise ValueError("数据文件必须包含'date'列")

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        
        # 安全地移除目标列和日期列
        if self.target in cols:
            cols.remove(self.target)
        if 'date' in cols:
            cols.remove('date')
        
        df_raw = df_raw[['date'] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        if num_train <= self.seq_len + self.pred_len:
            raise ValueError("The training set data is too small to meet the requirements of seq_len and pred_len.")
        if num_vali <= self.seq_len + self.pred_len:
            num_vali = self.seq_len + self.pred_len
            num_train = len(df_raw) - num_vali - num_test
        if num_test <= self.seq_len + self.pred_len:
            num_test = self.seq_len + self.pred_len
            num_train = len(df_raw) - num_vali - num_test

        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        border1 = max(border1, 0)
        border2 = min(border2, len(df_raw))

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)

            adj_path = os.path.join(self.root_path, 'adj_mx.npy')
            if os.path.exists(adj_path):
                adj = np.load(adj_path)
                data = graph_based_interpolation(data, adj)
            else:
                data = pd.DataFrame(data).interpolate(method='linear', axis=0).fillna(method='bfill').fillna(method='ffill').values
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday())
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour)
            df_stamp['minute'] = df_stamp['date'].apply(lambda row: row.minute)
            # 使用新的pandas语法
            data_stamp = df_stamp.drop(columns=['date']).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

def load_and_process_dataset(root_path, data_file, target_column=None, features='M', 
                           seq_len=96, label_len=48, pred_len=24, 
                           scale_to_01=True, batch_size=32, freq='h',
                           timeenc=0, use_graph_interpolation=False):
    """
    使用Dataset_Opennem类加载并处理数据集，确保数据范围在0-1之间
    
    Args:
        root_path: 根目录路径
        data_file: 数据文件名
        target_column: 目标列名，如果为None则自动检测
        features: 特征模式 'M' (multivariate) 或 'S' (univariate)
        seq_len: 序列长度
        label_len: 标签长度
        pred_len: 预测长度
        scale_to_01: 是否将数据缩放到0-1范围
        batch_size: 批次大小
        freq: 频率字符串
        timeenc: 时间编码模式
        use_graph_interpolation: 是否使用图基插值
        
    Returns:
        dict: 包含训练、验证、测试数据加载器和相关信息
    """
    
    # 获取数据集名称
    dataset_name = os.path.basename(data_file).replace('.csv', '')
    print(f"正在处理数据集: {dataset_name}")
    
    # 读取并检查数据文件
    data_path = os.path.join(root_path, data_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 预处理数据文件，确保格式符合Dataset_Opennem的要求
    processed_file = preprocess_data_file(root_path, data_file, target_column)
    
    size = [seq_len, label_len, pred_len]
    
    # 自动检测目标列
    if target_column is None:
        df_temp = pd.read_csv(processed_file)
        # 假设最后一列为目标列（除了date列）
        cols = [col for col in df_temp.columns if col.lower() != 'date']
        target_column = cols[-1]
        print(f"自动检测目标列: {target_column}")
    
    # 创建数据集实例
    try:
        # 训练集
        train_dataset = CompatibleDataset_Opennem(
            root_path=root_path,
            flag='train',
            size=size,
            features=features,
            data_path=os.path.basename(processed_file),
            target=target_column,
            scale=True,
            timeenc=timeenc,
            freq=freq
        )
        
        # 验证集
        val_dataset = CompatibleDataset_Opennem(
            root_path=root_path,
            flag='val',
            size=size,
            features=features,
            data_path=os.path.basename(processed_file),
            target=target_column,
            scale=True,
            timeenc=timeenc,
            freq=freq
        )
        
        # 测试集
        test_dataset = CompatibleDataset_Opennem(
            root_path=root_path,
            flag='test',
            size=size,
            features=features,
            data_path=os.path.basename(processed_file),
            target=target_column,
            scale=True,
            timeenc=timeenc,
            freq=freq
        )
        
        print(f"数据集创建成功:")
        print(f"  训练集大小: {len(train_dataset)}")
        print(f"  验证集大小: {len(val_dataset)}")
        print(f"  测试集大小: {len(test_dataset)}")
        
    except Exception as e:
        print(f"使用Dataset_Opennem创建数据集失败: {e}")
        raise e
    
    # 如果需要将数据缩放到0-1范围
    if scale_to_01:
        train_dataset, val_dataset, test_dataset = apply_01_scaling(
            train_dataset, val_dataset, test_dataset, use_graph_interpolation, root_path)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False
    )
    
    # 获取数据形状信息
    sample_x, sample_y, sample_x_mark, sample_y_mark = train_dataset[0]
    
    feature_dim = sample_x.shape[-1]
    time_feature_dim = sample_x_mark.shape[-1] if sample_x_mark is not None else 0
    
    # 返回结果
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'dataset_name': data_file.replace('.csv', ''),
        'target_column': target_column,
        'feature_dim': feature_dim,
        'time_feature_dim': time_feature_dim,
        'total_samples': len(train_dataset) + len(val_dataset) + len(test_dataset),
        'scaler': train_dataset.scaler if scale_to_01 else None,
        'data_shape': (len(train_dataset) + len(val_dataset) + len(test_dataset), feature_dim),
        'original_columns': [f'feature_{i}' for i in range(feature_dim)]
    }

def preprocess_data_file(root_path, data_file, target_column=None):
    """
    预处理数据文件，确保格式符合Dataset_Opennem的要求
    
    Args:
        root_path: 根目录路径
        data_file: 原始数据文件名
        target_column: 目标列名
        
    Returns:
        str: 处理后的文件路径
    """
    
    data_path = os.path.join(root_path, data_file)
    
    # 读取原始数据
    try:
        df = pd.read_csv(data_path)
        print(f"原始数据形状: {df.shape}")
        print(f"原始列名: {list(df.columns)}")
    except Exception as e:
        print(f"读取数据文件失败: {e}")
        raise e
    
    # 检查是否有日期列
    date_columns = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    if not date_columns:
        # 如果没有日期列，创建一个假的日期列
        print("未发现日期列，创建虚拟日期索引...")
        df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
        date_column = 'date'
    else:
        date_column = date_columns[0]
        print(f"使用日期列: {date_column}")
        # 确保日期列名为'date'
        if date_column != 'date':
            df = df.rename(columns={date_column: 'date'})
            date_column = 'date'
    
    # 确保日期列是datetime类型
    try:
        df['date'] = pd.to_datetime(df['date'])
    except Exception as e:
        print(f"日期转换失败: {e}")
        # 创建默认日期序列
        df['date'] = pd.date_range(start='2020-01-01', periods=len(df), freq='H')
    
    # 处理数值列
    numeric_columns = []
    for col in df.columns:
        if col != 'date':
            # 尝试转换为数值类型
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                numeric_columns.append(col)
            except:
                print(f"列 {col} 无法转换为数值，将被跳过")
    
    # 保留日期列和数值列
    df = df[['date'] + numeric_columns]
    
    # 处理缺失值
    print("处理缺失值...")
    nan_counts = df.isna().sum()
    total_nan = nan_counts.sum()
    
    if total_nan > 0:
        print(f"发现 {total_nan} 个NaN值")
        
        for col in numeric_columns:
            col_nan_count = df[col].isna().sum()
            if col_nan_count == 0:
                continue
                
            col_nan_percentage = (col_nan_count / len(df)) * 100
            
            if col_nan_percentage > 50:
                print(f"  {col}: 缺失率{col_nan_percentage:.1f}% > 50%，用0填充")
                df[col] = df[col].fillna(0)
            elif col_nan_percentage > 10:
                print(f"  {col}: 缺失率{col_nan_percentage:.1f}%，使用线性插值")
                df[col] = df[col].interpolate(method='linear').fillna(method='bfill').fillna(method='ffill')
            else:
                print(f"  {col}: 缺失率{col_nan_percentage:.1f}%，使用前向/后向填充")
                df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        
        # 最终检查
        remaining_nan = df.isna().sum().sum()
        if remaining_nan > 0:
            print(f"⚠ 仍有 {remaining_nan} 个NaN值，全部填充为0")
            df = df.fillna(0)
    
    # 处理目标列检测
    print(f"可用的数值列: {numeric_columns}")
    
    # 如果指定了目标列但不存在，尝试智能匹配
    if target_column is not None and target_column not in numeric_columns:
        print(f"指定的目标列 '{target_column}' 不存在，尝试智能匹配...")
        
        # 尝试模糊匹配
        possible_matches = []
        target_lower = target_column.lower()
        
        for col in numeric_columns:
            col_lower = col.lower()
            # 检查是否包含price, target等关键词
            if any(keyword in col_lower for keyword in ['price', 'target', 'label', 'y']):
                possible_matches.append(col)
            # 检查部分匹配
            elif any(part in col_lower for part in target_lower.split()):
                possible_matches.append(col)
        
        if possible_matches:
            target_column = possible_matches[0]
            print(f"找到可能的目标列: {target_column}")
        else:
            print(f"无法找到匹配的目标列，使用最后一列: {numeric_columns[-1]}")
            target_column = numeric_columns[-1]
    
    # 确定目标列
    if target_column is None:
        target_column = numeric_columns[-1]  # 默认最后一列为目标
        print(f"使用默认目标列: {target_column}")
    
    if target_column not in numeric_columns:
        raise ValueError(f"目标列 '{target_column}' 不在数值列中: {numeric_columns}")
    
    # 重新排列列顺序: date, 其他特征列, 目标列
    other_cols = [col for col in numeric_columns if col != target_column]
    df = df[['date'] + other_cols + [target_column]]
    
    # 保存处理后的文件
    processed_file = os.path.join(root_path, f"processed_{data_file}")
    df.to_csv(processed_file, index=False)
    
    print(f"数据预处理完成:")
    print(f"  处理后形状: {df.shape}")
    print(f"  目标列: {target_column}")
    print(f"  保存到: {processed_file}")
    
    return processed_file

def apply_01_scaling(train_dataset, val_dataset, test_dataset, use_graph_interpolation=False, root_path=None):
    """
    对数据集应用0-1缩放
    
    Args:
        train_dataset, val_dataset, test_dataset: 数据集实例
        use_graph_interpolation: 是否使用图基插值
        root_path: 根路径（用于加载邻接矩阵）
        
    Returns:
        tuple: 缩放后的数据集
    """
    
    print("应用0-1缩放...")
    
    # 获取训练数据用于拟合MinMaxScaler
    train_data = train_dataset.data_x
    
    # 如果使用图基插值处理缺失值
    if use_graph_interpolation and root_path:
        adj_path = os.path.join(root_path, 'adj_mx.npy')
        if os.path.exists(adj_path):
            print("使用图基插值处理缺失值...")
            adj = np.load(adj_path)
            train_data = graph_based_interpolation(train_data, adj)
    
    # 创建MinMaxScaler
    minmax_scaler = MinMaxScaler(feature_range=(0, 1))
    
    # 拟合训练数据
    minmax_scaler.fit(train_data)
    
    # 应用缩放到所有数据集
    train_dataset.data_x = minmax_scaler.transform(train_dataset.data_x)
    train_dataset.data_y = minmax_scaler.transform(train_dataset.data_y)
    
    val_dataset.data_x = minmax_scaler.transform(val_dataset.data_x)
    val_dataset.data_y = minmax_scaler.transform(val_dataset.data_y)
    
    test_dataset.data_x = minmax_scaler.transform(test_dataset.data_x)
    test_dataset.data_y = minmax_scaler.transform(test_dataset.data_y)
    
    # 保存MinMaxScaler以便后续反变换
    train_dataset.minmax_scaler = minmax_scaler
    val_dataset.minmax_scaler = minmax_scaler
    test_dataset.minmax_scaler = minmax_scaler
    
    # 检查数据范围
    train_min, train_max = train_dataset.data_x.min(), train_dataset.data_x.max()
    print(f"缩放后训练数据范围: [{train_min:.6f}, {train_max:.6f}]")
    
    return train_dataset, val_dataset, test_dataset

def load_and_clean_dataset(data_file):
    """
    向后兼容函数：加载并清理数据集，返回缩放到0-1范围的numpy数组
    
    Args:
        data_file: 数据文件名或路径
        
    Returns:
        numpy.ndarray: 缩放到0-1范围的数据
    """
    print(f"使用兼容模式加载数据集: {data_file}")
    
    try:
        # 智能路径处理
        if os.path.isabs(data_file):
            # 绝对路径，直接使用
            file_path = data_file
        elif data_file.startswith('./dataset/'):
            # 已经包含 ./dataset/ 前缀
            file_path = data_file
        elif data_file.startswith('dataset/'):
            # 包含 dataset/ 前缀，添加 ./
            file_path = './' + data_file
        else:
            # 只是文件名，添加完整路径
            file_path = os.path.join('./dataset', data_file)
        
        # 检查文件是否存在
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件不存在: {file_path}")
        
        # 读取原始数据
        df = pd.read_csv(file_path)
        
        # 移除非数值列
        numeric_columns = []
        for col in df.columns:
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_columns.append(col)
            except:
                continue
        
        if not numeric_columns:
            raise ValueError("没有找到数值列")
        
        # 提取数值数据
        data = df[numeric_columns].values
        
        # 应用MinMax缩放到0-1范围
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        print(f"兼容模式加载完成:")
        print(f"  数据形状: {scaled_data.shape}")
        print(f"  数据范围: [{scaled_data.min():.6f}, {scaled_data.max():.6f}]")
        print(f"  使用的数值列数: {len(numeric_columns)}")
        
        return scaled_data
        
    except Exception as e:
        print(f"兼容模式处理失败: {e}")
        raise e

def get_dataset_info():
    """
    返回所有可用数据集的信息
    """
    dataset_info = {
        'ECG_data': {
            'description': 'ECG时间序列数据',
            'format': 'pure_numeric',
            'features': 140,
            'recommended_params': {
                'seq_len': 96,
                'pred_len': 24,
                'features': 'M'
            }
        },
        'ECG_data_0': {
            'description': 'ECG时间序列数据（版本0）',
            'format': 'pure_numeric', 
            'features': 140,
            'recommended_params': {
                'seq_len': 96,
                'pred_len': 24,
                'features': 'M'
            }
        },
        'PeMS07': {
            'description': 'PeMS交通流量数据',
            'format': 'pure_numeric',
            'features': 'variable',
            'recommended_params': {
                'seq_len': 96,
                'pred_len': 12,
                'features': 'M',
                'freq': '5min'
            }
        },
        'France_processed_0': {
            'description': '法国电力消费数据',
            'format': 'pure_numeric',
            'features': 10,
            'recommended_params': {
                'seq_len': 96,
                'pred_len': 24,
                'features': 'M'
            }
        },
        'V_France_processed_0': {
            'description': '法国电力消费数据（处理版本）',
            'format': 'pure_numeric',
            'features': 10,
            'recommended_params': {
                'seq_len': 96,
                'pred_len': 24,
                'features': 'M'
            }
        },
        'Germany_processed_0': {
            'description': '德国电力生产数据（含日期列）',
            'format': 'with_headers_and_dates',
            'features': 16,
            'note': '包含日期列和列名，需要预处理',
            'recommended_params': {
                'seq_len': 96,
                'pred_len': 24,
                'features': 'M',
                'target_column': 'Day-ahead Price [EUR/MWh]'
            }
        },
        'V_Germany_processed_0': {
            'description': '德国电力生产数据（处理版本）',
            'format': 'pure_numeric',
            'features': 16,
            'recommended_params': {
                'seq_len': 96,
                'pred_len': 24,
                'features': 'M'
            }
        }
    }
    return dataset_info

def print_dataset_info():
    """
    打印所有数据集信息
    """
    info = get_dataset_info()
    print("="*80)
    print("可用数据集信息:")
    print("="*80)
    
    for name, details in info.items():
        print(f"\n数据集: {name}")
        print(f"  描述: {details['description']}")
        print(f"  格式: {details['format']}")
        print(f"  特征数: {details['features']}")
        if 'note' in details:
            print(f"  注意: {details['note']}")
        if 'recommended_params' in details:
            print("  推荐参数:")
            for key, value in details['recommended_params'].items():
                print(f"    {key}: {value}")
    
    print("\n" + "="*80)

def create_simple_dataloader(root_path, data_file, **kwargs):
    """
    简化的数据加载器创建函数
    
    Args:
        root_path: 根目录路径
        data_file: 数据文件名
        **kwargs: 其他参数
        
    Returns:
        dict: 数据加载器信息
    """
    
    # 设置默认参数
    default_params = {
        'seq_len': 96,
        'label_len': 48,
        'pred_len': 24,
        'features': 'M',
        'batch_size': 32,
        'scale_to_01': True,
        'freq': 'h',
        'timeenc': 0
    }
    
    # 更新参数
    params = {**default_params, **kwargs}
    
    # 获取数据集推荐参数
    dataset_info = get_dataset_info()
    dataset_name = os.path.basename(data_file).replace('.csv', '')
    
    if dataset_name in dataset_info and 'recommended_params' in dataset_info[dataset_name]:
        recommended = dataset_info[dataset_name]['recommended_params']
        for key, value in recommended.items():
            if key not in kwargs:  # 只使用未被用户指定的推荐参数
                params[key] = value
    
    print(f"使用参数: {params}")
    
    return load_and_process_dataset(root_path, data_file, **params) 