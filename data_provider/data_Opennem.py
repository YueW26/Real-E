import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_provider.data_loader import Dataset_Custom
from utils.timefeatures import time_features

def graph_based_interpolation(data, adj, max_iter=5):
    data = data.copy()
    T, N = data.shape

    adj = adj.astype(float)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    adj_norm = adj / row_sums

    for _ in range(max_iter):
        nan_mask = np.isnan(data)
        if not nan_mask.any():
            break

        for t in range(T):
            for n in range(N):
                if nan_mask[t, n]:
                    neighbors = adj_norm[n]
                    neighbor_values = data[t]
                    known_mask = ~np.isnan(neighbor_values)
                    if known_mask.sum() > 0:
                        weighted_vals = neighbors[known_mask] * neighbor_values[known_mask]
                        data[t, n] = weighted_vals.sum() / neighbors[known_mask].sum()

    for n in range(N):
        col = data[:, n]
        if np.isnan(col).any():
            idx = np.arange(T)
            known = ~np.isnan(col)
            data[:, n] = np.interp(idx, idx[known], col[known])

    return data

class Dataset_Opennem(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='Merged_cleaned_germany.csv',
                 target='Day-ahead Price [EUR/MWh]', scale=True, timeenc=0, freq='month', seasonal_patterns=None):
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), parse_dates=['date'])

        cols = list(df_raw.columns)
        cols.remove(self.target)
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
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday, 1)
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp['date'].apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        length = len(self.data_x) - self.seq_len - self.pred_len + 1
        if length < 0:
            return 0
        return length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



'''
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_provider.data_loader import Dataset_Custom
from utils.timefeatures import time_features

class Dataset_Opennem(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='Merged_cleaned_germany.csv',
                 target='Day-ahead Price [EUR/MWh]', scale=True, timeenc=0, freq='month', seasonal_patterns=None):
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), parse_dates=['date'])

        # df_raw.columns: ['date', ...(other features), target feature]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        # Debug: check for any NaN values
        if df_raw.isnull().sum().sum() > 0:
            print("Warning: Data contains NaN values. Consider preprocessing to handle NaNs.")
    
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        # Make sure each part has enough data length for training, validation, and testing
        
        if num_train <= self.seq_len + self.pred_len:
            raise ValueError("The training set data is too small to meet the requirements of seq_len and pred_len.")
        if num_vali <= self.seq_len + self.pred_len:
            num_vali = self.seq_len + self.pred_len  # Adjust validation set size
            num_train = len(df_raw) - num_vali - num_test  # Adjust the training set size
            print("The adjusted validation set size is:", num_vali)
        if num_test <= self.seq_len + self.pred_len:
            num_test = self.seq_len + self.pred_len  # Adjusting the test set size
            num_train = len(df_raw) - num_vali - num_test  # Adjust the training set size
            print("The adjusted test set size is:", num_test)


        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if border1 < 0:
            border1 = 0
        if border2 > len(df_raw):
            border2 = len(df_raw)

        print(f"Dataset split: {self.set_type}, border1: {border1}, border2: {border2}, num_train: {num_train}, num_vali: {num_vali}, num_test: {num_test}")

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)

            # 
            data_df = pd.DataFrame(data)
            data_df = data_df.interpolate(method='linear', axis=0)
            data_df = data_df.fillna(method='ffill').fillna(method='bfill')
            data = data_df.values

        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday, 1)
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp['date'].apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # Add debug information
        print(f"data_x shape: {self.data_x.shape}")
        print(f"data_y shape: {self.data_y.shape}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # When calculating length, make sure the bounds are correct
        length = len(self.data_x) - self.seq_len - self.pred_len + 1
        print(f"Dataset length calculation: len(data_x)={len(self.data_x)}, seq_len={self.seq_len}, pred_len={self.pred_len}, calculated length={length}")
        if length < 0:
            print(f"Error: Dataset length is negative. Adjusting to 0.")
            return 0
        return length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
        
        
        
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################     
        
        
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_provider.data_loader import Dataset_Custom
from utils.timefeatures import time_features

class Dataset_Opennem(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='Merged_cleaned_germany.csv',
                 target='Day-ahead Price [EUR/MWh]', scale=True, timeenc=0, freq='month', seasonal_patterns=None):
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), parse_dates=['date'])


        # df_raw.columns: ['date', ...(other features), target feature]

        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        # Debug: check for any NaN values
        if df_raw.isnull().sum().sum() > 0:
            print("Warning: Data contains NaN values. Consider preprocessing to handle NaNs.")
    
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        # Make sure each part has enough data length for training, validation, and testing
        
        if num_train <= self.seq_len + self.pred_len:
            raise ValueError("The training set data is too small to meet the requirements of seq_len and pred_len.")
        if num_vali <= self.seq_len + self.pred_len:
            num_vali = self.seq_len + self.pred_len  # Adjust validation set size
            num_train = len(df_raw) - num_vali - num_test  # Adjust the training set size
            print("The adjusted validation set size is:", num_vali)
        if num_test <= self.seq_len + self.pred_len:
            num_test = self.seq_len + self.pred_len  # Adjusting the test set size
            num_train = len(df_raw) - num_vali - num_test  # Adjust the training set size
            print("The adjusted test set size is:", num_test)


        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if border1 < 0:
            border1 = 0
        if border2 > len(df_raw):
            border2 = len(df_raw)

        print(f"Dataset split: {self.set_type}, border1: {border1}, border2: {border2}, num_train: {num_train}, num_vali: {num_vali}, num_test: {num_test}")

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
            
        # Debug: check for NaN values after scaling
            if pd.isnull(data).sum().sum() > 0:
                print("Warning: Scaled data contains NaN values.")
                data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values

        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday, 1)
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp['date'].apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # Add debug information
        print(f"data_x shape: {self.data_x.shape}")
        print(f"data_y shape: {self.data_y.shape}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # When calculating length, make sure the bounds are correct
        length = len(self.data_x) - self.seq_len - self.pred_len + 1
        print(f"Dataset length calculation: len(data_x)={len(self.data_x)}, seq_len={self.seq_len}, pred_len={self.pred_len}, calculated length={length}")
        if length < 0:
            print(f"Error: Dataset length is negative. Adjusting to 0.")
            return 0
        return length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

##################################################################
##################################################################
##################################################################
##################################################################
##################################################################
##################################################################

class Dataset_Opennem(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='20240517_OpenNEM.csv',
                 target='OT', scale=True, timeenc=0, freq='5min', seasonal_patterns=None):
        
        # Default size values
        default_seq_len = 24
        default_label_len = 12
        default_pred_len = 24
        
        # Read raw data to determine appropriate size
        df_raw = pd.read_csv(os.path.join(root_path, data_path))
        total_length = len(df_raw)
        
        if size is None:
            self.seq_len = default_seq_len
            self.label_len = default_label_len
            self.pred_len = default_pred_len
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        
        # Check if training, validation, and test sizes are adequate
        num_train = int(total_length * 0.7)
        num_test = int(total_length * 0.2)
        num_vali = total_length - num_train - num_test

        if num_vali <= self.seq_len + self.pred_len:
            self.label_len = min(self.label_len, num_vali - self.seq_len - self.pred_len)
            print("Adjusted validation label length to:", self.label_len)
        
        if num_test <= self.seq_len + self.pred_len:
            self.label_len = min(self.label_len, num_test - self.seq_len - self.pred_len)
            print("Adjusted test label length to:", self.label_len)
        
        # Initialize parent class with possibly adjusted values
        size = [self.seq_len, self.label_len, self.pred_len]
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        
        # df_raw.columns: ['date', ...(other features), target feature]
        
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        # Make sure each part has enough data length for training, validation, and testing
        if num_train <= self.seq_len + self.pred_len:
            raise ValueError("The training set data is too small to meet the requirements of seq_len and pred_len.")
        if num_vali <= self.seq_len + self.pred_len:
            num_vali = self.seq_len + self.pred_len  # Adjust validation set size
            num_train = len(df_raw) - num_vali - num_test  # Adjust the training set size
            print("The adjusted validation set size is:", num_vali)
        if num_test <= self.seq_len + self.pred_len:
            num_test = self.seq_len + self.pred_len  # Adjusting the test set size
            num_train = len(df_raw) - num_vali - num_test  # Adjust the training set size
            print("The adjusted test set size is:", num_test)

        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if border1 < 0:
            border1 = 0
        if border2 > len(df_raw):
            border2 = len(df_raw)

        print(f"Dataset split: {self.set_type}, border1: {border1}, border2: {border2}, num_train: {num_train}, num_vali: {num_vali}, num_test: {num_test}")

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp['date'])
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp['date'].apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp['date'].apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp['date'].apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp['date'].apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp['date'].apply(lambda row: row.minute, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # Add debug information
        print(f"data_x shape: {self.data_x.shape}")
        print(f"data_y shape: {self.data_y.shape}")

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        # When calculating length, make sure the bounds are correct
        length = len(self.data_x) - self.seq_len - self.pred_len + 1
        print(f"Dataset length calculation: len(data_x)={len(self.data_x)}, seq_len={self.seq_len}, pred_len={self.pred_len}, calculated length={length}")
        if length < 0:
            print(f"Error: Dataset length is negative. Adjusting to 0.")
            return 0
        return length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
'''