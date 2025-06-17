############# long short
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_provider.data_loader import Dataset_Custom
from utils.timefeatures import time_features

class Dataset_Opennem(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='Germany_processed_0.csv',
                 target='Fossil Gas  - Actual Aggregated [MW]', scale=True, timeenc=0, freq='month', seasonal_patterns=None):
        super().__init__(root_path, flag, size, features, data_path, target, scale, timeenc, freq, seasonal_patterns)

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), parse_dates=['date'])

        # Ensure proper column order
        #cols = list(df_raw.columns)
        #cols.remove(self.target)
        #cols.remove('date')
        #df_raw = df_raw[['date'] + cols + [self.target]]

        # 确保列名存在并按正确顺序排列
        cols = list(df_raw.columns)
        #print(f"Initial columns in the dataset: {cols}")  # 调试信息

        # 检查目标列是否存在
        if self.target not in cols:
            raise ValueError(f"Target column '{self.target}' not found in the dataset columns: {cols}")

        # 检查 'date' 列是否存在
        if 'date' not in cols:
            raise ValueError("'date' column not found in the dataset columns.")

        # 移除目标列和日期列后重新排列
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # 调试信息：检查调整后的列顺序
        #print(f"Reordered columns: {df_raw.columns.tolist()}")





        # Handle NaN values
        if df_raw.isnull().sum().sum() > 0:
            print("Warning: Data contains NaN values. Filling with forward and backward fill.")
            df_raw = df_raw.fillna(method='ffill').fillna(method='bfill')

        # Split the dataset
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        if num_train <= self.seq_len + self.pred_len:
            raise ValueError("Training data too small. Adjust `seq_len` and `pred_len`.")
        if num_vali <= self.seq_len + self.pred_len:
            num_vali = self.seq_len + self.pred_len
            num_train = len(df_raw) - num_vali - num_test
            print(f"Adjusted validation set size: {num_vali}")
        if num_test <= self.seq_len + self.pred_len:
            num_test = self.seq_len + self.pred_len
            num_train = len(df_raw) - num_vali - num_test
            print(f"Adjusted test set size: {num_test}")

        border1s = [0, num_train, num_train + num_vali]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1, border2 = border1s[self.set_type], border2s[self.set_type]

        # Extract features and target
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
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            df_stamp['weekday'] = df_stamp['date'].dt.weekday
            df_stamp['hour'] = df_stamp['date'].dt.hour
            print("Original hour values:")
            print(df_stamp['hour'].head(20))
            df_stamp['minute'] = df_stamp['date'].dt.minute
            print("df_stamp columns after adding minute:", df_stamp.columns)
            # Debug：检查是否有缺失的时间特征
            print("df_stamp columns before drop:", df_stamp.columns)
            print("Before normalization:")
            print(df_stamp[['month', 'day', 'weekday', 'hour', 'minute']].head())
            data_stamp = df_stamp.drop(['date'], axis=1).values
            print("After normalization:")
            print("data_stamp shape:", data_stamp.shape)
            print("data_stamp example:\n", data_stamp[:5])
            # Debug：检查生成的 data_stamp 形状和内容
            #print("data_stamp shape:", data_stamp.shape)
            #print("data_stamp example:\n", data_stamp[:5])
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

        # Debug: Print dataset details
        print(f"Dataset split: {self.set_type}, border1: {border1}, border2: {border2}")
        print(f"data_x shape: {self.data_x.shape}, data_y shape: {self.data_y.shape}, data_stamp shape: {self.data_stamp.shape}")

        print("data_stamp shape:", data_stamp.shape)
        print("data_stamp example:\n", data_stamp[:5])
        print("df_stamp columns:", df_stamp.columns)
        print("df_stamp head:\n", df_stamp.head())


        # 检查长度是否一致
        assert len(self.data_x) == len(self.data_stamp), "data_x 和 data_stamp 长度不一致！"
        assert len(self.data_y) == len(self.data_stamp), "data_y 和 data_stamp 长度不一致！"



    # def __getitem__(self, index):
    #     s_begin = index
    #     s_end = s_begin + self.seq_len
    #     r_begin = s_end - self.label_len
    #     r_end = r_begin + self.label_len + self.pred_len

    #     seq_x = self.data_x[s_begin:s_end]
    #     seq_y = self.data_y[r_begin:r_end]
    #     seq_x_mark = self.data_stamp[s_begin:s_end]
    #     seq_y_mark = self.data_stamp[r_begin:r_end]

    #     return seq_x, seq_y, seq_x_mark, seq_y_mark

  

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        # Add placeholders for x_dec and x_mark_dec if needed
        seq_x_dec = seq_y[:self.label_len]
        seq_x_mark_dec = seq_y_mark[:self.label_len]

        #return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_x_dec, seq_x_mark_dec
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        length = len(self.data_x) - self.seq_len - self.pred_len + 1
        return max(length, 0)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



