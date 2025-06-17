import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from data_provider.data_loader import Dataset_Custom
from utils.timefeatures import time_features


class Dataset_FourierGNN(Dataset_Custom):
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='Merged_Data_germany.csv',
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

        self.train_data = data[border1:border2]
        self.valid_data = data[border2:border2+num_vali]
        self.test_data = data[border2+ num_vali:border2 + num_vali + num_test]

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_begin = end
        next_end = next_begin + self.pred_len
        if self.flag == 'train':
            data = self.train_data[begin:end]
            next_data = self.train_data[next_begin:next_end]
        elif self.flag == 'val':
            data = self.valid_data[begin:end]
            next_data = self.valid_data[next_begin:next_end]
        else:
            data = self.test_data[begin:end]
            next_data = self.test_data[next_begin:next_end]
        return data, next_data

    def __len__(self):
        # minus the label length
        if self.flag == 'train':
            return len(self.train_data) - self.seq_len - self.pred_len
        elif self.flag == 'val':
            return len(self.valid_data) - self.seq_len - self.pred_len
        else:
            return len(self.test_data) - self.seq_len - self.pred_len

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


if __name__ == "__main__":
    dataset = Dataset_FourierGNN(root_path='datasets', flag='train', size=None, features='M', data_path='Merged_Data_germany.csv',
                              target='Day-ahead Price [EUR/MWh]', scale=True, timeenc=0, freq='month', seasonal_patterns=None)
    print(len(dataset))