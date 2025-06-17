import os
import pandas as pd
from sklearn.preprocessing import StandardScaler

class Dataset_FourierGNN:
    def __init__(self, root_path, flag='train', size=None,
                 features='M', data_path='Merged_Data_germany.csv',
                 target='Day-ahead Price [EUR/MWh]', scale=True, timeenc=0, freq='month', seasonal_patterns=None):
        self.root_path = root_path
        self.flag = flag
        self.size = size
        self.features = features
        self.data_path = data_path
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.seasonal_patterns = seasonal_patterns
        
        # These are the additional params required for the length and sequence
        self.seq_len = 24  # Example sequence length
        self.pred_len = 24  # Example prediction length
        
        # Call the data reading function to initialize data
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path), parse_dates=['date'])

        # Debug: check for any NaN values
        if df_raw.isnull().sum().sum() > 0:
            print("Warning: Data contains NaN values. Consider preprocessing to handle NaNs.")
        
        # Drop the date column for simplicity in feature handling
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]

        # Debug: Check the shape and initial data
        print("Data shape:", df_raw.shape)
        print(df_raw.head())

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

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
        border1 = border1s[0]  # Change this based on 'flag'
        border2 = border2s[0]  # Change this based on 'flag'

        if border1 < 0:
            border1 = 0
        if border2 > len(df_raw):
            border2 = len(df_raw)

        print(f"Data split borders: border1={border1}, border2={border2}")

        # Select features based on input flags
        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        # Scaling
        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)

            # Check for NaN values after scaling
            if pd.isnull(data).sum().sum() > 0:
                print("Warning: Scaled data contains NaN values.")
                data = pd.DataFrame(data).fillna(method='ffill').fillna(method='bfill').values
        else:
            data = df_data.values

        # Assign train, validation, and test data
        self.train_data = data[border1:border2]
        self.valid_data = data[border2:border2+num_vali]
        self.test_data = data[border2 + num_vali:border2 + num_vali + num_test]

        # Debug: Check the sizes of each dataset
        print(f"Train data length: {len(self.train_data)}")
        print(f"Validation data length: {len(self.valid_data)}")
        print(f"Test data length: {len(self.test_data)}")

    def __getitem__(self, index):
        begin = index
        end = index + self.seq_len
        next_begin = end
        next_end = next_begin + self.pred_len

        print(f"Getting item with index: {index}, begin: {begin}, end: {end}, next_begin: {next_begin}, next_end: {next_end}")
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
        print(f"Calculating length for {self.flag} data...")
        if self.flag == 'train':
            length = len(self.train_data) - self.seq_len - self.pred_len
        elif self.flag == 'val':
            length = len(self.valid_data) - self.seq_len - self.pred_len
        else:
            length = len(self.test_data) - self.seq_len - self.pred_len
        print(f"Calculated length: {length}")
        return length

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# Testing the dataset class
if __name__ == "__main__":
    # You can adjust root_path and data_path according to your environment
    dataset = Dataset_FourierGNN(root_path="datasets", flag='train', data_path='Merged_Data_germany.csv')

    # Test dataset length
    print("Dataset length (train):", len(dataset))

    # Test fetching a sample
    sample_data, target_data = dataset[0]
    print("Sample data shape:", sample_data.shape)
    print("Target data shape:", target_data.shape)