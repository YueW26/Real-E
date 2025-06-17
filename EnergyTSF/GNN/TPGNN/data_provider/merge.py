import pandas as pd

# 加载CSV文件
day_ahead_prices = pd.read_csv('/Users/wangbo/Documents/Time-Series-Library-git/Time-Series-Library-1/datasets/Day-ahead Prices_202301010000-202401010000.csv')
actual_generation = pd.read_csv('/Users/wangbo/Documents/Time-Series-Library-git/Time-Series-Library-1/datasets/Actual Generation per Production Type_202301010000-202401010000.csv')

# 重命名日期列以便合并
day_ahead_prices.rename(columns={'MTU (CET/CEST)': 'date'}, inplace=True)
actual_generation.rename(columns={'MTU': 'date'}, inplace=True)

# 删除时间信息以进行正确合并
day_ahead_prices['date'] = day_ahead_prices['date'].str.replace(r" \(CET/CEST\)", "", regex=True)
actual_generation['date'] = actual_generation['date'].str.replace(r" \(CET/CEST\)", "", regex=True)

# 处理日期格式并删除不符合格式的行
def parse_dates(df, column_name):
    df[column_name] = df[column_name].apply(lambda x: x.split(' - ')[0].strip())
    try:
        df[column_name] = pd.to_datetime(df[column_name], format='%d.%m.%Y %H:%M')
    except ValueError as e:
        print(f"Error parsing dates: {e}")
        df = df[~df[column_name].str.contains(' - ')]

    return df

day_ahead_prices = parse_dates(day_ahead_prices, 'date')
actual_generation = parse_dates(actual_generation, 'date')

# 合并数据
merged_data = pd.merge(day_ahead_prices, actual_generation, on='date', how='inner')

# 显示合并后的数据
print(merged_data.head())

# 保存合并后的数据到新CSV文件
merged_data.to_csv('/Users/wangbo/Documents/Time-Series-Library-git/Time-Series-Library-1/datasets/Merged_Data.csv', index=False)

'''
import pandas as pd

# 加载CSV文件
day_ahead_prices = pd.read_csv('/Users/wangbo/Documents/Time-Series-Library-git/Time-Series-Library-1/datasets/Day-ahead Prices_202301010000-202401010000.csv')
actual_generation = pd.read_csv('/Users/wangbo/Documents/Time-Series-Library-git/Time-Series-Library-1/datasets/Actual Generation per Production Type_202301010000-202401010000.csv')

# 重命名日期列以便合并
day_ahead_prices.rename(columns={'MTU (CET/CEST)': 'Date'}, inplace=True)
actual_generation.rename(columns={'MTU': 'Date'}, inplace=True)

# 删除时间信息以进行正确合并
day_ahead_prices['Date'] = day_ahead_prices['Date'].str.replace(r" \(CET/CEST\)", "", regex=True)
actual_generation['Date'] = actual_generation['Date'].str.replace(r" \(CET/CEST\)", "", regex=True)

# 合并数据
merged_data = pd.merge(day_ahead_prices, actual_generation, on='Date', how='inner')

# 显示合并后的数据
print(merged_data.head())

# 保存合并后的数据到新CSV文件
merged_data.to_csv('/Users/wangbo/Documents/Time-Series-Library-git/Time-Series-Library-1/datasets/Merged_Data.csv', index=False)
'''
