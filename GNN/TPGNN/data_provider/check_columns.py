import pandas as pd

# 加载CSV文件
df = pd.read_csv('/Users/wangbo/Documents/Time-Series-Library-git/Time-Series-Library-1/datasets/Merged_Data.csv')

# 打印列名
print(df.columns)