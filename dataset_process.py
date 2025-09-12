import pandas as pd

# 读取CSV文件
df = pd.read_csv('../dataset/buffer/buffer_detect.csv')

# 将is_vul列中的True和False映射为1和0
df['label'] = df['is_vul'].map({True: 1, False: 0})

# 保存修改后的DataFrame到新的CSV文件
df.to_csv('../dataset/buffer/buffer_detect_modified.csv', index=False)