import math

import pandas as pd
import numpy as np
import torch

'''

'''
path = '../datasets/playground-series-s3e12/'

train_dataset = pd.read_csv(path + 'train.csv')
test_dataset = pd.read_csv(path + 'test.csv')
col_num = len(train_dataset.keys())
# 去掉id列
train_dataset = train_dataset.iloc[:, 1:col_num]
test_dataset = test_dataset.iloc[:, 1:col_num - 1]
# 增加新特征


# 丢弃
# train_dataset = train_dataset.drop(['coffee_bar', 'salad_bar', 'prepared_food'], axis=1)
# test_dataset = test_dataset.drop(['coffee_bar', 'salad_bar', 'prepared_food'], axis=1)

# 填充缺失
train_dataset = train_dataset.fillna(train_dataset.mean())
test_dataset = test_dataset.fillna(train_dataset.mean())
# 离散信息转成one-hot
train_dataset = pd.get_dummies(train_dataset, dummy_na=True)
test_dataset = pd.get_dummies(test_dataset, dummy_na=True)
# 转成张量
train_dataset, test_dataset = np.array(train_dataset.values), np.array(test_dataset.values)
num = int(len(train_dataset) * 0.8)
val_dataset = train_dataset[num:]
train_dataset = torch.tensor(train_dataset, dtype=torch.float)
val_dataset = torch.tensor(val_dataset, dtype=torch.float)
test_dataset = torch.tensor(test_dataset, dtype=torch.float)
print('d=', test_dataset.shape[1])
torch.save(train_dataset, path + 'train.pt')
torch.save(train_dataset, path + 'val.pt')
torch.save(test_dataset, path + 'test.pt')
# d=20
