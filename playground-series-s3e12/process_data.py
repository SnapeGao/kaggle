import math

import pandas as pd
import numpy as np
import torch

'''

gravity,比重，尿液相对于水的密度；
ph,pH，氢离子的负对数；
osmo,渗透压(mOsm)，生物学和医学中使用但物理化学中不使用的单位,渗透压与溶液中分子的浓度成正比
cond,电导率(mMho milliMho)。1 Mho 是 1欧姆的倒数。电导率与溶液中带电离子的浓度成正比
urea,尿素浓度，单位为毫摩尔每升；
calc,钙浓度 (CALC)，单位为毫摩尔升。
    
'''
def process(path, train=False):
    dataset = pd.read_csv(path)
    col_num = len(dataset.keys())
    x, y = dataset.iloc[:, 1:col_num - 1], dataset.iloc[:, col_num - 1]

    if (not train):
        x, y = dataset.iloc[:, 1:col_num], dataset.iloc[:, col_num - 1]
    # 填充缺失
    x = x.fillna(x.mean())
    # 离散信息转成one-hot
    x = pd.get_dummies(x, dummy_na=True)
    # 转成张量
    x, y = np.array(x.values), np.array(y.values)
    num_features = x.shape[1]
    x = torch.tensor(x, dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)
    BN = torch.nn.BatchNorm1d(num_features)
    x = BN(x)

    if (train):
        p = 0.8
        train_nums = int(x.shape[0] * p)
        train_x, train_y, val_x, val_y = x[:train_nums], y[:train_nums], x[train_nums:], y[train_nums:]
        return train_x, train_y, val_x, val_y
    else:
        return x, y


path = '../datasets/playground-series-s3e12/'

train_x, train_y, val_x, val_y = process(path + 'train.csv', train=True)
y_ = torch.mean(train_y)
feature_num = train_x.shape[1]
nums = train_x.shape[0]
x = train_x.reshape(feature_num, -1)
y = train_y.reshape(1, -1)
ps = []
for i in range(feature_num):
    x_ = torch.mean(x[i])
    conxy = 0
    x_s = 0
    y_s = 0
    for j in range(nums):
        conxy += (x[i][j] - x_) * (y[0][j] - y_)
        x_s += (x[i][j] - x_) ** 2
        y_s += (y[0][j] - y_) ** 2
    x_s = math.sqrt(x_s)
    y_s = math.sqrt(y_s)
    p = conxy / (x_s * y_s)
    ps.append(p.item())
for i in range(len(ps)):
    print(i,'->',ps[i])
print(ps)
'''
0 -> -0.07181735336780548
1 -> 0.02776183746755123
2 -> 0.006226686295121908
3 -> -0.0685705617070198
4 -> 0.024791644886136055
5 -> 0.005612515844404697
'''
'''

'''