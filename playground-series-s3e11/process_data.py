import math

import pandas as pd
import numpy as np
import torch


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
        x, y = x[:5000], y[:5000]
        p = 0.8
        train_nums = int(x.shape[0] * p)
        train_x, train_y, val_x, val_y = x[:train_nums], y[:train_nums], x[train_nums:], y[train_nums:]
        return train_x, train_y, val_x, val_y
    else:
        return x, y


path = '../datasets/playground-series-s3e11/'

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
0 -> -0.045626118779182434
1 -> -0.021471159532666206
2 -> -0.011157243512570858
3 -> -0.016487937420606613
4 -> -0.025638721883296967
5 -> 0.005794562865048647
6 -> -0.016741015017032623
7 -> -0.03606819361448288
8 -> 0.015626659616827965
9 -> -0.017984820529818535
10 -> -0.0002995639806613326
11 -> -0.02345416694879532
12 -> 0.002680982928723097
13 -> 0.005211129318922758
14 -> 0.029328785836696625
去掉 5,10,12,13,
'''
'''
比率：
每单位收入：store_sales / unit_sales
每平方英尺销售额：store_sales / Store_sqft
每平方英尺单位数：unit_sales / Store_sqft
儿童比例：Total_children / Num_children_at_home 每
单位重量：Gross_weight / unit_sales
多项式：
为 store_sales、unit_sales、Total_children、avg_cars_at_home、Num_children_at_home、Gross_weight、Units_per_case 和 Store_sqft 等数值列创建多项式特征，以捕获潜在的非线性。
交互：
乘以二元特征 (Coffee_bar * Video_store) 以捕获商店设施之间的潜在交互。
组合数值特征（store_sales * Store_sqft 或 Total_children * Num_children_at_home）以捕获它们之间的其他交互。
聚合：
使用 store_sales、unit_sales、Total_children 和 Num_children_at_home 创建聚合特征，例如具有相似特征的每组商店（具有相似 store_sales 或 store_sqft 范围的商店）的均值、中位数或总和。
Bins：
Bin 数值列，例如 store_sales、unit_sales、Total_children、avg_cars_at_home、Num_children_at_home、Gross_weight、Units_per_case 和 Store_sqft。尝试调整 bin 大小并创建更大的类别，如低、中、高销售额，这将减少这些特征的基数，并可能减少一些噪音。
'''