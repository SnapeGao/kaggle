import math

import pandas as pd
import numpy as np
import torch

'''
id- 资料编辑
store_sales(in millions)- 该产品的店内销售价
unit_sales(in millions)- 该产品的销售量
total_children- 孩子数
num_children_at_home- 孩子在家的数量
avg_cars_at home(approx)- 平均车子数量
gross_weight- 总重
recyclable_package- 是否有可回收包装
low_fat- 是否为低脂肪产品
units_per_case- 每包装数量
store_sqft- 店铺平数
coffee_bar- 是不是有咖啡馆
video_store- 是不是有影音馆
salad_bar- 是不是有沙拉吧
prepared_food- 是无调理食品
florist- 是否有花季

每单位收入：store_sales / unit_sales
每平方英尺销售额：store_sales / Store_sqft
每平方英尺单位数：unit_sales / Store_sqft
儿童比例：Total_children / Num_children_at_home 每
单位重量：Gross_weight / unit_sales
'''
'''
去掉 5,10,12,13,
'''
path = '../datasets/playground-series-s3e11/'

train_dataset = pd.read_csv(path + 'train.csv')
test_dataset = pd.read_csv(path + 'test.csv')
col_num = len(train_dataset.keys())
# 去掉id列
train_dataset = train_dataset.iloc[:, 1:col_num]
test_dataset = test_dataset.iloc[:, 1:col_num - 1]
# 增加新特征
store_sales_per_unit_sales_train = train_dataset['store_sales(in millions)'] / train_dataset['unit_sales(in millions)']
store_sales_per_unit_sales_test = test_dataset['store_sales(in millions)'] / test_dataset['unit_sales(in millions)']
train_dataset.insert(3,"store_sales_per_unit_sales_train",store_sales_per_unit_sales_train)
test_dataset.insert(3,"store_sales_per_unit_sales_test",store_sales_per_unit_sales_test)

store_sales_per_Store_sqft_train = train_dataset['store_sales(in millions)'] / train_dataset['store_sqft']
store_sales_per_Store_sqft_test = test_dataset['store_sales(in millions)'] / test_dataset['store_sqft']
train_dataset.insert(3,"store_sales_per_Store_sqft_train",store_sales_per_Store_sqft_train)
test_dataset.insert(3,"store_sales_per_Store_sqft_test",store_sales_per_Store_sqft_test)
unit_sales_per_Store_sqft_train = train_dataset['unit_sales(in millions)'] / train_dataset['store_sqft']
unit_sales_per_Store_sqft_test = test_dataset['unit_sales(in millions)'] / test_dataset['store_sqft']
train_dataset.insert(3,"unit_sales_per_Store_sqft_train",unit_sales_per_Store_sqft_train)
test_dataset.insert(3,"unit_sales_per_Store_sqft_test",unit_sales_per_Store_sqft_test)
# Total_children_per_Num_children_at_home_train=train_dataset['total_children']/train_dataset['num_children_at_home']
# Total_children_per_Num_children_at_home_test=test_dataset['total_children']/test_dataset['num_children_at_home']
# train_dataset["Total_children_per_Num_children_at_home_train"]=Total_children_per_Num_children_at_home_train
# test_dataset["Total_children_per_Num_children_at_home_test"]=Total_children_per_Num_children_at_home_test
Gross_weight_per_unit_sales_train = train_dataset['gross_weight'] / train_dataset['unit_sales(in millions)']
Gross_weight_per_unit_sales_test = test_dataset['gross_weight'] / test_dataset['unit_sales(in millions)']
train_dataset.insert(3,"Gross_weight_per_unit_sales_train",Gross_weight_per_unit_sales_train)
test_dataset.insert(3,"Gross_weight_per_unit_sales_test",Gross_weight_per_unit_sales_test)

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
