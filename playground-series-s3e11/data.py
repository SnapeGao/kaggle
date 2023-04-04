import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from torch import nn


class MyDatasets(Dataset):
    def __init__(self, dataset, test=False):
        super(MyDatasets, self).__init__()
        self.test = test
        self.nums = 40000
        if not test:
            col = dataset.shape[1]
            self.x = dataset[:, :col - 1]
            self.y = dataset[:, -1]
            self.num_features = col - 1
            self.y = self.y
            if len(self.x) > self.nums:
                self.y = self.y[:self.nums]
            self.BN = nn.BatchNorm1d(self.num_features)
            self.x = self.BN(self.x)
            if len(self.x) > self.nums:
                self.x = self.x[:self.nums]
            self.len = len(self.x)

        else:
            self.x = dataset
            self.num_features = self.x.shape[1]
            self.BN = nn.BatchNorm1d(self.num_features)
            self.x = self.BN(self.x)
            self.len = len(self.x)

    def __getitem__(self, item):
        if not self.test:
            return self.x[item], self.y[item]
        return self.x[item]

    def __len__(self):
        return self.len


if __name__ == '__main__':
    ds = MyDatasets('../datasets/playground-series-s3e11/train.csv')
