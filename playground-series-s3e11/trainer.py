import json
import time

from data import MyDatasets
from model import Net
import sys
import pandas as pd

sys.path.append("")
print(sys.path)
import torch
import numpy as np
from timeit import default_timer as timer
from torch.utils.data import DataLoader
import random
from utils import write_log_file
from myparser import parsed_args


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)


class Trainer(object):
    def __init__(self, args):
        super(Trainer, self).__init__()
        self.args = args
        setup_seed(self.args.seed)
        self.path = self.args.path
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

        self.f = 'logs/log'+time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())+'.txt'
        train_dataset = torch.load(self.path + 'train.pt')
        val_dataset = torch.load(self.path + 'val.pt')
        test_dataset = torch.load(self.path + 'test.pt')
        self.train_dataset = MyDatasets(train_dataset)
        self.val_dataset = MyDatasets(val_dataset)
        self.test_dataset = MyDatasets(test_dataset, test=True)
        self.args.num_features = self.train_dataset.num_features
        self.model = Net(args).to(args.device)

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.args.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(self.test_dataset, batch_size=self.args.batch_size)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.loss_function = torch.nn.L1Loss()
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                    factor=args.lr_reduce_factor,
                                                                    patience=args.lr_schedule_patience,
                                                                    min_lr=args.min_lr,
                                                                    verbose=True)

    def process(self, path, train=False):
        dataset = pd.read_csv(path)
        col_num = len(dataset.keys())
        x, y = dataset.iloc[:, 1:col_num - 1], dataset.iloc[:, col_num - 1]

    def train(self):
        self.model.train()
        for epoch in range(1, self.args.epochs + 1):
            tic = timer()
            print(tic)
            loss_all = 0
            for index, data in enumerate(self.train_dataloader):
                self.optimizer.zero_grad()
                x, y = data[0].to(self.args.device), data[1]
                y_pred = self.model(x)
                loss = self.loss_function(y_pred.cpu(), y)
                loss_all += loss.item()
                loss.backward(retain_graph=True)
                self.optimizer.step()
                self.scheduler.step(loss)
            toc = timer()
            write_log_file(self.f, 'Epoch {}, Loss {:.4f},time:{:.2f}s'.format(epoch, loss_all / len(
                self.train_dataloader), toc - tic))
            if epoch % self.args.log_interval == 0:
                self.val()
                self.test()

    def val(self):
        self.model.eval()
        tic = timer()
        loss_all = 0
        for data in self.val_dataloader:
            self.optimizer.zero_grad()
            x, y = data[0].to(self.args.device), data[1]
            y_pred = self.model(x)

            loss = self.loss_function(y_pred.cpu(), y)
            loss_all += loss.item()
            # print('batch')
        toc = timer()
        write_log_file(self.f, ' val Loss {:.4f},time:{:.2f}s'.format(loss_all / len(
            self.val_dataloader), toc - tic))

    def test(self):
        self.model.eval()
        with torch.no_grad():
            tic = timer()
            y_list = []
            for data in self.test_dataloader:
                self.optimizer.zero_grad()
                x = data.to(self.args.device)
                y_pred = self.model(x)
                y_list.extend(y_pred.cpu())
            y = np.array(y_list)
            submission = pd.read_csv(self.path + 'sample_submission.csv')
            submission['cost'] = y
            submission_name = 'submission_feature_net_atten_40000.csv'
            submission.to_csv(self.path + submission_name, index_label=False)

            toc = timer()
            write_log_file(self.f, submission_name + ' finish! ,time:{:.2f}s'.format(toc - tic))


if __name__ == '__main__':
    args = parsed_args
    print(args)
    trainer = Trainer(args)
    trainer.train()
    trainer.test()
