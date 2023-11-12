# -*- coding: utf-8 -*-
# @Author: xueqiao
# @Date:   2023-02-09 12:50:33
# @Last Modified by:   xueqiao
# @Last Modified time: 2023-08-31 22:06:18
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import argparse


class CovidDataset(Dataset):
    """loader train data"""

    def __init__(self, csv_file1, csv_file2):
        """
        Args:
            csv_file (string): Path to the csv file.
        """
        self.symptom = pd.read_csv(csv_file1, header=None).dropna()
        self.infection = pd.read_csv(csv_file2, header=None).dropna()
        self.samples = []
        symptom_list = []
        for i in range(0, len(self.symptom), 9):
            symptom = np.array(self.symptom[i:i + 9])
            # pad = np.zeros((9,1))
            pad = np.zeros((9, 1))
            symptom = np.c_[pad, symptom]
            symptom_list.append(symptom)

        infection_list = np.array(self.infection)
        for k in range(0, len(infection_list)):
            self.samples.append([symptom_list[k], infection_list[k]])         

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sym, infection = self.samples[idx]
        return sym, infection


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv = nn.Sequential(
            # nn.Conv1d(in_channels=5, out_channels=1, kernel_size=3),
            # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(7, 2)),
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(9, 2)),
            # nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(2, 3)),
            # nn.Conv2d(in_channels=3, out_channels=1, kernel_size=(1, 3)),
            # nn.Sigmoid()
        )

        self.fc = nn.Sequential(
            nn.LazyLinear(30),
            # nn.Linear(9,1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv(x)
        # x = torch.flatten(x)
        out = self.fc(x)
        return out


def train_loop(dataloader, model, loss_fn, optimizer):
    model = model.double()
    label_train = []
    for X, y in dataloader:
        #input size
        # X = X.view(X.shape[0],1, 7, 31)
        X = X.view(X.shape[0], 1, 9, 31)
        # Compute prediction and loss
        pred = model(X)
        # print(pred.size())
        pred = pred.view(pred.shape[0], 30)
        loss = loss_fn(pred, y.double())

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.item()
        label_train.extend(y)


def test_loop(dataloader, model, loss_fn):
    num_batches = len(dataloader)
    test_loss = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for X, y in dataloader:
            # X = X.view(X.shape[0],1, 7, 31)
            X = X.view(X.shape[0], 1, 9, 31)
            pred = model(X)
            # pred = pred.squeeze(1)
            pred = pred.view(pred.shape[0], 30)
            test_loss += loss_fn(pred, y.double()).item()
            y_true.append(y.numpy().flatten())
            y_pred.append(pred.numpy().flatten())
    test_loss /= num_batches
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    print(f"Avg loss: {test_loss:>8f} \n")
    return y_true, y_pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, default='./data/feature.csv')
    parser.add_argument('--test', type=str, default='./data/infection.csv')
    parser.add_argument('--model', type=str, default='./model/model_831.pth')
    parser.add_argument('--epoch', type=int, default=200)
    args = parser.parse_args()
    full_data = CovidDataset(args.train, args.test)
    # train_data, test_data = torch.utils.data.random_split(full_data, [, 4000])
    train_size = int(0.7 * len(full_data))
    test_size = len(full_data) - train_size
    train_data, test_data = torch.utils.data.random_split(full_data, [train_size, test_size])
    
    train_dataloader = DataLoader(train_data, batch_size=30)
    test_dataloader = DataLoader(test_data, batch_size=30)
    model = NeuralNetwork()
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = args.epoch
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        y_true, y_pred = test_loop(test_dataloader, model, loss_fn)
        # print(y_true.shape)
        # print(y_pred)
        auc = roc_auc_score(y_true, y_pred)
        print(y_pred)
        print(f"AUC: {auc:4f} \n")

    print("Done!")
    torch.save(model.state_dict(), args.model)
