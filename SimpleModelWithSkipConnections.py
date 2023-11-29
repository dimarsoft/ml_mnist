"""
Модель НС
"""
import torch

from torch import nn


class SimpleModelWithSkipConnections(nn.Module):
    """
    Модель простая, но со Skip Connections
    """

    def __init__(self, norm=False, dropout=False, batch_norm=False):
        super().__init__()
        self.flatten = nn.Flatten()

        if norm:
            self.norm = nn.LayerNorm(784)
        else:
            self.norm = None

        self.fc1 = nn.Linear(in_features=784, out_features=128)
        self.relu = nn.ReLU()

        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(num_features=128)
        else:
            self.batch_norm = None

        if dropout:
            self.dropout = nn.Dropout(0.5)
        else:
            self.dropout = None

        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=192, out_features=10)  # Изменено out_features на 192

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Шаг обучения модели
        :param x:
        :return:
        """
        x = self.flatten(x)

        if self.norm is not None:
            x = self.norm(x)

        x1 = x = self.relu(self.fc1(x))

        if self.batch_norm is not None:
            x1 = self.batch_norm(x1)

        if self.dropout is not None:
            x1 = self.dropout(x1)

        x2 = self.relu(self.fc2(x1))

        # Добавление skip-connection, конкатенация результатов двух слоев
        concatenated = torch.cat((x, x2), dim=1)
        x3 = self.fc3(concatenated)
        x3 = self.sigmoid(x3)

        return x3
