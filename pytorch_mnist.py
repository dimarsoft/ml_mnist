"""
Разпознование цифр Mnist с помощью PyTorch
"""

import numpy as np
from matplotlib import pyplot as plt

import torch

from torchvision import datasets, transforms

from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import torch.nn.functional as F

from prettytable import PrettyTable


def create_model():
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=784, out_features=128),
        nn.ReLU(),
        nn.Linear(in_features=128, out_features=64),
        nn.ReLU(),
        nn.Linear(in_features=64, out_features=10),
        nn.Sigmoid()
    )
    return model


def create_model_2(norm=False):
    model = nn.Sequential()
    model.append(nn.Flatten())

    if norm:
        model.append(nn.LayerNorm(784))

    model.append(nn.Linear(in_features=784, out_features=128))
    model.append(nn.ReLU())
    model.append(nn.Linear(in_features=128, out_features=64))
    model.append(nn.ReLU())
    model.append(nn.Linear(in_features=64, out_features=10))
    model.append(nn.Sigmoid())

    return model


def print_count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def show_losses(train_loss_hist, test_loss_hist):
    # clear_output()

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.title('Train Loss')
    plt.plot(np.arange(len(train_loss_hist)), train_loss_hist)
    plt.yscale('log')
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.title('Test Loss')
    plt.plot(np.arange(len(test_loss_hist)), test_loss_hist)
    plt.yscale('log')
    plt.grid()

    plt.show()


def show_graphs(train_loss_hist, test_loss_hist, train_accuracy_hist, test_accuracy_hist):
    # clear_output()

    plt.figure(figsize=(12, 4))

    plt.subplot(2, 2, 1)
    plt.title('Train Loss')
    plt.plot(np.arange(len(train_loss_hist)), train_loss_hist)
    plt.yscale('log')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.title('Test Loss')
    plt.plot(np.arange(len(test_loss_hist)), test_loss_hist)
    plt.yscale('log')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.title('Train Accuracy')
    plt.plot(np.arange(len(train_accuracy_hist)), train_accuracy_hist)
    plt.yscale('log')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.title('Test Accuracy')
    plt.plot(np.arange(len(test_accuracy_hist)), test_accuracy_hist)
    plt.yscale('log')
    plt.grid()

    plt.show()


def train(model, train_loader, loss_function, optimizer, device, epoch, log_interval=-1):
    model.train()

    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)

        prediction = output.argmax(axis=1, keepdims=True)
        y_n = target.argmax(axis=1, keepdims=True)
        correct += prediction.eq(y_n.view_as(prediction)).sum().item()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if log_interval > 0 & (batch_idx % log_interval == 0):
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item()/len(data):.6f}')

    total_items = len(train_loader.dataset)

    total_loss /= total_items
    correct /= total_items

    return total_loss, correct


def eval_model(model, train_loader, loss_function, device, epoch, log_interval=-1):
    model.eval()

    total_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_function(output, target)

        prediction = output.argmax(axis=1, keepdims=True)
        y_n = target.argmax(axis=1, keepdims=True)

        correct += prediction.eq(y_n.view_as(prediction)).sum().item()

        loss.backward()

        total_loss += loss.item()

        if log_interval > 0 & (batch_idx % log_interval == 0):
            print(f'Eval Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item()/len(data):.6f}')

    total_items = len(train_loader.dataset)

    total_loss /= total_items
    correct /= total_items

    return total_loss, correct


def run(model, dataloader, loss_function, optimizer=None):
    # set the model to evaluation or training mode
    if optimizer == None:
        model.eval()
    else:
        model.train()

    total_loss = 0

    for X, y in dataloader:
        # compute prediction
        pred = model(X)
        # compute loss
        loss = loss_function(pred, y)

        # print(f"loss = {loss}")
        # save loss
        total_loss += loss.item()
        if optimizer != None:
            # compute gradients
            loss.backward()
            # do optimizer step
            optimizer.step()
            # clear gradients
            optimizer.zero_grad()

    return total_loss / len(dataloader)


def one_hot_transform(target):
    return F.one_hot(torch.tensor(target), num_classes=10).to(dtype=torch.float)


def get_train_and_test_data(batch_size=10, batch_size_test=4):
    # x (входы) трасформируем в тензоры
    transform = transforms.ToTensor()
    """
    так же можно применяь несколько трасформатором и данные нормаливать и/или конвертировать в диапазон [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
    ])
    """

    # y (ответы) кодируем в one_hot, т.к категории

    target_transform = one_hot_transform

    print(f"mnist start load ...")

    mnist_train = datasets.MNIST(root='mnist', download=True, train=True, transform=transform,
                                 target_transform=target_transform)
    mnist_test = datasets.MNIST(root='mnist', download=True, train=False, transform=transform,
                                target_transform=target_transform)

    train_loader = DataLoader(mnist_train, batch_size=batch_size)
    test_loader = DataLoader(mnist_test, batch_size=batch_size_test)

    print(f"mnist loaded ...")

    return train_loader, test_loader


def train_model(model_to_train, train_loader, test_loader, device, epochs=10, show_graph=True) -> None:
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model_to_train.parameters(), lr=1e-1)

    train_loss_hist = []
    test_loss_hist = []

    train_accuracy_hist = []
    test_accuracy_hist = []

    print(f"start train: epochs = {epochs}, device = {device}")

    for i in range(epochs):

        train_loss, train_accuracy = train(model_to_train, train_loader, loss_function, optimizer, device, i)
        test_loss, test_accuracy = eval_model(model_to_train, test_loader, loss_function, device, i)

        print(f"epoch: {i + 1}/{epochs}, "
              f"train_loss = {train_loss:.6f}, train_accuracy = {train_accuracy:.6f}, "
              f"test_loss = {test_loss:.6f}, test_accuracy = {test_accuracy:.6f}")

        if show_graph:
            train_loss_hist.append(train_loss)
            train_accuracy_hist.append(train_accuracy)
            test_loss_hist.append(test_loss)
            test_accuracy_hist.append(train_accuracy)

    if show_graph:
        show_graphs(train_loss_hist, test_loss_hist, train_accuracy_hist, test_accuracy_hist)


def get_device():
    use_cuda = torch.cuda.is_available()

    print(f"use_cuda = {use_cuda}")

    if use_cuda:
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def model_1():
    print("Test simple model")

    # устройство на котором обучаем, CPU/GPU
    my_device = get_device()

    # создаем модель и переносим на устройство
    model = create_model().to(my_device)

    # загрузка и подготовка датасета
    train_loader, test_loader = get_train_and_test_data()

    # запуск обучения

    train_model(model, train_loader, test_loader, device=my_device, epochs=2)


np.random.seed(123)

model_1()
'''

x = torch.tensor(1)
xx = F.one_hot(x, num_classes=10).to(dtype=torch.float)

transform = transforms.ToTensor()
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.1307,), (0.3081,))
])

target_transform = lambda target: F.one_hot(torch.tensor(target), num_classes=10).to(dtype=torch.float)

mnist_train = datasets.MNIST(root='mnist', download=True, train=True, transform=transform,
                             target_transform=target_transform)
mnist_test = datasets.MNIST(root='mnist', download=True, train=False, transform=transform,
                            target_transform=target_transform)

for item in mnist_train:
    x, y = item

    break
use_cuda = torch.cuda.is_available()

print(f"use_cuda = {use_cuda}")

if use_cuda:
    my_device = torch.device("cuda")
else:
    my_device = torch.device("cpu")

batch_size = 64

train_loader = DataLoader(mnist_train, batch_size=batch_size)
test_loader = DataLoader(mnist_test, batch_size=8)

my_model = create_model_2(norm=True)

my_model = my_model.to(my_device)

print_count_parameters(my_model)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(my_model.parameters(), lr=1e-1, momentum=0.9)

BATCH_SIZE = 8
NUM_EPOCHS = 10

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
# test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

train_loss_hist = []
test_loss_hist = []

train_accuracy_hist = []
test_accuracy_hist = []

for i in range(NUM_EPOCHS):
    # train_loss = run(my_model, train_loader, loss_function, optimizer)

    train_loss, train_accuracy = train(my_model, train_loader, loss_function, optimizer, my_device, i, log_interval=100)

    # print(f"epoch = {i + 1}, train_loss = {train_loss}")
    # print(f"epoch: {i + 1}/{NUM_EPOCHS}, train_loss = {train_loss}, train_accuracy = {train_accuracy}")
    train_loss_hist.append(train_loss)
    train_accuracy_hist.append(train_accuracy)

    # test_loss = run(my_model, test_loader, loss_function)

    test_loss, test_accuracy = eval_model(my_model, test_loader, loss_function, my_device, i, log_interval=100)

    print(f"epoch: {i + 1}/{NUM_EPOCHS}, "
          f"train_loss = {train_loss:.6f}, train_accuracy = {train_accuracy:.6f}, "
          f"test_loss = {test_loss:.6f}, test_accuracy = {test_accuracy:.6f}")
    test_loss_hist.append(test_loss)
    test_accuracy_hist.append(train_accuracy)

    # if i % 2 == 0: show_losses(train_loss_hist, test_loss_hist)

show_graphs(train_loss_hist, test_loss_hist, train_accuracy_hist, test_accuracy_hist)

"""

my_model.eval()

correct = 0

for X, y in test_loader:
    y = y.to(my_device)
    # for i in np.random.choice(np.arange(0, 100), size=(10,)):
    probs = my_model(X.to(my_device))
    prediction = probs.argmax(axis=1, keepdims=True)
    y_n = y.argmax(axis=1, keepdims=True)

    correct += prediction.eq(y_n.view_as(prediction)).sum().item()

    # correct = len(y)

    # image = (X * 255).reshape((28, 28)).astype("uint8")

    print(f"Actual digit is {y}({y_n}), predicted {prediction}, correct = {correct}")
    # cv2.imshow("Digit", image)
    # cv2.waitKey(0)

    # break

correct /= len(test_loader.dataset)

print(f'Test set: Average loss: {test_loss:.4f}, '
      f'Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct:.0f}%)')
      
"""

'''
