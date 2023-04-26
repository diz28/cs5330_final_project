import os
import random

# Di Zhang
# April 22, 2023
# CS5330 - Computer Vision

# imports
import torch
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import ConcatDataset
from torchsummary import summary
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet101_Weights
from torchvision.models import ResNet152_Weights
from torchvision.models import AlexNet_Weights
from torchvision.models import VGG16_Weights
import numpy as np
from torchvision.transforms import ToTensor
import cv2


# build the network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout(0.5)
        self.conv3 = nn.Conv2d(20, 128, kernel_size=5)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(self.conv2_drop(self.conv3(x)))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv4(x)), 2))
        x = x.view(-1, 1024)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def plot_image(dataset, targets, n_images, truth_or_pred):
    plt.figure()
    for i in range(n_images):
        # set the plot to be 2 by 3
        plt.subplot(int(n_images / 3), 3, i + 1)
        # set it to be a tight plot
        plt.tight_layout()
        # set a few parameters
        image = dataset[i][0]
        plt.imshow(image, cmap='viridis', interpolation='none')
        plt.title("{}: {}".format(truth_or_pred, targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


# plot the training and testing loss
def plot(train_counter, train_losses, test_counter, test_losses):
    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.show()


# greek letter data set transform
class Transform:
    # initiate the transform
    def __init__(self):
        pass

    # actually implementation of the transform
    def __call__(self, x):
        # x = torchvision.transforms.functional.affine(x, 0, (0, 0), 1, 0)
        x = torchvision.transforms.functional.center_crop(x, (64, 64))
        return x


# get the data
def get_data(batch_size):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('./original64JPG/trainset',
                                         transform=torchvision.transforms.Compose(
                                             [torchvision.transforms.CenterCrop((64, 64)),
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                               std=[0.229, 0.224, 0.225])])),
        batch_size=batch_size,
        shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('./original64JPG/testset',
                                         transform=torchvision.transforms.Compose(
                                             [torchvision.transforms.CenterCrop((64, 64)),
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                               std=[0.229, 0.224, 0.225])])),

        shuffle=True)
    dev_loader = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('./original64JPG/devset',
                                         transform=torchvision.transforms.Compose(
                                             [torchvision.transforms.CenterCrop((64, 64)),
                                              torchvision.transforms.ToTensor(),
                                              torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                               std=[0.229, 0.224, 0.225])])),
        batch_size=batch_size,
        shuffle=True)
    return train_loader, test_loader, dev_loader


# testing method to test the accuracy of the trained mode with 1000 images
def test(network, test_loader, test_losses):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            # cross_entropy is much better than F.nll_loss()
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# training method to train the network to identify digits on a dataset of 60000 digit images
def train(epoch, network, optimizer, train_loader, train_losses, train_counter, log_interval, batch_size):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size) + ((epoch - 1) * len(train_loader.dataset)))


# main/driver class of the file
def main():
    n_epoch = 10
    # vgg16 = torchvision.models.vgg16(weights=VGG16_Weights)
    # resnet = torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    # alexnet = torchvision.models.alexnet(weights=AlexNet_Weights.IMAGENET1K_V1)
    learning_rate = 0.01
    momentum = 0.5
    batch_size = 32
    log_interval = 10

    # learning_rates = [0.001, 0.005, 0.05, 0.01]
    # momentums = [0.5, 0.7, 0.9, 0.99]
    # batch_sizes = [24, 32, 64]

    # network = resnet
    # network = alexnet

    # lock the weights of the network
    # for param in network.parameters():
    #     param.requires_grad = False
    # network.fc = nn.Linear(512, 10)

    # Modify the last layer of the model to include dropout
    # num_ftrs = network.fc.in_features
    # network.fc = nn.Sequential(
    #     nn.Dropout(p=0.5),
    #     nn.Linear(num_ftrs, 10) # modify the output size based on the number of classes in your problem
    # )

    network = Net()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    train_loader, test_loader, dev_loader = get_data(batch_size)

    examples = enumerate(dev_loader)
    batch_idx, (example_data, example_targets) = next(examples)
    print(example_data.shape)
    print(type(example_data))
    # plot the first 6 digit images
    plot_image(example_data, example_targets, 9, 'Ground Truth')

    train_loss = []
    train_counter = []
    test_loss = []
    test_counter = [i * len(dev_loader.dataset) for i in range(n_epoch + 1)]
    print(test_counter)

    test(network, test_loader, test_loss)
    for k in range(1, n_epoch + 1):
        train(k, network, optimizer, dev_loader, train_loss, train_counter, log_interval, batch_size)
        test(network, test_loader, test_loss)

    # training and testing plot
    plot(train_counter, train_loss, test_counter, test_loss)
    print('learning rates: {}, momentum: {}, batch size: {}'.format(learning_rate, momentum, batch_size))


# where the program starts
if __name__ == '__main__':
    main()
