import os
import sys
import time
import math
import torch.nn.init as init
from torchvision import datasets, transforms
from src.vision import *
from torch.utils.data.dataset import *
import numpy as np



class TinyImageNet(VisionDataset):
    @property
    def train_labels(self):
        warnings.warn("train_labels has been renamed targets")
        return self.targets

    @property
    def test_labels(self):
        warnings.warn("test_labels has been renamed targets")
        return self.targets

    @property
    def train_data(self):
        warnings.warn("train_data has been renamed data")
        return self.data

    @property
    def test_data(self):
        warnings.warn("test_data has been renamed data")
        return self.data

    def __init__(self, root, train=True, transform=None):
        super(TinyImageNet, self).__init__(root)
        training_data_file = 'tiny_imagenet_train_x.npy'
        training_label_file = 'tiny_imagenet_train_y.npy'
        test_data_file = 'tiny_imagenet_train_x.npy'
        test_label_file = 'tiny_imagenet_train_y.npy'
        self.train = train  # training set or test set

        if self.train:
            self.data = np.load(training_data_file, allow_pickle=True)
            self.targets = np.load(training_label_file, allow_pickle=True)
        else:
            self.data = np.load(test_data_file, allow_pickle=True)
            self.targets = np.load(test_label_file, allow_pickle=True)
        # self.data = [x.reshape(50,50) for x in self.data]
        self.targets = [x for x in self.targets]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        return img, target

    def __len__(self):
        return len(self.data)


def get_dataset(dataset_name, args):
    dataset_name = dataset_name.lower()
    print("dataset_name",dataset_name)
    if dataset_name == "tiny_imagenet":
        train_dataset = TinyImageNet(args.root, train=True)
        test_dataset = TinyImageNet(args.root, train=False)

        
    elif dataset_name ==  "mnist":
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_dataset = datasets.MNIST(args.root, train=True, transform=transform, download=True)
        test_dataset = datasets.MNIST(args.root, train=False, transform=transform, download=True)

    elif dataset_name=='cifar10':
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            #transforms.Normalize((0.1307,), (0.3081,)),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        train_dataset = datasets.CIFAR10(args.root, train=True, transform=transform, download=True)
        test_dataset = datasets.CIFAR10(args.root, train=False, transform=transform, download=True)

    else:
        raise ValueError("Not support dataset: {}!".format(dataset_name))
    return train_dataset, test_dataset