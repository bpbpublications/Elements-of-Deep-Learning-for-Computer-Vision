# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 02:12:48 2020

@author: bhara
"""
import torch
import torchvision
import torchvision.transforms as transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
cifar_data = torchvision.datasets.CIFAR10(root=r'C:\Users\bhara\OneDrive\Desktop\Datasets-torchvision', train=True,
                                        download=True, transform=transform)
data_loader = torch.utils.data.DataLoader(cifar_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=0)