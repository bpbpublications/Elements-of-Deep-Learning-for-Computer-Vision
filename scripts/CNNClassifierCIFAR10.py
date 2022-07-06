# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 01:24:16 2020

@author: bhara
"""

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./CIFAR1', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./CIFAR1', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #First Convolutional Layer
        self.conv1 = nn.Conv2d(3, 8, 5)
        #MaxPool Layer
        self.pool = nn.MaxPool2d(2, 2)
        #Second Convolutional Layer
        self.conv2 = nn.Conv2d(8, 16, 5)
        #Linear Flattening Layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #Further Flattening of the layer
        self.fc2 = nn.Linear(120, 84)
        #Output with 10 classes 
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adamax(net.parameters(), lr=0.01)

num_epoch = 5
total_step = len(trainloader)
for epoch in range(num_epoch):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('Epoch [%d/%d], Step [%d/%d], Loss: %.3f' %
                  (epoch + 1, num_epoch, i + 1,total_step, running_loss / 2000))
            running_loss = 0.0

true = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        true += (predicted == labels).sum().item()

print('Accuracy: %d %%' % (
    100 * true / total))

true_class = [0.0] * len(classes)
total_classes = [0.0] * len(classes)
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            true_class[label] += c[i].item()
            total_classes[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * true_class[i] / total_classes[i]))