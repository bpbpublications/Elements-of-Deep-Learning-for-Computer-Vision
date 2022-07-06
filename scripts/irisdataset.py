# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 02:18:30 2020

@author: bhara
"""

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_dim):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(input_dim,50)
        self.layer2 = nn.Linear(50, 30)
        self.layer3 = nn.Linear(30, 20)
        self.layer4 = nn.Linear(20, 3)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.softmax(self.layer4(x))
        return x
    
iris = load_iris()
xtrain,xtest,ytrain,ytest = train_test_split(iris.data,iris.target, random_state=0, shuffle=True)


model = Model(xtrain.shape[1])
optimiser = torch.optim.RMSprop(model.parameters(), lr=0.01)
loss_function = nn.CrossEntropyLoss()
epochs = 200

    
x_train, y_train = Variable(torch.from_numpy(xtrain)).float(), Variable(torch.from_numpy(ytrain)).long()
for epoch in range(1, epochs+1):
    print ("Epoch Number: ",epoch)
    y_pred = model(x_train)
    loss = loss_function(y_pred, y_train)
    print('Loss Value: ', loss.item())
    optimiser.zero_grad()
    loss.backward()
    optimiser.step()
    
x_test = Variable(torch.from_numpy(xtest)).float()
pred = model(x_test)
pred = pred.detach().numpy()

print ("The accuracy is", np.mean(ytest == np.argmax(pred, axis=1)))
