#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:34:40 2021

@author: kunal
"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, lr, inputDim, conv1Dim, conv2Dim, numActions):
        super(NeuralNetwork, self).__init__()
        self.inputDim = inputDim
        
        # conv dimensions are a tuple (int, (int, int))
        self.conv1Dim = conv1Dim
        self.conv2Dim = conv2Dim
        self.numActions = numActions
        
        self.conv1 = nn.Linear(self.inputDim, self.conv1Dim)
        self.conv2 = nn.Linear(self.conv1Dim, self.conv2Dim)
        self.conv3 = nn.Linear(self.conv2Dim, self.numActions)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        
        self.device = 'cuda' if T.cuda.is_available() else 'cpu'
        self.to(self.device)
    def forward(self, state):
        x = F.relu(self.conv1(state.float()), inplace=False)
        x = F.relu(self.conv2(x.float()), inplace=False)
        actions = self.conv3(x.float())
        #actions = F.softmax(actions, dim=1)
        #print(actions)
        return actions
        