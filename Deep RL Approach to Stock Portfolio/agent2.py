#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 17:54:14 2021

@author: kunal
"""

from simpleDQN import NeuralNetwork
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class dqnAgent:
    def __init__(self, gamma, epsilon, lr, inputDim, batchSize, numActions, 
                 max_mem_size=100000, eps_end = 0.01, eps_dec = 0.0005):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.actionSpace = [i for i in range(numActions)]
        self.inputDim = inputDim
        self.batchSize = batchSize
        self.numAction = numActions
        self.max_mem_size = max_mem_size
        self.memCount = 0
        self.eps_end = eps_end
        self.eps_dec = eps_dec

        self.Q_eval = NeuralNetwork(lr, self.inputDim, 5, 50, self.numAction)
        
        self.stateMemory = np.zeros((self.max_mem_size, self.inputDim), dtype=np.float32)
        self.newStateMemory = np.zeros((self.max_mem_size, self.inputDim), dtype=np.float32)
        self.actionMemory = np.zeros(self.max_mem_size, dtype=np.int32)
        self.rewardMemory = np.zeros(self.max_mem_size, dtype=np.float32)
        self.terminalMemory = np.zeros(self.max_mem_size, dtype=np.bool)
    
    def storeTransition(self, state, action, reward, newState, done):
        index = self.memCount % self.max_mem_size
        self.stateMemory[index] = state
        self.actionMemory[index] = action
        self.rewardMemory[index] = reward
        self.newStateMemory[index] = newState
        self.terminalMemory[index] = done
        
        self.memCount += 1
    
    def chooseAction(self, observation):
        if (np.random.random() > self.epsilon):
            state = T.tensor([observation]).to(self.Q_eval.device)
            actions = self.Q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.actionSpace)
        
        return action
    
    def learn(self):
        if self.memCount < self.batchSize:
            return
        
        self.Q_eval.optimizer.zero_grad()
        
        maxMem = min(self.memCount, self.max_mem_size)
        batch = np.random.choice(maxMem, self.batchSize, replace=False)
        
        batchIdx = np.arange(self.batchSize, dtype=np.int32)
        
        stateBatch = T.tensor(self.stateMemory[batch]).to(self.Q_eval.device)
        newStateBatch = T.tensor(self.newStateMemory[batch]).to(self.Q_eval.device)
        rewardBatch = T.tensor(self.rewardMemory[batch]).to(self.Q_eval.device)
        terminalBatch = T.tensor(self.terminalMemory[batch]).to(self.Q_eval.device)
        
        actionBatch = self.actionMemory[batch]
        
        q_eval = self.Q_eval.forward(stateBatch)[batchIdx, actionBatch]
        q_next = self.Q_eval.forward(newStateBatch)
        q_next[terminalBatch] = 0.0
  

        
        q_target = rewardBatch + self.gamma * T.max(q_next, dim=-1)[0]
        
        #T.autograd.set_detect_anomaly(True)
        
        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()
    
        if self.epsilon >= self.eps_end:
            self.epsilon *= self.eps_dec 
        else:
            self.epsilon = self.eps_end
            