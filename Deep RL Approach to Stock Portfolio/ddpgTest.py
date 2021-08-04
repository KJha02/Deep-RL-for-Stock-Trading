#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  4 12:04:33 2021

@author: kunal
"""

from env import env
import ddpg
from ddpg import DDPGAgent
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
dataPath = "/Users/kunal/OneDrive - Dartmouth College/Research/Mutli-Agent-Reinforcement-Learning/Deep RL Approach to Stock Portfolio/"

#%%
trainDf = pd.read_csv(dataPath + "data/AAPL.csv")
INITIAL_BALANCE = 100000
actions = [0,1,2] # buy, sell, or hold
myEnv = env(trainDf, INITIAL_BALANCE, actions)
agent = DDPGAgent(inputDims=(5,), env=myEnv, numActions=len(actions))

#%%
numGames = 100
figureFile = 'charts/100epDDPG.png'
scoreHistory, balanceHistory = [], []
loadCheckpoint = False

if loadCheckpoint:
    numSteps = 0
    while numSteps <= agent.batchSize:
        observation = myEnv.reset()
        action = np.random.choice(env.actionSpace)
        newState, reward, done, info = env.step(action)
        agent.remember(observation, action, reward, newState, done)
        numSteps += 1
    agent.learn()
    agent.loadModels()
    evaluate = True
else:
    evaluate = False

for i in range(numGames):
    observation = myEnv.reset()
    done = False
    score = 0
    while not done:
        action = agent.chooseAction(observation, evaluate)
        newState, reward, done = myEnv.step(action)
        score += reward
        agent.remember(observation, action, reward, newState, done)
        if not loadCheckpoint:
            agent.learn()
        observation = newState
    scoreHistory.append(score)
    balanceHistory.append(myEnv.currBalance)
    avgScore = np.mean(scoreHistory[-100:])
    print("episode ", i, "score %.2f" % score, "average score %.2f" % avgScore)

#%%
if not loadCheckpoint:
    x = [i+1 for i in range(numGames)]
    plt.scatter(x, scoreHistory)
    plt.xlabel("Episode Number")
    plt.ylabel("Score")
    plt.title("Change in Score Over 100 Episodes")
    plt.show()
    plt.scatter(x, balanceHistory)
    plt.xlabel("Episode Number")
    plt.ylabel("Final Account Balance")
    plt.title("Change in Final Balance Over 100 Episodes")
    plt.savefig(dataPath + "charts/100epDDPGBalance.png")
    plt.show()

#%%
agent.saveModels()