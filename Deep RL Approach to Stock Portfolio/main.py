#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 19:24:30 2021

@author: kunal
"""

from agent2 import dqnAgent
import pandas as pd
import numpy as np
import yfinance as yf
from env import env
dataPath = "/Users/kunal/OneDrive - Dartmouth College/Research/Mutli-Agent-Reinforcement-Learning/Deep RL Approach to Stock Portfolio/data/"
#%%
# downloading stock info
trainDf = yf.download("AAPL", start="2010-01-01", end="2018-01-01", interval= "1d")
# saving df to csv for ease of access later
trainDf.to_csv(dataPath + "AAPL.csv")

#%%
trainDf = pd.read_csv(dataPath + "AAPL.csv")

#%%
INITIAL_BALANCE = 100000
actions = [0,1,2] # buy, sell, or hold
myEnv = env(trainDf, INITIAL_BALANCE, actions)

#%%
myAgent = dqnAgent(0.99, 1.0, 0.0001, (5), 64, 3, eps_end=0.01, eps_dec=0.0005)
scores, epsHistory = [], []
n_games = 500
for i in range(n_games):
    score = 0
    done = False
    observation = myEnv.reset()
    while not done:
        action = myAgent.chooseAction(observation)
        nextObs, reward, done = myEnv.step(action)
        score += reward
        myAgent.storeTransition(observation, action, reward, nextObs, done)
        myAgent.learn()
        observation = nextObs
    scores.append(score)
    epsHistory.append(myAgent.epsilon)
    averageScore = np.mean(scores[-100:])
    print("episode ", i, "score %.2f" % score, "average score %.2f" % averageScore, "epsilon %.2f" % myAgent.epsilon)
x = [i+1 for i in range(n_games)]