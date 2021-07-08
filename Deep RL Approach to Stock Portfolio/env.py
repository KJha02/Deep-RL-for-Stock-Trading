#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 18:28:22 2021

@author: kunal
"""

from agent2 import dqnAgent
import pandas as pd
import numpy as np

class env:
    def __init__(self, df, initBalance, actions, transactionFeePercent=0.001, tradeNormalization=100):
        self.initialAccountBalance = initBalance # initial money held
        self.currBalance = self.initialAccountBalance # how much is currently held
        self.actions = actions # 3 possible actions
        self.transactionFeePercent = transactionFeePercent # percentage expenditure for buying or selling
        self.tradeNormalization = tradeNormalization # 100 shares per trade
        self.numShares = 0 # number of shares currently held
        self.currentDay = 0
        
        self.df = df
        self.openPrices = df["Open"].to_numpy()
        self.closePrices = df["Adj Close"].to_numpy()
        self.dates = df["Date"].to_numpy()
        self.volume = df["Volume"].to_numpy()
        
        
    def buy(self):
        openPrice = self.openPrices[self.currentDay]
        buyCost = self.tradeNormalization * openPrice
        transactionCost = self.transactionFeePercent * self.tradeNormalization
        netCost = buyCost + transactionCost
        
        self.numShares += self.tradeNormalization
        closePrice = self.closePrices[self.currentDay]
        reward = (openPrice - closePrice) * self.tradeNormalization
        
        
        self.currBalance -= netCost
        
        return reward # our reward
        
    def sell(self):
        openPrice = self.openPrices[self.currentDay]
        shareValue = self.numShares * openPrice
        transactionCost = self.transactionFeePercent * self.numShares
        netGain = shareValue - transactionCost
        closePrice = self.closePrices[self.currentDay]
        reward = (openPrice - closePrice) * self.numShares
        
        self.numShares = 0
        self.currBalance += netGain
        
        return reward # our reward
        
    def getObservation(self, i):
        month = float(self.dates[i][-5:-3]) # get the month as a float
        day = float(self.dates[i][-2:]) # get the day as a float
        openPrice = float(self.openPrices[i]) # get the open price as a float
        tradeVolume = float(self.volume[i]) # get the trading volume as a float
        return (month, day, openPrice, tradeVolume, self.currBalance)
        
    def step(self, action):
        reward = 0
        done = False
        if action == 0:
            reward = self.buy()
        elif action == 1:
            reward = self.sell()
        if (self.currBalance <= 0):
            done = True
        if (self.currentDay + 2 > len(self.df)-1):
            done = True
        nextObservation = self.getObservation(self.currentDay + 1)
        self.currentDay += 1
        
        return nextObservation, reward, done
    def reset(self):
        self.currBalance = self.initialAccountBalance # how much is currently held
        self.numShares = 0 # number of shares currently held
        self.currentDay = 0
        return self.getObservation(0)
        
            
            
        
        