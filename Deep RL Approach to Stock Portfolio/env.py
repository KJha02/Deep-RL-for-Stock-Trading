#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 18:28:22 2021

@author: kunal
"""

from agent2 import dqnAgent
import pandas as pd
import numpy as np
import math

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
        
        self.boughtPrices = []
        self.profit = 0.0
        
        self.actionSpace = list(i for i in range(10))
        
        
    def buy(self):
        # price you bought at
        openPrice = self.openPrices[self.currentDay]
        # cost of 100 shares
        buyCost = self.tradeNormalization * openPrice
        transactionCost = self.transactionFeePercent * buyCost
        netCost = buyCost + transactionCost
        
        # gain 100 shares
        self.numShares += self.tradeNormalization
        # store the net cost of buying 100 shares
        self.boughtPrices.append(netCost)
        
        return 0.0 # our reward
        
    def sell(self):
        try :
            assert len(self.boughtPrices) > 0
            assert self.numShares > 0
            # open price compared to bought price
            openPrice = self.openPrices[self.currentDay]
            boughtPrice = self.boughtPrices.pop(0)
            
            # value of selling 100 shares
            shareValue = self.tradeNormalization * openPrice
            transactionCost = self.transactionFeePercent * shareValue
            netGain = shareValue - transactionCost
            
            # reward is netGain of selling 100 - netCost of buying 100
            reward = netGain - boughtPrice
            self.profit += reward
            
            # have 100 less shares
            self.numShares -= self.tradeNormalization
            # gain netGain to total balance
            self.currBalance += reward
            
            
            return max(reward, 0.0) # our reward
        except:
            return -1.0 # disincentivize trying to sell when you don't have any shares
        
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
        self.netPurchase = 0.0
        self.profit = 0.0
        self.boughtPrices = []
        
        
        return self.getObservation(0)
        
            
            
        
        