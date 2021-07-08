#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 11:48:53 2021

@author: kunal
"""

import pandas as pd
import yfinance as yf
#path for data folder
dataPath = "/Users/kunal/OneDrive - Dartmouth College/Research/Mutli-Agent-Reinforcement-Learning/Deep RL Approach to Stock Portfolio/data/"

#%%
# pulling a table of all companies in the S&P 500 
symbols = ["AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "NVDA", "JPM", "JNJ", "V", "UNH", "HD", "PG", "MA", "DIS", "BAC", "ADBE", "XOM", "CMCSA", "NFLX", "VZ", "INTC"]

#%%
stringStocks = "" # creating a string of all of the stocks to be downloaded
for names in symbols:
    stringStocks += names + " "
stringStocks = stringStocks[:-1] # removing the last space

# downloading stock info
trainDf = yf.download(stringStocks, start="2010-01-01", end="2018-01-01", group_by='Ticker', interval= "1d")
# reshaping df to avoid multi layer column names
trainDf = trainDf.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
# saving df to csv for ease of access later
trainDf.to_csv(dataPath + "trainStockInfo.csv")

# downloading stock info
testDf = yf.download(stringStocks, start="2018-01-02", end="2021-01-01", group_by='Ticker', interval= "1d")
# reshaping df to avoid multi layer column names
testDf = testDf.stack(level=0).rename_axis(['Date', 'Ticker']).reset_index(level=1)
# saving df to csv for ease of access later
testDf.to_csv(dataPath + "testStockInfo.csv")


#%%
# Load basic stock data
trainDf = pd.read_csv(dataPath + "trainStockInfo.csv")
testDf = pd.read_csv(dataPath + "testStockInfo.csv")



