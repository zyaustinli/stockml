#everything based on dataquest tutorial

import yfinance as yf
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import precision_score

data_path = "tsla_data.json"
if os.path.exists(data_path):
    with open(data_path) as f:
        tsla_history = pd.read_json(data_path)
else:
    tsla = yf.Ticker("TSLA")
    tsla_history=tsla.history(period="max")

    tsla_history.to_json(data_path)

'''
plt.plot(tsla_history["Close"], color="r")
plt.show()'''


closed_data = tsla_history[['Close']]
closed_data['Target']= tsla_history.rolling(2).apply(lambda close: close.iloc[1]>close.iloc[0])['Close']
#0 in target means increase, 1 means decrease
tsla_shifted = tsla_history.copy().shift(1) #shifts data one ahead to predict one day in future

predictors = ["Close", "Volume", "Open", "High", "Low"]
closed_data = closed_data.join(tsla_shifted[predictors]).iloc[1:] #start at one to use data from previous day

model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)
def backtest(data, model, predictors, start=1000, step=750):
    predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()

        model.fit(train[predictors], train["Target"])

        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > .6] = 1
        preds[preds<=.6] = 0

        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)

        predictions.append(combined)

    return pd.concat(predictions)
predictions = backtest(data, model, predictors)
weekly_mean = data.rolling(7).mean()["Close"]
quarterly_mean = data.rolling(90).mean()["Close"]
annual_mean = data.rolling(365).mean()["Close"]
weekly_trend = data.shift(1).rolling(7).sum()["Target"]
data["weekly_mean"] = weekly_mean / data["Close"]
data["quarterly_mean"] = quarterly_mean / data["Close"]
data["annual_mean"] = annual_mean / data["Close"]
data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
data["weekly_trend"] = weekly_trend
data["open_close_ratio"] = data["Open"] / data["Close"]
data["high_close_ratio"] = data["High"] / data["Close"]
data["low_close_ratio"] = data["Low"] / data["Close"]
full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "hig
predictions = backtest(data.iloc[365:], model, full_predictors)
print(precision_score(predictions["Target"], predictions["Predictions"]))
