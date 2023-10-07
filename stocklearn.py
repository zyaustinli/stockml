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
train = closed_data.iloc[:-100]
test = closed_data.iloc[-100:] 
model.fit(train[predictors], train["Target"])

predicts= model.predict(test[predictors])
predicts = pd.Series(predicts, index=test.index)
precision_score(test["Target"], predicts)



