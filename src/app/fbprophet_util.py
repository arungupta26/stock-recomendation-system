import yfinance as yf
import pandas as pd
import numpy as np
from fbprophet import Prophet
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import pickle
import datetime

import yfinance as yf

import pandas as pd
import numpy as np
from fbprophet import Prophet
from pathlib import Path

import linear_regression_util as lru


def get_predicted_price(symbol, period=10):
    model_file_path = './../resources/fbprophet/model/' + symbol + '.model'

    print("loading model ", model_file_path)

    # read the Prophet model object
    with open(model_file_path, 'rb') as f:
        m = pickle.load(f)

    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    data = forecast.set_index('ds')
    data = data[['yhat', 'yhat_upper', 'yhat_lower']].dropna().tail(500)
    data['yhat'] = np.exp(data.yhat)
    data['yhat_lower'] = np.exp(data.yhat_lower)
    data['yhat_upper'] = np.exp(data.yhat_upper)

    data.rename(columns={'yhat': 'Predicted Price', 'yhat_lower': 'Predicted Price(Lower)',
                         'yhat_upper': 'Predicted Price(Upper)'}, inplace=True)

    return data[-period:]
