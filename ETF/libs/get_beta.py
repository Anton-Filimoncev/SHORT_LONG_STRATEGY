import pandas as pd
import numpy as np
import yfinance as yf
import requests
import datetime
import time
import datetime as dt
from scipy.signal import argrelextrema
from dateutil.relativedelta import relativedelta

pd.options.mode.chained_assignment = None  # default='warn'
import sqlite3
from scipy import stats
import pandas_ta as pta
import os
from sklearn.linear_model import LinearRegression


def calculate_beta(stock_ticker, market_ticker):
    stock_data = stock_ticker["Close"].pct_change()[1:]
    market_data = market_ticker["Close"].pct_change()[1:]

    covariance = np.cov(stock_data, market_data)[0][1]
    var = np.var(market_data)

    beta = covariance / var
    return beta


def run_beta(ticker_list, stock_yahoo):
    beta_list = []

    start_date = datetime.datetime.now().date()
    spy_yahoo = yf.download("SPY")

    limit_date = start_date - relativedelta(years=+3)
    limit_date = limit_date.strftime("%Y-%m-%d")

    for tick in ticker_list:
        try:
            beta_list.append(calculate_beta(stock_yahoo[tick][limit_date:], spy_yahoo[limit_date:]))
        except:
            beta_list.append(np.nan)

    return beta_list
