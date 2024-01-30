import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from dateutil.relativedelta import relativedelta
import datetime
from scipy import stats

def volatility_calc(stock_yahoo):
    # ======= HIST volatility calculated ===========
    try:
        TRADING_DAYS = 252
        returns = np.log(stock_yahoo / stock_yahoo.shift(1))
        returns.fillna(0, inplace=True)
        volatility = returns.rolling(window=TRADING_DAYS).std() * np.sqrt(TRADING_DAYS)
        hist_vol = volatility.iloc[-1]
    except:
        hist_vol = 0

    return hist_vol, volatility

def vol_stage(stock_yahoo_solo):
    stock_yahoo_solo = stock_yahoo_solo[-300:]
    df_quantile = stock_yahoo_solo.rolling(252).apply(
        lambda x: stats.percentileofscore(x, x.iloc[-1]))

    quantile = df_quantile.iloc[-1]

    regime = 1
    if 25 < quantile < 50:
        regime = 2
    elif 50 < quantile < 75:
        regime = 3
    elif quantile > 75:
        regime = 4

    return regime


def get_volatility_run(tick_list, stock_yahoo):
    print('---------------------------')
    print('------------- Getting HV --------------')
    print('---------------------------')
    hist_vol_list = []
    hist_vol_stage_list = []
    for tick in tick_list:
        stock_yahoo_solo = stock_yahoo[tick]['Close']
        hist_vol, vol_df = volatility_calc(stock_yahoo_solo)
        hist_vol_stage_list.append(vol_stage(vol_df))
        hist_vol_list.append(hist_vol)

    return hist_vol_list, hist_vol_stage_list


