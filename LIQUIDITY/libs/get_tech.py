import pandas_ta as pta
import pandas as pd
import numpy as np
import yfinance as yf
pd.options.mode.chained_assignment = None


def get_tech_data(df):
    df['RSI'] = pta.rsi(df['Close'])
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_100'] = df['Close'].rolling(window=100).mean()

    sma_20 = df['SMA_20'].dropna().iloc[-1]
    sma_100 = df['SMA_100'].dropna().iloc[-1]
    rsi = df['RSI'].dropna().iloc[-1]

    return sma_20, sma_100, rsi

def trend(price_df):
    trend = ''
    current_price = price_df['Close'].iloc[-1]
    SMA_20 = price_df['Close'].rolling(window=20).mean().iloc[-1]
    SMA_100 = price_df['Close'].rolling(window=100).mean().iloc[-1]

    if current_price > SMA_100 and current_price > SMA_20 and SMA_20 > SMA_100:
        trend = 'Strong Uptrend'
    if current_price > SMA_100 and current_price > SMA_20 and SMA_20 < SMA_100:
        trend = 'Uptrend'
    if current_price > SMA_100 and current_price < SMA_20 and SMA_20:
        trend = 'Weak Uptrend'
    if current_price < SMA_100 and current_price < SMA_20 and SMA_20 < SMA_100:
        trend = 'Strong Downtrend'
    if current_price < SMA_100 and current_price < SMA_20 and SMA_20 > SMA_100:
        trend = 'Downtrend'
    if current_price < SMA_100 and current_price > SMA_20:
        trend = 'Weak downtrend'

    return trend



def get_tech_run(stock_yahoo, tick_list):
    print('---------------------------')
    print('------------- Getting Tech Signals ... --------------')
    print('---------------------------')
    trend_signal_list = []
    rsi_list = []
    cur_price_list = []

    for tick in tick_list:
        try:
            trend_signal = trend(stock_yahoo[tick])
            sma_20, sma_100, rsi = get_tech_data(stock_yahoo[tick])
            cur_price = stock_yahoo[tick]['Close'].iloc[-1]
        except:
            trend_signal = 'Empty'
            rsi = 'Empty'
            cur_price = 'Empty'

        trend_signal_list.append(trend_signal)
        rsi_list.append(rsi)
        cur_price_list.append(cur_price)


    return trend_signal_list, rsi_list, cur_price_list

 