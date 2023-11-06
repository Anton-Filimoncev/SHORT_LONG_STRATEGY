import pandas as pd
import numpy as np
import os
import time
import pickle
import gspread as gd
from ib_insync import *
from scipy import stats
from sklearn import mixture as mix
import yfinance as yf
import pandas_ta as pta
import datetime
from dateutil.relativedelta import relativedelta
import aiohttp
import asyncio
import requests
import math
import mibian
import tqdm
pd.options.mode.chained_assignment = None


def calculate_probability(ticker, nb_simulations, days, lower_bound, upper_bound, target_price_df, side):
    # Fetch historical market data
    hist_data = yf.download(ticker)['2000-01-01':]

    # Calculate the log returns
    log_returns = np.log(1 + hist_data['Adj Close'].pct_change())

    # Define the variables
    u = log_returns.mean()
    var = log_returns.var()

    # Calculate drift and standard deviation
    drift = u - (0.5 * var)
    stddev = log_returns.std()

    # Generate a random variable
    daily_returns = np.exp(drift + stddev * np.random.standard_normal((days, nb_simulations)))

    # Simulate the price paths
    s0 = hist_data['Adj Close'][-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = s0
    for t in range(1, days):
        price_list[t] = price_list[t - 1] * daily_returns[t]

    # Calculate probabilities
    # final_in_range = np.logical_and(lower_bound <= price_list[-1],
    #                                 price_list[-1] <= upper_bound).sum() / nb_simulations
    # during_out_of_range = (np.logical_or(price_list < lower_bound,
    #                                      price_list > upper_bound).sum(axis=0) > 0).sum() / nb_simulations
    touch_target_price_list = []
    for target_price in target_price_df:
        if side == 'C':
            touch_target_price = (price_list >= target_price).any(axis=0).sum() / nb_simulations
        if side == 'P':
            touch_target_price = (price_list <= target_price).any(axis=0).sum() / nb_simulations

        touch_target_price_list.append(touch_target_price)


    return touch_target_price_list


def nearest_equal_abs(lst, target):
    return min(lst, key=lambda x: abs(abs(x) - target))


def option_price_calc(current_price, strike_list, close_exp_date, volatility_list, side):

    price_list = []
    for strike, vol in zip(strike_list, volatility_list):
        c = mibian.BS([current_price, strike, 4, close_exp_date], volatility=vol*100)
        if side == 'C':
            price_list.append(c.callPrice)
        if side == 'P':
            price_list.append(c.putPrice)

    return price_list


def get_data_and_calc_long(pool_input):
    KEY = "ckZsUXdiMTZEZVQ3a25TVEFtMm9SeURsQ1RQdk5yWERHS0RXaWNpWVJ2cz0"
    # try:
    start_df, stock_yahoo_long = pool_input
    if int(start_df['IV DIA year']) == 1:
        tick = start_df['Symbol']
        current_price = float(start_df['Current Price'])
        hv = float(start_df['HV 100'])
        print(tick)

        # ----------- get Exp date list  -----------------

        url_exp = f"https://api.marketdata.app/v1/options/expirations/{tick}/?token={KEY}"
        response_exp = requests.request("GET", url_exp).json()
        exp_date_df = pd.DataFrame(response_exp)
        exp_date_df['expirations'] = pd.to_datetime(exp_date_df['expirations'])
        exp_date_df['Days_to_exp'] = (exp_date_df['expirations'] - datetime.datetime.now()).dt.days
        days_to_exp = nearest_equal_abs(exp_date_df['Days_to_exp'], 300)
        needed_exp_date = exp_date_df[exp_date_df['Days_to_exp'] == days_to_exp]['expirations'].reset_index(drop=True).iloc[
            0].date()

        # ----------- Chains -----------------
        url_call = f"https://api.marketdata.app/v1/options/chain/{tick}/?expiration={needed_exp_date}&side=call&token={KEY}"
        response_chains_call = requests.request("GET", url_call).json()
        chains_call = pd.DataFrame(response_chains_call)
        print('chains_call')
        print(chains_call[['strike', 'delta']])
        delta_call = nearest_equal_abs(chains_call['delta'], 0.2)
        needed_chains_call = chains_call[chains_call['delta'] == delta_call]
        print(needed_chains_call[['strike', 'delta']])

        url_put = f"https://api.marketdata.app/v1/options/chain/{tick}/?expiration={needed_exp_date}&side=put&token={KEY}"
        response_chains_put = requests.request("GET", url_put).json()
        chains_put = pd.DataFrame(response_chains_put)
        print('chains_put')
        print(chains_put[['strike', 'delta']])
        delta_put = nearest_equal_abs(chains_put['delta'], 0.2)
        needed_chains_put = chains_put[chains_put['delta'] == delta_put]
        print(needed_chains_put[['strike', 'delta']])

        close_exp_date = int(days_to_exp / 2)
        exp_move = hv * current_price * math.sqrt(close_exp_date / 365)

        price_up = current_price + exp_move
        price_down = current_price - exp_move

        price_up_options_list_call = option_price_calc(price_up, chains_call['strike'].tolist(), days_to_exp,
                                                  chains_call['iv'].tolist(), 'C')
        # price_down_options_list_call = option_price_calc(price_down, chains_call['strike'].tolist(), days_to_exp,
        #                                             chains_call['iv'].tolist(), 'C')

        # price_up_options_list_put = option_price_calc(price_up, chains_call['strike'].tolist(), days_to_exp,
        #                                           chains_call['iv'].tolist(), 'P')
        # price_down_options_list_put = option_price_calc(price_down, chains_put['strike'].tolist(), days_to_exp,
        #                                             chains_put['iv'].tolist(), 'P')

        otm_value = np.where(chains_put['strike'] > current_price, 0, current_price - chains_put['strike'])
        chains_put['Margin'] = np.where((chains_put['strike'] * 0.1) > (current_price * 0.2 - otm_value),
                                    (chains_put['strike'] * 0.1), (current_price * 0.2 - otm_value))


        # Считаем ROC = цена колла с 4 этапа *100/MARGIN

        roc = price_up_options_list_call / chains_put['Margin']

        print('price_up_options_list_call', price_up_options_list_call)
        print(chains_put['Margin'])
        # Считаем вероятность касания за пол срока до экспирации для верхней границы expected move

        pot = calculate_probability_call(stock_yahoo_long[tick], 10000, close_exp_date, 0, 0,     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                                    price_up, 'C')                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        print('pot', pot)

        poe = calculate_probability_put(stock_yahoo_long[tick], 10000, days_to_exp, 0, 0,     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                                    chains_put['strike'], 'P')                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        print('poe', poe)
        pot_poe = pot/poe

        score = pot_poe * roc
        print('score')
        print(score)
        print(score.max())


    else:
        score = np.nan
    # except:
    #     score = 'EMPTY'

    return score


def calculate_probability_call(hist_data, nb_simulations, days, lower_bound, upper_bound, target_price, side):
    # Fetch historical market data
    # hist_data = yf.download(ticker, start='2000-01-01')

    # Calculate the log returns
    log_returns = np.log(1 + hist_data['2000-01-01':]['Adj Close'].pct_change())

    # Define the variables
    u = log_returns.mean()
    var = log_returns.var()

    # Calculate drift and standard deviation
    drift = u - (0.5 * var)
    stddev = log_returns.std()

    # Generate a random variable
    daily_returns = np.exp(drift + stddev * np.random.standard_normal((days, nb_simulations)))

    # Simulate the price paths
    s0 = hist_data['Adj Close'][-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = s0
    for t in range(1, days):
        price_list[t] = price_list[t - 1] * daily_returns[t]

    # Calculate probabilities
    # final_in_range = np.logical_and(lower_bound <= price_list[-1],
    #                                 price_list[-1] <= upper_bound).sum() / nb_simulations
    # during_out_of_range = (np.logical_or(price_list < lower_bound,
    #                                      price_list > upper_bound).sum(axis=0) > 0).sum() / nb_simulations

    if side == 'C':
        touch_target_price = (price_list >= float(target_price)).any(axis=0).sum() / nb_simulations
    if side == 'P':
        touch_target_price = (price_list <= float(target_price)).any(axis=0).sum() / nb_simulations


    return touch_target_price


def calculate_probability_put(hist_data, nb_simulations, days, lower_bound, upper_bound, below_price_list, side):
    # Fetch historical market data
    # hist_data = yf.download(ticker, start='2000-01-01')

    # Calculate the log returns
    log_returns = np.log(1 + hist_data['2000-01-01':]['Adj Close'].pct_change())

    # Define the variables
    u = log_returns.mean()
    var = log_returns.var()

    # Calculate drift and standard deviation
    drift = u - (0.5 * var)
    stddev = log_returns.std()

    # Generate a random variable
    daily_returns = np.exp(drift + stddev * np.random.standard_normal((days, nb_simulations)))

    # Simulate the price paths
    s0 = hist_data['Adj Close'][-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = s0
    for t in range(1, days):
        price_list[t] = price_list[t - 1] * daily_returns[t]

    below_end_price_list = []
    for below_price in below_price_list:
        below_end_price_list.append((price_list[-1] <= below_price).sum() / nb_simulations)

    return below_end_price_list



if __name__ == '__main__':

    KEY = "ckZsUXdiMTZEZVQ3a25TVEFtMm9SeURsQ1RQdk5yWERHS0RXaWNpWVJ2cz0"

    tick = 'GOOGL' # LLY

    stock_yahoo_long = yf.download(tick)

    start_df = pd.read_excel('Active_Stock_Screaner.xlsx')

    start_df = start_df[start_df['Symbol'] == tick].reset_index(drop=True)
    i = 0
    tick = start_df['Symbol'].iloc[i]
    current_price = stock_yahoo_long['Close'].iloc[-1]
    iv = start_df['IV ATM'].iloc[i]
    hv = start_df['HV 100'].iloc[i]

    # ----------- get Exp date list  -----------------

    url_exp = f"https://api.marketdata.app/v1/options/expirations/{tick}/?token={KEY}"
    response_exp = requests.request("GET", url_exp).json()
    exp_date_df = pd.DataFrame(response_exp)
    exp_date_df['expirations'] = pd.to_datetime(exp_date_df['expirations'])
    exp_date_df['Days_to_exp'] = (exp_date_df['expirations'] - datetime.datetime.now()).dt.days
    days_to_exp = nearest_equal_abs(exp_date_df['Days_to_exp'], 300)
    needed_exp_date = exp_date_df[exp_date_df['Days_to_exp'] == days_to_exp]['expirations'].reset_index(drop=True).iloc[
        0].date()

    # ----------- Chains -----------------
    url_call = f"https://api.marketdata.app/v1/options/chain/{tick}/?expiration={needed_exp_date}&side=call&token={KEY}"
    response_chains_call = requests.request("GET", url_call).json()
    chains_call = pd.DataFrame(response_chains_call)
    delta_call = nearest_equal_abs(chains_call['delta'], 0.2)
    needed_chains_call = chains_call[chains_call['delta'] == delta_call]

    url_put = f"https://api.marketdata.app/v1/options/chain/{tick}/?expiration={needed_exp_date}&side=put&token={KEY}"
    response_chains_put = requests.request("GET", url_put).json()
    chains_put = pd.DataFrame(response_chains_put)

    delta_put = nearest_equal_abs(chains_put['delta'], 0.2)
    needed_chains_put = chains_put[chains_put['delta'] == delta_put]

    close_exp_date = int(days_to_exp / 2)
    exp_move = float(hv) * current_price * math.sqrt(close_exp_date / 365)

    price_up = current_price + exp_move
    price_down = current_price - exp_move

    price_up_options_list_call = option_price_calc(price_up, needed_chains_call['strike'].tolist(), close_exp_date,
                                                   needed_chains_call['iv'].tolist(), 'C')
    print('current_price', current_price)
    print('price_up', price_up)
    print('price_up_options_list_call', price_up_options_list_call)
    print('price_up_options_list_call', price_up_options_list_call)

    otm_value = np.where(needed_chains_put['strike'] > current_price, 0, current_price - needed_chains_put['strike'])
    margin = np.where((needed_chains_put['strike'] * 0.1) > (current_price * 0.2 - otm_value),
                                    (needed_chains_put['strike'] * 0.1), (current_price * 0.2 - otm_value))

    # Считаем ROC = цена колла с 4 этапа *100/MARGIN

    roc = price_up_options_list_call / margin

    # Считаем вероятность касания за пол срока до экспирации для верхней границы expected move

    pot = calculate_probability_call(stock_yahoo_long, 10000, close_exp_date, 0, 0,
                                     price_up, 'C')

    poe = calculate_probability_put(stock_yahoo_long, 10000, days_to_exp, 0, 0,
                                    needed_chains_put['strike'], 'P')

    pot_poe = pot / poe
    score = pot_poe * roc

    print('symbol: ', tick)
    print('margin: ', margin)
    print('roc: ', roc)
    print('poe: ', poe)
    print('pot: ', pot)
    print('score: ', score)

    needed_chains_call['expiration'] = pd.to_datetime(needed_chains_call['expiration'], unit='s')
    needed_chains_put['expiration'] = pd.to_datetime(needed_chains_put['expiration'], unit='s')

    print_df = pd.DataFrame({
        'Symbol': [needed_chains_call['underlying'].iloc[0]],
        'CALL Strike': [needed_chains_call['strike'].iloc[0]],
        'PUT Strike': [needed_chains_put['strike'].iloc[0]],
        'EXP_date': [needed_chains_call['expiration'].iloc[0].date()],
    })
    print(print_df)

