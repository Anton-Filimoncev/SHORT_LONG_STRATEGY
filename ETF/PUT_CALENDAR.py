import pandas as pd
import numpy as np
import os
import time
import pickle
import gspread as gd
import scipy
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
from math import log, e
from scipy.stats import norm
pd.options.mode.chained_assignment = None
pd.set_option('display.max_columns', None)


def nearest_equal_abs(lst, target):
    return min(lst, key=lambda x: abs(abs(x) - target))


def option_price_calc(current_price, strike_list, close_exp_date, volatility_list, side):

    price_list = []
    for strike, vol in zip(strike_list, volatility_list):
        c = mibian.BS([current_price, strike, 4, close_exp_date], volatility=vol*100)
        if side == 'C':
            price_list.append(c.callPrice)
        else:
            price_list.append(c.putPrice)

    return price_list

def option_price_batch(volatility, daysToExpiration, underlyingPrice, strikePrice, interestRate, side):
#     interestRate = interestRate /100
    daysToExpiration = daysToExpiration /365
#     volatility = volatility / 100
    a = volatility * daysToExpiration**0.5
    d1 = (np.log(underlyingPrice / strikePrice) + (interestRate + (volatility**2) / 2) * daysToExpiration) / a
    d2 = d1 - a
    if side == 'P':
        price = strikePrice * e**(-interestRate * daysToExpiration) * norm.cdf(-d2) - underlyingPrice * norm.cdf(-d1)
    if side == 'C':
        price = underlyingPrice * norm.cdf(d1) - strikePrice * e** (-interestRate * daysToExpiration) * norm.cdf(d2)
    return price


def calculate_probability(ticker, nb_simulations, days, lower_bound, upper_bound, target_price_df, side):
    # Fetch historical market data
    hist_data = yf.download(ticker, start='2000-01-01')

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

def get_atm_strikes(chains_df, current_price):
    chains_df['ATM_strike_volatility'] = np.nan
    for exp in chains_df['EXP_date'].unique():
        solo_exp_df = chains_df[chains_df['EXP_date'] == exp]
        atm_strike = nearest_equal(solo_exp_df['strike'].tolist(), current_price)
        atm_put_volatility = solo_exp_df[solo_exp_df['strike'] == atm_strike]['iv'].reset_index(drop=True).iloc[0]
        chains_df.loc[chains_df['EXP_date'] == exp, "ATM_strike_volatility"] = atm_put_volatility

    return chains_df

def stock_monte_carlo(hist, init_price, N_days, N_sims, dt, reshape=True):
    # Compute daily returns
    daily_returns = np.log(1 + hist.pct_change())

    # Compute mean (r)
    u = daily_returns.mean()
    var = daily_returns.var()
    r = u - (0.5 * var)

    # Compute standard deviation (sigma)
    sigma = daily_returns.std()

    # Rest of the code (as above)
    epsilon = np.random.normal(size=(N_sims * N_days + N_sims - 1))
    ds_s = r * dt + sigma * epsilon
    ones = -np.ones((N_sims * N_days + N_sims))
    ones[0:-1:N_days + 1] = 1
    ds_s[N_days:N_days * N_sims + N_sims:N_days + 1] = -1
    d = [ds_s + 1, ones]
    K = [-1, 0]
    M = scipy.sparse.diags(d, K, format='csc')
    p = np.zeros((N_sims * N_days + N_sims, 1))
    p[0:-1:N_days + 1] = init_price
    s = scipy.sparse.linalg.spsolve(M, p)

    if reshape == True:
        s = np.reshape(s, (N_sims, N_days + 1))

    return s


def get_abs_return(price_array, type_option, days_to_exp, days_to_exp_short, history_vol, current_price, strike, prime, vol_opt):
    put_price_list = []
    call_price_list = []
    proba_list = []
    price_gen_list = []

    for stock_price_num in range(len(price_array)):
        try:
            P_below = stats.norm.cdf(
                (np.log(price_array[stock_price_num] / current_price) / (
                        history_vol * math.sqrt(days_to_exp_short / 365))))
            P_current = stats.norm.cdf(
                (np.log(price_array[stock_price_num + 1] / current_price) / (
                        history_vol * math.sqrt(days_to_exp_short / 365))))
            proba_list.append(P_current - P_below)
            if type_option == 'Short':
                c = mibian.BS([price_array[stock_price_num + 1], strike, 4, 1], volatility=vol_opt * 100)
            if type_option == 'Long':
                c = mibian.BS([price_array[stock_price_num + 1], strike, 4, days_to_exp], volatility=vol_opt * 100)

            put_price_list.append(c.putPrice)
            call_price_list.append(c.callPrice)
            price_gen_list.append(price_array[stock_price_num + 1])
        except:
            pass

    put_df = pd.DataFrame({
        'gen_price': price_gen_list,
        'put_price': put_price_list,
        'call_price': call_price_list,
        'proba': proba_list,
    })

    put_df['return'] = (put_df['put_price'] - prime)

    if type_option == 'Short':
        return ((prime - put_df['put_price']) * put_df['proba']).sum()

    if type_option == 'Long':
        return ((put_df['put_price'] - prime) * put_df['proba']).sum()

def expected_return_calc(vol_put_short, vol_put_long, current_price, history_vol, days_to_exp_short, days_to_exp_long, strike_put_long, strike_put_short, prime_put_long, prime_put_short):

    # print('expected_return CALCULATION ...')

    price_array = np.arange(current_price - current_price / 2, current_price + current_price, 0.2)
    # print('price_array', price_array)
    short_finish = get_abs_return(price_array, 'Short', days_to_exp_short, days_to_exp_short, history_vol, current_price, strike_put_short,
                                prime_put_short,
                                vol_put_short)


    long_finish = get_abs_return(price_array, 'Long', days_to_exp_long, days_to_exp_short, history_vol, current_price, strike_put_long,
                                 prime_put_long,
                                 vol_put_long)

    expected_return = (short_finish + long_finish) * 100

    return expected_return

def get_BS_prices(current_price, type_option, option_chains_short_FULL):
    price_gen_list = []

    for i in range(len(option_chains_short_FULL)):
        try:
            strike = option_chains_short_FULL['strike'].iloc[i]
            dte = option_chains_short_FULL['days_to_exp'].iloc[i]
            atm_IV = option_chains_short_FULL['ATM_strike_volatility'].iloc[i]

            c = mibian.BS([current_price, strike, 4, dte], volatility=atm_IV * 100)
            if type_option == 'P':
                price_gen_list.append(c.putPrice)
            if type_option == 'C':
                price_gen_list.append(c.callPrice)
        except Exception as e:
            print(e)
            pass

    option_chains_short_FULL['BS_PRICE'] = price_gen_list
    return option_chains_short_FULL

if __name__ == '__main__':
    KEY = "ckZsUXdiMTZEZVQ3a25TVEFtMm9SeURsQ1RQdk5yWERHS0RXaWNpWVJ2cz0"

    tick = 'MSFT' # LLY

    start_df = pd.read_excel('Active_Stock_Screaner.xlsx')
    start_df = start_df[start_df['Symbol'] == tick].reset_index(drop=True)

    current_price = float(start_df['Current Price'])
    hv = float(start_df['HV 100'])
    print(tick)

    # ----------- get Exp date SELL  -----------------

    url_exp = f"https://api.marketdata.app/v1/options/expirations/{tick}/?token={KEY}"
    response_exp = requests.request("GET", url_exp).json()
    print(response_exp)
    exp_date_df = pd.DataFrame(response_exp)
    exp_date_df['expirations'] = pd.to_datetime(exp_date_df['expirations'])
    exp_date_df['Days_to_exp'] = (exp_date_df['expirations'] - datetime.datetime.now()).dt.days
    days_to_exp = nearest_equal_abs(exp_date_df['Days_to_exp'], 300)
    needed_exp_date = \
        exp_date_df[exp_date_df['Days_to_exp'] == days_to_exp]['expirations'].reset_index(drop=True).iloc[
            0].date()

    # ----------- get Exp date BUY  -----------------

    url_exp = f"https://api.marketdata.app/v1/options/expirations/{tick}/?token={KEY}"
    response_exp = requests.request("GET", url_exp).json()
    exp_date_df = pd.DataFrame(response_exp)
    exp_date_df['expirations'] = pd.to_datetime(exp_date_df['expirations'])
    exp_date_df['Days_to_exp'] = (exp_date_df['expirations'] - datetime.datetime.now()).dt.days
    # days_to_exp = nearest_equal_abs(exp_date_df['Days_to_exp'], 300)
    exp_dates_short = exp_date_df[exp_date_df['Days_to_exp'] >= 20]
    exp_dates_short = exp_dates_short[exp_dates_short['Days_to_exp'] <= 90]

    all_needed_exp_df_sell = pd.DataFrame()
    for exp_date_shortus in exp_dates_short['expirations']:
        # ----------- Chains -----------------
        url = f"https://api.marketdata.app/v1/options/chain/{tick}/?expiration={exp_date_shortus}&side=put&token={KEY}"
        response_chains = requests.request("GET", url).json()
        chains = pd.DataFrame(response_chains)
        chains['expiration'] = pd.to_datetime(chains['expiration'], unit='s')
        chains['Days_to_exp'] = (chains['expiration'] - datetime.datetime.now()).dt.days
        # chains = chains[chains['strike'] < current_price * 1.20]
        # chains = chains[chains['strike'] > current_price * 0.8].reset_index(drop=True)
        all_needed_exp_df_sell = pd.concat([all_needed_exp_df_sell, chains])

    exp_move = 0.5 * hv * current_price * math.sqrt(180 / 365)

    # print('exp_move')
    # print(exp_move)
    # print(current_price-exp_move)

    needed_strike_sell = nearest_equal_abs(all_needed_exp_df_sell['strike'], current_price - exp_move)

    all_needed_exp_df_sell = all_needed_exp_df_sell[all_needed_exp_df_sell['strike'] == needed_strike_sell]
    needed_short = \
    all_needed_exp_df_sell[all_needed_exp_df_sell['iv'] == all_needed_exp_df_sell['iv'].max()].iloc[0]

    # print('needed_short')
    # print(needed_short)

    exp_dates_long = exp_date_df[exp_date_df['Days_to_exp'] >= needed_short['Days_to_exp'] + 30]
    exp_dates_long = exp_dates_long[exp_dates_long['Days_to_exp'] <= 200]

    all_needed_exp_df_buy = pd.DataFrame()
    for exp_date_longus in exp_dates_long['expirations']:
        # ----------- Chains -----------------
        url = f"https://api.marketdata.app/v1/options/chain/{tick}/?expiration={exp_date_longus}&side=put&token={KEY}"
        response_chains = requests.request("GET", url).json()
        chains = pd.DataFrame(response_chains)
        chains['expiration'] = pd.to_datetime(chains['expiration'], unit='s')
        chains['Days_to_exp'] = (chains['expiration'] - datetime.datetime.now()).dt.days
        # chains = chains[chains['strike'] < current_price * 1.20]
        # chains = chains[chains['strike'] > current_price * 0.8].reset_index(drop=True)
        all_needed_exp_df_buy = pd.concat([all_needed_exp_df_buy, chains])

    all_needed_exp_df_buy = all_needed_exp_df_buy[all_needed_exp_df_buy['strike'] == needed_strike_sell]
    needed_long = all_needed_exp_df_buy[all_needed_exp_df_buy['iv'] == all_needed_exp_df_buy['iv'].min()].iloc[
        0]

    # print('needed_long')
    # print(needed_long)

    debet = needed_long['ask'] - needed_short['bid']

    # Считаем expected return позиции на последний день экспирации шорта
    vol_put_short = needed_short['iv']
    vol_put_long = needed_long['iv']
    days_to_exp_short = needed_short['Days_to_exp']
    days_to_exp_long = needed_long['Days_to_exp'] - needed_short['Days_to_exp']
    strike_put_short = needed_short['strike']
    strike_put_long = needed_long['strike']
    prime_put_short = needed_short['bid']
    prime_put_long = needed_long['ask']
    print('current_price', current_price)
    print('hv', hv)
    print('vol_put_short', vol_put_short)
    print('vol_put_long', vol_put_long)
    print('days_to_exp_long', days_to_exp_long)
    print('strike_put_short', strike_put_short)
    print('strike_put_long', strike_put_long)
    print('prime_put_short', prime_put_short)
    print('prime_put_long', prime_put_long)

    expected_return = expected_return_calc(vol_put_short, vol_put_long, current_price, hv, days_to_exp_short,
                                           days_to_exp_long, strike_put_long, strike_put_short, prime_put_long,
                                           prime_put_short)

    # Считаем итоговый score (ожидаемый ROC в годовом формате) = (expected return/(debet*100)/DTE short * 365

    caledar_put_score = (expected_return / (debet * 100)) / needed_short['Days_to_exp'] * 365

    print('caledar_put_score', caledar_put_score)

    print_df = pd.DataFrame({
        'Symbol': [needed_short['underlying']],
        'Short Strike': [needed_short['strike']],
        'Long Strike': [needed_long['strike']],
        'EXP_date Short': [needed_short['expiration'].date()],
        'EXP_date Long': [needed_long['expiration'].date()],
        'Expected_Return': [expected_return],
    })
    print(print_df)
    # print(return_50 * prob)



