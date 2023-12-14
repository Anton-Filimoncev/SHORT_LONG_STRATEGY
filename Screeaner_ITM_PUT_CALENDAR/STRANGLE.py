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

    current_price = start_df['Current Price'].values[0]
    hv = start_df['HV 100'].values[0]
    # ----------- get Exp date list  -----------------

    url_exp = f"https://api.marketdata.app/v1/options/expirations/{tick}/?token={KEY}"
    response_exp = requests.request("GET", url_exp).json()
    exp_date_df = pd.DataFrame(response_exp)
    exp_date_df['expirations'] = pd.to_datetime(exp_date_df['expirations'])
    exp_date_df['Days_to_exp'] = (exp_date_df['expirations'] - datetime.datetime.now()).dt.days
    days_to_exp = nearest_equal_abs(exp_date_df['Days_to_exp'], 300)
    needed_exp_date = exp_date_df[exp_date_df['Days_to_exp'] == days_to_exp]['expirations'].reset_index(drop=True).iloc[0].date()

    # ----------- Chains -----------------
    url = f"https://api.marketdata.app/v1/options/chain/{tick}/?expiration={needed_exp_date}&token={KEY}"
    response_chains = requests.request("GET", url).json()
    chains = pd.DataFrame(response_chains)
    chains['updated'] = pd.to_datetime(chains['updated'], unit='s')
    chains['EXP_date'] = pd.to_datetime(chains['expiration'], unit='s', errors='coerce')
    chains['days_to_exp'] = (chains['EXP_date'] - chains['updated']).dt.days
    #

    put_df = chains[chains['side'] == 'put'].reset_index(drop=True)
    # ограничиваем пут бидом не менее 1 бакса
    put_df = put_df[put_df['bid'] >= 1].reset_index(drop=True)
    atm_put_strike = nearest_equal_abs(put_df['strike'], current_price)
    put_df = put_df[put_df['strike'] <= atm_put_strike].reset_index(drop=True)

    put_df['ATM_strike_volatility'] = put_df[put_df['strike'] == atm_put_strike]['iv'].values[0]
    call_df = chains[chains['side'] == 'call'].reset_index(drop=True)


    put_df = get_BS_prices(current_price, 'P', put_df)
    put_df['Difference'] = put_df['bid'] - put_df['BS_PRICE']
    # print(put_df[['strike', 'bid', 'BS_PRICE', 'Difference']])
    needed_put = put_df[put_df['Difference'] == put_df['Difference'].max()].reset_index(drop=True).iloc[0]

    put_delta = needed_put['delta']
    needed_call_delta = nearest_equal_abs(call_df['delta'], abs(put_delta))
    needed_call = call_df[call_df['delta'] == needed_call_delta].reset_index(drop=True).iloc[0]

    total_prime = needed_call.bid + needed_put.bid

    # put
    otm_value_put = np.where(needed_put['strike'] > current_price, 0, current_price - needed_put['strike'])
    needed_put['Margin PUT'] = np.where((needed_put['strike'] * 0.1) > (current_price * 0.2 - otm_value_put),
                                (needed_put['strike'] * 0.1), (current_price * 0.2 - otm_value_put))

    # call
    otm_value_call = np.where(needed_call['strike'] < current_price, 0, needed_call['strike'] - current_price)
    needed_call['Margin CALL'] = np.where((needed_call['strike'] * 0.1) > (current_price * 0.2 - otm_value_call),
                                        (needed_call['strike'] * 0.1), (current_price * 0.2 - otm_value_call))

    total_margin = np.where((needed_call['Margin CALL']) > (needed_put['Margin PUT']), (needed_call['Margin CALL']+needed_put.bid), (needed_put['Margin PUT']+needed_call.bid))
    return_50 = (total_prime / 2) / total_margin

    # ----------------------------    Считаем 50%POP для стренгла

    close_exp_date = days_to_exp

    # Для каждого страйка считаем вероятность получения 50% прибыли по монте-карло (POP50)

    proba_list = []
    proba_50_list = []
    hist_data = yf.download(tick, start='2000-01-01')['Adj Close']
    # for strike, iv, bid in zip(chains['strike'], chains['iv'], chains['bid']):
    r = 0.04
    N_sims = 1000
    dt =1

    #  Call our stock modeling code.  Returns a matrix (N_sims, N_days) in shape
    S = stock_monte_carlo(hist_data, current_price, close_exp_date, N_sims, dt, reshape=True)

    #     P = np.array(put_price(S, K_put, r, range(N_days+1), sigma_put))
    P = option_price_batch(needed_put.iv, np.array([np.arange(close_exp_date + 2)[::-1][:-1]] * N_sims), S, needed_put.strike,
                           r, 'P')

    C = option_price_batch(needed_call.iv, np.array([np.arange(close_exp_date + 2)[::-1][:-1]] * N_sims), S, needed_call.strike,
                           r, 'C')

    total_price = C + P

    init_price = total_price[0][0]
    half_max = init_price / 2

    #  Create a matrix with booleans if the cutoff has been reached
    reached_half_max = total_price[:, -1] <= half_max

    #  Do a column-wise sum.  If we reached 50% during any time during the trade.  The sum will be at least one if not greater.
    # reached_half_max = np.sum(reached_half_max, axis = 1)

    #  Calulcate the probability by summing the number of winners and dividing by the total number of simulations
    prob = np.sum(reached_half_max > 0) / N_sims
    # Получаем итоговую таблицу RPOP = 50%RETURN *50%POP

    # print('needed_call')
    # print(needed_call)
    # print('needed_put')
    # print(needed_put)
    print_df = pd.DataFrame({
        'Symbol': [needed_call['underlying']],
        'Call Strike': [needed_call['strike']],
        'Put Strike': [needed_put['strike']],
        'EXP_date': [needed_put['EXP_date'].date()],
    })
    print(print_df)
    # print(return_50 * prob)



