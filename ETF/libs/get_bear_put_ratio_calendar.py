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
from multiprocessing import Pool
pd.options.mode.chained_assignment = None
from .MARKET_DATA import *
from .popoption.PutCalendar_template import putCalendar_template
from .popoption.CallCalendar_template import callCalendar_template


def nearest_equal_abs(lst, target):
    return min(lst, key=lambda x: abs(abs(x) - target))


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

    return hist_vol


def get_exp_move(tick, stock_yahoo):
    print('---------------------------')
    print('------------- Getting HV --------------')
    print('---------------------------')

    stock_yahoo_solo = stock_yahoo['Close']
    hist_vol = volatility_calc(stock_yahoo_solo)

    return hist_vol

def get_yahoo_price(ticker):
    yahoo_data = yf.download(ticker, progress=False)['2018-01-01':]
    return yahoo_data

def get_calendar_diagonal(tick, rate, days_to_expiration_long, days_to_expiration_short, closing_days_array,
                          percentage_array, position_type, quotes_short, quotes_long, short_count, long_count):
    yahoo_stock = get_yahoo_price(tick)
    underlying = yahoo_stock['Close'].iloc[-1]
    trials = 2000

    sum_df = pd.DataFrame()

    hv = get_exp_move(tick, yahoo_stock)

    if position_type == 'put':
        quotes_short = quotes_short[quotes_short['side'] == 'put'].reset_index(drop=True)
        quotes_short = quotes_short[quotes_short['strike'] <= underlying * 1].reset_index(drop=True)
        quotes_short = quotes_short[quotes_short['strike'] >= underlying * 0.93].reset_index(drop=True)

        quotes_long = quotes_long[quotes_long['side'] == 'put'].reset_index(drop=True)
        quotes_long = quotes_long[quotes_long['strike'] <= underlying * 1].reset_index(drop=True)
        quotes_long = quotes_long[quotes_long['strike'] >= underlying * 0.93].reset_index(drop=True)
        for num, quotes_short_row in quotes_short.iterrows():
            for num_long, quotes_long_row in quotes_long.iterrows():
                sigma_short = quotes_short_row['iv'] * 100
                short_strike = quotes_short_row['strike']
                short_price = quotes_short_row['bid'] * short_count
                sigma_long = quotes_long_row['iv'] * 100
                long_strike = quotes_long_row['strike']
                long_price = quotes_long_row['ask'] * long_count

                if quotes_short_row['strike'] == quotes_long_row['strike']:
                    calendar_diagonal_data, max_profit, percentage_type = putCalendar_template(underlying, sigma_short, sigma_long, rate, trials,
                                                         days_to_expiration_short, days_to_expiration_long,
                                                         [closing_days_array],
                                                         [percentage_array], long_strike, long_price, short_strike,
                                                         short_price, yahoo_stock, short_count, long_count)
                    # print('calendar_diagonal_data', calendar_diagonal_data)
                    calendar_diagonal_data = pd.DataFrame(calendar_diagonal_data)
                    calendar_diagonal_data['Strike_Short'] = [short_strike]
                    calendar_diagonal_data['Strike_Long'] = [long_strike]
                    sum_df = pd.concat([sum_df, calendar_diagonal_data])


    if position_type == 'call':
        quotes_short = quotes_short[quotes_short['side'] == 'call'].reset_index(drop=True)
        quotes_short = quotes_short[quotes_short['strike'] <= underlying * 1.05].reset_index(drop=True)
        quotes_short = quotes_short[quotes_short['strike'] >= underlying * 1].reset_index(drop=True)

        quotes_long = quotes_long[quotes_long['side'] == 'call'].reset_index(drop=True)
        quotes_long = quotes_long[quotes_long['strike'] <= underlying * 1.05].reset_index(drop=True)
        quotes_long = quotes_long[quotes_long['strike'] >= underlying * 1].reset_index(drop=True)

        # if position_options['structure'] == 'calendar':
        #     quotes_long =

        for num, quotes_short_row in quotes_short.iterrows():
            for num_long, quotes_long_row in quotes_long.iterrows():
                sigma_short = quotes_short_row['iv'] * 100
                short_strike = quotes_short_row['strike']
                short_price = quotes_short_row['bid'] * short_count

                sigma_long = quotes_long_row['iv'] * 100
                long_strike = quotes_long_row['strike']
                long_price = quotes_long_row['ask'] * long_count

                # print('short bid', quotes_short_row['bid'])
                # print('short_price', short_price)
                # print('long ask', quotes_long_row['ask'])
                # print('long_price', long_price)
                # print('long_strike', long_strike)
                # print('short_strike', short_strike)
                # print('long_strike', sigma_long)
                # print('short_strike', sigma_short)
                # print('days_to_expiration_short', days_to_expiration_short)
                # print('days_to_expiration_long', days_to_expiration_long)

                if quotes_short_row['strike'] == quotes_long_row['strike']:
                    calendar_diagonal_data, max_profit, percentage_type = callCalendar_template(underlying, sigma_short, sigma_long, rate, trials,
                                                         days_to_expiration_short, days_to_expiration_long,
                                                         [closing_days_array],
                                                         [percentage_array], long_strike, long_price, short_strike,
                                                         short_price, yahoo_stock,
                                                         short_count, long_count)

                    calendar_diagonal_data = pd.DataFrame(calendar_diagonal_data)
                    calendar_diagonal_data['Strike_Short'] = [short_strike]
                    calendar_diagonal_data['Strike_Long'] = [long_strike]
                    sum_df = pd.concat([sum_df, calendar_diagonal_data])


    nearest_atm_strike = nearest_equal_abs(quotes_short['strike'].astype('float'), underlying)
    iv = quotes_short[quotes_short['strike'] == nearest_atm_strike]['iv'].values.tolist()[0]
    print('nearest_strike', nearest_atm_strike)
    print('current_iv', iv)
    print('sum_df')
    print(sum_df)
    sum_df['top_score'] = sum_df['pop'] * sum_df['exp_return']
    best_df = sum_df[sum_df['top_score'] == sum_df['top_score'].max()]
    exp_move_hv = hv * underlying * math.sqrt(days_to_expiration_short / 365)
    exp_move_iv = iv * underlying * math.sqrt(days_to_expiration_short / 365)

    return sum_df, best_df, exp_move_hv, exp_move_iv, max_profit, percentage_type

def get_data_and_calc(pool_input):
    KEY = "ckZsUXdiMTZEZVQ3a25TVEFtMm9SeURsQ1RQdk5yWERHS0RXaWNpWVJ2cz0"
    # try:
    start_df, stock_yahoo_short, RISK_RATE, side = pool_input
    # if int(start_df['IV DIA year']) == 1:
    ticker = start_df['Symbol']
    current_price = float(start_df['Current Price'])
    print(ticker)

    nearest_dte_short = 30
    nearest_dte_long = 90
    percentage_array = 10
    short_count = 2
    long_count = 1


    needed_exp_date_short, dte_short, exp_df_short = hedginglab_get_exp_date(ticker, nearest_dte_short)
    needed_exp_date_long, dte_long, exp_df_long = hedginglab_get_exp_date(ticker, nearest_dte_long)

    # ----------------------------------   get skewness optimal DTE SHORT
    exp_date_df = pd.DataFrame()
    exp_date_df['expirations'] = exp_df_short.values
    exp_date_df['expirations'] = pd.to_datetime(exp_date_df['expirations'])
    exp_date_df['Days_to_exp'] = (exp_date_df['expirations'] - datetime.datetime.now()).dt.days
    # exp_df['Days_to_exp'] = (exp_df['expirations'] - datetime.datetime.now()).dt.days
    print('exp_date_df')
    print(exp_date_df)
    exp_dates_short = exp_date_df[exp_date_df['Days_to_exp'] >= 20]
    exp_dates_short = exp_dates_short[exp_dates_short['Days_to_exp'] < 90]

    all_needed_exp_df_sell = pd.DataFrame()
    for exp_date_shortus in exp_dates_short['expirations']:
        # ----------- Chains -----------------
        url = f"https://api.marketdata.app/v1/options/chain/{ticker}/?expiration={exp_date_shortus}&side={side}&token={KEY}"
        print(url)
        response_chains = requests.request("GET", url).json()
        chains = pd.DataFrame(response_chains)
        chains['expiration'] = pd.to_datetime(chains['expiration'], unit='s')
        chains['Days_to_exp'] = (chains['expiration'] - datetime.datetime.now()).dt.days
        # chains = chains[chains['strike'] < current_price * 1.20]
        # chains = chains[chains['strike'] > current_price * 0.8].reset_index(drop=True)
        all_needed_exp_df_sell = pd.concat([all_needed_exp_df_sell, chains])

    needed_strike_sell = nearest_equal_abs(all_needed_exp_df_sell['strike'], current_price)
    all_needed_exp_df_sell = all_needed_exp_df_sell[all_needed_exp_df_sell['strike'] == needed_strike_sell]
    needed_short = all_needed_exp_df_sell[all_needed_exp_df_sell['iv'] == all_needed_exp_df_sell['iv'].max()].iloc[0]

    # ----------------------------------   get skewness optimal DTE LONG
    exp_date_df = pd.DataFrame()
    exp_date_df['expirations'] = exp_df_long.values
    exp_date_df['expirations'] = pd.to_datetime(exp_date_df['expirations'])
    exp_date_df['Days_to_exp'] = (exp_date_df['expirations'] - datetime.datetime.now()).dt.days
    # exp_df['Days_to_exp'] = (exp_df['expirations'] - datetime.datetime.now()).dt.days
    print('exp_date_df')
    print(exp_date_df)
    exp_dates_long = exp_date_df[exp_date_df['Days_to_exp'] >= needed_short['Days_to_exp']+30]
    exp_dates_long = exp_dates_long[exp_dates_long['Days_to_exp'] <= 200]

    all_needed_exp_df_long = pd.DataFrame()
    for exp_date_longus in exp_dates_long['expirations']:
        # ----------- Chains -----------------
        url = f"https://api.marketdata.app/v1/options/chain/{ticker}/?expiration={exp_date_longus}&side={side}&token={KEY}"
        response_chains = requests.request("GET", url).json()
        chains = pd.DataFrame(response_chains)
        chains['expiration'] = pd.to_datetime(chains['expiration'], unit='s')
        chains['Days_to_exp'] = (chains['expiration'] - datetime.datetime.now()).dt.days
        # chains = chains[chains['strike'] < current_price * 1.20]
        # chains = chains[chains['strike'] > current_price * 0.8].reset_index(drop=True)
        all_needed_exp_df_long = pd.concat([all_needed_exp_df_long, chains])

    needed_strike_long = nearest_equal_abs(all_needed_exp_df_long['strike'], current_price)
    all_needed_exp_df_long = all_needed_exp_df_long[all_needed_exp_df_long['strike'] == needed_strike_long]
    needed_long = all_needed_exp_df_long[all_needed_exp_df_long['iv'] == all_needed_exp_df_long['iv'].max()].iloc[
        0]

    print('needed_long')
    print(needed_short['Days_to_exp'])
    print('needed_long')
    print(needed_long['Days_to_exp'])

    quotes_short = hedginglab_get_quotes(ticker, needed_short['Days_to_exp'])
    quotes_long = hedginglab_get_quotes(ticker, needed_long['Days_to_exp'])


    strengle_data, best_df, exp_move_hv, exp_move_iv, profit_for_percent, percentage_type = get_calendar_diagonal(
        ticker, RISK_RATE,
        dte_long,
        dte_short,
        dte_short,
        percentage_array, side,
        quotes_short, quotes_long,
        short_count, long_count)

    print('strengle_data', strengle_data)
    print('best_df', best_df)
    print('exp_move_hv', exp_move_hv)
    print('exp_move_iv', exp_move_iv)
    print('profit_for_percent', profit_for_percent)
    print('percentage_type', percentage_type)

    caledar_put_score =best_df.iloc[0]['top_score']
# else:
#     caledar_put_score = np.nan
#     except Exception as err:
#         caledar_put_score = 'EMPTY'

    return caledar_put_score


def bear_put_ration_calendar_run(active_stock_df, stock_yahoo, poll_num, RISK_RATE, side):
    print('---------------------------')
    print('------------- Getting Bear Calendar ... --------------')
    print('---------------------------')

    with Pool(poll_num) as p:
        bear_cal_out = p.map(get_data_and_calc, [(active_stock_df.iloc[i], stock_yahoo, RISK_RATE, side) for i in range(len(active_stock_df))])
    print('----------bear_cal_out-----------------')
    print(bear_cal_out)
    return bear_cal_out
