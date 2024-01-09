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
from .popoption.PutDebitSpread import putDebitSpread
pd.options.mode.chained_assignment = None


def nearest_equal_abs(lst, target):
    return min(lst, key=lambda x: abs(abs(x) - target))


def option_price_calc(current_price, strike_list, close_exp_date, volatility_list, side, RISK_RATE):

    price_list = []
    for strike, vol in zip(strike_list, volatility_list):
        c = mibian.BS([current_price, strike, RISK_RATE, close_exp_date], volatility=vol*100)
        if side == 'C':
            price_list.append(c.callPrice)
        else:
            price_list.append(c.putPrice)

    return price_list


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


def get_abs_return(price_array, type_option, days_to_exp, days_to_exp_short, history_vol, current_price, strike, prime, vol_opt, RISK_RATE):
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
                c = mibian.BS([price_array[stock_price_num + 1], strike, RISK_RATE, 1], volatility=vol_opt * 100)
            if type_option == 'Long':
                c = mibian.BS([price_array[stock_price_num + 1], strike, RISK_RATE, 1], volatility=vol_opt * 100)

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

def expected_return_calc(vol_put_short, vol_put_long, current_price, history_vol, days_to_exp_short, days_to_exp_long, strike_put_long, strike_put_short, prime_put_long, prime_put_short, RISK_RATE):

    # print('expected_return CALCULATION ...')

    price_array = np.arange(current_price - current_price / 2, current_price + current_price, 0.2)
    # print('price_array', price_array)
    short_finish = get_abs_return(price_array, 'Short', days_to_exp_short, days_to_exp_short, history_vol, current_price, strike_put_short,
                                prime_put_short,
                                vol_put_short, RISK_RATE)

    long_finish = get_abs_return(price_array, 'Long', days_to_exp_long, days_to_exp_short, history_vol, current_price, strike_put_long,
                                 prime_put_long,
                                 vol_put_long, RISK_RATE)

    expected_return = (short_finish + long_finish) * 100

    return expected_return

def get_data_and_calc_long(pool_input):
    KEY = "ckZsUXdiMTZEZVQ3a25TVEFtMm9SeURsQ1RQdk5yWERHS0RXaWNpWVJ2cz0"
    try:
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

            delta_call = nearest_equal_abs(chains_call['delta'], 0.2)
            needed_chains_call = chains_call[chains_call['delta'] == delta_call]

            url_put = f"https://api.marketdata.app/v1/options/chain/{tick}/?expiration={needed_exp_date}&side=put&token={KEY}"
            response_chains_put = requests.request("GET", url_put).json()
            chains_put = pd.DataFrame(response_chains_put)

            delta_put = nearest_equal_abs(chains_put['delta'], 0.2)
            needed_chains_put = chains_put[chains_put['delta'] == delta_put]
            close_exp_date = int(days_to_exp / 2)
            exp_move = hv * current_price * math.sqrt(close_exp_date / 365)

            price_up = current_price + exp_move

            price_up_options_list_call = option_price_calc(price_up, needed_chains_call['strike'].tolist(), days_to_exp,
                                                      needed_chains_call['iv'].tolist(), 'C')[0]
            # price_down_options_list_call = option_price_calc(price_down, chains_call['strike'].tolist(), days_to_exp,
            #                                             chains_call['iv'].tolist(), 'C')

            # price_up_options_list_put = option_price_calc(price_up, chains_call['strike'].tolist(), days_to_exp,
            #                                           chains_call['iv'].tolist(), 'P')
            # price_down_options_list_put = option_price_calc(price_down, chains_put['strike'].tolist(), days_to_exp,
            #                                             chains_put['iv'].tolist(), 'P')

            otm_value = np.where(needed_chains_put['strike'] > current_price, 0, current_price - needed_chains_put['strike'])
            margin = np.where((needed_chains_put['strike'] * 0.1) > (current_price * 0.2 - otm_value),
                                        (needed_chains_put['strike'] * 0.1), (current_price * 0.2 - otm_value))[0]


            # Считаем ROC = цена колла с 4 этапа *100/MARGIN

            roc = price_up_options_list_call / margin

            # Считаем вероятность касания за пол срока до экспирации для верхней границы expected move

            pot = calculate_probability_call(stock_yahoo_long[tick], 10000, close_exp_date, 0, 0,     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                                        price_up, 'C')                    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            print('pot', pot)

            poe = calculate_probability_put(stock_yahoo_long[tick], 10000, days_to_exp, 0, 0,     # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                                                        needed_chains_put['strike'], 'P')[0]                # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            print('poe', poe)
            pot_poe = pot/poe

            score = pot_poe * roc
            print('score')
            print(score)

        else:
            score = np.nan
    except:
        score = 'EMPTY'

    return score

def get_proba_debit_spread(current_price, yahoo_data, put_long_strike, put_long_price, put_short_strike, put_short_price,
                          sigma_short, sigma_long, days_to_expiration_short, days_to_expiration_long, risk_rate):
    closing_days_array = [days_to_expiration_short]
    percentage_array = [30]
    trials = 3000

    proba_50 = putDebitSpread(current_price, sigma_short, sigma_long, risk_rate, trials, days_to_expiration_short,
                days_to_expiration_long, closing_days_array, percentage_array, put_long_strike,
                put_long_price, put_short_strike, put_short_price, yahoo_data)

    return proba_50

def get_data_and_calc_debit_spread(pool_input):
    KEY = "ckZsUXdiMTZEZVQ3a25TVEFtMm9SeURsQ1RQdk5yWERHS0RXaWNpWVJ2cz0"
    try:
        start_df, stock_yahoo_short, RISK_RATE = pool_input

        tick = start_df['Symbol']
        hv = float(start_df['HV 100'])
        print(tick)

        # ----------- get Exp date SELL  -----------------

        url_exp = f"https://api.marketdata.app/v1/options/expirations/{tick}/?token={KEY}"
        response_exp = requests.request("GET", url_exp).json()
        exp_date_df = pd.DataFrame(response_exp)
        exp_date_df['expirations'] = pd.to_datetime(exp_date_df['expirations'])
        exp_date_df['Days_to_exp'] = (exp_date_df['expirations'] - datetime.datetime.now()).dt.days
        days_to_exp_short = nearest_equal_abs(exp_date_df['Days_to_exp'], 40)
        needed_exp_date_short = \
        exp_date_df[exp_date_df['Days_to_exp'] == days_to_exp_short]['expirations'].reset_index(drop=True).iloc[0].date()

        # ----------- get Exp date BUY  -----------------

        url_exp = f"https://api.marketdata.app/v1/options/expirations/{tick}/?token={KEY}"
        response_exp = requests.request("GET", url_exp).json()
        exp_date_df = pd.DataFrame(response_exp)
        exp_date_df['expirations'] = pd.to_datetime(exp_date_df['expirations'])
        exp_date_df['Days_to_exp'] = (exp_date_df['expirations'] - datetime.datetime.now()).dt.days
        days_to_exp_long = nearest_equal_abs(exp_date_df['Days_to_exp'], 40)
        needed_exp_date_long = \
        exp_date_df[exp_date_df['Days_to_exp'] == days_to_exp_long]['expirations'].reset_index(drop=True).iloc[0].date()

        # ----------- Chains -----------------
        url = f"https://api.marketdata.app/v1/options/chain/{tick}/?expiration={needed_exp_date_short}&side=put&token={KEY}"
        response_chains = requests.request("GET", url).json()
        chains_short = pd.DataFrame(response_chains)
        chains_short['expiration'] = pd.to_datetime(chains_short['expiration'], unit='s')
        chains_short['Days_to_exp'] = (chains_short['expiration'] - datetime.datetime.now()).dt.days

        current_price = chains_short['underlyingPrice'].iloc[0]
        # exp_move = hv * current_price * math.sqrt(180 / 365)

        print(chains_short['delta'])

        needed_delta_sell = nearest_equal_abs(chains_short['delta'].dropna(), 0.3)
        print('needed_delta_sell', needed_delta_sell)
        print('current_price', current_price)
        needed_short = chains_short[chains_short['delta'] == needed_delta_sell].reset_index(drop=True).iloc[0]
        # needed_short = all_needed_exp_df_sell[all_needed_exp_df_sell['iv'] == all_needed_exp_df_sell['iv'].max()].iloc[0]


        # ----------- Chains -----------------
        url = f"https://api.marketdata.app/v1/options/chain/{tick}/?expiration={needed_exp_date_long}&side=put&token={KEY}"
        response_chains = requests.request("GET", url).json()
        chains_long = pd.DataFrame(response_chains)
        chains_long['expiration'] = pd.to_datetime(chains_long['expiration'], unit='s')
        chains_long['Days_to_exp'] = (chains_long['expiration'] - datetime.datetime.now()).dt.days

        needed_delta_buy = nearest_equal_abs(chains_long['delta'].dropna(), 0.6)
        needed_long = chains_long[chains_long['delta'] == needed_delta_buy].reset_index(drop=True).iloc[0]

        #  300 и 600 дней
        # print(needed_long)

        debit = needed_long['ask'] - needed_short['bid']

        max_profit = needed_long['strike'] - needed_short['strike'] - debit

        RR = max_profit/debit

        # Считаем expected return позиции на последний день экспирации шорта
        vol_put_short = needed_short['iv']
        vol_put_long = needed_long['iv']
        days_to_exp_short = needed_short['Days_to_exp']
        days_to_exp_long = needed_long['Days_to_exp']
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

        print('---------- needed_short -----------------')
        print(needed_short)
        print('---------- needed_long -----------------')
        print(needed_long)

        expected_return = expected_return_calc(vol_put_short, vol_put_long, current_price, hv, days_to_exp_short, days_to_exp_long, strike_put_long, strike_put_short, prime_put_long, prime_put_short, RISK_RATE)

        print('current_price', current_price, type(current_price))
        print('stock_yahoo_short', stock_yahoo_short[tick])
        print('needed_long strike',
              needed_long['strike'])
        print('needed_long price', needed_long['ask'])
        print('needed_short strike', needed_short['strike'])
        print('needed_short price', needed_short['bid'])
        print('needed_short iv', needed_short['iv']*100)
        print('needed_long iv', needed_long['iv']*100)
        print('days_to_exp_short', days_to_exp_short)
        print('days_to_exp_long', days_to_exp_long)

        proba, avg_dtc = get_proba_debit_spread(current_price, stock_yahoo_short[tick], needed_long['strike'], needed_long['ask'],
                              needed_short['strike'], needed_short['bid'],  needed_short['iv']*100, needed_long['iv']*100,
                                                  days_to_exp_short, days_to_exp_long, RISK_RATE)

        print('proba', proba)
        # Считаем итоговый score (ожидаемый ROC в годовом формате) = (expected return/(debet*100)/DTE short * 365
        # caledar_put_score = (expected_return/(debet*100))/needed_short['Days_to_exp']*365
        # Считаем RR = profit 30/open cost
        # RR = (price_up_long*0.3) / abs(needed_short['bid']-needed_long['ask'])

        needed_strike_sell, needed_strike_buy = needed_short['strike'], needed_long['strike']

    except Exception as err:
        print(err)
        RR, needed_strike_sell, needed_strike_buy, proba, expected_return = np.nan, np.nan, np.nan, np.nan, np.nan
        pass

    return RR, needed_strike_sell, needed_strike_buy, proba, expected_return


def debit_spread_run(active_stock_df, stock_yahoo, tick_list, poll_num, RISK_RATE):
    print('---------------------------')
    print('------------- Getting BEAR Debit Spread ... --------------')
    print('---------------------------')

    with Pool(poll_num) as p:
        itm_calendar_out = p.map(get_data_and_calc_debit_spread, [(active_stock_df.iloc[i], stock_yahoo, RISK_RATE) for i in range(len(active_stock_df))])

    RR, needed_strike_sell, needed_strike_buy, proba_30, expected_return = zip(*itm_calendar_out)
    RR = np.array([*RR])
    RR = np.reshape(RR, len(RR))
    needed_strike_sell = np.array([*needed_strike_sell])
    needed_strike_sell = np.reshape(needed_strike_sell, len(needed_strike_sell))
    needed_strike_buy = np.array([*needed_strike_buy])
    needed_strike_buy = np.reshape(needed_strike_buy, len(needed_strike_buy))
    proba_30 = np.array([*proba_30])
    proba_30 = np.reshape(proba_30, len(proba_30))
    expected_return = np.array([*expected_return])
    expected_return = np.reshape(expected_return, len(expected_return))

    return RR, needed_strike_sell, needed_strike_buy, proba_30, expected_return
