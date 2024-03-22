import numpy as np
import pandas as pd
import requests
import datetime
from dateutil.relativedelta import relativedelta
import mibian
import math
import tqdm
from scipy.stats import norm

import streamlit as st

def implied_volatility(row, side, option_price):

    P = option_price
    S = float(row['underlyingPrice'])
    E = float(row['strike'])
    T = row['days_to_exp']/365
    r = 0.002
    sigma = 0.01

    while sigma < 1:
        # try:
        d_1 = float(float((math.log(S/E)+(r+(sigma**2)/2)*T))/float((sigma*(math.sqrt(T)))))
        d_2 = float(float((math.log(S/E)+(r-(sigma**2)/2)*T))/float((sigma*(math.sqrt(T)))))

        if side == 'C':
            P_implied = float(S*norm.cdf(d_1) - E*math.exp(-r*T)*norm.cdf(d_2))

        if side == 'P':
            P_implied = float(norm.cdf(-d_2)*E*math.exp(-r*T) - norm.cdf(-d_1)*S)

        if P-(P_implied) < 0.001:
            return sigma * 100

        sigma +=0.001
        # except:
        #     return 0

    return 0

def greek_calc(input_df):
    delta_put_list = []
    delta_call_list = []
    iv_put_list = []
    iv_call_list = []

    tqdm_params = {
        'total': len(input_df),
        'miniters': 1,
        'unit': 'it',
        'unit_scale': True,
        'unit_divisor': 1024,
    }

    with tqdm.tqdm(**tqdm_params) as pb:
        for number_row, row in input_df.iterrows():
            current_price = row['underlyingPrice']
            strike = row['strike']
            days_to_exp = row['days_to_exp']
            call_price = row['call_bid_quote']
            put_price = row['put_bid_quote']

            # print('volatility calc...')
            # if side == 'C':
                # c = mibian.BS([current_price, strike, 1.5, days_to_exp], callPrice=option_price)
                # volatility = c.impliedVolatility
                # print('volatility1', volatility)
            call_volatility = implied_volatility(row, 'C', call_price)


            # elif side == 'P':
                # c = mibian.BS([current_price, strike, 1.5, days_to_exp], putPrice=option_price)
                # volatility = c.impliedVolatility
                # print('volatility1', volatility)
            # print('0')
            # print(row)
            # print(put_price)
            # print('0')
            put_volatility = implied_volatility(row, 'P', put_price)

            iv_put_list.append(put_volatility)
            iv_call_list.append(call_volatility)

            # BS([underlyingPrice, strikePrice, interestRate, daysToExpiration], volatility=x, callPrice=y, putPrice=z)
            # print('delta calc...')
            c_call = mibian.BS([current_price, strike, 1, days_to_exp], volatility=call_volatility)
            c_put = mibian.BS([current_price, strike, 1, days_to_exp], volatility=put_volatility)

            delta_call_list.append(c_call.callDelta)
            delta_put_list.append(c_put.putDelta)

            pb.update(1)

    input_df['delta_put'] = delta_put_list
    input_df['delta_call'] = delta_call_list
    input_df['iv_put'] = iv_put_list
    input_df['iv_call'] = iv_call_list
    return input_df

def nearest_equal_abs(lst, target):
    return min(lst, key=lambda x: abs(abs(x) - target))


def get_exp_dates(ticker, trade_date, KEY, look_dte):
    while True:
        try:
            url = f"https://www.hedginglab.com:/api/beta/v1/option_expire_date/{ticker}/?trade_date={trade_date}&apikey={KEY}"
            print(url)
            response = requests.request("GET", url).json()
            print(response)
            exp_df = pd.to_datetime(pd.Series(response['expire_date_list']), format="%Y/%m/%d")
            break
        except:
            trade_date = (datetime.datetime.strptime(trade_date, '%Y-%m-%d') - relativedelta(days=1)).strftime('%Y-%m-%d')

    exp_df_days = (exp_df - datetime.datetime.strptime(trade_date, '%Y-%m-%d')).dt.days
    print('exp_df_days', exp_df_days)
    print('look_dte', look_dte)
    needed_dte = nearest_equal_abs(exp_df_days.values.tolist(), look_dte)
    needed_exp_date_index = (exp_df_days[exp_df_days == needed_dte]).index
    needed_exp_date = exp_df.iloc[needed_exp_date_index].dt.date.iloc[0]
    needed_exp_date = needed_exp_date.strftime('%Y-%m-%d')
    print('needed_exp_date', needed_exp_date)

    return needed_exp_date, exp_df

# def get_quotes(ticker, trade_date, KEY, exp_date):
#     url = f"https://www.hedginglab.com:/api/beta/v1/option_quote/{ticker}/?trade_date={trade_date}&expire_date={exp_date}&apikey={KEY}"
#     response = requests.request("GET", url).json()
#     quotes_df = pd.DataFrame(response)
#     return quotes_df

def get_quotes(ticker, trade_date, KEY, exp_date):
    url = f"https://api.marketdata.app/v1/options/chain/{ticker}/?expiration={exp_date}&token={KEY}"
    print(url)
    response = requests.request("GET", url).json()
    quotes_df = pd.DataFrame(response)
    return quotes_df

def get_earning_history(ticker, KEY):
    url = f"https://www.hedginglab.com/api/beta/v1/earning_history/{ticker}/?apikey={KEY}"
    response = requests.request("GET", url).json()
    earning_df = pd.DataFrame(response)
    return earning_df

def get_position(ticker, entry_type, combo_name, start_date_str, end_date_str, KEY):
    url = f"https://www.hedginglab.com/api/beta/v1/earning_batch_trade/?apikey={KEY}&symbol={ticker}&start_date_str={start_date_str}&end_date_str={end_date_str}&entry_type={entry_type}&combo_name={combo_name}"
    print(url)
    response = requests.request("GET", url).json()
    df = pd.DataFrame(response)
    return df

def get_pln(ticker, trade_type, pnl_trade_date, strike_1, strike_2, expire_date_1, expire_date_2, KEY):
    url = f"https://www.hedginglab.com:/api/beta/v1/option_trade/{ticker}/?trade_type={trade_type}&trade_date={pnl_trade_date}&strike_1={strike_1}&strike_2={strike_2}&expire_date_1={expire_date_1}&expire_date_2={expire_date_2}&apikey={KEY}"
    print(url)
    response = requests.request("GET", url).json()
    pnl_df = pd.DataFrame(response['trade_result'])
    return pnl_df

def get_strikes(ticker, trade_date, expire_date_1, KEY):
    url = f"https://www.hedginglab.com:/api/beta/v1/option_strike/{ticker}/?trade_date={trade_date}&expire_date={expire_date_1}&apikey={KEY}"
    print(url)
    response = requests.request("GET", url).json()
    pnl_df = pd.DataFrame(response)
    return pnl_df

def hedginglab_get_exp_date(ticker, nearest_dte):
    KEY = "qxQGV7V5.FaXhxdmhOyahWidhOoDlxYGEpSXIEXmX"
    trade_date = (datetime.date.today() - relativedelta(days=1)).strftime('%Y-%m-%d')
    needed_exp_date, exp_df = get_exp_dates(ticker, trade_date, KEY, nearest_dte)
    needed_exp_date = datetime.datetime.strptime(needed_exp_date, '%Y-%m-%d')
    dte = (needed_exp_date - datetime.datetime.now()).days

    return needed_exp_date, dte, exp_df


def hedginglab_get_quotes(ticker, nearest_dte):
    needed_exp_date, dte, exp_df = hedginglab_get_exp_date(ticker, nearest_dte)
    KEY = "ckZsUXdiMTZEZVQ3a25TVEFtMm9SeURsQ1RQdk5yWERHS0RXaWNpWVJ2cz0"
    trade_date = (datetime.date.today() - relativedelta(days=1)).strftime('%Y-%m-%d')
    # strike_list = get_strikes(ticker, trade_date, expire_date_1, KEY)
    # print(strike_list)
    quotes = get_quotes(ticker, trade_date, KEY, needed_exp_date.strftime('%Y-%m-%d'))
    print(quotes)
    return quotes