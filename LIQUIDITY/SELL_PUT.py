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
from libs.popoption.ShortPut import shortPut
pd.options.mode.chained_assignment = None


def nearest_equal_abs(lst, target):
    return min(lst, key=lambda x: abs(abs(x) - target))


def option_price_calc(
    current_price, strike_list, close_exp_date, volatility_list, side
):
    price_list = []
    for strike, vol in zip(strike_list, volatility_list):
        c = mibian.BS([current_price, strike, 4, close_exp_date], volatility=vol * 100)
        if side == "C":
            price_list.append(c.callPrice)
        else:
            price_list.append(c.putPrice)

    return price_list


def option_price_batch(
    volatility, daysToExpiration, underlyingPrice, strikePrice, interestRate, side
):
    #     interestRate = interestRate /100
    daysToExpiration = daysToExpiration / 365
    #     volatility = volatility / 100
    a = volatility * daysToExpiration**0.5
    d1 = (
        np.log(underlyingPrice / strikePrice)
        + (interestRate + (volatility**2) / 2) * daysToExpiration
    ) / a
    d2 = d1 - a
    if side == "P":
        price = strikePrice * e ** (-interestRate * daysToExpiration) * norm.cdf(
            -d2
        ) - underlyingPrice * norm.cdf(-d1)
    if side == "C":
        price = underlyingPrice * norm.cdf(d1) - strikePrice * e ** (
            -interestRate * daysToExpiration
        ) * norm.cdf(d2)
    return price


def calculate_probability(
    ticker, nb_simulations, days, lower_bound, upper_bound, target_price_df, side
):
    # Fetch historical market data
    hist_data = yf.download(ticker, start="2000-01-01")

    # Calculate the log returns
    log_returns = np.log(1 + hist_data["Adj Close"].pct_change())

    # Define the variables
    u = log_returns.mean()
    var = log_returns.var()

    # Calculate drift and standard deviation
    drift = u - (0.5 * var)
    stddev = log_returns.std()

    # Generate a random variable
    daily_returns = np.exp(
        drift + stddev * np.random.standard_normal((days, nb_simulations))
    )

    # Simulate the price paths
    s0 = hist_data["Adj Close"][-1]
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
        if side == "C":
            touch_target_price = (price_list >= target_price).any(
                axis=0
            ).sum() / nb_simulations
        if side == "P":
            touch_target_price = (price_list <= target_price).any(
                axis=0
            ).sum() / nb_simulations

        touch_target_price_list.append(touch_target_price)

    return touch_target_price_list


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
    ones[0 : -1 : N_days + 1] = 1
    ds_s[N_days : N_days * N_sims + N_sims : N_days + 1] = -1
    d = [ds_s + 1, ones]
    K = [-1, 0]
    M = scipy.sparse.diags(d, K, format="csc")
    p = np.zeros((N_sims * N_days + N_sims, 1))
    p[0 : -1 : N_days + 1] = init_price
    s = scipy.sparse.linalg.spsolve(M, p)

    if reshape == True:
        s = np.reshape(s, (N_sims, N_days + 1))

    return s


if __name__ == "__main__":
    KEY = "ckZsUXdiMTZEZVQ3a25TVEFtMm9SeURsQ1RQdk5yWERHS0RXaWNpWVJ2cz0"

    tick = "NVDA"  # LLY

    yahoo_df = yf.download(tick)
    current_price = yahoo_df["Close"].iloc[-1]

    print(f"---------    {tick}")
    # ----------- get Exp date list  -----------------

    url_exp = f"https://api.marketdata.app/v1/options/expirations/{tick}/?token={KEY}"
    response_exp = requests.request("GET", url_exp).json()
    exp_date_df = pd.DataFrame(response_exp)
    exp_date_df["expirations"] = pd.to_datetime(exp_date_df["expirations"])
    exp_date_df["Days_to_exp"] = (
        exp_date_df["expirations"] - datetime.datetime.now()
    ).dt.days
    days_to_exp = nearest_equal_abs(exp_date_df["Days_to_exp"], 300)
    needed_exp_date = (
        exp_date_df[exp_date_df["Days_to_exp"] == days_to_exp]["expirations"]
        .reset_index(drop=True)
        .iloc[0]
        .date()
    )

    # ----------- Chains -----------------
    url = f"https://api.marketdata.app/v1/options/chain/{tick}/?expiration={needed_exp_date}&side=put&token={KEY}"
    response_chains = requests.request("GET", url).json()
    chains = pd.DataFrame(response_chains)
    chains = chains[chains["strike"] < current_price * 1.10]
    chains = chains[chains["strike"] > current_price * 0.7].reset_index(drop=True)
    print(chains)
    # Для каждого страйка считаем маржинальные требования по формуле =MAX((STRIKE*0.1),(PRICE*0.2-otm value))*100.
    # OTM VALUE считается как =IF(STRIKE>PRICE, 0 ,PRICE-STRIKE)
    # print(chains.columns)
    close_exp_date = days_to_exp
    # exp_move = hv * current_price * math.sqrt(close_exp_date/365)
    otm_value = np.where(
        chains["strike"] > current_price, 0, current_price - chains["strike"]
    )

    chains["Margin"] = np.where(
        (chains["strike"] * 0.1) > (current_price * 0.2 - otm_value),
        (chains["strike"] * 0.1),
        (current_price * 0.2 - otm_value),
    )
    chains["50% RETURN"] = (chains["bid"] / 2) / chains["Margin"]

    # Для каждого страйка считаем вероятность получения 50% прибыли по монте-карло (POP50)

    monte_carlo_proba_50 = []
    atm_strike = nearest_equal_abs(chains["strike"], current_price)
    atm_volatility = chains[chains["strike"] == atm_strike]["iv"].values[0] * 100

    for index, row in chains.iterrows():
        yahoo_stock = yahoo_df
        short_strike = row["strike"]
        short_price = row["bid"]
        rate = 4.6
        sigma = atm_volatility
        days_to_expiration = close_exp_date
        closing_days_array = [close_exp_date]
        percentage_array = [50]
        trials = 2000
        proba_50 = shortPut(
            current_price,
            sigma,
            rate,
            trials,
            days_to_expiration,
            closing_days_array,
            percentage_array,
            short_strike,
            short_price,
            yahoo_stock,
        )
        monte_carlo_proba_50.append(proba_50)


    chains["Probability of making 50%"] = monte_carlo_proba_50
    chains["50 RPOP"] = chains["Probability of making 50%"] * chains["50% RETURN"]

    print(
        chains[
            ["strike", "Margin", "50% RETURN", "Probability of making 50%", "50 RPOP"]
        ]
    )
    print("max_val: ", chains["50 RPOP"].max())
