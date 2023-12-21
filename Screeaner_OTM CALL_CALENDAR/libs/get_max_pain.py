import pandas as pd
import numpy as np
import scipy.stats as stats
import datetime
import time
from dateutil.relativedelta import relativedelta
import yfinance as yf
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
import gspread as gd
import requests
import asyncio
import aiohttp
from multiprocessing import Pool




def nearest_equal(lst, target):
    # ближайшее значение к таргету относительно переданного списка
    return min(lst, key=lambda x: abs(x - target))


def get_atm_strikes(chains_df, current_price):
    chains_df["ATM_strike_volatility"] = np.nan
    for exp in chains_df["EXP_date"].unique():
        solo_exp_df = chains_df[chains_df["EXP_date"] == exp]
        atm_strike = nearest_equal(solo_exp_df["strike"].tolist(), current_price)
        atm_put_volatility = (
            solo_exp_df[solo_exp_df["strike"] == atm_strike]["iv"]
            .reset_index(drop=True)
            .iloc[0]
        )
        chains_df.loc[
            chains_df["EXP_date"] == exp, "ATM_strike_volatility"
        ] = atm_put_volatility

    return chains_df


def total_loss_on_strike(chain, expiry_price, opt_type):
    """
    Get's the total loss at the given strike price
    """
    # call options with strike price below the expiry price -> loss for option writers
    if opt_type == "call":
        in_money = chain[chain["strike"] < expiry_price][["openInterest", "strike"]]
        in_money["Loss"] = (expiry_price - in_money["strike"]) * in_money[
            "openInterest"
        ]

    if opt_type == "put":
        in_money = chain[chain["strike"] > expiry_price][["openInterest", "strike"]]
        in_money["Loss"] = (in_money["strike"] - expiry_price) * in_money[
            "openInterest"
        ]

    return in_money["Loss"].sum()


def get_max_pain(df_chains_for_max_pain):
    call_max_pain_list = []
    put_max_pain_list = []
    strike_list = []

    max_pain_put = df_chains_for_max_pain[
        df_chains_for_max_pain["side"] == "put"
    ].reset_index(drop=True)
    max_pain_call = df_chains_for_max_pain[
        df_chains_for_max_pain["side"] == "call"
    ].reset_index(drop=True)

    for i in range(len(max_pain_put)):
        put_max_pain_list.append(
            total_loss_on_strike(max_pain_put, max_pain_put["strike"][i], "put")
        )
        call_max_pain_list.append(
            total_loss_on_strike(max_pain_call, max_pain_put["strike"][i], "call")
        )
        strike_list.append(max_pain_put["strike"][i])

    max_pain = pd.DataFrame(
        {"PUT": put_max_pain_list, "CALL": call_max_pain_list, "Strike": strike_list}
    )

    max_pain_value = (max_pain["PUT"] + max_pain["CALL"]).min()
    max_pain["Sum"] = max_pain["PUT"] + max_pain["CALL"]
    max_pain_strike = (
        max_pain[max_pain["Sum"] == max_pain_value]["Strike"]
        .reset_index(drop=True)
        .iloc[0]
    )

    return max_pain_strike


async def get_market_data(session, url):
    async with session.get(url) as resp:
        market_data = await resp.json(content_type=None)
        option_chain_df = pd.DataFrame(market_data)

        return option_chain_df


async def get_prime(exp_date_list, tick):
    KEY = "ckZsUXdiMTZEZVQ3a25TVEFtMm9SeURsQ1RQdk5yWERHS0RXaWNpWVJ2cz0"
    option_chain_df = pd.DataFrame()
    async with aiohttp.ClientSession() as session:
        tasks = []
        for exp in exp_date_list:
            url = f"https://api.marketdata.app/v1/options/chain/{tick}/?token={KEY}&expiration={exp}"  #
            tasks.append(asyncio.create_task(get_market_data(session, url)))

        solo_exp_chain = await asyncio.gather(*tasks)

        for chain in solo_exp_chain:
            option_chain_df = pd.concat([option_chain_df, chain])

    option_chain_df["updated"] = pd.to_datetime(option_chain_df["updated"], unit="s")
    option_chain_df["EXP_date"] = pd.to_datetime(
        option_chain_df["expiration"], unit="s", errors="coerce"
    )
    option_chain_df["days_to_exp"] = (
        option_chain_df["EXP_date"] - option_chain_df["updated"]
    ).dt.days
    option_chain_df = option_chain_df.reset_index(drop=True)

    return option_chain_df


def get_df_chains(tick, limit_date_min, limit_date_max):
    KEY = "ckZsUXdiMTZEZVQ3a25TVEFtMm9SeURsQ1RQdk5yWERHS0RXaWNpWVJ2cz0"
    url_exp = f"https://api.marketdata.app/v1/options/expirations/{tick}/?token={KEY}"
    response_exp = requests.request("GET", url_exp)
    expirations_df = pd.DataFrame(response_exp.json())
    expirations_df["expirations"] = pd.to_datetime(
        expirations_df["expirations"], format="%Y-%m-%d"
    )
    expirations_df = expirations_df[expirations_df["expirations"] > limit_date_min]
    expirations_df = expirations_df[expirations_df["expirations"] < limit_date_max]
    print(expirations_df)
    option_chain_df = asyncio.run(get_prime(expirations_df["expirations"], tick))

    return option_chain_df


def get_max_pain_values(pool_input):
    stock_yahoo,  tick = pool_input
    # ------------------------------------------ Max Pain  ---------------------
    print(stock_yahoo)
    print(tick)
    limit_date_min = datetime.datetime.now() + relativedelta(days=+3)
    limit_date_max = datetime.datetime.now() + relativedelta(days=+31)
    df_chains = get_df_chains(tick, limit_date_min, limit_date_max)
    #
    current_price = stock_yahoo['Close'].iloc[-1]
    # print('---------------- df_chains  ---------------------')
    # print(df_chains)
    df_chains = get_atm_strikes(df_chains, current_price)
    max_pain_val = get_max_pain(df_chains)
    print("max_pain_val", max_pain_val)
    return max_pain_val

def get_max_pain_run(tick_list, stock_yahoo, poll_num):
    with Pool(poll_num) as p:
         pool_out = p.map(get_max_pain_values, [(stock_yahoo[tick], tick) for tick in tick_list])
         print('pool_out', pool_out)
         return pool_out