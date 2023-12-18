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
from thetadata import ThetaClient, OptionReqType, OptionRight, DateRange, SecType, StockReqType
import requests
from multiprocessing import Pool


def market_stage_ticker(pool_input):
    try:
        # ==================================================== IB ==============================================================
        id_num, ticker = pool_input
        print('id_num', id_num)
        print('tick', ticker)

        ib = IB()
        try:
            ib.connect('127.0.0.1', 4002, clientId=id_num)  # 7497

        except:
            ib.connect('127.0.0.1', 7497, clientId=id_num)

        try:

            exch = 'NYSE'
            if ticker == 'GDX' or ticker == 'GDXJ' or ticker == 'REMX':
                exch = 'ARCA'
            contract = Stock(ticker, exch, 'USD')
            bars = ib.reqHistoricalData(
                contract, endDateTime='', durationStr='365 D',
                barSizeSetting='1 day', whatToShow='OPTION_IMPLIED_VOLATILITY', useRTH=True)

            df = util.df(bars)

        except:
            df = pd.DataFrame()

        # HIST VOLATILITY ====================================
        try:
            print('==========================================   ', ticker,
                  '  =============================================')

            #  ========================================  получение волатильности
            contract = Stock(ticker, exch, 'USD')
            bars_hist = ib.reqHistoricalData(
                contract, endDateTime='', durationStr='365 D',
                barSizeSetting='1 day', whatToShow='HISTORICAL_VOLATILITY', useRTH=True)

            df_hist = util.df(bars_hist)

        except:
            df_hist = pd.DataFrame()

        df = df.set_index('date')

        df_hist = df_hist.set_index('date')

        df = df.dropna()
        # print(df)

        unsup = mix.GaussianMixture(n_components=4,
                                    covariance_type="spherical",
                                    n_init=100,
                                    random_state=42)

        unsup.fit(np.reshape(df, (-1, df.shape[1])))

        regime = unsup.predict(np.reshape(df, (-1, df.shape[1])))
        df['Return'] = np.log(df['close'] / df['close'].shift(1))
        Regimes = pd.DataFrame(regime, columns=['Regime'], index=df.index) \
            .join(df, how='inner') \
            .assign(market_cu_return=df.Return.cumsum()) \
            .reset_index(drop=False) \
            .rename(columns={'index': 'Date'})

        Regimes = Regimes.dropna()

        one_period_price_mean = Regimes[Regimes['Regime'] == 0]['close'].max()
        two_period_price_mean = Regimes[Regimes['Regime'] == 1]['close'].max()
        three_period_price_mean = Regimes[Regimes['Regime'] == 2]['close'].max()
        four_period_price_mean = Regimes[Regimes['Regime'] == 3]['close'].max()

        lisusss = [one_period_price_mean, two_period_price_mean, three_period_price_mean, four_period_price_mean]
        lisusss_sort = sorted(lisusss)

        period_set = {0: lisusss_sort.index(lisusss[0]) + 1,
                      1: lisusss_sort.index(lisusss[1]) + 1,
                      2: lisusss_sort.index(lisusss[2]) + 1,
                      3: lisusss_sort.index(lisusss[3]) + 1,
                      }

        Regimes_plot = Regimes.copy()

        Regimes_plot['Regime'] = Regimes_plot['Regime'].replace(period_set)

        # ================================================     hist vol ======================
        unsup_hist = mix.GaussianMixture(n_components=4,
                                         covariance_type="spherical",
                                         n_init=100,
                                         random_state=42)

        unsup_hist.fit(np.reshape(df_hist, (-1, df_hist.shape[1])))

        regime = unsup_hist.predict(np.reshape(df_hist, (-1, df_hist.shape[1])))
        df_hist['Return'] = np.log(df_hist['close'] / df_hist['close'].shift(1))
        Regimes_hist = pd.DataFrame(regime, columns=['Regime'], index=df_hist.index) \
            .join(df_hist, how='inner') \
            .assign(market_cu_return=df_hist.Return.cumsum()) \
            .reset_index(drop=False) \
            .rename(columns={'index': 'Date'})

        Regimes_hist = Regimes_hist.dropna()

        one_period_price_mean_hist = Regimes_hist[Regimes_hist['Regime'] == 0]['close'].max()
        two_period_price_mean_hist = Regimes_hist[Regimes_hist['Regime'] == 1]['close'].max()
        three_period_price_mean_hist = Regimes_hist[Regimes_hist['Regime'] == 2]['close'].max()
        four_period_price_mean_hist = Regimes_hist[Regimes_hist['Regime'] == 3]['close'].max()

        lisusss_hist = [one_period_price_mean_hist, two_period_price_mean_hist, three_period_price_mean_hist,
                        four_period_price_mean_hist]
        lisusss_sort_hist = sorted(lisusss_hist)

        period_set_hist = {0: lisusss_sort_hist.index(lisusss_hist[0]) + 1,
                           1: lisusss_sort_hist.index(lisusss_hist[1]) + 1,
                           2: lisusss_sort_hist.index(lisusss_hist[2]) + 1,
                           3: lisusss_sort_hist.index(lisusss_hist[3]) + 1,
                           }

        Regimes_hist['Regime'] = Regimes_hist['Regime'].replace(period_set_hist)

        IV_Regime = float(Regimes_plot['Regime'].iloc[-1])
        HV_Regime = Regimes_hist['Regime'].iloc[-1]
        IV_Median = Regimes_plot[int(len(Regimes_plot) / 2):]['close'].median()
        HV_Median = Regimes_hist[int(len(Regimes_hist) / 2):]['close'].median()
        IV = Regimes_plot['close'].iloc[-1]

        HV_20 = Regimes_hist['close'].rolling(window=20).mean().iloc[-1]
        HV_50 = Regimes_hist['close'].rolling(window=50).mean().iloc[-1]
        HV_100 = Regimes_hist['close'].rolling(window=100).mean().iloc[-1]

        df['IV_percentile'] = df['close'].rolling(364).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]))
        IV_percentile = df['IV_percentile'].iloc[-1]


    except:
        IV_percentile = 'EMPTY'
        IV_Regime = 'EMPTY'
        IV_Median = 'EMPTY'
        IV = 'EMPTY'
        HV_Regime = 'EMPTY'
        HV_20 = 'EMPTY'
        HV_50 = 'EMPTY'
        HV_100 = 'EMPTY'

    ib.disconnect()

    print('IV_Regime: ', IV_Regime,
          '  =============================================')

    return IV_percentile, IV_Regime, IV_Median, IV, HV_Regime, HV_20, HV_50, HV_100


def total_loss_on_strike(chain, expiry_price, opt_type):
    '''
    Get's the total loss at the given strike price
    '''
    # call options with strike price below the expiry price -> loss for option writers
    if opt_type == 'call':
        in_money = chain[chain['Strike'] < expiry_price][["callOpenInterest", "Strike"]]
        in_money["Loss"] = (expiry_price - in_money['Strike']) * in_money["callOpenInterest"]

    if opt_type == 'put':
        in_money = chain[chain['Strike'] > expiry_price][["putOpenInterest", "Strike"]]
        in_money["Loss"] = (in_money['Strike'] - expiry_price) * in_money["putOpenInterest"]

    return in_money["Loss"].sum()


def chain_converter(tickers):
    # разбиваем контракты по грекам и страйкам, для того, что бы удобнее было строить позиции
    # со сложными условиями входа

    iv_bid = []
    iv_ask = []
    delta_bid = []
    delta_ask = []
    gamma_bid = []
    gamma_ask = []
    vega_bid = []
    vega_ask = []
    theta_bid = []
    theta_ask = []
    strike_list = []
    right_list = []
    exp_date_list = []

    for ticker in tickers:
        try:

            iv_bid.append(ticker.bidGreeks.impliedVol)
            iv_ask.append(ticker.askGreeks.impliedVol)
            delta_bid.append(ticker.bidGreeks.delta)
            delta_ask.append(ticker.askGreeks.delta)
            gamma_bid.append(ticker.bidGreeks.gamma)
            gamma_ask.append(ticker.askGreeks.gamma)
            vega_bid.append(ticker.bidGreeks.vega)
            vega_ask.append(ticker.askGreeks.vega)
            theta_bid.append(ticker.bidGreeks.theta)
            theta_ask.append(ticker.askGreeks.theta)
            strike_list.append(ticker.contract.strike)
            right_list.append(ticker.contract.right)
            exp_date_list.append(ticker.contract.lastTradeDateOrContractMonth)

        except:
            pass

    greek_df = pd.DataFrame(
        {
            'IV bid': iv_bid,
            'IV ask': iv_ask,
            'Delta bid': delta_bid,
            'Delta ask': delta_ask,
            'Gamma bid': gamma_bid,
            'Gamma ask': gamma_ask,
            'Vega bid': vega_bid,
            'Vega ask': vega_ask,
            'Theta bid': theta_bid,
            'Theta ask': theta_ask,
            'Strike': strike_list,
            'Right': right_list,
            'EXP_date': exp_date_list,
        }
    )

    df_chains = util.df(tickers)
    df_chains['time'] = df_chains['time'].dt.tz_localize(None)
    df_chains = pd.concat([df_chains, greek_df], axis=1)

    return df_chains


def get_df_chains(ticker_contract, limit_date_min, limit_date_max, tick, rights, ib):
    expirations_filter_list_date = []
    expirations_filter_list_strike = []

    chains = ib.reqSecDefOptParams(ticker_contract.symbol, '', ticker_contract.secType, ticker_contract.conId)
    chain = next(c for c in chains if c.tradingClass == tick and c.exchange == 'SMART')

    # фильтрация будущих контрактов по времени
    for exp in chain.expirations:
        year = exp[:4]
        month = exp[4:6]
        day = exp[6:]
        date = year + '-' + month + '-' + day
        datime_date = datetime.datetime.strptime(date, "%Y-%m-%d")
        print('---')
        print(datime_date)
        print(limit_date_min)
        print(limit_date_max)
        if datime_date >= limit_date_min and datime_date <= limit_date_max:
            expirations_filter_list_date.append(exp)

    # print('expirations_filter_list_date', expirations_filter_list_date)
    print('strikes', chain.strikes)
    print('expirations', chain.expirations)
    # фильтрация страйков относительно текущей цены
    time.sleep(4)

    for strikus in chain.strikes:
        expirations_filter_list_strike.append(strikus)

    time.sleep(4)

    contracts = [Option(tick, expiration, strike, right, 'SMART', tradingClass=tick)
                 for right in rights
                 for expiration in [expirations_filter_list_date[0]]
                 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                 for strike in expirations_filter_list_strike]

    contracts = ib.qualifyContracts(*contracts)
    ib.sleep(3)
    print('contracts')
    print(contracts)

    return contracts


def get_avalible_exp_dates(ticker, limit_date_expiration_min):
    with client.connect():  # Make any requests for data inside this block. Requests made outside this block wont run.
        exp_date_data = client.get_expirations(ticker).dt.date
        all_exp_date = exp_date_data[exp_date_data > limit_date_expiration_min].reset_index(drop=True)

        strike_list = []

        for exp_date in all_exp_date:
            strike_list.append(client.get_strikes(ticker, exp_date).astype('float').values.tolist())

    return all_exp_date, strike_list


def get_option_chains(exp_date, strike, side, ticker) -> pd.DataFrame:
    try:
        with client.connect():

            if side == 'C':
                right = OptionRight.CALL
            if side == 'P':
                right = OptionRight.PUT

            # Make the request
            out = client.get_hist_option(
                req=OptionReqType.EOD,  # End of day data   GREEKS
                root=ticker,
                exp=exp_date,
                strike=strike,
                right=right,
                date_range=DateRange(datetime.date(2020, 1, 1), exp_date),
                interval_size=3_600_000
            )

        out.columns = ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'COUNT', 'DATE']

        # out['days_to_exp'] = (exp_date - out['DATE']).dt.days
        out['Side'] = [side] * len(out)

    except:
        out = pd.DataFrame()

    return out


def nearest_equal(lst, target):
    # ближайшее значение к таргету относительно переданного списка
    return min(lst, key=lambda x: abs(x - target))


def nearest_equal_abs(lst, target):
    return min(lst, key=lambda x: abs(abs(x) - target))


def get_ib_run(tick_list, poll_num):
    print('---------------------------')
    print('------------- Getting IB Volatilities ... --------------')
    print('---------------------------')

    with Pool(poll_num) as p:
        pool_out = p.map(market_stage_ticker,
                         [(id_num, tick) for id_num, tick in
                          enumerate(tick_list)])

    IV_percentile, IV_Regime, IV_Median, IV, HV_Regime, HV_20, HV_50, HV_100 = zip(*pool_out)
    IV_percentile = np.array([*IV_percentile])
    IV_percentile = np.reshape(IV_percentile, len(IV_percentile))
    IV_Regime = np.array([*IV_Regime])
    IV_Regime = np.reshape(IV_Regime, len(IV_Regime))
    IV_Median = np.array([*IV_Median])
    IV_Median = np.reshape(IV_Median, len(IV_Median))
    IV = np.array([*IV])
    IV = np.reshape(IV, len(IV))
    HV_Regime = np.array([*HV_Regime])
    HV_Regime = np.reshape(HV_Regime, len(HV_Regime))
    HV_20 = np.array([*HV_20])
    HV_20 = np.reshape(HV_20, len(HV_20))
    HV_50 = np.array([*HV_50])
    HV_50 = np.reshape(HV_50, len(HV_50))
    HV_100 = np.array([*HV_100])
    HV_100 = np.reshape(HV_100, len(HV_100))

    return IV_percentile, IV_Regime, IV_Median, IV, HV_20, HV_50, HV_100, HV_Regime

