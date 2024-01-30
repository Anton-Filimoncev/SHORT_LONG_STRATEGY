import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression

def beta_calc(tick, stock_yahoo, BETA_BENCHMARK):
    if tick != BETA_BENCHMARK:
        # symbols = [stock, market]
        # start date for historical prices
        symbols = [tick, BETA_BENCHMARK]
        data = stock_yahoo[symbols].dropna()[-252:]

        # Convert historical stock prices to daily percent change
        df_tick = data[tick]
        df_bench = data[BETA_BENCHMARK]

        stock_data = df_tick['Close'].pct_change()[1:]
        market_data = df_bench['Close'].pct_change()[1:]

        covariance = np.cov(stock_data, market_data)[0][1]
        var = np.var(market_data)

        beta = covariance / var
        return beta

def get_beta_run(tick_list, stock_yahoo, BETA_BENCHMARK):
    print('---------------------------')
    print('------------- Getting Beta --------------')
    print('---------------------------')
    beta_list = []
    for tick in tick_list:
        beta_list.append(beta_calc(tick, stock_yahoo, BETA_BENCHMARK))

    return beta_list

