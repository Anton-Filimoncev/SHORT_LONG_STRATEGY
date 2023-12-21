import pandas_ta as pta
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

pd.options.mode.chained_assignment = None


# Define a function to calculate stock momentum
def calculate_stock_momentum(yahoo_data):
    end_date = datetime.now() - timedelta(days=30)  # Exclude the most recent month
    start_date = end_date - timedelta(days=365)  # Go back 13 months

    # Fetch historical prices
    stock_data = yahoo_data[start_date:end_date]

    # Compute monthly returns
    df_monthly = stock_data['Adj Close'].resample('M').ffill()
    df_monthly_returns = df_monthly.pct_change()

    # Exclude the last month
    df_monthly_returns = df_monthly_returns[:-1]

    # Calculate average growth rate
    avg_growth_rate = df_monthly_returns.mean()

    # Calculate standard deviation
    sd = df_monthly_returns.std()

    # Calculate momentum
    momentum = avg_growth_rate / sd if sd != 0 else 0

    return momentum


def get_momentum_run(stock_yahoo, tick_list):
    print("---------------------------")
    print("------------- Getting Momentum ... --------------")
    print("---------------------------")
    momentum_list = []

    for tick in tick_list:
        try:
            momentum_signal = calculate_stock_momentum(stock_yahoo[tick])
        except:
            momentum_signal = "Empty"

        momentum_list.append(momentum_signal)

    return momentum_list
