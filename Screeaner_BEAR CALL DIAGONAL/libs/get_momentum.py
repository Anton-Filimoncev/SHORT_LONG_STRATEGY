import pandas_ta as pta
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn import preprocessing
from scipy.stats import pearsonr
from hurst import compute_Hc
pd.options.mode.chained_assignment = None

def calc_corr(df, lookback, hold):
    # Create a copy of the dataframe
    data = df.copy()

    # Calculate lookback returns
    data['lookback_returns'] = data['Close'].pct_change(lookback)

    # Calculate hold returns
    data['future_hold_period_returns'] = data['Close'].pct_change(hold).shift(-hold)

    data = data.dropna()
    data = data.iloc[::hold]

    # Calculate correlation coefficient and p-value
    corr, p_value = pearsonr(data.lookback_returns,
                             data.future_hold_period_returns)
    return corr, p_value



# Define a function to calculate stock momentum
def calculate_stock_momentum(yahoo_data):
    crude_data = yahoo_data

    # Define lookback periods
    lookback = [15, 30, 60, 90, 150, 240, 360]

    # Define holding periods
    hold = [5, 10, 15, 30, 45, 60]

    # Create a dataframe which stores price of a security
    crude_data.dropna()

    # Create an array of length lookback*hold
    corr_grid = np.zeros((len(lookback), len(hold)))
    p_value_grid = np.zeros((len(lookback), len(hold)))

    # Run through a length of lookback and holding periods
    for i in range(len(lookback)):
        for j in range(len(hold)):
            # Call calc_corr function and calculate correlation coefficient and p-value
            corr_grid[i][j], p_value_grid[i][j] = calc_corr(
                crude_data, lookback[i], hold[j])

    opt = np.where(corr_grid == np.max(corr_grid))
    opt_lookback = lookback[opt[0][0]]
    opt_hold = hold[opt[1][0]]

    hurst_exp = compute_Hc(crude_data["Close"][-222:].values, kind='price', simplified=False)[0]
    print(f"Hurst exponent with {opt_hold} lags: {hurst_exp:.4f}")

    print('opt_lookback', opt_lookback)

    # end_date = datetime.now() - timedelta(days=30)  # Exclude the most recent month
    start_date = datetime.now() - timedelta(days=opt_lookback)  # Go back

    # Fetch historical prices
    stock_data = yahoo_data[start_date:]

    # Compute monthly returns
    df_monthly = stock_data['Adj Close'].ffill()
    df_monthly_returns = df_monthly.pct_change()

    print('df_monthly_returns', df_monthly_returns)

    # Exclude the last month
    df_monthly_returns = df_monthly_returns[:-1]

    # Calculate average growth rate
    avg_growth_rate = df_monthly_returns.mean()

    # Calculate standard deviation
    sd = df_monthly_returns.std()

    # Calculate momentum
    momentum = avg_growth_rate / sd if sd != 0 else 0

    return momentum, opt_hold, hurst_exp


def get_momentum_run(stock_yahoo, tick_list):
    print("---------------------------")
    print("------------- Getting Momentum ... --------------")
    print("---------------------------")
    momentum_list = []
    hold_list = []
    hurst_list = []

    for tick in tick_list:
        try:
            momentum_signal, opt_hold, hurst_exp = calculate_stock_momentum(stock_yahoo[tick])
        except:
            momentum_signal, opt_hold, hurst_exp = "Empty", "Empty", "Empty"

        momentum_list.append(momentum_signal)
        hold_list.append(opt_hold)
        hurst_list.append(hurst_exp)

    return momentum_list, hold_list, hurst_list
