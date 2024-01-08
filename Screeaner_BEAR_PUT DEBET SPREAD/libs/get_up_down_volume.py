import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import time
from sklearn import preprocessing
import numpy as np
from sklearn.linear_model import LinearRegression

pd.options.mode.chained_assignment = None


def up_down_volume_run(tick_list, stock_yahoo):
    val_list = []
    regression_list = []
    for tick in tick_list:

        # Calculate the date 50 trading days ago (approx. 70 calendar days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=70)
        start_date = start_date.strftime("%Y-%m-%d")
        last_year = end_date - timedelta(days=365)

        # Download historical data from Yahoo Finance
        data_native = stock_yahoo[tick][last_year:]

        # Keep only the last 50 trading days data
        data = data_native[start_date:].tail(50)

        # Calculate the daily returns
        data['Returns'] = data['Adj Close'].pct_change()

        # Calculate volumes on up and down days
        volume_up = data.loc[data['Returns'] > 0, 'Volume'].sum()
        volume_down = data.loc[data['Returns'] < 0, 'Volume'].sum()

        # Calculate the Volume Up/Down Ratio
        volume_up_down_ratio = volume_up / volume_down if volume_down != 0 else float('inf')

        value_plot = []
        index_plot = []

        for i in range(len(data_native)):
            if len(data_native.iloc[i:i + 50]) < 50:
                break
            # Calculate volumes on up and down days
            data = data_native.iloc[i:i + 50]
            data = data.tail(50)
            data['Returns'] = data_native.iloc[i:i + 50]['Adj Close'].pct_change()
            volume_up = data.loc[data['Returns'] > 0, 'Volume'].sum()
            volume_down = data.loc[data['Returns'] < 0, 'Volume'].sum()

            # Calculate the Volume Up/Down Ratio
            volume_up_down_ratio_local = volume_up / volume_down if volume_down != 0 else float('inf')

            value_plot.append(volume_up_down_ratio_local)
            index_plot.append(data_native.iloc[i:i + 50].index.tolist()[-1])

        # get linear regression line
        index_plot = pd.DataFrame({'Date': index_plot})
        index_plot['Date'] = index_plot['Date'].map(datetime.toordinal)

        x = np.array(index_plot[-6:]).reshape((-1, 1))
        y = np.array(value_plot[-6:])
        model = LinearRegression()
        model.fit(x, y)

        y_pred = model.predict(x)

        liniar_coeff = y_pred[-1] / y_pred[0]


        val_list.append(round(volume_up_down_ratio, 3))
        regression_list.append(round(liniar_coeff, 3))

    return val_list, regression_list

