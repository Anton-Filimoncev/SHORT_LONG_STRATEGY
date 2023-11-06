import pandas as pd
import numpy as np
import yfinance as yf
from scipy.signal import argrelextrema
import datetime as dt


def calcBollinger(data, size, std_type, ma_type):
    df = data.copy()
    if ma_type == "SMA":
        df["ma"] = df["Close"].rolling(size).mean()

    if ma_type == "EMA":
        df["ma"] = df["Close"].ewm(com=0.8).mean()

    if std_type == "Close":
        df["bolu_3"] = df["ma"] + 3 * df["Close"].rolling(size).std(ddof=0)
        df["bolu_4"] = df["ma"] + 4 * df["Close"].rolling(size).std(ddof=0)
        df["bold_3"] = df["ma"] - 3 * df["Close"].rolling(size).std(ddof=0)
        df["bold_4"] = df["ma"] - 4 * df["Close"].rolling(size).std(ddof=0)

    if std_type == "Close-Open":
        df["bolu_3"] = df["ma"] + 3 * (df["Close"] - df["Open"]).rolling(size).std(
            ddof=0
        )
        df["bolu_4"] = df["ma"] + 4 * (df["Close"] - df["Open"]).rolling(size).std(
            ddof=0
        )
        df["bold_3"] = df["ma"] - 3 * (df["Close"] - df["Open"]).rolling(size).std(
            ddof=0
        )
        df["bold_4"] = df["ma"] - 4 * (df["Close"] - df["Open"]).rolling(size).std(
            ddof=0
        )

    df.dropna(inplace=True)
    return df


def rur_bollinger(ticker_list, stock_yahoo):
    windowSizeBoll_signal = 5
    startBoll_signal = dt.datetime.today() - dt.timedelta(16)
    endBoll_signal = dt.datetime.today()
    std_type = "Close-Open"
    sma_type = "SMA"
    numYearBoll = 20
    windowSizeBoll = 20

    startBoll = dt.datetime.today() - dt.timedelta(numYearBoll * 365)
    startBoll = startBoll.strftime("%Y-%m-%d")

    signals_list = []

    for ticker in ticker_list:
        try:
            dataBoll = stock_yahoo[ticker]
            # dataBoll['updated'] = pd.to_datetime(dataBoll['t'], unit='s', errors='ignore').dt.strftime('%Y-%m-%d')
            dataBoll["updated"] = dataBoll.index.tolist()
            dataBoll["updated"] = dataBoll["updated"].dt.strftime("%Y-%m-%d")
            dataBoll["Date"] = dataBoll.index.tolist()
            dataBoll["Date"] = dataBoll["Date"].dt.strftime("%Y-%m-%d")
            # dataBoll = dataBoll.rename(columns={"c": "Close", "h": "High", "l": "Low", "o": "Open", "updated": "Date"})
            dataBoll = dataBoll.set_index("Date")[["Close", "High", "Low", "Open"]]
            dataBoll = dataBoll[startBoll:]

            df_boll = calcBollinger(dataBoll, windowSizeBoll, std_type, sma_type)
            df_boll = df_boll.reset_index()

            # добавление точек экстремума
            max_idx = argrelextrema(
                np.array(df_boll["Close"].values), np.greater, order=3
            )
            min_idx = argrelextrema(np.array(df_boll["Close"].values), np.less, order=3)
            # print(max_idx)
            df_boll["peaks"] = np.nan
            df_boll["lows"] = np.nan
            for i in max_idx:
                df_boll["peaks"][i] = df_boll["Close"][i]
            for i in min_idx:
                df_boll["lows"][i] = df_boll["Close"][i]

            op = df_boll["Open"].astype(float)
            hi = df_boll["High"].astype(float)
            lo = df_boll["Low"].astype(float)
            cl = df_boll["Close"].astype(float)

            signal = []

            for row_num in range(len(df_boll)):
                row_bolinger = df_boll.iloc[row_num]
                if row_bolinger["peaks"] > row_bolinger["bolu_3"]:
                    signal.append(1)
                elif row_bolinger["lows"] < row_bolinger["bold_3"]:
                    signal.append(-1)
                else:
                    signal.append(0)

            df_boll["signal"] = signal
            df_boll = df_boll.set_index("Date")

            signals_list.append(df_boll["signal"].iloc[-1])
        except:
            signals_list.append('EMPTY')
    return signals_list
