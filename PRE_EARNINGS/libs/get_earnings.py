import pandas as pd
import numpy as np
import yfinance as yf
import datetime
import requests
from dateutil.relativedelta import relativedelta


def peat_run(tickers):
    KEY = "ckZsUXdiMTZEZVQ3a25TVEFtMm9SeURsQ1RQdk5yWERHS0RXaWNpWVJ2cz0"

    peat_df = pd.DataFrame()
    for ticker in tickers:
        try:
            print(ticker)
            url = f"https://api.marketdata.app/v1/stocks/earnings/{ticker}/?from=2023-01-01&token={KEY}"
            response = requests.request("GET", url).json()
            print(response)
            earnings_data = pd.DataFrame(response)
            earnings_data['reportDate'] = pd.to_datetime(earnings_data['reportDate'], unit='s', errors='ignore').dt.strftime('%Y-%m-%d')
            # earnings_data.to_excel('earnings_data.xlsx')
            peat_df = pd.concat([peat_df, earnings_data])
            print(earnings_data['reportDate'])
            print(earnings_data)
        except:
            pass
    peat_df['reportDate'] = pd.to_datetime(peat_df['reportDate'], errors='ignore')

    # print(peat_df[peat_df['reportDate'] >= datetime.datetime.now() - relativedelta(days=7)])
    peat_df = peat_df[peat_df['reportDate'] <= datetime.datetime.now() + relativedelta(days=14)]
    print(peat_df)
    peat_df.to_excel('peat.xlsx')
    tickers_clean = peat_df['symbol'].unique().tolist()
    # yahoo_native = yf.download(tickers_clean, group_by="Ticker")
    return_df = pd.DataFrame()

    for ticker in tickers_clean:
        try:

            earnings_date = peat_df[peat_df['symbol'] == ticker]['reportDate'].iloc[-1]
            last_reportedEPS = peat_df[peat_df['symbol'] == ticker]['reportedEPS'].iloc[-2]
            estimatedEPS = peat_df[peat_df['symbol'] == ticker]['estimatedEPS'].iloc[-1]
            print('ticker', ticker)
            print('earnings_date', earnings_date)
            print('estimatedEPS', estimatedEPS)
            print('last_reportedEPS', last_reportedEPS)
        except:
            earnings_date = np.nan
            last_reportedEPS = np.nan
            estimatedEPS = np.nan

        local_df = pd.DataFrame({
            'Symbol': [ticker],
            'Earnings_Date': [earnings_date],
            'EstimatedEPS': [estimatedEPS],
            'Prev_ReportedEPS': [last_reportedEPS],
        })

        return_df = pd.concat([return_df, local_df])

    print(return_df)
    print(return_df[return_df['Earnings_Date'] > datetime.datetime.now()])
    return_df = return_df[return_df['Earnings_Date'] > datetime.datetime.now()]

    return return_df

