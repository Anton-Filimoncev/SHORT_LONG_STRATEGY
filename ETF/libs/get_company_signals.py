import pandas as pd
import numpy as np
import datetime
import time
from .short_selling import *
from dateutil.relativedelta import relativedelta
import os
from pathlib import Path
from multiprocessing import Pool


def get_company_signals_run(stock_yahoo, tick_list, poll_num):
    print('---------------------------')
    print('------------- Getting Company Rel. Signals  ... --------------')
    print('---------------------------')

    start_date = datetime.datetime.now().date()
    start_date = start_date.strftime('%Y-%m-%d')

    regime_start_date = (datetime.datetime.strptime(start_date, '%Y-%m-%d') - relativedelta(years=5)).strftime('%Y-%m-%d')
    benchmark = round(yf.download(tickers='^GSPC', start=regime_start_date, interval="1d",
                                      group_by='column', auto_adjust=True, prepost=True,
                                      proxy=None)['Close'], 2)

    with Pool(poll_num) as p:
         pool_out = p.map(get_current_regime, [(stock_yahoo[tick], benchmark, regime_start_date, tick) for tick in tick_list])
    print('pool_out')
    print(pool_out)
    regime_list, relative_regime_list = zip(*pool_out)

    return regime_list, relative_regime_list


