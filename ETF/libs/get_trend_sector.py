import pandas as pd
import numpy as np
import asyncio
import scipy.stats as stats
import datetime
import time
from dateutil.relativedelta import relativedelta
import yfinance as yf
import pickle
# from support import *
from .apendix import *
import nest_asyncio
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium import webdriver
import os
import tqdm
nest_asyncio.apply()
import mibian
from contextvars import ContextVar
from scipy.stats import norm
import math

pd.options.mode.chained_assignment = None



def get_trend_df(symb_for_sector_trend, sector_for_sector_trend):

    start_date = datetime.datetime.now().date().strftime('%Y-%m-%d')

    filtered_tickers = get_sorted_table(symb_for_sector_trend, sector_for_sector_trend, start_date)
    return filtered_tickers
