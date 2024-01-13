import pandas as pd
import numpy as np
import os
import time
import pickle
import gspread as gd
from ib_insync import *
from scipy import stats
import yfinance as yf
import pandas_ta as pta
import datetime
from dateutil.relativedelta import relativedelta
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from concurrent.futures.thread import ThreadPoolExecutor
from time import sleep
from selenium.webdriver.support.ui import WebDriverWait



def strategist_pcr_signal():
    barcode = 'dranatom'
    password = 'MSIGX660'
    # ____________________ Работа с Selenium ____________________________
    path = os.path.join(os.getcwd(), 'chromedriver.exe')
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument('headless')

    checker = webdriver.Chrome(options=chrome_options)

    checker.get(f'https://www.optionstrategist.com/subscriber-content/put-call-ratios')
    time.sleep(2)

    sign_in_userName = checker.find_element(by=By.ID, value="edit-name")
    sign_in_userName.send_keys(barcode)
    sign_in_password = checker.find_element(by=By.ID, value="edit-pass")
    sign_in_password.send_keys(password)
    sign_in = checker.find_element(by=By.ID,
                                   value='''edit-submit''')
    sign_in.click()
    time.sleep(2)
    try:
        close_popup = checker.find_element(By.XPATH, '//*[@id="PopupSignupForm_0"]/div[2]/div[1]')
        close_popup.click()
    except:
        pass

    # select_fresh_txt = checker.find_element(By.XPATH,
    #                                         '''//*[@id="node-35"]/div/div/div/div/table[1]/tbody/tr[1]/td[1]''')
    #
    # select_fresh_txt.click()
    time.sleep(2)

    html_txt = checker.find_element(By.XPATH,
                                    '''//*[@id="node-47"]/div/div/div/div/pre''').text.split('\n')

    ticker_list = []
    strategist_signals = []
    strategist_signals_percentile = []
    plot_links_list = []

    for piece in html_txt:
        try:
            tick_dirty = piece.split(' ')[0].split('_W')[0]
            signals_percentile = int(piece.split('/')[1].split('%')[0])
            signals = piece.split('.')[-1]

            if '@' not in tick_dirty and '$' not in tick_dirty and '1' not in tick_dirty and 'BRKB' not in tick_dirty and 'NYSE' not in tick_dirty:
                ticker_list.append(tick_dirty)
                strategist_signals_percentile.append(signals_percentile)
                strategist_signals.append(signals)
        except:
            pass

        # if tick == piece.split(' ')[0] or tick + '_W' == piece.split(' ')[0]:
        #     strategist_signals.append(piece[len(tick) + 3:])
        #     strategist_signals_percentile.append(int(piece.split('/')[1].split('%')[0]))


    # print(strategist_signals)

    # получаем ссылки на графики
    all_link = checker.find_elements(By.XPATH, "//a[@href]")
    total_links = []
    for elem in all_link:
        total_links.append(elem.get_attribute("href"))

    # print(total_links)

    for tick in ticker_list:
        local_flag = 0
        for elem in total_links:
            if '=' + tick + '.' in elem or '=' + tick + '_W' in elem:
                plot_links_list.append(elem)
                local_flag = 1
                break
        if local_flag == 0:
            plot_links_list.append('Empty')
            # print('Empty')

    # print(plot_links_list)

    return_df = pd.DataFrame({
        'Symbol': ticker_list,
        'PCR SIGNAL': strategist_signals,
        'PCR PERCENTILE': strategist_signals_percentile,
        'PCR Link': plot_links_list,
    })

    return return_df


def get_pcr_run():
    print('---------------------------')
    print('------------- Getting PCR ... --------------')
    print('---------------------------')

    return_df = strategist_pcr_signal()

    return return_df


 