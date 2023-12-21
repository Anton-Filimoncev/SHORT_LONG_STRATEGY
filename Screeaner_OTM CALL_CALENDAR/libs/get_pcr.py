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



def strategist_pcr_signal(ticker_list):
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

    strategist_signals = []
    plot_links_list = []

    for tick in ticker_list:
        local_flag = 0
        for piece in html_txt:
            if tick in piece:
                if tick == piece.split(' ')[0] or tick + '_W' == piece.split(' ')[0]:
                    strategist_signals.append(piece[len(tick) + 3:])
                    # print(piece)
                    local_flag = 1
                    break
        if local_flag == 0:
            strategist_signals.append('Empty')
            # print('Empty')

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

    return strategist_signals, plot_links_list

def run_strategist():
    # # ================ раббота с таблицей============================================
    gc = gd.service_account(filename='Seetus.json')
    worksheet = gc.open("IBKR").worksheet("ETF")
    worksheet_spark = gc.open("IBKR").worksheet("sparkline")

    worksheet_df_len = pd.DataFrame(worksheet.get_all_records())[:-1]
    worksheet_df = pd.DataFrame(worksheet.get_all_records())[:-1]
    company_list = worksheet_df_len['ETF COMPLEX POSITION'].values.tolist()
    strategist_pcr_signals, plot_links_list = strategist_pcr_signal(company_list)
    hist_vol_df = hist_vol_start()

    for i in range(len(worksheet_df_len)):

        # заполняем столбцы с формулами
        for k in range(len(worksheet_df_len)):
            worksheet_df['CURRENT PRICE'].iloc[k] = f'=GOOGLEFINANCE(A{k + 2},"price")'
            worksheet_df['WEIGHT PCR sparkline'].iloc[k] = f'=sparkline(sparkline!B{k + 1}:W{k + 1})'
            # worksheet_df['% BETA DELTA'].iloc[k] = f'=C{k + 2}/$C$27'

        # print(worksheet_df)

        tick = worksheet_df['ETF COMPLEX POSITION'].iloc[i]
        # print('TICKER = ', tick)
        # print(hist_vol_df[hist_vol_df['Symbol'] == tick]['Percentile'])
        iv_os = hist_vol_df[hist_vol_df['Symbol'] == tick]['Percentile']

        if iv_os.empty == True:
            iv_os = 'Empty'
        else:
            iv_os = iv_os.values[0] / 100

        # ----------------------------
        worksheet_df['O_S WEIGHT PCR SIGNAL'].iloc[i] = strategist_pcr_signals[i]
        worksheet_df['O_S Plot Link'].iloc[i] = plot_links_list[i]
        worksheet_df['IV O_S'].iloc[i] = iv_os


def get_pcr_run(tick_list):
    print('---------------------------')
    print('------------- Getting PCR ... --------------')
    print('---------------------------')

    strategist_pcr_signals, plot_links_list = strategist_pcr_signal(tick_list)

    return(strategist_pcr_signals, plot_links_list)


 