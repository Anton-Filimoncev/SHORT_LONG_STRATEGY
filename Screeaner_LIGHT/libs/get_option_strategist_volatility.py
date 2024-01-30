import pandas as pd
import numpy as np
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from concurrent.futures.thread import ThreadPoolExecutor
import os
from time import sleep
from selenium.webdriver.support.ui import WebDriverWait
import pickle


def hist_vol():
    barcode = 'dranatom'
    password = 'MSIGX660'
    # ____________________ Работа с Selenium ____________________________
    path = os.path.join(os.getcwd(), 'chromedriver.exe')
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument("--start-maximized")
    # chrome_options.add_argument('headless')

    checker = webdriver.Chrome(options=chrome_options)

    checker.get(f'https://www.optionstrategist.com/subscriber-content/volatility-history')
    sleep(1)

    try:
        close_popup_start = checker.find_element(By.XPATH, '/html/body/div[3]/div[2]/button')
        close_popup_start.click()
    except:
        pass

    sign_in_userName = checker.find_element(by=By.ID, value="edit-name")
    sign_in_userName.send_keys(barcode)
    sign_in_password = checker.find_element(by=By.ID, value="edit-pass")
    sign_in_password.send_keys(password)
    sign_in = checker.find_element(by=By.ID,
                                   value='''edit-submit''')
    sign_in.click()
    sleep(1)
    try:
        close_popup = checker.find_element(By.XPATH, '/html/body/div[3]/div[2]/button')
        close_popup.click()
    except:
        pass

    select_fresh_txt = checker.find_element(By.XPATH,
                                            '''//*[@id="node-35"]/div/div/div/div/table[1]/tbody/tr[1]/td[1]''')

    select_fresh_txt.click()
    sleep(1)

    try:
        close_popup = checker.find_element(By.XPATH, '/html/body/div[3]/div[2]/button')
        close_popup.click()
    except:
        pass

    html_txt = checker.page_source
    full_txt = (html_txt[html_txt.index('800-724-1817'):html_txt.rindex('800-724-1817')]).replace('* ', '').replace(
        '^ ', '')


    print(full_txt)

    # with open('data.txt', 'r') as file:
    #     full_txt = file.read()

    col_name_replace = 'Symbol (option symbols)           hv20  hv50 hv100    DATE   curiv Days/Percentile Close'
    full_txt = full_txt.replace(col_name_replace, '').replace('\n', ' ')

    # with open('data.txt', 'w') as f:
    #     f.write(full_txt)

    checker.close()

    return full_txt, col_name_replace


def hist_vol_analysis(full_txt, col_name_replace):
    data_list = list(filter(len, full_txt.split(' ')))

    print('data_list')
    print(data_list)

    row_list = []

    for value_num in range(len(data_list)):
        try:
            float(data_list[value_num + 1])
            float(data_list[value_num + 2])
            float(data_list[value_num + 3])
            float(data_list[value_num + 4])
            float(data_list[value_num + 5])
            float(data_list[value_num + 8])
            row_list.append(data_list[value_num:value_num + 9])
        except:
            pass

    unnamed_df = pd.DataFrame(row_list)
    print('unnamed_df')
    print(unnamed_df)

    print('col_name_replace')
    print(col_name_replace)

    unnamed_df.columns = ['Symbol', 'hv20', 'hv50', 'hv100', 'DATE', 'curiv', 'Days', 'Percentile', 'Close']
    # unnamed_df = unnamed_df.set_index('Sc')

    int_percentile = []

    for perc in range(len(unnamed_df)):
        try:
            int_percentile.append(float(unnamed_df.iloc[perc]['Percentile'].replace('%ile', '')))
        except:
            int_percentile.append(0)

    unnamed_df['Percentile'] = int_percentile

    for i in unnamed_df.columns.tolist():
        try:
            unnamed_df[i] = unnamed_df[i].astype(float)
        except Exception as err:
            pass

    print('unnamed_df')
    print(unnamed_df)

    return unnamed_df


def volatility_skew():
    barcode = 'dranatom'
    password = 'MSIGX660'
    # ____________________ Работа с Selenium ____________________________
    path = os.path.join(os.getcwd(), 'chromedriver.exe')
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument("--start-maximized")
    # chrome_options.add_argument('headless')

    checker = webdriver.Chrome(options=chrome_options)

    checker.get(f'https://www.optionstrategist.com/subscriber-content/volatility-skewing')
    sleep(1)

    sign_in_userName = checker.find_element(by=By.ID, value="edit-name")
    sign_in_userName.send_keys(barcode)
    sign_in_password = checker.find_element(by=By.ID, value="edit-pass")
    sign_in_password.send_keys(password)
    sign_in = checker.find_element(by=By.ID,
                                   value='''edit-submit''')
    sign_in.click()
    sleep(1)
    try:
        close_popup = checker.find_element(By.XPATH, '//*[@id="PopupSignupForm_0"]/div[2]/div[1]')
        close_popup.click()
    except:
        pass

    select_fresh_txt = checker.find_element(By.XPATH,
                                            '''//*[@id="node-34"]/div/div/div/div/table[1]/tbody/tr[1]/td[1]''')

    select_fresh_txt.click()
    sleep(1)

    html_txt = checker.page_source
    # print("The current url is:" + str(checker.page_source))

    full_txt = (html_txt[html_txt.index('volatilities)'):html_txt.rindex('</pre>')]).replace('volatilities)',
                                                                                             '').replace('^ ', '')

    # full_txt = full_txt

    print(full_txt)

    # with open('data.txt', 'r') as file:
    #     full_txt = file.read()

    col_name_replace = 'Stock    Price   TotOptVolu     Put Volu         Implied   VolofIV      Skew    Stk Hi    Stk Lo  Stk Last'
    full_txt = full_txt.replace(col_name_replace, '').replace('\n', ' ')

    # with open('data.txt', 'w') as f:
    #     f.write(full_txt)

    return full_txt, col_name_replace


def volatility_skew_analysis(full_txt, col_name_replace):
    data_list = list(filter(len, full_txt.split(' ')))

    print('data_list')
    print(data_list)

    row_list = []

    for value_num in range(len(data_list)):
        try:
            float(data_list[value_num + 2])
            float(data_list[value_num + 5])
            float(data_list[value_num + 7])
            float(data_list[value_num + 8])
            float(data_list[value_num + 9])
            float(data_list[value_num + 10])
            float(data_list[value_num + 11])
            float(data_list[value_num + 12])
            row_list.append(data_list[value_num:value_num + 13])
        except:
            pass

    unnamed_df = pd.DataFrame(row_list)
    print('unnamed_df')
    print(unnamed_df)

    print('col_name_replace')
    print(col_name_replace)
    # print(unnamed_df[1])

    unnamed_df = unnamed_df.drop(columns=1)
    unnamed_df.columns = ['Symbol', 'Price', 'TotOpt', 'Volu', 'Put', 'Volu', 'Implied', 'VolofIV', 'Skew', 'Stk Hi',
                          'Stk Lo', 'Stk Last']
    # unnamed_df = unnamed_df.set_index('Sc')

    int_percentile = []

    for i in unnamed_df.columns.tolist():
        try:
            unnamed_df[i] = unnamed_df[i].astype(float)
        except Exception as err:
            pass

    print('unnamed_df')
    print(unnamed_df)

    return unnamed_df


def option_strategist_volatility_start(tick_list):
    html_txt, col_name_replace = hist_vol()
    history_vol_df = hist_vol_analysis(html_txt, col_name_replace)

    curiv = []
    Percentile = []

    for tick in tick_list:
        try:
            curiv.append(history_vol_df[history_vol_df['Symbol'] == tick]['curiv'].reset_index(drop=True).iloc[0])
            Percentile.append(history_vol_df[history_vol_df['Symbol'] == tick]['Percentile'].reset_index(drop=True).iloc[0])
        except:
            curiv.append(np.nan)
            Percentile.append(np.nan)

    return curiv, Percentile







