import threading
import random
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
import os
import datetime
from dateutil.relativedelta import relativedelta
from time import sleep
from pathlib import Path


def get_div_data(tickers_list):
    print('Remove old files...')
    for filename in os.listdir(f'div_data'):
        os.remove(f'div_data\{filename}')

    barcode = 'fin@ss-global-group.com'
    password = 'MSIGX660'
    # ____________________ Работа с Selenium ____________________________
    path = os.path.join(os.getcwd(), 'chromedriver.exe')
    chrome_options = Options()
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument("--start-maximized")
    # chrome_options.add_argument('headless')
    path_folder = f'{Path(__file__).parent.absolute()}\div_data'
    print(path_folder)

    prefs = {"download.default_directory": f'''{Path(__file__).parent.absolute()}\div_data''',
             "download.prompt_for_download": False,
             'profile.managed_default_content_settings.javascript': 1,
             'profile.managed_default_content_settings.images': 1,
             "download.directory_upgrade": True}  # set path

    chrome_options.add_experimental_option("prefs", prefs)  # set option

    checker = webdriver.Chrome(options=chrome_options)
    checker.get('chrome://settings/')
    checker.execute_script('chrome.settingsPrivate.setDefaultZoom(0.6);')

    checker.get(f'https://www.gurufocus.com/')
    sleep(random.randint(10, 25))
    sign_in_click = checker.find_element(by=By.LINK_TEXT, value='''Login''')
    sign_in_click.click()
    sleep(5)
    sign_in_userName = checker.find_element(by=By.ID, value="login-dialog-name-input")
    sign_in_userName.send_keys(barcode)
    sign_in_password = checker.find_element(by=By.ID, value="login-dialog-pass-input")
    sign_in_password.send_keys(password)
    sleep(4)
    sign_in = checker.find_element(by=By.XPATH,
                                   value='''//*[@id="__layout"]/div/div/div[2]/form/div[6]/button/span/span/div''')
    sign_in.click()
    sleep(10)

    for ticker in tickers_list:
        try:


            print(f'Downloading: {ticker} ...')
            url_f = f"https://www.gurufocus.com/etf/{ticker}/summary#etf-main|fundamental-chart"
            checker.get(url_f)
            # sleep(0.5)
            # html = checker.find_element(by=By.XPATH, value='/html/body')
            # html.click()

            sleep(7)

            select_all = checker.find_element(By.XPATH,
                                              '''/html/body/div/div/section/section/main/div[1]/div[4]/div[2]/div/div[1]/div[1]/div[2]/div/div/div[3]/div[2]/div/div/div[2]/div/span/div/span''')
            select_all.click()
            sleep(2)

            export = checker.find_element(By.XPATH, '''/html/body/ul/li[2]/span''')

            export.click()
            sleep(5)

            try:
                os.remove(f'guru_files\{ticker}.xlsx')
            except:
                pass

            os.chdir(path_folder)
            files = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)
            newest = files[-1]
            os.rename(newest, ticker+'.xlsx')


            # print('Go to next ticker')
        except Exception as e:
            print(e)
            print('Cant download file')

    sleep(3)

def run_guru(tick_list, download_fresh_data):
    # download div data
    if download_fresh_data:
        get_div_data(tick_list)

    div_score_list = []

    for tick in tick_list:
        print('TICKER = ', tick)
        try:
            if download_fresh_data:
                div_data = pd.read_excel(f'{Path(__file__).parent.absolute()}\div_data\{tick}.xlsx')[2:]
                div_data = div_data.T[::-1][:-1].reset_index(drop=True)
                div_data.columns = [['Date', 'Dividend Yield']]

                data_val_list = []
                div_list = []
                for div_num in range(len(div_data)):
                    data_val_list.append(div_data['Date'].iloc[div_num].values[0])
                    div_list.append(div_data['Dividend Yield'].iloc[div_num].values[0])


                new_df = pd.DataFrame({
                    'Date': data_val_list,
                    'Dividend Yield': div_list,
                })

                new_df['Date'] = pd.to_datetime(new_df['Date'], format='%Y-%m-%d', errors='coerce')
                new_df = new_df.set_index('Date')
                new_df.to_excel(f'{Path(__file__).parent.absolute()}\div_data\{tick}.xlsx')

            else:
                new_df = pd.read_excel(f'{Path(__file__).parent.absolute()}\div_data\{tick}.xlsx')
                new_df = new_df.set_index('Date')
            current_div_data = new_df['Dividend Yield'].iloc[-1]
            median_last_days = datetime.datetime.now() - relativedelta(days=450)
            median_div_data = new_df['Dividend Yield'][median_last_days:].median()
            div_score_list.append(float(current_div_data) / median_div_data)

        except:
            div_score_list.append('EMPTY')

    return div_score_list

