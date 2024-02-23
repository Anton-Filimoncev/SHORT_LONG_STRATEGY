import numpy as np
import pandas as pd
import yfinance as yf
from libs.get_company_signals import get_company_signals_run
# from get_strategist_vol import get_strategist_vol_run
from libs.get_tech import get_tech_run
from libs.get_pcr import get_pcr_run
from libs.get_ib_volatility import get_ib_run
from libs.get_up_down_volume import up_down_volume_run
# from get_revard_risk import revard_risk_run
from libs.get_bear_call_diagonal import bear_call_diagonal_run
from libs.get_max_pain import get_max_pain_run
from libs.get_momentum import get_momentum_run
from libs.get_earnings import peat_run
from libs.get_itm_calendar import itm_calendar_run
from libs.get_otm_calendar import otm_calendar_run
from libs.get_option_strategist_volatility import option_strategist_volatility_start
from libs.get_volatilty_calc import get_volatility_run
from libs.get_proba_touch import get_proba_touch_run
from libs.get_beta import get_beta_run
# from get_mprp_call import mprp_call_run
import time
import os
from pathlib import Path


if __name__ == "__main__":
    poll_num = 4  # Количество потоков
    RISK_RATE = 4
    PCR_PERCENTILE_start = 80
    PCR_PERCENTILE_end = 100
    BETA_BENCHMARK = 'SPY'
    PROBA_TOUCH_DAYS = 300
    PROBA_TOUCH_DIR = 'UP'
    STD_NUM = 1

    tick_data = pd.read_excel('active_stock_df.xlsx')
    tick_list = tick_data['Symbol']

    # Получаем дни до отчетности
    active_stock_df = peat_run(tick_list)
    tick_list = active_stock_df["Symbol"].values.tolist()

    # active_stock_df = pd.read_excel("active_stock_df.xlsx")[:10]

    curiv, Percentile = option_strategist_volatility_start(tick_list)
    active_stock_df["IV"] = curiv
    active_stock_df["IV_Percentile"] = Percentile

    for num in range(len(active_stock_df["Symbol"])):
        symb = active_stock_df["Symbol"].iloc[num]
        if "." in symb:
            active_stock_df["Symbol"].iloc[num] = symb.replace(".", "-")
    #

    print('tick_list')
    print(tick_list)
    # Загружаем ценовые ряды из яхуу
    stock_yahoo = yf.download(tick_list+[BETA_BENCHMARK], group_by="ticker").dropna(axis=1, how='all')
    # фильтруем датасет, оставляем тикеры которые есть на yf
    tick_list = list(stock_yahoo.columns.get_level_values(0).unique())
    print(len(tick_list))
    print(tick_list)
    print(len(active_stock_df))
    print(len(active_stock_df[active_stock_df['Symbol'].isin(tick_list)]))
    active_stock_df = active_stock_df[active_stock_df['Symbol'].isin(tick_list)]
    tick_list = active_stock_df['Symbol'].values.tolist()
    stock_yahoo = yf.download(tick_list + [BETA_BENCHMARK], group_by="ticker").dropna(axis=1, how='all')

    hist_vol_list, hist_vol_stage_list = get_volatility_run(tick_list, stock_yahoo)
    print(len(hist_vol_list), len(tick_list))
    active_stock_df["HV"] = hist_vol_list
    active_stock_df["HV_Stage"] = hist_vol_stage_list

    beta_list = get_beta_run(tick_list, stock_yahoo, BETA_BENCHMARK)
    active_stock_df["Beta"] = beta_list

    # Получаем краткосрочный тренд и RSI
    time.sleep(3)
    trend_signal_list, rsi_list, cur_price_list = get_tech_run(stock_yahoo, tick_list)

    active_stock_df["Current Price"] = cur_price_list
    active_stock_df["RSI"] = rsi_list
    active_stock_df["Trend"] = trend_signal_list
    #
    # proba, target_price = get_proba_touch_run(tick_list, stock_yahoo, PROBA_TOUCH_DAYS, PROBA_TOUCH_DIR, STD_NUM, active_stock_df)
    # active_stock_df['Probability of touching'] = proba
    # active_stock_df['Probability of touching Price'] = target_price


    # time.sleep(3)
    # Получаем таблицы с трендом в папку Scored_dfs файлы score_LONG.xlsx и score_SHORT.xlsx
    # качаем приведенные данные по сплиту и дивам с яхи

    stock_yahoo_regime = yf.download(tick_list, group_by="ticker", interval="1d", auto_adjust=True)
    regime_list, relative_regime_list = get_company_signals_run(
        stock_yahoo_regime, tick_list, poll_num
    )  # pooled

    active_stock_df["Regime"] = regime_list
    active_stock_df["Relative Regime"] = relative_regime_list
    #

    # Получаем UP DOWN trend
    val_list, regression_list = up_down_volume_run(tick_list, stock_yahoo)
    active_stock_df["UP DOWN Trend"] = val_list
    active_stock_df["UP DOWN Line"] = regression_list

    # momentum, hold, hurst = get_momentum_run(stock_yahoo, tick_list)
    #
    # active_stock_df["Momentum"] = momentum
    # active_stock_df["Hold"] = hold
    # active_stock_df["Hurst"] = hurst

    # PUT
    active_stock_df_put = active_stock_df[active_stock_df['Relative Regime'] == -1]
    # Для тикеров с 1 первым диапазоном и релативным регтаймом -1 - собираем медвежий календарь
    RR, needed_strike_sell, needed_strike_buy, proba_30, expected_return = itm_calendar_run(
        active_stock_df_put, stock_yahoo, tick_list, poll_num, RISK_RATE
    )

    active_stock_df_put["Strike_SHORT"] = needed_strike_sell
    active_stock_df_put["Strike_LONG"] = needed_strike_buy
    active_stock_df_put["RR"] = RR
    active_stock_df_put["Proba 30"] = proba_30
    active_stock_df_put["Expected Return"] = expected_return
    active_stock_df_put['Strat'] = 'PUT_Calendar'


    # CALL
    active_stock_df_call = active_stock_df[active_stock_df['Relative Regime'] == 1]
    print('active_stock_df_call')
    print(active_stock_df_call)
    # Для тикеров с 1 первым диапазоном и релативным регтаймом -1 - собираем медвежий календарь
    # Для тикеров с 1 первым диапазоном и релативным регтаймом -1 - собираем медвежий календарь
    caledar_call_score, needed_strike_sell, needed_strike_buy, proba_30, expected_return = otm_calendar_run(
        active_stock_df_call, stock_yahoo, tick_list, poll_num, RISK_RATE
    )

    active_stock_df_call["Caledar Call Score"] = caledar_call_score
    active_stock_df_call["Strike_SHORT"] = needed_strike_sell
    active_stock_df_call["Strike_LONG"] = needed_strike_buy
    active_stock_df_call["Proba 30"] = proba_30
    active_stock_df_call["Expected Return"] = expected_return
    active_stock_df_call['Strat'] = 'CALL_Calendar'


    pd.concat([active_stock_df_put, active_stock_df_call], axis=0).to_excel("PRE_PEAT_SCREENER_RESULT.xlsx", index=False)