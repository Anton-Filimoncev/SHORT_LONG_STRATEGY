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
from libs.get_earnings import run_earnings_get
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

    # active_stock_df = pd.read_excel("active_stock_df.xlsx")[:10]

    # Получаем PCR сигналы со стратегиста
    active_stock_df = get_pcr_run()

    # # переопределяем список тикеров после отсева по PCR PERCENTILE
    active_stock_df = active_stock_df[active_stock_df['PCR PERCENTILE'] <= PCR_PERCENTILE_end].reset_index(drop=True)
    active_stock_df = active_stock_df[active_stock_df['PCR PERCENTILE'] >= PCR_PERCENTILE_start].reset_index(drop=True)
    active_stock_df = active_stock_df.drop_duplicates(subset=['Symbol'], keep=False)
    tick_list = active_stock_df["Symbol"].values.tolist()
    print(active_stock_df)

    curiv, Percentile = option_strategist_volatility_start(tick_list)
    active_stock_df["IV"] = curiv
    active_stock_df["IV_Percentile"] = Percentile

    for num in range(len(active_stock_df["Symbol"])):
        symb = active_stock_df["Symbol"].iloc[num]
        if "." in symb:
            active_stock_df["Symbol"].iloc[num] = symb.replace(".", "-")
    #
    tick_list = active_stock_df["Symbol"].values.tolist()
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

    proba, target_price = get_proba_touch_run(tick_list, stock_yahoo, PROBA_TOUCH_DAYS, PROBA_TOUCH_DIR, STD_NUM, active_stock_df)
    active_stock_df['Probability of touching'] = proba
    active_stock_df['Probability of touching Price'] = target_price


    # Получаем дни до отчетности и EVR
    earnings_list, evr_list = run_earnings_get(tick_list)
    active_stock_df["Earnings"] = earnings_list
    active_stock_df["EVR"] = evr_list

    time.sleep(3)
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

    momentum, hold, hurst = get_momentum_run(stock_yahoo, tick_list)

    active_stock_df["Momentum"] = momentum
    active_stock_df["Hold"] = hold
    active_stock_df["Hurst"] = hurst

    print(active_stock_df)

    active_stock_df.to_excel("LIGHT_SCREENER_RESULT.xlsx", index=False)
