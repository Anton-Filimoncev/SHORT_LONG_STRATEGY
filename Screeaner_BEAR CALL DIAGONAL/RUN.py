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
# from get_mprp_call import mprp_call_run
import time
import os
from pathlib import Path


if __name__ == "__main__":
    poll_num = 4  # Количество потоков
    # Загружаем датафрейм с компаниями

    RISK_RATE = 4

    # active_stock_df = pd.read_excel("active_stock_df.xlsx")[:10]

    active_stock_df = pd.read_excel("active_stock/HIGH_CAP_COMPANY.xlsx")

    print(active_stock_df)

    active_stock_df = active_stock_df[active_stock_df["Last"] > 10].reset_index(
        drop=True
    )

    for num in range(len(active_stock_df["Symbol"])):
        symb = active_stock_df["Symbol"].iloc[num]
        if "." in symb:
            active_stock_df["Symbol"].iloc[num] = symb.replace(".", "-")
    #
    tick_list = active_stock_df["Symbol"].values.tolist()
    # Загружаем ценовые ряды из яхуу
    stock_yahoo = yf.download(tick_list, group_by="ticker")


    # Получаем PCR сигналы со стратегиста
    strategist_pcr_signals, plot_links_list, strategist_signals_percentile = get_pcr_run(tick_list)
    active_stock_df['PCR SIGNAL'] = strategist_pcr_signals
    active_stock_df['PCR PERCENTILE'] = strategist_signals_percentile
    active_stock_df['PCR Link'] = plot_links_list
    print(active_stock_df)
    # # переопределяем список тикеров после отсева по PCR PERCENTILE
    active_stock_df = active_stock_df[active_stock_df['PCR PERCENTILE'] >= 20].reset_index(drop=True)
    tick_list = active_stock_df["Symbol"].values.tolist()
    print(active_stock_df)

    # # Получаем волатильность с ИБ
    # (
    #     IV_percentile,
    #     IV_Regime,
    #     IV_Median,
    #     IV,
    #     HV_20,
    #     HV_50,
    #     HV_100,
    #     HV_Regime,
    # ) = get_ib_run(tick_list, poll_num)
    # active_stock_df["IV % year"] = IV_percentile
    # active_stock_df["IV DIA year"] = IV_Regime
    # active_stock_df["IV median 6 m"] = IV_Median
    # active_stock_df["IV ATM"] = IV
    # active_stock_df["HV 20"] = HV_20
    # active_stock_df["HV 50"] = HV_50
    # active_stock_df["HV 100"] = HV_100
    # active_stock_df["HV DIA"] = HV_Regime
    # time.sleep(3)
    # print(active_stock_df)
    # active_stock_df.to_excel("active_stock_df.xlsx")
    # # ###############################################
    # print(active_stock_df['IV DIA year'])
    # for i in range(len(active_stock_df['IV DIA year'])):
    #     try:
    #         active_stock_df['IV DIA year'].iloc[i] = float(active_stock_df['IV DIA year'].iloc[i])
    #     except:
    #         active_stock_df['IV DIA year'].iloc[i] = float(active_stock_df['IV DIA year'].iloc[i])
    #         pass
    #
    # # переопределяем список тикеров после отсева по волатильности
    # active_stock_df = active_stock_df[active_stock_df['IV DIA year'] >= 2].reset_index(drop=True)
    # tick_list = active_stock_df["Symbol"].values.tolist()
    # print(active_stock_df)

    #
    # # Получаем краткосрочный тренд и RSI
    # time.sleep(3)
    # trend_signal_list, rsi_list, cur_price_list = get_tech_run(stock_yahoo, tick_list)
    #
    # active_stock_df["Current Price"] = cur_price_list
    # active_stock_df["RSI"] = rsi_list
    # active_stock_df["Trend"] = trend_signal_list
    # print(active_stock_df)
    # #
    #
    # # Получаем дни до отчетности и EVR
    # earnings_list, evr_list = run_earnings_get(tick_list)
    # active_stock_df["Earnings"] = earnings_list
    # active_stock_df["EVR"] = evr_list
    #
    # time.sleep(3)
    # # Получаем таблицы с трендом в папку Scored_dfs файлы score_LONG.xlsx и score_SHORT.xlsx
    # # качаем приведенные данные по сплиту и дивам с яхи
    #
    # stock_yahoo_regime = yf.download(tick_list, group_by="ticker", interval="1d", auto_adjust=True)
    # regime_list, relative_regime_list = get_company_signals_run(
    #     stock_yahoo_regime, tick_list, poll_num
    # )  # pooled
    #
    # active_stock_df["Regime"] = regime_list
    # active_stock_df["Relative Regime"] = relative_regime_list
    # #
    # # Получаем UP DOWN trend
    #
    # val_list, regression_list = up_down_volume_run(tick_list, stock_yahoo)
    # active_stock_df["UP DOWN Trend"] = val_list
    # active_stock_df["UP DOWN Line"] = regression_list
    #
    # print(active_stock_df)
    #
    # # MAX PAIN
    # #
    # # max_pain = get_max_pain_run(tick_list, stock_yahoo, poll_num)
    # # active_stock_df["Max_Pain"] = max_pain
    #
    # print(active_stock_df['HV 100'])
    #
    # # Для тикеров с 1 первым диапазоном и релативным регтаймом -1 - собираем медвежий календарь
    # rr_ratio,  proba_30, expected_return = bear_call_diagonal_run(
    #     active_stock_df, stock_yahoo, tick_list, poll_num, RISK_RATE
    # )
    #
    # active_stock_df["Reward/Risk Ratio"] = rr_ratio
    # active_stock_df["Proba 30"] = proba_30
    # active_stock_df["Expected Return"] = expected_return
    # #
    # momentum, hold, hurst = get_momentum_run(stock_yahoo, tick_list)
    #
    # active_stock_df["Momentum"] = momentum
    # active_stock_df["Hold"] = hold
    # active_stock_df["Hurst"] = hurst

    print(active_stock_df)

    active_stock_df.to_excel("SCREENER_RESULT.xlsx", index=False)
