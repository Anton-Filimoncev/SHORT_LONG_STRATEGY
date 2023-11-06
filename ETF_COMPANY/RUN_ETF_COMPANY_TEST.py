import numpy as np
import pandas as pd
import yfinance as yf
from libs.get_company_signals import get_company_signals_run

# from get_strategist_vol import get_strategist_vol_run
from libs.get_tech import get_tech_run
from libs.get_pcr import get_pcr_run
from libs.get_ib_volatility import get_ib_run

# from get_revard_risk import revard_risk_run
from libs.get_sell_put import sell_put_run
from libs.get_risk_reversal import risk_reversal_run
from libs.get_bear_calendar import bear_calendar_run
from libs.get_sell_call import sell_call_run
from libs.get_strangle import strangle_run

# from get_mprp_call import mprp_call_run
import time
import os
from pathlib import Path


if __name__ == "__main__":
    poll_num = 8  # Количество потоков
    # Загружаем датафрейм с компаниями
    active_stock_df = pd.read_excel("ETF_COMPANY.xlsx")[:20]

    print(active_stock_df)
    #
    # active_stock_df = active_stock_df[active_stock_df["Last"] > 10].reset_index(
    #     drop=True
    # )
    #
    # for num in range(len(active_stock_df["Symbol"])):
    #     symb = active_stock_df["Symbol"].iloc[num]
    #     if "." in symb:
    #         active_stock_df["Symbol"].iloc[num] = symb.replace(".", "-")

    tick_list = active_stock_df["Symbol"].values.tolist()
    # Загружаем ценовые ряды из яхуу
    stock_yahoo = yf.download(tick_list, group_by="ticker")
    #
    # # Получаем краткосрочный тренд и RSI
    # trend_signal_list, rsi_list, cur_price_list = get_tech_run(stock_yahoo, tick_list)
    #
    # active_stock_df["Current Price"] = cur_price_list
    # active_stock_df["RSI"] = rsi_list
    # active_stock_df["Trend"] = trend_signal_list
    #
    # # Получаем таблицы с трендом в папку Scored_dfs файлы score_LONG.xlsx и score_SHORT.xlsx
    # regime_list, relative_regime_list = get_company_signals_run(
    #     stock_yahoo, tick_list, poll_num
    # )  # pooled
    #
    # active_stock_df["Regime"] = regime_list
    # active_stock_df["Relative Regime"] = relative_regime_list
    #
    # # Получаем PCR сигналы со стратегиста
    # # strategist_pcr_signals, plot_links_list = get_pcr_run(tick_list)
    # # active_stock_df['PCR SIGNAL'] = strategist_pcr_signals
    # # print(active_stock_df)
    #
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

    # Для тикеров у которых 2-3-4 диапазон по волатильности - собираем данные по продаже путов и коллов

    sell_put_result_rpop_50, sell_put_result_ROCday = sell_put_run(
        active_stock_df, stock_yahoo, tick_list, poll_num
    )
    time.sleep(3),

    sell_call_result = sell_call_run(active_stock_df, stock_yahoo, tick_list, poll_num)

    active_stock_df["SELL_PUT_PoP"] = sell_put_result_rpop_50
    active_stock_df["SELL_PUT_ROCday"] = sell_put_result_ROCday
    active_stock_df["SELL_CALL"] = sell_call_result

    # Для тикеров с 2-3 диа собираем стренглы
    strangle_data = strangle_run(
        active_stock_df, stock_yahoo, tick_list, poll_num
    )  # pooled
    active_stock_df["STRANGLE"] = strangle_data
    #
    # # Для тикеров с 1 первым диапазоном и релативным регтаймом 1 - собираем риск реверсал
    risk_reversal_data = risk_reversal_run(
        active_stock_df, stock_yahoo, tick_list, poll_num
    )

    # active_stock_df[["RISK/REVERSAL Score", "RISK/REVERSAL PRICE"]] = np.nan

    print(risk_reversal_data)
    active_stock_df[["RISK/REVERSAL Score", "RISK/REVERSAL PRICE"]] = risk_reversal_data
    # active_stock_df["RISK/REVERSAL PRICE"] = risk_reversal_data
    print(active_stock_df[["RISK/REVERSAL Score", "RISK/REVERSAL PRICE"]])
    #
    # Для тикеров с 1 первым диапазоном и релативным регтаймом -1 - собираем медвежий календарь
    bear_calendar_data = bear_calendar_run(
        active_stock_df, stock_yahoo, tick_list, poll_num
    )
    active_stock_df["BEAR_CALENDAR"] = bear_calendar_data

    active_stock_df.to_excel("ETF_COMPANY.xlsx", index=False)
