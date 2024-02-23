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
from libs.get_sell_put import sell_put_run
from libs.get_risk_reversal import risk_reversal_run
from libs.get_bear_calendar import bear_calendar_run
from libs.get_sell_call import sell_call_run
from libs.get_strangle import strangle_run
from libs.get_beta import run_beta
from libs.get_trend_sector import get_trend_df
from libs.get_bollinger import rur_bollinger

# from get_mprp_call import mprp_call_run
import time
import os
from pathlib import Path


if __name__ == "__main__":
    poll_num = 4  # Количество потоков
    # Загружаем датафрейм с компаниями
    main_data = pd.read_excel("ETF_start.xlsx")
    print(main_data)

    tikers_for_sector_trend = main_data[main_data["Type"] == "Секторальные"][
        "Symbol"
    ].values.tolist()
    sector_for_sector_trend = main_data[main_data["Type"] == "Секторальные"][
        "Sector"
    ].values.tolist()
    full_tikers_list = main_data["Symbol"].values.tolist()

    sector_score = get_trend_df(tikers_for_sector_trend, sector_for_sector_trend)
    main_data.loc[main_data['Type'] == 'Секторальные', 'Sector_Trend'] = sector_score

    stock_yahoo = yf.download(full_tikers_list, group_by="ticker")
    stock_yahoo.index = pd.to_datetime(stock_yahoo.index)
    print(stock_yahoo)
    tick_list_reg = main_data[main_data['Type'] != 'Секторальные']['Symbol'].values.tolist()

    stock_yahoo_regime = yf.download(tick_list_reg, group_by="ticker", interval="1d", auto_adjust=True)
    reg_trend_signals, rel_reg_trend_signals = get_company_signals_run(stock_yahoo_regime, tick_list_reg, poll_num)
    print('reg_trend_signals')
    print(reg_trend_signals)
    print('rel_reg_trend_signals')
    print(rel_reg_trend_signals)
    main_data.loc[main_data['Type'] != 'Секторальные', 'Regime'] = reg_trend_signals
    main_data.loc[main_data['Type'] != 'Секторальные', 'Relative Regime'] = rel_reg_trend_signals

    # Получаем краткосрочный тренд и RSI
    trend_signal_list, rsi_list, cur_price_list = get_tech_run(
        stock_yahoo, full_tikers_list
    )

    main_data["Current Price"] = cur_price_list
    main_data["RSI"] = rsi_list
    main_data["Trend"] = trend_signal_list

    # Получаем Бэту
    main_data["Beta"] = run_beta(full_tikers_list, stock_yahoo)

    # Получаем UP DOWN trend
    val_list, regression_list = up_down_volume_run(full_tikers_list, stock_yahoo)
    main_data["UP DOWN Trend"] = val_list
    main_data["UP DOWN Line"] = regression_list


    print(main_data)

    # Получаем сигнал по Боллинжеру
    main_data["Bollinger"] = rur_bollinger(full_tikers_list, stock_yahoo)
    #
    IV_percentile, IV_Regime, IV_Median, IV, HV_20, HV_50, HV_100, HV_200, HV_Regime = get_ib_run(full_tikers_list, poll_num)
    main_data['IV % year'] = IV_percentile
    main_data['IV DIA year'] = IV_Regime
    main_data['IV median 6 m'] = IV_Median
    main_data['IV ATM'] = IV
    main_data['HV 20'] = HV_20
    main_data['HV 50'] = HV_50
    main_data['HV 100'] = HV_100
    main_data['HV 200'] = HV_200
    main_data['HV DIA'] = HV_Regime
    #
    # print(main_data)

    # # Для тикеров у которых 2-3-4 диапазон по волатильности - собираем данные по продаже путов и коллов
    sell_put_result_rpop_50, sell_put_result_ROCday = sell_put_run(
        main_data, stock_yahoo, full_tikers_list, poll_num
    )
    time.sleep(3)

    sell_call_result = sell_call_run(main_data, stock_yahoo, full_tikers_list, poll_num)

    main_data["SELL_PUT_PoP"] = sell_put_result_rpop_50
    main_data["SELL_PUT_ROCday"] = sell_put_result_ROCday
    main_data["SELL_CALL"] = sell_call_result
    print(main_data)
    #
    # # # Для тикеров с 2-3-4 диа собираем стренглы
    strangle_data = strangle_run(
        main_data, stock_yahoo, full_tikers_list, poll_num
    )  # pooled
    main_data["STRANGLE"] = strangle_data
    print(main_data)
    main_data.to_excel("ETF_result.xlsx", index=False)
