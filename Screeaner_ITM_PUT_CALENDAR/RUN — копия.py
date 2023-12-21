import numpy as np
import pandas as pd
import yfinance as yf
from libs.get_company_signals import get_company_signals_run



if __name__ == "__main__":
    active_stock_df = pd.read_excel("active_stock_df.xlsx")

    # # ###############################################
    print(active_stock_df['IV DIA year'])
    active_stock_df = active_stock_df.dropna(subset=['IV DIA year']).reset_index(drop=True)
    # переопределяем список тикеров после отсева по волатильности


    for i in range(len(active_stock_df['IV DIA year'])):
        try:
            active_stock_df['IV DIA year'].iloc[i] = float(active_stock_df['IV DIA year'].iloc[i])
        except:
            pass
    active_stock_df = active_stock_df[active_stock_df['IV DIA year'] == 1].reset_index(drop=True)
    tick_list = active_stock_df["Symbol"].values.tolist()
    print(active_stock_df)
    # Получаем краткосрочный тренд и RSI
    time.sleep(3)
    trend_signal_list, rsi_list, cur_price_list = get_tech_run(stock_yahoo, tick_list)

    active_stock_df["Current Price"] = cur_price_list
    active_stock_df["RSI"] = rsi_list
    active_stock_df["Trend"] = trend_signal_list
    print(active_stock_df)
    #
    time.sleep(3)
    # Получаем таблицы с трендом в папку Scored_dfs файлы score_LONG.xlsx и score_SHORT.xlsx
    regime_list, relative_regime_list = get_company_signals_run(
        stock_yahoo, tick_list, poll_num
    )  # pooled

    active_stock_df["Regime"] = regime_list
    active_stock_df["Relative Regime"] = relative_regime_list
    #
    print(active_stock_df)
    # Получаем PCR сигналы со стратегиста
    # strategist_pcr_signals, plot_links_list = get_pcr_run(tick_list)
    # active_stock_df['PCR SIGNAL'] = strategist_pcr_signals
    # active_stock_df['PCR Link'] = plot_links_list
    # print(active_stock_df)


    # MAX PAIN
    #
    max_pain = get_max_pain_run(tick_list, stock_yahoo, poll_num)
    active_stock_df["Max_Pain"] = max_pain

    print(active_stock_df)

    # Для тикеров с 1 первым диапазоном и релативным регтаймом -1 - собираем медвежий календарь
    RR, needed_strike_sell, proba_30 = itm_calendar_run(
        active_stock_df, stock_yahoo, tick_list, poll_num
    )

    active_stock_df["Strike"] = needed_strike_sell
    active_stock_df["RR"] = RR
    active_stock_df["Proba 30"] = proba_30

    momentum = get_momentum_run(stock_yahoo, tick_list)

    active_stock_df["Momentum"] = momentum

    active_stock_df.to_excel("SCREENER_RESULT.xlsx", index=False)
