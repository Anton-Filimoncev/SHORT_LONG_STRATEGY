import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


def calculate_probability(ticker, nb_simulations, days, target_price, below_price,
                          above_price, STD_NUM, PROBA_TOUCH_DIR):
    # Fetch historical market data
    hist_data = yf.download(ticker, start='2000-01-01')

    # Calculate the log returns
    log_returns = np.log(1 + hist_data['Adj Close'].pct_change())

    # Define the variables
    u = log_returns.mean()
    var = log_returns.var()

    # Calculate drift and standard deviation
    drift = u - (0.5 * var)
    stddev = log_returns.std()

    # Generate a random variable
    daily_returns = np.exp(drift + (stddev * STD_NUM) * np.random.standard_normal((days, nb_simulations)))

    # Simulate the price paths
    s0 = hist_data['Adj Close'][-1]
    price_list = np.zeros_like(daily_returns)
    price_list[0] = s0
    for t in range(1, days):
        price_list[t] = price_list[t - 1] * daily_returns[t]

    # plt.hist(price_list[-1], bins=50)
    # plt.show()

    # Calculate probabilities
    # final_in_range = np.logical_and(lower_bound <= price_list[-1],
    #                                 price_list[-1] <= upper_bound).sum() / nb_simulations
    # during_out_of_range = (np.logical_or(price_list < lower_bound,
    #                                      price_list > upper_bound).sum(axis=0) > 0).sum() / nb_simulations
    if PROBA_TOUCH_DIR == 'UP':
        touch_target_price = (price_list >= target_price).any(axis=0).sum() / nb_simulations
    else:
        touch_target_price = (price_list <= target_price).any(axis=0).sum() / nb_simulations
    below_end_price = (price_list[-1] <= below_price).sum() / nb_simulations
    above_end_price = (price_list[-1] >= above_price).sum() / nb_simulations

    return touch_target_price, below_end_price, above_end_price


def get_proba_touch_run(tick_list, stock_yahoo, PROBA_TOUCH_DAYS, PROBA_TOUCH_DIR, STD_NUM, active_stock_df):
    print('---------------------------')
    print('------------- Probability of Touching --------------')
    print('---------------------------')
    proba_list = []
    target_price_list = []
    for ticker in tick_list:
        hv = active_stock_df[active_stock_df['Symbol'] == ticker]['HV'].reset_index(drop=True).iloc[0]
        # lower_bound = 55
        # upper_bound = 70
        nb_simulations = 10000
        current_price = stock_yahoo[ticker]['Close'].iloc[-1]
        exp_move = hv * current_price * math.sqrt(PROBA_TOUCH_DAYS / 365)
        if PROBA_TOUCH_DIR == 'UP':
            target_price = current_price + exp_move # add your target touch price here
        else:
            target_price = current_price - exp_move

        below_price = current_price - exp_move  # price below at the end
        above_price = current_price + exp_move  # price above at the end

        p3, p4, p5 = calculate_probability(ticker, nb_simulations, PROBA_TOUCH_DAYS, target_price, below_price, above_price, STD_NUM, PROBA_TOUCH_DIR)

        # print(f'Probability of being in the range at the end of the period: {p1 * 100}%')
        # print(f'Probability of leaving the range at some point during the period: {p2 * 100}%')
        # print(f'Probability of touching the target price at some point during the period: {p3 * 100}%')
        # print(f'Probability of being below the target price at the end of the period: {p4 * 100}%')
        # print(f'Probability of being above the target price at the end of the period: {p5 * 100}%')
        proba_list.append(p3)
        target_price_list.append(target_price)

    return proba_list, target_price_list