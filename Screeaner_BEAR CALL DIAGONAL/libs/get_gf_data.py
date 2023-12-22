import pandas as pd
import numpy as np
import requests



def get_gf_data(tick_list):
    api_token = '34b27ff5b013ecb09d2eeafdf8724472:2ab9f1f92f838c3c431fc85e772d0f6c'
    gf_score_list = []
    gf_valuation_list = []

    for tick in tick_list:

        url = f"https://api.gurufocus.com/public/user/{api_token}/stock/{tick}/summary"
        response_exp = requests.request("GET", url).json()
        print(response_exp)
        guru_df = pd.DataFrame(response_exp['summary'])
        gf_score = guru_df['general']['gf_score']
        gf_valuation = guru_df['general']['gf_valuation']
        gf_score_list.append(gf_score)
        gf_valuation_list.append(gf_valuation)

    return gf_score_list, gf_valuation_list

