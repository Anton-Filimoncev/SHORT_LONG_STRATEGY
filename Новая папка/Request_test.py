import pandas as pd
import requests
import json

KEY = '8cY6RMPP.l03n604cbpLaGLCZJOXjTO8Tqfl5aCGF'
ticker = 'AAPL'

x = requests.get(f"https://www.hedginglab.com:/api/beta/v1/option_trade/AAPL/?trade_type=DoubleDiagonal&apikey=demo&trade_date=2018-05-21")
print(x.text)
print(x.json())
print(pd.DataFrame(x.json()))
pd.DataFrame(x.json()).to_excel('test2.xlsx')