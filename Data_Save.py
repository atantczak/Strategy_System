'''
Andrew Antczak
July 2020


This code will download daily pricing data, on a minutely basis, to our SP500 directory.
We are choosing to limit this data storage to the SP500 with the obvious knowledge that his can be changed and re-organized at any time.
'''

import os
import functools
import requests
import bs4 as bs
import datetime as dt
from datetime import timedelta
import math
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import style
import alpaca_trade_api as tradeapi
import numpy as np
import random as rand
import pandas as pd
import pickle
import multiprocessing as mp
from multiprocessing import Lock, Value
from dateutil.parser import parse
import time as tm
from Functions.Alpaca_Key_Store import initiate_API_keys
# ---------------------------------------------------------------------------------------------------------------------#
'''
This block of code is connecting the code to our Alpaca account which gives us unlimited access to Polygon's API.
The data we use from Polygon is strictly pricing data; in this case, minutely. 

NOTE: This code will not function in its raw form as the API Keys have not been included, for obvious reasons.
'''

ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_API_BASE_URL = initiate_API_keys()
ALPACA_PAPER = True
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_API_BASE_URL, 'v2')
# ---------------------------------------------------------------------------------------------------------------------#
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class Data_Save():
    def __init__(self, start_date, end_date):
        self.start_date = start_date
        self.end_date = end_date

        return

    def new_upload(self, tickers):

        today = time.mktime(dt.datetime.strptime(self.start_date, "%Y-%m-%d").timetuple())
        s_today = str(dt.datetime.fromtimestamp(today))[0:10]
        end_date = False
        # We are saving files per stock, per day.

        while end_date is False:

            today = s_today
            today = time.mktime(dt.datetime.strptime(today, "%Y-%m-%d").timetuple())

            s_ts = str(dt.datetime.fromtimestamp(today) + dt.timedelta(minutes=570))[0:19]
            s_ts = time.mktime(dt.datetime.strptime(s_ts, "%Y-%m-%d %H:%M:%S").timetuple())
            e_ts = str(dt.datetime.fromtimestamp(today) + dt.timedelta(minutes=960))[0:19]
            e_ts = time.mktime(dt.datetime.strptime(e_ts, "%Y-%m-%d %H:%M:%S").timetuple())

            for ticker in tickers:
                try:
                    data = api.polygon.historic_agg_v2(str(ticker), 1, timespan='minute', _from=s_today,
                                                       to=s_today).df
                except:
                    continue

                df = data.reset_index()

                if len(df) == 0:
                    continue

                for i in range(0,len(df)):
                    curr_time = str(df['timestamp'][i])[0:19]
                    curr_time_ts = time.mktime(dt.datetime.strptime(curr_time, "%Y-%m-%d %H:%M:%S").timetuple())

                    if s_ts >= curr_time_ts:
                        in_s = i
                    if curr_time_ts <= e_ts:
                        in_e = i

                try:
                    df = df.truncate(before=in_s, after=in_e)
                    l_df = len(df)
                    # Should a certain stock not have enough data on a certain day, we skip that stock on that day.
                    # I created the threshold to be roughly 85% of the day's pricing data. 
                    if l_df < 330:
                        df = df.empty
                        continue
                    else:
                        pass
                except:
                    continue

                df = df.reset_index(drop=True)
                filename = '../Data_Store/SP500_Price_Data/{}_{}.pk'.format(ticker, s_today)
                # This depends on where the pricing data is located on your local drive.
                # Here, I've left it in its original form for example.
                with open(filename, 'wb') as file:
                    pickle.dump(df, file)

                print("{}_{}.pk".format(ticker, s_today) + " has been downloaded.")

            s_today = str(dt.datetime.fromtimestamp(today) + dt.timedelta(days=1))[0:10]
            if s_today == self.end_date:
                end_date = True
            else:
                continue

        return

'''
This chunk of code is originally from sentdex (YouTuber). It simply grabs the tickers for the SP500

resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, "lxml")
table = soup.find('table', {'class': 'wikitable sortable'})
tickers_trial = []
for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text.strip()
    mapping = str.maketrans(".", "-")
    ticker = ticker.translate(mapping)
    ticker = ticker.replace("-", ".")
    tickers_trial.append(ticker)

tickers = tickers_trial
'''

# Add any date range you want.
ds = Data_Save('2020-08-05', '2020-10-14')


# The remaining lines of code are meant to activate the above function such that the download will begin. 
# More importantly, however, is that we are utilizing multiprocessing capabilities to make the download occur quicker.

t_len = float(len(tickers))

groups = 16
groups = groups
g = math.floor((t_len) / groups)
if g < 1:
    g = 1


def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i + n]


tick_chunks = list(chunks(tickers, g))

if __name__ == '__main__':
    pool = mp.Pool(groups)
    pool.map(functools.partial(ds.new_upload), tick_chunks)


