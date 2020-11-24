'''
Andrew Antczak
November 23rd, 2020

The purpose of this code is to show a somewhat generalized framework of the experimental set up I have created and
utilized throughout the past 5 or so months (05-01-2020 :: 11-23-2020).

This experimental framework is meant to track various strategies for trading equity and enable the user to design and
analyze the output of said experiment.

The function, within the experiment class, that is key is the "experiment()" function. This holds the algorithm
responsible for going through the day to day prices of each stock and, based on your input signals, determining if a
given stock at a given time is a buy/sell/hold. From here, you can track pretty much anything you want.

In the code below, I have left a version of an experiment I ran in tact such that one could see how this would
potentially work.
'''

import os
import datetime as dt
from datetime import timedelta
import requests
import bs4 as bs
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib import style
import alpaca_trade_api as tradeapi
import numpy as np
import random as rand
import pandas as pd
import time
import pickle
import multiprocessing as mp
from multiprocessing import Lock, Value
from dateutil.parser import parse
import time as tm
from Functions.Alpaca_Key_Store import initiate_API_keys
# ---------------------------------------------------------------------------------------------------------------------#
ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_API_BASE_URL = initiate_API_keys()
ALPACA_PAPER = True
api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, ALPACA_API_BASE_URL, 'v2')
# ---------------------------------------------------------------------------------------------------------------------#
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


class Experiment():
    def __init__(self, tickers, start_date, end_date):

        self.tickers = tickers

        # Times and Dates
        self.start_date = start_date
        self.end_date = end_date

        # Data Frame Initialization
        self.df = {}

        # Price Tracking
        self.price = {}
        self.max_change = {}
        self.r_pct_change = {}
        self.start_price = {}
        self.close_prices = {}
        self.n_open = {}
        self.buy_price = {}
        self.curr_max = {}

        # Iterative Measures
        self.c = 0
        self.b = {}
        self.max = {}
        self.min = {}
        self.buy_search = {}
        self.buy_act = {}
        self.sell_sig = {}
        self.avg_up = 0
        self.avg_down = 0

        # Counters
        self.count_seg_u = 0
        self.count_seg_d = 0

        # --------------------------------------------------------- #
        # Experimental Input/Output
        # Buy Signal, Positive Sell Signal, Negative Sell Signal
        self.b_sig = -0.0625
        self.p_s_sig = 0.065625
        self.n_s_sig = -0.24

        # Original Expection Value of return such that the first iteration will automatically reset the value.
        self.exp_ret = float("-inf")

        # This variable controls for how much we will change our sell signal by.
        self.delt_c = 0.05

        # This is a counter of how many runs we've been through such that we can avoid an infinite loop by restraining
        # this number from exceeding a preset value.
        self.experiment_count = 0

        #--------------------------------------------------------- #
        # This portion is initializing the output dataframe for your experimental results.
        self.experiment_df = pd.DataFrame()

        # One of these three will be used as the experimental index.
        self.pos_sell = []
        self.neg_sell = []
        self.buy_sig = []

        # These are the parameters we are tracking and recording for our experimental output dataframe.
        self.avg_f_l = []
        self.avg_up_l = []
        self.avg_down_l = []
        self.portion_u = []
        self.w_pct = []
        self.w_avg = []
        self.l_avg = []
        self.exp_val = []
        self.n_trades = []

    # On whichever day is given, this function is called from below and produces a dataframe for each stock of interest.
    def df_func_new(self, day):
        for ticker in self.tickers:
            self.df["{}".format(ticker)] = False
            if os.path.exists('Data_Storage/SP500_Price_Data/{}_{}.pk'.format(ticker, day)):
                filename = 'Data_Storage/SP500_Price_Data/{}_{}.pk'.format(ticker, day)
                with open(filename, 'rb') as file:
                    try:
                        self.df["{}".format(ticker)] = pd.DataFrame(pickle.load(file))
                    except:
                        self.df["{}".format(ticker)] = False
                continue
            else:
                self.df["{}".format(ticker)] = False

        return

    # Below is an example of a buying signal. This signal should have inputs/outputs set up such that the main algo
    # ... can run without any extra additions.
    def dip_signal(self, bpct, spct_up, spct_down, ticker):
        # This version of the dip signal buys when the price has fallen by some (defined) amount and sells
        # ... when the return is > +1% or < -20%.

        if self.c == 0:
            self.curr_max["{}".format(ticker)] = self.price["{}".format(ticker)]
        else:
            if self.price["{}".format(ticker)] > self.curr_max["{}".format(ticker)]:
                self.curr_max["{}".format(ticker)] = self.price["{}".format(ticker)]

        self.max_change["{}".format(ticker)] = (self.price["{}".format(ticker)] - self.curr_max["{}".format(ticker)])/\
                     (self.curr_max["{}".format(ticker)])

        try:
            self.r_pct_change["{}".format(ticker)] = (self.price["{}".format(ticker)] - self.buy_price["{}".format(ticker)])/\
                           (self.buy_price["{}".format(ticker)])
        except:
            self.r_pct_change["{}".format(ticker)] = 0.0

        if self.buy_search["{}".format(ticker)]:
            if self.max_change["{}".format(ticker)] < bpct:
                mc = self.max_change["{}".format(ticker)]*100
                print("Buy signal on {} due to drop of {}%.".format(ticker, "%.2f" % mc))
                self.buy_act["{}".format(ticker)] = True
            else:
                pass
        elif self.buy_search["{}".format(ticker)] is False and self.buy_act["{}".format(ticker)] is False:
            if self.r_pct_change["{}".format(ticker)] > spct_up or self.r_pct_change["{}".format(ticker)] < spct_down:
                if self.r_pct_change["{}".format(ticker)] > spct_up:
                    self.count_seg_u+=1
                    print("Sell of {} at {} -- UP Signal".format(ticker, self.n_open["{}".format(ticker)]))
                    self.avg_up = 1
                    self.avg_down = 0
                if self.r_pct_change["{}".format(ticker)] < spct_down:
                    self.count_seg_d+=1
                    print("Sell of {} at {} -- DOWN Signal".format(ticker, self.n_open["{}".format(ticker)]))
                    self.avg_down = 1
                    self.avg_up = 0

                self.sell_sig["{}".format(ticker)] = True
                self.curr_max["{}".format(ticker)] = self.n_open["{}".format(ticker)]
                self.buy_search["{}".format(ticker)] = True

        return

    def experiment(self):
        # Various local counters.
        day_count = 0
        period_count = 0
        sale_count = 0

        # These are the original states for experimental averages.
        Avg_Fin = 0
        Avg_Up = 0
        Avg_Down = 0

        # This is an empty array that will be appended to, recording each trades final percentage return.
        Final = []

        max = {}
        min = {}

        # Grabs the pricing data from the first day in the experimental run period.
        self.df_func_new(self.start_date)

        # This section initializes all necessary variables that we will be tracking through the experiment.
        for ticker in self.tickers:
            max["{}".format(ticker)] = np.nan
            min["{}".format(ticker)] = np.nan

            self.close_prices["{}".format(ticker)] = []
            self.b["{}".format(ticker)] = 0
            self.max["{}".format(ticker)] = 0
            self.min["{}".format(ticker)] = 0
            self.max_change["{}".format(ticker)] = 0
            self.r_pct_change["{}".format(ticker)] = 0
            self.count_seg_u = 0
            self.count_seg_d = 0
            self.c = 0

            # These two variables determine if a "buy" will take place.
            self.buy_search["{}".format(ticker)] = True
            self.buy_act["{}".format(ticker)] = False
            self.sell_sig["{}".format(ticker)] = False
            try:
                self.price["{}".format(ticker)] = self.df["{}".format(ticker)]['open'][0]
                self.start_price["{}".format(ticker)] = self.price["{}".format(ticker)]
            except:
                self.start_price["{}".format(ticker)] = np.nan
                continue

        today = time.mktime(dt.datetime.strptime(self.start_date, "%Y-%m-%d").timetuple())
        s_today = str(dt.datetime.fromtimestamp(today))[0:10]
        end_date = False

        while end_date is False:

            today = s_today
            today = time.mktime(dt.datetime.strptime(today, "%Y-%m-%d").timetuple())
            # Above sets the appropriate date, below grabs the pricing data for that day.
            self.df_func_new(s_today)

            for period in range(0,391):
                i = (period + 1) + (day_count) * 390.0

                for ticker in self.tickers:

                    try:
                        self.price["{}".format(ticker)] = self.df["{}".format(ticker)]['close'][period]
                    except:
                        continue
                    try:
                        self.n_open["{}".format(ticker)] = self.df["{}".format(ticker)]['open'][period + 1]
                    except:
                        self.n_open["{}".format(ticker)] = self.df["{}".format(ticker)]['close'][period]

                    if self.start_price["{}".format(ticker)] is np.nan:
                        self.start_price["{}".format(ticker)] = self.price["{}".format(ticker)]
                    else:
                        pass

                    # For various indicators, we will need to keep track of N previous prices. These two lines ...
                    # ... take care of that while cutting it at a length of 30 (alterable) in order to keep it O(1) ST.
                    self.close_prices["{}".format(ticker)].append(self.price["{}".format(ticker)])
                    self.close_prices["{}".format(ticker)] = self.close_prices["{}".format(ticker)][-30:]


                    # The b variable corresponds to tell the program that the stock is currently "bought" and in need
                    # ... of selling. This is really not needed within the experimental framework but given that this is
                    # ... the sister code of the simulation code, we're going to leave it.

                    try:
                        self.dip_signal(self.b_sig, self.p_s_sig, self.n_s_sig, ticker)
                    except (KeyError):
                        continue

                    # All print statements within the algorithm below, as well as the signal algorithm above, are there
                    # for sanity checks of a new signal. I've left these as is such that they can be easily followed
                    # by someone when implementing their own signal. It is important to have a way to sanity check the
                    # progression of buys/sells.

                    if self.buy_search["{}".format(ticker)] or self.sell_sig["{}".format(ticker)]:
                        if self.buy_act["{}".format(ticker)] or self.sell_sig["{}".format(ticker)]:
                            if self.b["{}".format(ticker)] == 1:
                                try:
                                    pct_change = (self.n_open["{}".format(ticker)] - self.buy_price["{}".format(ticker)])/(self.buy_price["{}".format(ticker)]) * 100.0
                                except:
                                    pct_change = np.nan

                                if pct_change > 10.0 or pct_change < -50.0:
                                    pct_change = 0

                                Avg_Fin = (Avg_Fin*sale_count + pct_change)/(sale_count+1)
                                sale_count+=1

                                Final.append(pct_change)

                                print("Sale of {} for a {}% return.".format(ticker, "%.2f" % pct_change))
                                print(s_today)
                                print("---------------------------------------------------")
                                self.sell_sig["{}".format(ticker)] = False
                                self.b["{}".format(ticker)] = 0

                                # I wanted to assess the average return for up signal trades and down signal trades.
                                # Doing it as seen helps minimize the space complexity of the problem ... as opposed
                                # to saving each iteration to an array and averaging at the end.
                                if self.avg_up == 1:
                                    Avg_Up = (Avg_Up*(self.count_seg_u-1) + pct_change)/(self.count_seg_u)
                                elif self.avg_down == 1:
                                    Avg_Down = (Avg_Down*(self.count_seg_d-1) + pct_change)/(self.count_seg_d)
                                continue

                            else:
                                pass

                            if self.buy_act["{}".format(ticker)]:

                                self.buy_price["{}".format(ticker)] = self.n_open["{}".format(ticker)]
                                print("Buy of {} at {}, {} minutes into the day".format(ticker, self.buy_price["{}".format(ticker)], period))
                                self.max["{}".format(ticker)] = self.price["{}".format(ticker)]
                                self.min["{}".format(ticker)] = self.price["{}".format(ticker)]

                                self.buy_search["{}".format(ticker)] = False
                                self.buy_act["{}".format(ticker)] = False

                                self.b["{}".format(ticker)] = 1
                            else:
                                continue
                        else:
                            continue

                    # In many use cases for this algorithm, I was interested in seeing the maximum/minimun percent
                    # change throughout the lifetime in which I "held" a stock.
                    if self.buy_search["{}".format(ticker)] is False:
                        if self.b["{}".format(ticker)] == 1:
                            if self.price["{}".format(ticker)] > self.max["{}".format(ticker)]:
                                self.max["{}".format(ticker)] = self.price["{}".format(ticker)]
                                max["{}".format(ticker)] = (self.max["{}".format(ticker)] - self.buy_price["{}".format(ticker)])/(self.buy_price["{}".format(ticker)]) * 100.0
                            if self.price["{}".format(ticker)] < self.min["{}".format(ticker)]:
                                self.min["{}".format(ticker)] = self.price["{}".format(ticker)]
                                min["{}".format(ticker)] = (self.min["{}".format(ticker)] - self.buy_price["{}".format(ticker)])/(self.buy_price["{}".format(ticker)]) * 100.0
                            continue
                    else:
                        continue

                    period_count += 1

            s_today = str(dt.datetime.fromtimestamp(today) + dt.timedelta(days=1))[0:10]
            if s_today == self.end_date:
                end_date = True
            else:
                day_count += 1
                self.c += 1
                continue

        for ticker in self.tickers:
            if self.buy_act["{}".format(ticker)] is False:
                if self.b["{}".format(ticker)] == 1:
                    pct_change = (self.n_open["{}".format(ticker)] - self.buy_price["{}".format(ticker)]) / (
                    self.buy_price["{}".format(ticker)]) * 100.0

                    Avg_Fin = (Avg_Fin * sale_count + pct_change) / (sale_count + 1)
                    sale_count += 1

                    Final.append(pct_change)

        return Avg_Fin, Avg_Up, Avg_Down, Final

    def run_exp(self):

        # This function is the most easily "alterable" in the sense that whatever output you are interested in for a
        # particular experiment, you can create here. In this case, I have displayed a few results, thrown a few more
        # into dataframes, and forced the algorithm to run in a recursive nature until a certain condition was met.

        Avg_Final, Avg_Up, Avg_Down, Final = self.experiment()

        if self.experiment_count >= 1:
            self.experiment_df = pd.DataFrame({'Avg Final %': self.avg_f_l, 'Avg Up Sig %': self.avg_up_l,
                                               'Avg Down Sig %': self.avg_down_l, 'Up Sig Portion': self.portion_u,
                                               'Winning %': self.w_pct, 'Average Win %': self.w_avg, 'Average Loss %':
                                                   self.l_avg, 'E[% per Trade]': self.exp_val,
                                               '# of Trades': self.n_trades},
                                              columns=['Avg Final %', 'Avg Up Sig %', 'Avg Down Sig %',
                                                       'Up Sig Portion',
                                                       'Winning %', 'Average Win %', 'Average Loss %', 'E[% per Trade]',
                                                       '# of Trades'], index=self.buy_sig)
            self.experiment_df.to_excel('Experimental_Run_1.xlsx')
            return

        print("Average Up Signal: {}%".format("%.2f" % Avg_Up))
        x = float(self.count_seg_u)/(float(self.count_seg_d) + float(self.count_seg_u)) * 100.0
        x_ = x/100.0
        print("{}% of Signals from Up Signals".format("%.2f" % x))

        print("Average Down Signal: {}%".format("%.2f" % Avg_Down))
        y = 1-x

        w = 0
        w_avg = 0
        l = 0
        l_avg = 0
        for item in Final:
            if item >= 0.0:
                w_avg = ((w_avg * w) + item)/(w+1)
                w+=1
            else:
                l_avg = ((l_avg*l) + item)/(l+1)
                l+=1

        win_pct = (float(w))/(float(w+l)) * 100.0
        print("Winning Percentage: {}%".format("%.2f" % win_pct))
        print("Win Avg: {}% -- Loss Avg: {}%".format("%.2f" % w_avg, "%.2f" % l_avg))
        print("Average Final: {}%".format("%.2f" % Avg_Final))

        C = 100

        exp_val = C*x_*((Avg_Up/100.0) + 1.0) + C*(1-x_)*((Avg_Down/100.0) + 1.0)
        exp_ret = (exp_val - C)

        print("The expected value of a single trade is {}%".format("%.2f" % exp_ret))

        l_f = len(Final)
        print("There were {} total trades.".format(l_f))

        # Append data going into dataframe.
        # self.pos_sell.append(self.p_s_sig)
        # self.neg_sell.append(self.n_s_sig)
        self.buy_sig.append(self.b_sig)
        self.avg_f_l.append(Avg_Final)
        self.avg_up_l.append(Avg_Up)
        self.avg_down_l.append(Avg_Down)
        self.portion_u.append(x_)
        self.w_pct.append(win_pct)
        self.w_avg.append(w_avg)
        self.l_avg.append(l_avg)
        self.exp_val.append(exp_ret)
        self.n_trades.append(l_f)

        if exp_ret >= self.exp_ret:
            self.b_sig = self.b_sig + self.delt_c
            self.delt_c = self.delt_c / 2.0
            self.exp_ret = exp_ret

            return self.run_exp()
        else:
            self.experiment_count += 1
            self.b_sig = self.b_sig - self.delt_c
            self.delt_c = self.delt_c / 2.0

            return self.run_exp()


# These next few lines help append the tickers of the SP500 to a local list. This list then becomes our group of
# tickers for our experiment.

resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, "lxml")
table = soup.find('table', {'class': 'wikitable sortable'})
tickers = []
for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text.strip()
    mapping = str.maketrans(".", "-")
    ticker = ticker.translate(mapping)
    tickers.append(ticker)

# This line simply selects N random stocks out of the 500 within the SP500.
tickers = rand.sample(tickers, 100)
#tickers = ['AAPL', 'AMZN', 'GOOG']

# Finally, we run the experiment. We first have to initialize our class by giving it the list of tickers, a desired
# start date, and a desired end date.
ind_env = Experiment(tickers, '2017-01-03', '2020-10-19')
# Then, we "tell" the class which function we would like to run.
ind_env.run_exp()


