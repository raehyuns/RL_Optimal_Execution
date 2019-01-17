import datetime
import random
import copy
import uuid
import pdb
import logging
import pandas as pd
import numpy as np

from utils import *

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Market:
    def __init__(self, args, logger):
        self.logger = logger
        self.transaction_fee = args.transaction_fee
        self.pu = args.pu
        self.console_display = args.console_display

    def place_order(self, raw_q, dt_time, qu, prc, order_type,
                            time_cap=None, mkt_order=False, force_fill=False):

        '''
         "Submit and leave" order
         a = place_order(rq, mkt_min.head(1).dt_time, qu = -500, prc=0, mkt_order = True)
         INPUT:
           `raw_q`    : a data.frame for raw quote database
           'cls_raw_q`    : a data.frame for raw quote database
           `dt_time`  : a datetime.time object for the initial placement time
           `qu`       : quantity (always positive value)
           `prc` = 1  : placing an order at the best price
           `prc` = 2  : placing an order at the next best price
           'cls_prc'  : price at the end of the day
           `mkt_order`: sweeping the OB immediately. (voiding `prc`)
           `time_cap` : If not filled until this length of the time, place market order
         price unit (0.05 for KOSPI2, 0.01 for US)
         ASSUMPTIONS:

         OUTPUT:
           `prc_avg` : average price of traded quantities
           `qu_exe`  : total executed quantity
           `last_exe`: `dt` of last execution (possibly needed for order duration)
           `revenue`    : revenue or Cash flow from this order
        '''
        if order_type =='ask':
            qu *= -1

        def r4(x): return(round(x, 4))

        if mkt_order:
            if self.console_display:
                self.logger.info("Plaicing Market Order quantity of", qu)
        else:
            if self.console_display:
                self.logger.info("Placing Limit Order quantity of {} at price {} with time_cap {}"\
                .format(qu, prc, time_cap))

        pu = self.pu
        last_exe = None
        bid_Ps = ['bid_P1', 'bid_P2', 'bid_P3', 'bid_P4', 'bid_P5']
        bid_Qs = ['bid_Q1', 'bid_Q2', 'bid_Q3', 'bid_Q4', 'bid_Q5']
        ask_Ps = ['ask_P1', 'ask_P2', 'ask_P3', 'ask_P4', 'ask_P5']
        ask_Qs = ['ask_Q1', 'ask_Q2', 'ask_Q3', 'ask_Q4', 'ask_Q5']

        if mkt_order:
            # raw data at placement time
            raw_q_now = raw_q.iloc[min(np.where(raw_q.dt_time >= dt_time)[0]), :]
            pq_array = []
            if (qu > 0): # buy order sweeps the ask queue
                # perform orders within best-5 price
                for lv in range(5):
                    if (qu > 0):
                        ask_prc = raw_q_now[ask_Ps[lv]]
                        ask_q = raw_q_now[ask_Qs[lv]]
                        pq_array.append([ask_prc, min(qu, ask_q)])
                        qu -= min(qu, ask_q)
                    else: break
                # residual volumnes --> assume uniform OB with average quantity of top 5 prices
                if (qu > 0):
                    # TODO Check logic here
                    # What is the appropriate price level for residuals
                    pq_array.append([
                        raw_q_now.ask_P5 + 0.5 * pu * qu / np.mean(raw_q_now[ask_Qs]), qu])

            else:  # sell order sweeps the bid queue
                # perform orders within best-5 price
                if (qu < 0): # buy order sweeps the ask queue
                    for lv in range(5):
                        if (qu < 0):
                            bid_prc = raw_q_now[bid_Ps[lv]]
                            bid_q = -raw_q_now[bid_Qs[lv]]
                            pq_array.append([bid_prc, max(qu, bid_q)])
                            qu -= max(qu, bid_q)
                        else: break
                # residual volumnes --> assume uniform OB with average quantity of top 5 prices
                if (qu < 0):
                    pq_array.append([
                        raw_q_now.bid_P5 + 0.5 * pu * qu / np.mean(raw_q_now[bid_Qs]), qu])

            pq_array = np.array(pq_array)
            if self.console_display:
                self.logger.info(pq_array)
            qu_exe = np.sum(pq_array, axis=0)[1]
            payment = sum(pq_array[:, 0] * pq_array[:, 1]) # previously def as revenue
            prc_avg = payment / qu_exe
            order_result = {'prc_avg': r4(prc_avg), 'qu_exe': qu_exe, 'last_exe': raw_q_now.dt_time}
            return (order_result)

        # limit order processing follows:
        # raw_q = rq
        # dt_time = mkt_min.head(1).dt_time
        # prc = 1 # if prc > 5
        # pu = 0.05
        # mkt_order = False
        if qu > 0:  # if bid order
            """
            [intermediate variables]
            idx: current index in `raw_q`
            raw_q_idx: `raw_q` at current index
            quote_prc: quote price of this order
            qu_prior: quantities of higher priority than mine
            [output]
            qu_exe: number of quantities executed so far
            last_exe: last time of execution
            """
            qu_exe = 0
            idx = min(np.where(raw_q.dt_time >= dt_time)[0])
            idx_max = raw_q.shape[0]-1
            raw_q_idx = raw_q.iloc[idx, :]
            quote_prc = raw_q_idx["bid_P1"] - pu * prc
            try:
                qu_prior = raw_q_idx[bid_Qs[int(np.where(raw_q_idx[bid_Ps] == quote_prc)[0])]]
            except:
                qu_prior = 0

            while (qu_exe < qu) & (idx < idx_max):  # while not filled
                idx = idx + 1
                raw_q_idx = raw_q.iloc[idx, :]
                trd_prc = raw_q_idx['trd_prc']
                if trd_prc > quote_prc:
                    pass
                elif trd_prc == quote_prc: # A transaction occurs at `quote_prc`
                    qu_trd = raw_q_idx['trd_vol']
                    if qu_prior > 0:  # if there is quantities of higher priority
                        if qu_prior > qu_trd:
                            qu_prior = qu_prior - qu_trd

                        else:
                            qu_trd = qu_trd - qu_prior
                            qu_prior = 0 # all prior queue cleared
                            if self.console_display:
                                self.logger.info("Prior queue cleared")
                    if qu_prior == 0:  # my order is at the front
                        qu_exe = min(qu_exe + qu_trd, qu)
                        last_exe = raw_q_idx.dt_time
                        if self.console_display:
                            self.logger.info("partial execution {} {}".format(qu_trd, last_exe))
                else:
                    # Assume that if market price went down below my qoute prices
                    # my quote would have been traded before
                    last_exe = raw_q_idx.dt_time
                    if self.console_display:
                        self.logger.info("complete execution {} {}".format(qu - qu_exe, last_exe))
                    qu_exe = qu

            # seems to be unnecessary
            if force_fill and qu_exe < qu: # if not filled and forced to fill
                if self.console_display: print("Not filled, so placing market order")
                mo_result = self.place_order(self.train_set_raw_q, self.train_set_raw_q.iloc[idx_max].dt_time, qu-qu_exe, mkt_order=True)
                quote_prc = (quote_prc*qu_exe + mo_result['prc_avg']*mo_result['qu_exe'])/qu
                qu_exe = qu  # The order is filled now
                last_exe = mo_result['last_exe']

            order_result = {'qu_prc': r4(quote_prc),'qu_exe': qu_exe, 'last_exe': last_exe}

        else:  # qu < 0 if ask order
            """
            [intermediate variables]
            idx: current index in `raw_q`
            raw_q_idx: `raw_q` at current index
            quote_prc: quote price of this order
            qu_prior: quantities of higher priority
            [output]
            qu_exe: number of quantities executed so far
            last_exe: last time of execution
            """
            qu_exe = 0
            idx = min(np.where(raw_q.dt_time >= dt_time)[0])
            idx_max = raw_q.shape[0]-1
            raw_q_idx = raw_q.iloc[idx, :]
            quote_prc = raw_q_idx["ask_P1"] + pu * prc
            try:
                qu_prior = raw_q_idx[ask_Qs[int(np.where(raw_q_idx[ask_Ps] == quote_prc)[0])]]
            except:
                qu_prior = 0

            while (qu_exe < -qu) & (idx < idx_max):  # while not filled
                idx = idx + 1
                raw_q_idx = raw_q.iloc[idx, :]
                trd_prc = raw_q_idx['trd_prc']
                if trd_prc < quote_prc:
                    pass
                elif trd_prc == quote_prc:  # A trade occurs at `quote_prc`
                    qu_trd = raw_q_idx['trd_vol']
                    if qu_prior > 0: # if there is quantities of higher priority
                        if qu_prior > qu_trd:
                            qu_prior = qu_prior - qu_trd
                        else:
                            qu_trd = qu_trd - qu_prior
                            qu_prior = 0 # all prior queue cleared
                            if self.console_display:
                                self.logger.info("Prior queue cleared")
                    if qu_prior == 0:  # if this order is at the front
                        qu_exe = min(qu_exe + qu_trd, -qu)
                        last_exe = raw_q_idx.dt_time
                        if self.console_display:
                            self.logger.info("partial execution {} {}".format(qu_trd, last_exe))

                else:
                    last_exe = raw_q_idx.dt_time
                    if self.console_display:
                        self.logger.info("complete execution {} {}".format(-qu - qu_exe, last_exe))
                    qu_exe = -qu

            if force_fill and qu_exe < -qu: # if not filled and forced to fill
                if self.console_display: print("Not filled, so placing market order")
                mo_result = self.place_order(self.train_set_raw_q, self.train_set_raw_q.iloc[idx_max].dt_time, qu+qu_exe, mkt_order=True)
                quote_prc = (quote_prc*qu_exe - mo_result['prc_avg']*mo_result['qu_exe'])/(-qu)
                qu_exe = -qu  # The order is filled now
                last_exe = mo_result['last_exe']
            order_result = {'qu_prc': r4(quote_prc), 'qu_exe': -qu_exe, 'last_exe': last_exe}
        return order_result
