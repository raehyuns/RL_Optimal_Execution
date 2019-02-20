import os
import datetime as DT
from datetime import timedelta
import time
import argparse
import random
import uuid
import pdb
import logging
import pickle
import pymongo
from pymongo import MongoClient

from utils import *
from market import *
from agent import *
from config import *
#from order import *
import qaracs

import torch
import torch.nn as nn
import torch.optim as optim

import tensorflow as tf
from tensorboardX import SummaryWriter


class Trader:
    def __init__(self, args, logger, model_id):
        self.logger = logger
        #self.writer = tf.summary.FileWriter('./save/tf_board/'+args.order_type)
        self.writer = SummaryWriter('./save/tf_board/'+args.order_type)
        self.model_id = model_id

        self.console_display = args.console_display
        self.sample_batch_size = args.sample_batch_size
        self.n_amt_lv = args.n_amt_lv
        self.amt_unit = args.order_amt // args.n_amt_lv
        self.non_trd_beta = args.non_trade_penalty
        self.q_fac_r = args.quantity_penalty
        self.pu = args.pu
        self.lookback = args.min_lookback
        self.max_time = args.max_time
        self.order_type = args.order_type
        #self.state_pub_name = parse_state_pub(args.state_pub)
        self.grad_max_norm = args.grad_max_norm

        self.mkt = Market(args, logger)
        self.exp_memory = ReplayMemory(args.max_memory)
        self.agent = Agent(args).cuda()
        self.logger.info(self.agent)
        self.eval_prices = []
        # store all train, test dataset in list
        # each element of list is tuple of (raw dataframe, close price)

        if args.mode != 'real_trading':
            self.train_data_list, self.test_data_list = load_data(args.sec_type, args.data_dir,
                                                        args.test_sec_name, args.n_test_day,
                                                        args.trade_price_assumption,
                                                        args.mode, args.train_all)
            self.mkt.train_min1_agg, self.mkt.train_raw_q, self.mkt.train_simtr, self.mkt.cls_prc = select_data(self.train_data_list)

        self.logger.info("All data loaded")

        self.trans_hist = []

        self.logger.info("Solver object is initialized")

    def get_state(self, min1_agg_data, raw_data, cur_idx, remain_t=1, remain_q=1):
        """
        Get state information data at current point
        [ Input ]
        cur_idx         : index of current data point
        min1_agg_data   : data aggregated by every minute
        raw_data        : data with all quotes
        lookback        : How many minutes that we provide to agent (before init time)
        remain_t        : Remaining time
        remain_q        : Remaining quantities to order

        [ Output ]
        init_dt     : Randomly initialized starting point
        long_X      : Min aggregated information for lookback minutes (lookback, # features)
        short_X     : Latest 1 minute information aggregated by second (60, # features)
        """
        #cur_idx = random.randint(0+lookback, min1_agg_data.shape[0]-1)
        cur_dt = min1_agg_data.iloc[cur_idx, :]['dt_time']
        frt_dt = min1_agg_data.iloc[cur_idx-1, :]['dt_time']
        cur_midprc = min1_agg_data.iloc[cur_idx, :]['midprice']

        lb = min(self.lookback, cur_idx)
        target_df = min1_agg_data.iloc[cur_idx-lb:cur_idx].drop('dt_time', 1)
        bid1_prc = target_df.bid_P1.iloc[-1]
        ask1_prc = target_df.ask_P1.iloc[-1]

        # Aggregated minute information to provide Longer term
        #summary_cols = ['agg_vol', 'agg_trd_vol', 'pressure',
                        #'ask_P1', 'bid_P1', 'ask_Qa', 'bid_Qa']
        state = select_and_normalize(target_df)
        #short_X = raw_data.loc[(raw_data.time_hr==frt_dt.hour) & (raw_data.time_min==frt_dt.minute)]
        #short_X = normalize_target_cols(agg_to_sec1(short_X)[summary_cols])
        return cur_dt, state, cur_midprc, bid1_prc, ask1_prc


    def get_baseline(self, min1_agg_data, raw_data, cur_idx):
        start_dt = min1_agg_data.iloc[cur_idx]['dt_time']
        end_dt = min1_agg_data.iloc[min(min1_agg_data.shape[0]-1, cur_idx+self.max_time)]['dt_time']
        target_data = raw_data.loc[(raw_data.dt_time>=start_dt) & (raw_data.dt_time<=end_dt)]

        twap = (target_data.trd_prc.max()+target_data.trd_prc.min()+target_data.iloc[-1].trd_prc)/3
        vwap = (target_data.trd_prc * target_data.trd_vol).sum() / target_data.trd_vol.sum()
        end_prc = target_data.trd_prc.iloc[-1]
        return twap, vwap, end_prc

    def get_reward(self, trd_prc, trd_amt_lv, bc_prc, remain_t, remain_q):
        # all rewards are calculated based on difference btw close price & traded price
        t_factor = np.sqrt(remain_t)
        if trd_amt_lv == 0:
            # reward when agent decide not to trade, should consider opportunity cost
            if self.order_type =='buy':
                reward_rate = ((trd_prc - bc_prc) / bc_prc) * t_factor
            else: # ask
                reward_rate = ((bc_prc - trd_prc) / bc_prc) * t_factor
            reward = reward_rate * t_factor
        else:
            # reward when agent decide to trade
            q_factor = self.q_fac_r * np.log(trd_amt_lv//2+1) + 1
            if self.order_type =='buy':
                reward = ((bc_prc - trd_prc) / bc_prc) * t_factor * q_factor
            else: # ask
                reward = ((trd_prc - bc_prc) / bc_prc) * t_factor * q_factor
        if reward < 0:
            # risk aversion
            return reward * 2
        else:
            return reward

    def update_mkt_info(self, symbols):
        for s in symbols:
            self.mkt_data_dict[s] = get_data_from_db(s)

    def real_trading(self, args):

        self.agent.load_state_dict(torch.load('./save/model/20190219-3ff40e35'))
        self.agent.eval()

        delay = 60.0
        while True:
            self.update_mkt_info(symbols)
            start_dt = datetime.fromtimestamp(time.time()).replace(hour=9, minute=30)
            end_dt = datetime.fromtimestamp(time.time()).replace(hour=15, minute=15)
            final_orders, valid_orders = get_live_orders()
            self.final_clear_order(final_orders)
            self.normal_order(valid_orders)
            time.sleep(delay - ((time.time() - starttime) % delay))

        #if final_order: # placing all remain amt at market price user qaracs request_market_price_all
        #    request_final_order(final_orders)

        remain_q = args.order_amt
        remain_t = args.max_time
        date = DT.datetime.strptime(str(raw_q.iloc[0]['date']), '%Y%m%d')
        cur_idx = min1_agg.shape[0]-1

        while (remain_q and remain_t):
            rq = remain_q / args.order_amt
            rt = remain_t / args.max_time
            cur_t, x, mid_prc, bid1_prc, ask1_prc = self.get_state(min1_agg, raw_q, cur_idx, rt, rq)
            cur_dt = DT.datetime.combine(date, cur_t)
            act_value, agt_action = self.agent(x)
            act_idx = agt_action.cpu().item()
            if act_idx%2==0: # Quote price = Bid 1
                qt_prc = bid1_prc
            else: # Quote price = Ask 1
                qt_prc = ask1_prc
            qt_amt = (act_idx//2+1) * self.amt_unit

            ord_res = qaracs.request_order('005930', qt_prc, qt_amt, 'stock', self.order_type)
            print(ord_res)
            proc_time = time.time() + timedelta(hours=0,minutes=remain_t).total_seconds()
            resv_res = qaracs.request_market_price_all(ord_res['ordno'], proc_time)
            print(resv_res)
            pdb.set_trace()

            remain_t -= 1

    def final_clear_order():
        # 영묵님 br_order_no랑 같이 주문하는 부분 짜지면 그 함수 활용해서
        # 청산해야 하는 주문량에 대해 모두 Market Price에 주문
        pass

    def normal_order(valid_order):
        pass

if __name__ == "__main__":

    args = get_args()

    # set a logger
    model_id = time.strftime("%Y%m%d-") + str(uuid.uuid4())[:8]
    formatter = logging.Formatter('%(asctime)s: %(message)s ', '%m/%d/%Y %I:%M:%S %p')
    logger = logging.getLogger(model_id)
    logger.setLevel(logging.INFO)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    logger.addHandler(streamHandler)
    if args.save_log:
        file_path = os.path.join(args.save_dir, 'log/'+model_id+'.log')
        fileHandler = logging.FileHandler(file_path)
        fileHandler.setFormatter(formatter)
        logger.addHandler(fileHandler)
        logger.info('log file : {}'.format(file_path))
    logger.info(args)

    trader = Trader(args, logger, model_id)

    logger.info("Trading with AI agent at real market")
    trader.real_trading(args)
