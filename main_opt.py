import os
import pdb
import pandas as pd
import numpy as np
import datetime
import argparse
import random
import copy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import Adam
from collections import deque


class Market:
    def __init__(self, args = None):
        self.transaction_fee = 0.0
        self.pu = 0.05  # price_unit
        self.console_display = args.console_display

    def place_order(self, raw_q, dt_time, qu, prc=0, time_cap=None, mkt_order=False, force_fill=False):
        def r4(x): return(round(x, 4))
        if mkt_order:
            if self.console_display: print("Plaicing Market Order quantity of", qu)
        else:
            if self.console_display: print("Placing Limit Order quantity of", qu, "at price", prc, "with time_cap", time_cap)
        # "Submit and leave" order
        # a = place_order(rq, mkt_min.head(1).dt_time, qu = -500, prc=0, mkt_order = True)
        # INPUT:
        #   `raw_q`    : a data.frame for raw quote database
        #   `dt_time`  : a datetime.time object for the initial placement time
        #   `qu`       : quantity (+ for buy, - for sell)
        #   `prc` = 0  : placing an order at the best price
        #   `prc` = 1  : placing an order at the next best price
        #   `mkt_order`: sweeping the OB immediately. (voiding `prc`)
        #   `time_cap` : If not filled until this length of the time, place market order
        # price unit (0.05 for KOSPI2, 0.01 for US)
        # ASSUMPTIONS:
        #
        # OUTPUT:
        #   `prc_avg` : average price of traded quantities
        #   `qu_exe`  : total executed quantity
        #   `last_exe`: `dt` of last execution (possibly needed for order duration)
        #   `revenue`    : revenue or Cash flow from this order

        pu = self.pu
        last_exe = None
        bid_Ps = ['bid_P1', 'bid_P2', 'bid_P3', 'bid_P4', 'bid_P5']
        bid_Qs = ['bid_Q1', 'bid_Q2', 'bid_Q3', 'bid_Q4', 'bid_Q5']
        ask_Ps = ['ask_P1', 'ask_P2', 'ask_P3', 'ask_P4', 'ask_P5']
        ask_Qs = ['ask_Q1', 'ask_Q2', 'ask_Q3', 'ask_Q4', 'ask_Q5']
        if mkt_order:
            raw_q_now = raw_q.iloc[min(np.where(raw_q.dt_time >= dt_time)[0]), :]
            pq_array = []

            if (qu > 0): # buy order sweeps the ask queue
                pq_array.append([raw_q_now.ask_P1, min(qu, raw_q_now.ask_Q1)])
                qu = qu - min(qu, raw_q_now.ask_Q1)
                if (qu > 0):
                    pq_array.append([raw_q_now.ask_P2, min(qu, raw_q_now.ask_Q2)])
                    qu = qu - min(qu, raw_q_now.ask_Q2)
                    if (qu > 0):
                        pq_array.append([raw_q_now.ask_P3, min(qu, raw_q_now.ask_Q3)])
                        qu = qu - min(qu, raw_q_now.ask_Q3)
                        if (qu > 0):
                            pq_array.append([raw_q_now.ask_P4, min(qu, raw_q_now.ask_Q4)])
                            qu = qu - min(qu, raw_q_now.ask_Q4)
                            if (qu > 0):
                                pq_array.append([raw_q_now.ask_P5, min(qu, raw_q_now.ask_Q5)])
                                qu = qu - min(qu, raw_q_now.ask_Q5)
                                if (qu > 0):  # assume uniform OB with average quantity of top 5 prices
                                    pq_array.append([
                                        raw_q_now.ask_P5 + 0.5 * pu * qu / np.mean(raw_q_now[ask_Qs]), qu])

            else:  # sell order sweeps the bid  queue
                pq_array.append([raw_q_now.bid_P1, max(qu, -raw_q_now.bid_Q1)])
                qu = qu - max(qu, -raw_q_now.bid_Q1)
                if (qu < 0):
                    pq_array.append([raw_q_now.bid_P2, max(qu, -raw_q_now.bid_Q2)])
                    qu = qu - max(qu, -raw_q_now.bid_Q2)
                    if (qu < 0):
                        pq_array.append([raw_q_now.bid_P3, max(qu, -raw_q_now.bid_Q3)])
                        qu = qu - max(qu, -raw_q_now.bid_Q3)
                        if (qu < 0):
                            pq_array.append([raw_q_now.bid_P4, max(qu, -raw_q_now.bid_Q4)])
                            qu = qu - max(qu, -raw_q_now.bid_Q4)
                            if (qu < 0):
                                pq_array.append([raw_q_now.bid_P5, max(qu, -raw_q_now.bid_Q5)])
                                qu = qu - max(qu, -raw_q_now.bid_Q5)
                                if (qu < 0):  # assume uniform OB with average quantity of top 5 prices
                                    pq_array.append([
                                        raw_q_now.bid_P5 + 0.5 * pu * qu / np.mean(raw_q_now[bid_Qs]), qu])
            pq_array = np.array(pq_array)
            if self.console_display: print(pq_array)
            qu_exe = np.sum(pq_array, axis=0)[1]
            revenue = sum(pq_array[:, 0] * pq_array[:, 1])
            prc_avg = revenue/ qu_exe
            order_result = {'prc_avg': r4(prc_avg), 'qu_exe': qu_exe, 'last_exe': raw_q_now.dt_time, 'revenue': r4(-revenue)}
            return (order_result)
        # limit order processing follows:
        # raw_q = rq
        # dt_time = mkt_min.head(1).dt_time
        # qu = -400
        # prc = 1 # if prc > 5
        # pu = 0.05
        # mkt_order = False
        if qu > 0:  # if bid order
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
            raw_q_idx = raw_q.iloc[idx, :]
            quote_prc = raw_q_idx["bid_P1"] - pu * prc
            dead_line = (datetime.datetime.combine(datetime.date(1970, 1, 1), dt_time) + datetime.timedelta(minutes=time_cap)).time()
            idx_max = max(np.where(raw_q.dt_time < dead_line)[0])

            try:
                qu_prior = raw_q_idx[bid_Qs[int(np.where(raw_q_idx[bid_Ps] == quote_prc)[0])]]
            except:
                qu_prior = 0
            while (qu_exe < qu) & (idx < idx_max - 1):  # while not filled
                idx = idx + 1
                # print(idx)
                raw_q_idx = raw_q.iloc[idx, :]
                # checking duplicate and skip
                # while (raw_q_idx['trd_vol'] == raw_q.iloc[idx-1,:]['trd_vol']) & (raw_q_idx['trd_prc'] == raw_q.iloc[idx-1,:]['trd_prc']):
                #     idx = idx + 1
                #     # print(idx)
                #     raw_q_idx = raw_q.iloc[idx, :]
                trd_prc = raw_q_idx['trd_prc']
                if trd_prc > quote_prc:
                    pass
                elif trd_prc == quote_prc: # A trade occurs at `quote_prc`
                    qu_trd = raw_q_idx['trd_vol']
                    if qu_prior > 0:  # if there is quantities of higher priority
                        if qu_prior > qu_trd:
                            if self.console_display: print("queue is reduced by", qu_trd)
                            qu_prior = qu_prior - qu_trd
                        else:
                            if self.console_display: print("queue is depleted")
                            qu_trd = qu_trd - qu_prior
                            qu_prior = 0 # all prior queue cleared
                    if qu_prior == 0:  # if this order is at the front
                        qu_exe = min(qu_exe + qu_trd, qu)
                        last_exe = raw_q_idx.dt_time
                        if self.console_display: print("partial execution", qu_trd, last_exe)
                else:
                    last_exe = raw_q_idx.dt_time
                    if self.console_display: print("complete execution", qu - qu_exe, last_exe)
                    qu_exe = qu

            if force_fill and qu_exe < qu: # if not filled and forced to fill
                if self.console_display: print("Not filled, so placing market order")
                mo_result = self.place_order(self.train_set_raw_q, self.train_set_raw_q.iloc[idx_max].dt_time, qu-qu_exe, mkt_order=True)
                quote_prc = (quote_prc*qu_exe + mo_result['prc_avg']*mo_result['qu_exe'])/qu
                qu_exe = qu  # The order is filled now
                last_exe = mo_result['last_exe']
                # print(qu_exe)
            order_result = {'prc_avg': r4(quote_prc), 'qu_exe': qu_exe, 'last_exe': last_exe, 'revenue': r4(-quote_prc*qu_exe)}
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
            raw_q_idx = raw_q.iloc[idx, :]
            quote_prc = raw_q_idx["ask_P1"] + pu * prc
            dead_line = (datetime.datetime.combine(datetime.date(1970, 1, 1), dt_time) + datetime.timedelta(minutes=time_cap)).time()
            idx_max = max(np.where(raw_q.dt_time < dead_line)[0])
            try:
                qu_prior = raw_q_idx[ask_Qs[int(np.where(raw_q_idx[ask_Ps] == quote_prc)[0])]]
            except:
                qu_prior = 0
            while (qu_exe < -qu) & (idx < idx_max - 1):  # while not filled
                idx = idx + 1
                # print(idx)
                raw_q_idx = raw_q.iloc[idx, :]
                # checking duplicate and skip
                # while (raw_q_idx['trd_vol'] == raw_q.iloc[idx - 1, :]['trd_vol']) & (
                #         raw_q_idx['trd_prc'] == raw_q.iloc[idx - 1, :]['trd_prc']):
                #     idx = idx + 1
                #     # print(idx)
                #     raw_q_idx = raw_q.iloc[idx, :]
                trd_prc = raw_q_idx['trd_prc']
                if trd_prc < quote_prc:
                    pass
                elif trd_prc == quote_prc:  # A trade occurs at `quote_prc`
                    qu_trd = raw_q_idx['trd_vol']
                    if qu_prior > 0: # if there is quantities of higher priority
                        if qu_prior > qu_trd:
                            if self.console_display: print("queue is reduced by", qu_trd)
                            qu_prior = qu_prior - qu_trd
                        else:
                            if self.console_display: print("queue is depleted")
                            qu_trd = qu_trd - qu_prior
                            qu_prior = 0 # all prior queue cleared
                    if qu_prior == 0:  # if this order is at the front
                        qu_exe = min(qu_exe + qu_trd, -qu)
                        last_exe = raw_q_idx.dt_time
                        if self.console_display: print("partial execution", qu_trd, last_exe)
                else:
                    last_exe = raw_q_idx.dt_time
                    if self.console_display: print("complete execution", -qu - qu_exe, last_exe)
                    qu_exe = -qu
                # print(qu_exe)
            if force_fill and qu_exe < -qu: # if not filled and forced to fill
                if self.console_display: print("Not filled, so placing market order")
                mo_result = self.place_order(self.train_set_raw_q, self.train_set_raw_q.iloc[idx_max].dt_time, qu+qu_exe, mkt_order=True)
                quote_prc = (quote_prc*qu_exe - mo_result['prc_avg']*mo_result['qu_exe'])/(-qu)
                qu_exe = -qu  # The order is filled now
                last_exe = mo_result['last_exe']
                # print(qu_exe)
            order_result = {'prc_avg': r4(quote_prc), 'qu_exe': -qu_exe, 'last_exe': last_exe, 'revenue': r4(quote_prc*qu_exe)}
        return order_result
    def step_and_reward(self, time_t_dt, qu, prc=0, is_last = False):
        # Move forward 1 minute
        # Return the private state and return the revenue
        if is_last:
            order_result = self.place_order(self.train_set_raw_q, time_t_dt, qu, prc, time_cap=1, force_fill=True)
        else:
            order_result = self.place_order(self.train_set_raw_q, time_t_dt, qu, prc, time_cap=1, force_fill=False)
        return order_result
    # def gen_scaled_trans_hist_pub(self, trans_hist):
    #     trans_hist_pub = trans_hist[self.state_pub_name]
    #     self.scale_max = trans_hist_pub.max()
    #     self.scale_min = trans_hist_pub.min()
    #     return (trans_hist_pub - self.scale_min)/(self.scale_max-self.scale_min)

class Solver:
    def __init__(self, args = None):
        self.main_dir = '/www/ml/aibot/opt_trd'
        self.console_display = args.console_display
        self.episodes = args.episodes
        self.train_gap = args.train_gap
        self.sample_batch_size = args.sample_batch_size
        self.state_pub_name = self.parse_state_pub(args.state_pub)
        self.mkt = Market(args)
        self.agent = Agent(args)

        self.mkt.train_set_raw_q = self.load_train_set(args)
        self.train_set = self.agg_to_min1(self.mkt.train_set_raw_q)
        self.mkt.trans_hist_pub = self.load_scaled_trans_hist_pub(args)
        self.mkt.scale_max = self.scale_max
        self.mkt.scale_min = self.scale_min
        # self.mkt.trans_hist = self.load_scaled_trans_hist(args)

        self.pu = 0.05

        print("Main object is initialized")
    def load_scaled_trans_hist_pub(self, args):
        if args.new_trans_hist:
            list_files = os.listdir(os.getcwd() + '/' + args.train_set_dir)
            trans_hist = pd.DataFrame()
            for file in list_files:
                rq = self.load_train_set(args, file)
                rq_min = self.agg_to_min1(rq)
                trans_hist = trans_hist.append(rq_min)
            trans_hist = trans_hist.sort_index()
            trans_hist.to_pickle("transition_history.pkl")
        trans_hist_pub = pd.read_pickle("transition_history.pkl")[self.state_pub_name]

        self.scale_max = trans_hist_pub.max()
        self.scale_min = trans_hist_pub.min()

        return (trans_hist_pub - self.scale_min)/(self.scale_max-self.scale_min)
    def load_train_set(self, args, file=None):
        if file is None:
            raw_file = os.getcwd() + '/' + args.train_set_dir + '/' + 'trade2009_' + str(args.train_set_no) + '.csv'
        else:
            raw_file = os.getcwd() + '/' + args.train_set_dir + '/' + file
        rq = pd.read_csv(raw_file) # raw_tick # os.listdir(data_dir)
        print(raw_file, "is loaded")
        rq = rq[rq.time < 150000000]
        rq['dt_time'] = [
            datetime.time(int(str(x)[:-7]), int(str(x)[-7:-5]), int(str(x)[-5:-3]), int(str(x)[-3:]) * 1000) for x in
            rq['time']]
        rq['time_hr'] = [x.hour for x in rq['dt_time']]
        rq['time_min'] = [x.minute for x in rq['dt_time']]
        rq.rename(columns={
            'price': 'trd_prc', 'volume': 'trd_vol', 'ASK_TOT_ORD_RQTY': 'ask_Qa', 'BID_TOT_ORD_RQTY': 'bid_Qa',
            'ASK_STEP1_BSTORD_PRC': 'ask_P1', 'ASK_STEP1_BSTORD_RQTY': 'ask_Q1', 'BID_STEP1_BSTORD_PRC': 'bid_P1',
            'BID_STEP1_BSTORD_RQTY': 'bid_Q1',
            'ASK_STEP2_BSTORD_PRC': 'ask_P2', 'ASK_STEP2_BSTORD_RQTY': 'ask_Q2', 'BID_STEP2_BSTORD_PRC': 'bid_P2',
            'BID_STEP2_BSTORD_RQTY': 'bid_Q2',
            'ASK_STEP3_BSTORD_PRC': 'ask_P3', 'ASK_STEP3_BSTORD_RQTY': 'ask_Q3', 'BID_STEP3_BSTORD_PRC': 'bid_P3',
            'BID_STEP3_BSTORD_RQTY': 'bid_Q3',
            'ASK_STEP4_BSTORD_PRC': 'ask_P4', 'ASK_STEP4_BSTORD_RQTY': 'ask_Q4', 'BID_STEP4_BSTORD_PRC': 'bid_P4',
            'BID_STEP4_BSTORD_RQTY': 'bid_Q4',
            'ASK_STEP5_BSTORD_PRC': 'ask_P5', 'ASK_STEP5_BSTORD_RQTY': 'ask_Q5', 'BID_STEP5_BSTORD_PRC': 'bid_P5',
            'BID_STEP5_BSTORD_RQTY': 'bid_Q5'}, inplace=True)
        return rq
    def parse_state_pub(self, args_str):
        glossary = {'1vol': 'min1_vol', '2trd_vol': 'min1_trd_vol', '3pressure': 'pressure'}
        alias = args_str.split(', ')
        alias.sort()
        return [glossary[x] for x in alias]
    def gen_min1_vol(self, prc):
        vol = np.sqrt(sum(np.square(np.diff(np.log(prc))))) * 100
        return vol
    def agg_to_min1(self, rq):
        ## Aggregate to 1 minute (volatility, trading_volume, orderbook_pressure)
        mkt_min1_vol = rq.groupby(['date', 'time_hr', 'time_min']).midprice.agg(self.gen_min1_vol)
        mkt_min1_vol = pd.DataFrame(mkt_min1_vol)
        mkt_min1_vol.columns = ['min1_vol']
        mkt_min1_trd_vol = rq.groupby(['date', 'time_hr', 'time_min']).trd_vol.agg(np.sum)/10000
        mkt_min1_trd_vol = pd.DataFrame(mkt_min1_trd_vol)
        mkt_min1_trd_vol.columns = ['min1_trd_vol']
        mkt_agg = mkt_min1_vol.join(mkt_min1_trd_vol)

        ## Takes the last status
        mkt_last = rq.groupby(['date', 'time_hr', 'time_min']).last()
        mkt_last = mkt_last[['dt_time', 'midprice',
                             'bid_P1', 'bid_P2', 'bid_P3', 'bid_P4', 'bid_P5',
                             'ask_P1', 'ask_P2', 'ask_P3', 'ask_P4', 'ask_P5',
                             'bid_Qa', 'bid_Q1', 'bid_Q2', 'bid_Q3', 'bid_Q4', 'bid_Q5',
                             'ask_Qa', 'ask_Q1', 'ask_Q2', 'ask_Q3', 'ask_Q4', 'ask_Q5']]
        mkt_last['pressure'] = np.log(mkt_last.bid_Qa)-np.log(mkt_last.ask_Qa)
        mkt_min1 = mkt_agg.join(mkt_last)
        return mkt_min1

    def train(self, args):
        def r2s(x): return(str(round(x,2)))
        def r4s(x): return(str(round(x,4)))
        def r4(x): return(round(x,4))
        self.init_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        print("This train is named as:", self.init_time)
        trd_hist_file = open(self.main_dir + '/logs' + '/trd_hist_' + self.init_time + '.txt', 'w')
        trd_hist_file.write(str(args) + '\r\n')
        trd_hist_file.flush()
        print("Start training an agent")
        for idx_episode in range(10): # range(self.episodes):
            print("idx_episode:", idx_episode)
            # begin_time, len_problem, remain_qu, init_prc
            self.t, self.agent.T_t, self.agent.X, self.init_prc = self.parent_reset(args)
            self.t_dt = self.train_set.iloc[self.t, :]['dt_time']
            bm_revenue = \
                {'sweep': r4(self.mkt.place_order(self.mkt.train_set_raw_q, self.t_dt, self.agent.X, mkt_order=True)['revenue']),
                 'dump0': r4(self.mkt.place_order(self.mkt.train_set_raw_q, self.t_dt, self.agent.X, prc=0, time_cap=self.agent.T_t, force_fill=True)['revenue']),
                 'dump1': r4(self.mkt.place_order(self.mkt.train_set_raw_q, self.t_dt, self.agent.X, prc=1, time_cap=self.agent.T_t, force_fill=True)['revenue']),
                 'dump2': r4(self.mkt.place_order(self.mkt.train_set_raw_q, self.t_dt, self.agent.X, prc=2, time_cap=self.agent.T_t, force_fill=True)['revenue']),
                 'dump3': r4(self.mkt.place_order(self.mkt.train_set_raw_q, self.t_dt, self.agent.X, prc=3, time_cap=self.agent.T_t, force_fill=True)['revenue']),
                 'dump4': r4(self.mkt.place_order(self.mkt.train_set_raw_q, self.t_dt, self.agent.X, prc=4, time_cap=self.agent.T_t, force_fill=True)['revenue'])}

            remain_X = copy.deepcopy(self.agent.X)
            remain_t = copy.deepcopy(self.agent.T_t)

            agent_revenue = 0
            agent_reward = 0
            init_prc = self.train_set.iloc[self.t, :]['midprice']
            for time_t in range(self.t, self.t + self.agent.T_t):
                time_t_dt = self.train_set.iloc[time_t, :]['dt_time']
                # 1. prepare state and gen action
                state_pvt = np.array([remain_X, remain_t])
                state_pub = (self.train_set.iloc[time_t, :][self.state_pub_name] - self.mkt.scale_min)/(self.mkt.scale_max - self.mkt.scale_min)
                state_feat = None

                state = {'pvt': state_pvt, 'pub': np.array(state_pub), 'feat': np.array(state_feat)}
                #qu, prc = self.agent.act(state)
                qu = self.agent.act(state)
                prc = 0
                # print('time_t:', time_t, 'remain_X:', remain_X, 'remain_t:', remain_t, 'qu:', qu, 'prc:', prc)
                # 2. execute the child order
                if remain_t!=1:
                    child_result = self.mkt.step_and_reward(time_t_dt, qu, prc, is_last=False)
                else:
                    child_result = self.mkt.step_and_reward(time_t_dt, qu, is_last=True)
                # print(child_result)
                # 3. remember and go forward
                reward = (init_prc - child_result['prc_avg'])*child_result['qu_exe'] # positive reward is better
                # print(init_prc, child_result['prc_avg'], child_result['qu_exe'], reward)
                self.agent.remember(state, qu, prc, reward)
                revenue = child_result['revenue']
                agent_reward = agent_reward + reward
                agent_revenue = agent_revenue + revenue
                ideal_revenue = -init_prc * self.agent.X
                remain_X = remain_X - child_result['qu_exe']
                remain_t = remain_t - 1

            trd_hist_str = str(idx_episode) + ";" + r4s(self.agent.exploration_rate) + ";" + \
                           str(self.agent.X) + ";" + str(self.agent.T_t) + ";" + \
                           r4s(agent_revenue) + ";" + r4s(agent_reward) + ";" + \
                           str(bm_revenue) + ";" + r4s(ideal_revenue) + "\n"
            trd_hist_file.write(trd_hist_str)
            trd_hist_file.flush()

        print("a")

            # following is for replay
            # next_state_pub_multiple = self.mkt.run_kNN_simulator(state_pub=state_pub, k=10)
            # self.agent.decay_exploration_rate()

            # if self.agent.remain_X is not 0: # if parent order is not cleared
            #     self.agent.T_dt = self.train_set.iloc[self.t + self.agent.T_t, :]['dt_time']
            #     last_child = self.mkt.place_order(self.mkt.train_set_raw_q, self.agent.T_dt , self.agent.remain_X, mkt_order=True)
            #     self.agent.revenue = self.agent.revenue + last_child['revenue']
            #     self.agent.remain_X = self.agent.remain_X - last_child['qu_exe']



    def parent_reset(self, args):
        begin_time = random.randint(0, round(0.8*self.train_set.shape[0]))
        len_problem = random.randint(args.min_prob_len, min(args.max_prob_len, self.train_set.shape[0] - begin_time))
        remain_qu = random.randint(-args.max_order_size, args.max_order_size)
        init_prc = self.train_set.iloc[begin_time, :]['midprice']
        self.agent.revenue = 0
        return begin_time, len_problem, remain_qu, init_prc

if __name__ == "__main__":
    if True:
        argp = argparse.ArgumentParser(description='Optimal Trade Execution',
                                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        # Main Control
        argp.add_argument('--data_dir', type=str, default="./data")

        argp.add_argument('--train_set_dir', action="store", default="lvl2_samples")
        argp.add_argument('--train_set_no', type=int, default=66)

        argp.add_argument('--console_display', dest='console_display', action='store_false', default=True)

        argp.add_argument('--no_console_display', dest='console_display', action='store_false')
        argp.add_argument('--train_mode', action="store_true", default=True)
        argp.add_argument('--simulation_mode', action="store_true", default=False)
        argp.add_argument('--new_trans_hist', action="store_true", default=False)

        # Problem Setting (Parent Order)
        argp.add_argument('--max_prob_len', type=int, default=10)
        argp.add_argument('--min_prob_len', type=int, default=5)
        argp.add_argument('--max_order_size', type=int, default=300)
        argp.add_argument('--state_pub', action="store", default="1vol, 2trd_vol, 3pressure")

        # Agent
        argp.add_argument('--exploration_decay', type=float, default=0.999)
        argp.add_argument('--init_exploration_rate', type=float, default=0.9)
        argp.add_argument('--max_memory', type=int, default=700)
        argp.add_argument('--sample_batch_size', type=int, default=700)
        argp.add_argument('--train_gap', type=int, default=5)
        argp.add_argument('--deepQ_epoch', type=int, default=50)
        argp.add_argument('--deepQ_lr', type=int, default=0.01)
        argp.add_argument('--episodes', type=int, default=700)
        args = argp.parse_args()
        print(args)

    solver = Solver(args)
    if args.train_mode:
        print("Train Mode Begins")
        solver.train(args)
    elif args.simulation_mode:
        print("Simulation Mode Begins")
        solver.simulate(args)
    else:
        print("Something Else")
