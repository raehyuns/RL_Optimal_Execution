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


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, exp):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = exp
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        random.shuffle(self.memory)
        samples = self.memory[:batch_size]
        del self.memory[:batch_size]
        self.position = len(self.memory)
        return samples

    def __len__(self):
        return len(self.memory)



class Solver:
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

    def sim_trd_prc(self, cur_dt, end_dt, qt_prc, qt_amt, mkt_sim):
        '''
        Simulated Transaction at this game's date would mkt_sim
        (Loaded and Stored with raw market data)
        ISSUE : should check that the prices match !! (different DB)
        '''
        order_hr, order_min = cur_dt.hour, cur_dt.minute
        end_hr, end_min = end_dt.hour, end_dt.minute
        # temporary price only for debug
        order_hr, order_min = 12, 46
        end_dt = end_dt.replace(day=14, hour=13,minute=1)
        # Time & Price Condition
        executed = mkt_sim.loc[(mkt_sim.order_hr==order_hr)\
                                & (mkt_sim.order_min==order_min)]
                                #& (mkt_sim.OrdPrc==qt_prc)]
        # temporary price only for debug
        executed = executed.loc[executed.OrdPrc==executed.OrdPrc.iloc[0]]
        # Executed before game end point
        valid = executed.loc[(executed.ExecTime<=end_dt)]

        remain_q = qt_amt
        exe_info = []
        for idx, trans in valid.iterrows():
            exe_info.append((min(trans.execqty, remain_q), trans.execprc))
            remain_q -= (min(trans.execqty, remain_q))
            if remain_q <= 0:
                break
        qty, prc = np.array(exe_info)[:,0], np.array(exe_info)[:,1]
        trd_prc = (qty*prc).sum() / qty.sum()
        return trd_prc, qty.sum()

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

    def train(self, args):
        temp_reward_writer = []
        if args.save_log:
            trd_hist_file = open(args.save_dir + '/trd_hist/' + self.model_id + '.txt', 'w')
            trd_hist_file.write(str(args) + '\r\n')
            trd_hist_file.flush()
        self.logger.info("\n\n-----------------------Start training an agent")

        global_step = 0
        episode_step = 0
        update_step = 0
        eval_step = 0
        while episode_step < args.max_episode:
            # Initialize random point to start episode. Decision is made every minute
            date = dt.datetime.strptime(str(self.mkt.train_raw_q.iloc[0]['date']), '%Y%m%d')
            cur_idx = random.randint(0+args.min_lookback, self.mkt.train_min1_agg.shape[0]-args.max_time-1)
            end_idx = cur_idx + self.max_time
            end_t = self.mkt.train_min1_agg.iloc[end_idx]['dt_time']
            end_dt = dt.datetime.combine(date, end_t)
            twap_prc, vwap_prc, end_prc = self.get_baseline(self.mkt.train_min1_agg, self.mkt.train_raw_q, cur_idx)

            # set benchmark price for episode
            if args.benchmark_type == 'day_close':
                bc_prc = self.mkt.cls_prc
            elif args.benchmark_type == 'window_close':
                bc_prc = end_prc
            remain_q = args.order_amt
            remain_t = args.max_time
            while (remain_q and remain_t):
                # Continuely add (state, reward tuple to memory) while proceeding episode
                rq = remain_q / args.order_amt
                rt = remain_t / args.max_time
                cur_t, x, mid_prc, bid1_prc, ask1_prc = self.get_state(self.mkt.train_min1_agg, self.mkt.train_raw_q, cur_idx, rt, rq)
                cur_dt = dt.datetime.combine(date, cur_t)
                act_value, agt_action = self.agent(x)
                # epsilon-greedy
                if np.random.uniform()<self.agent.epsilon: # random action
                    act_idx = np.random.randint(args.n_amt_lv+1)
                else: # greedy action
                    act_idx = agt_action.cpu().item()

                if act_idx%2==0: # Quote price = Bid 1
                    qt_prc = bid1_prc
                else: # Quote price = Ask 1
                    qt_prc = ask1_prc
                qt_amt = (act_idx//2+1) * self.amt_unit
                if args.trade_price_assumption=='midprice':
                    trd_prc = mid_prc
                    trd_amt = qt_amt
                elif args.trade_price_assumption=='market_simulator':
                    trd_prc, trd_amt = self.sim_trd_prc(cur_dt, end_dt,
                                                qt_prc, qt_amt, self.mkt.train_simtr)
                    if trd_amt != qt_amt: # when not fully traded
                        res_amt = qt_amt - trd_amt
                        trd_prc = ((trd_amt*trd_prc) + (res_amt*bc_prc)) / qt_amt
                # calculating reward
                if act_idx != 0 : # if trade
                    reward = self.get_reward(trd_prc, act_idx, bc_prc, rt, rq)
                    remain_q -= min(remain_q, trd_amt)
                else: # should decide how to cal reward when not trade
                    # considering remain_q and current mid price to calculate opportunity cost (remain time & order)
                    reward = self.get_reward(trd_prc, 0, bc_prc, rt, rq)

                self.exp_memory.push((act_value, act_idx, reward))

                # private state update
                cur_idx += 1
                remain_t -= 1

            episode_step +=1

            # Update agent parameters
            if episode_step % args.update_interval==0 and episode_step!=0:
                # conduct update for every update interval
                self.agent.train()
                self.agent.opt.zero_grad()

                # experience replay
                sample_size = min(self.sample_batch_size, len(self.exp_memory.memory))
                sampled_exp = self.exp_memory.sample(sample_size)

                act_values = torch.cat([x[0].unsqueeze(0) for x in sampled_exp], 0)
                act_idx = torch.LongTensor([x[1] for x in sampled_exp]).cuda()
                true_rwd = torch.FloatTensor(np.array([x[2] for x in sampled_exp])).cuda()
                exp_rwd = torch.gather(act_values, 1, act_idx.unsqueeze(1)).squeeze()
                # update exploration rate
                if self.agent.epsilon > self.agent.min_eplr :
                    self.agent.epsilon *= self.agent.eplr_decay

                loss = torch.sqrt(self.agent.criterion(exp_rwd, true_rwd))
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_max_norm)
                self.agent.opt.step()
                message = "\n Epsilon {:2.3f} | RMSE Loss {:2.4f} | Average Reward {:2.4f}"\
                            .format(self.agent.epsilon, loss.cpu().item(), true_rwd.mean().cpu().item())
                self.logger.info(message)

                if args.tf_board:
                    act_dist = np.array([(act_idx.cpu().numpy()==x).sum() for x in range(args.n_amt_lv+1)])/sample_size
                    self.writer.add_scalar('Train Loss', loss.cpu().item(), update_step)
                    self.writer.add_scalar('Train Reward', true_rwd.mean().cpu().item()*100, update_step)
                    act_dist_summary(self.writer, act_dist, 7, update_step, 'Train')
                # Sample another dataset
                self.mkt.train_min1_agg, self.mkt.train_raw_q, self.mkt.train_simtr, self.mkt.cls_prc = select_data(self.train_data_list)
                update_step += 1

            # Evaluation
            if episode_step % args.eval_interval == 0 and episode_step !=0:
                result = self.evaluate(args, eval_step)
                torch.save(self.agent.state_dict(), './save/model/{}'.format(self.model_id))
                eval_step += 1


    def evaluate(self, args, eval_step):
        test_exp = []
        if args.save_log:
            trd_hist_file = open(args.save_dir + '/trd_hist/' + self.model_id + '.txt', 'w')
            trd_hist_file.write(str(args) + '\r\n')
            trd_hist_file.flush()
        self.mkt.test_min1_agg, self.mkt.test_raw_q, self.mkt.test_simtr, self.mkt.te_cls_prc = select_data(self.test_data_list)

        self.logger.info("\n\n-----------------------Start evaluating an agent")

        episode_step = 0
        report_list = []
        act_dist_list = []

        while episode_step < args.n_eval_episode:
            date = dt.datetime.strptime(str(self.mkt.test_raw_q.iloc[0]['date']), '%Y%m%d')
            cur_idx = random.randint(0+args.min_lookback, self.mkt.test_min1_agg.shape[0]-args.max_time)
            twap_prc, vwap_prc, end_prc = self.get_baseline(self.mkt.test_min1_agg, self.mkt.test_raw_q, cur_idx)
            # set benchmark price for episode
            if args.benchmark_type == 'day_close':
                bc_prc = self.mkt.cls_prc
            elif args.benchmark_type == 'window_close':
                bc_prc = end_prc
            remain_q = args.order_amt
            remain_t = args.max_time
            # For report
            n_execute = 0
            no_trd = 0
            act_dist = [0] * (args.n_amt_lv*2+1)
            episode_loss = []
            episode_reward = []
            episode_exe = []
            while (remain_q and remain_t):
                rq = remain_q / args.order_amt
                rt = remain_t / args.max_time
                cur_t, x, mid_prc, bid1_prc, ask1_prc = self.get_state(self.mkt.test_min1_agg, self.mkt.test_raw_q, cur_idx, rt, rq)
                cur_dt = dt.datetime.combine(date, cur_t)
                self.agent.eval()
                act_value, agt_action = self.agent(x)
                act_idx = agt_action.cpu().item()
                '''
                fix action to understand reward structure
                agt_action = torch.LongTensor([5]).cuda()
                agt_idx = 5
                '''
                if act_idx%2==0: # Quote price = Bid 1
                    qt_prc = bid1_prc
                else: # Quote price = Ask 1
                    qt_prc = ask1_prc
                qt_amt = (act_idx//2+1) * self.amt_unit
                if args.trade_price_assumption=='midprice':
                    trd_prc = mid_prc
                    trd_amt = qt_amt
                elif args.trade_price_assumption=='market_simulator':
                    trd_prc, trd_amt = self.sim_trd_prc(cur_dt, end_dt,
                                                qt_prc, qt_amt, self.mkt.train_simtr)
                    if trd_amt != qt_amt: # when not fully traded
                        res_amt = qt_amt - trd_amt
                        trd_prc = ((trd_amt*trd_prc) + (res_amt*bc_prc)) / qt_amt

                # calculating reward
                if act_idx != 0 : # if trade
                    true_rwd = self.get_reward(trd_prc, act_idx, bc_prc, rt, rq)
                    episode_exe.append((trd_prc, min(remain_q, trd_amt)))
                    remain_q -= min(remain_q, trd_amt)
                else: # should decide how to cal reward when not trade
                    true_rwd = self.get_reward(trd_prc, 0, bc_prc, rt, rq)
                    no_trd += 1
                exp_rwd = act_value[agt_action].cpu().item()
                episode_loss.append(exp_rwd-true_rwd)
                act_dist[agt_action] += 1
                remain_t -= 1
                n_execute += 1
            if remain_q:
                episode_exe.append((end_prc, remain_q))
            episode_exe = np.array(episode_exe)

            avg_prc = (episode_exe[:,0]*episode_exe[:,1]).sum() / args.order_amt
            loss = np.sqrt(np.square(episode_loss).mean())
            def price_diff(agt_prc, target_prc, type):
                if type == 'buy':
                    return target_prc-agt_prc
                else:
                    return agt_prc-target_prc
            self.eval_prices.append([avg_prc, twap_prc, vwap_prc, end_prc])
            with open('./save/res/{}.pkl'.format(self.model_id), 'wb') as f:
                pickle.dump(self.eval_prices, f)

            cls_reward = price_diff(avg_prc, end_prc, self.order_type)
            twap_reward = price_diff(avg_prc, twap_prc, self.order_type)
            vwap_reward = price_diff(avg_prc, vwap_prc, self.order_type)
            no_trd_rate = no_trd/n_execute
            act_dist_list.append(np.expand_dims(np.array(act_dist)/n_execute, 0))
            report_list.append((loss, cls_reward, twap_reward, vwap_reward, no_trd_rate))
            episode_step+=1

        loss, cls_reward, twap_reward, vwap_reward, no_trd = np.array(report_list).mean(0)
        act_dist = np.concatenate(act_dist_list, 0).mean(0)

        message = "\n Eval Step {} | No Trade Rate {:2.4f} | RMSE Loss {:2.4f} |  ".format(eval_step, no_trd_rate, loss)
        message += " Reward Close {:2.4f} | Reward TWAP {:2.4f} | Reward VWAP {:2.4f} |"\
                    .format(cls_reward, twap_reward, vwap_reward)
        message += "\n------------------------------------------- \n\n"

        self.logger.info(message)

        if args.tf_board:
            self.writer.add_scalar('Eval Loss', loss, eval_step)
            self.writer.add_scalar('Eval Close Reward', cls_reward, eval_step)
            self.writer.add_scalar('Eval TWAP Reward', twap_reward, eval_step)
            self.writer.add_scalar('Eval VWAP Reward', vwap_reward, eval_step)
            self.writer.add_scalar('Eval Not Trade Rate', no_trd, eval_step)
            self.writer.add_histogram('Eval Act Value', act_dist, eval_step)
            act_dist_summary(self.writer, act_dist, 7, eval_step, 'Eval')
        return

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
            final_order, valid_order = get_live_orders()

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

    def real_orders(valid_order):
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

    solver = Solver(args, logger, model_id)

    if 'train'in args.mode:
        logger.info("Train Mode Begins")
        solver.train(args)
    elif 'simulation' in args.mode:
        logger.info("Simulation Mode Begins")
        solver.simulate(args)
    elif 'debug' in args.mode:
        logger.info("Debug Mode Begins")
        args.update_interval = 3
        args.eval_interval = 5
        args.n_eval_episode = 2
        solver.train(args)
    elif 'real_trading' in args.mode:
        logger.info("Trading with AI agent at real market")
        solver.real_trading(args)
    else:
        logger.info("Something Else")
