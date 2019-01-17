
import os
import time
import argparse
import random
import uuid
import pdb
import logging
import pickle

from utils import *
from market import *
from agent import *
from config import *

import torch
import torch.nn as nn
import torch.optim as optim

import tensorflow as tf
#from tensorboardX import SummaryWriter


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
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)



class Solver:
    def __init__(self, args, logger, model_id):
        self.logger = logger
        self.writer = tf.summary.FileWriter('./save/tf_board')
        self.model_id = model_id

        self.console_display = args.console_display
        self.sample_batch_size = args.sample_batch_size
        self.pu = args.pu
        self.lookback = args.min_lookback
        self.order_type = args.order_type
        #self.state_pub_name = parse_state_pub(args.state_pub)
        self.grad_max_norm = args.grad_max_norm

        self.mkt = Market(args, logger)
        self.exp_memory = ReplayMemory(args.max_memory)
        self.agent = Agent(args).cuda()

        # store all train, test dataset in list
        # each element of list is tuple of (raw dataframe, close price)
        self.train_data_list, self.test_data_list = load_data(args)
        self.logger.info("All data loaded")

        self.mkt.train_min1_agg, self.mkt.train_raw_q, self.mkt.cls_raw_q, self.mkt.cls_prc = select_data(self.train_data_list)
        self.trans_hist = []

        self.logger.info("Solver object is initialized")


    def get_state(self, min1_agg_data, raw_data, cur_idx):
        """
        Get state information data at current point
        [ Input ]
        cur_idx         : index of current data point
        min1_agg_data   : data aggregated by every minute
        raw_data        : data with all quotes
        time_limit      : Maximum times agent have to make decision
        lookback        : How many minutes that we provide to agent (before init time)

        [ Output ]
        init_dt     : Randomly initialized starting point
        long_X      : Min aggregated information for lookback minutes (lookback, # features)
        short_X     : Latest 1 minute information aggregated by second (60, # features)
        """
        #cur_idx = random.randint(0+lookback, min1_agg_data.shape[0]-1)
        cur_dt = min1_agg_data.iloc[cur_idx, :]['dt_time']
        frt_dt = min1_agg_data.iloc[cur_idx-1, :]['dt_time']
        cur_midprc = min1_agg_data.iloc[cur_idx, :]['midprice']

        # Aggregated minute information to provide Longer term
        summary_cols = ['agg_vol', 'agg_trd_vol', 'pressure',
                        'ask_P1', 'bid_P1', 'ask_Qa', 'bid_Qa']
        lb = min(self.lookback, cur_idx)
        long_X = normalize_target_cols(min1_agg_data.iloc[cur_idx-lb:cur_idx][summary_cols])

        short_X = raw_data.loc[(raw_data.time_hr==frt_dt.hour) & (raw_data.time_min==frt_dt.minute)]
        short_X = normalize_target_cols(agg_to_sec1(short_X)[summary_cols])
        return cur_dt, long_X, short_X, cur_midprc

    def get_reward(self, mid_prc, trd_prc):
        if self.order_type =='bid':
            reward = mid_prc - trd_prc
        else: # ask
            reward = trd_prc - mid_prc
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
            # Initialize random point to start episode
            cur_idx = random.randint(0+args.min_lookback, self.mkt.train_min1_agg.shape[0]-args.max_time)
            remain_q = args.order_amt
            remain_t = args.max_time
            while (remain_q and remain_t):
                # Continuely add (state, reward tuple to memory) while proceeding episode
                dt, x1, x2, mid_prc = self.get_state(self.mkt.train_min1_agg, self.mkt.train_raw_q, cur_idx)
                act_value, agt_action = self.agent(x1.values)
                # epsilon-greedy
                if np.random.uniform()<self.agent.epsilon: # random action
                    act_idx = np.random.randint(args.n_prc_lev*2+2)
                else: # greedy action
                    act_idx = agt_action.cpu().item()

                # calculating reward
                if act_idx != 0 : # if trade
                    prc_lev = act_idx - args.n_prc_lev
                    order_result = self.mkt.place_order(self.mkt.train_raw_q, dt, args.one_trd_amt, prc_lev, args.order_type)
                    trd_prc, exe_qu = order_result['qu_prc'], order_result['qu_exe']
                    reward = self.get_reward(mid_prc, trd_prc)
                    remain_q -= exe_qu
                else: # should decide how to cal reward when not trade
                    reward = self.get_reward(mid_prc, self.mkt.cls_prc)
                self.exp_memory.push((act_value, act_idx, reward))

                # private state update
                remain_t -= 1

            episode_step +=1

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
                #loss = torch.sqrt(torch.mean((exp_rwd-true_rwd)**2))
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.grad_max_norm)
                self.agent.opt.step()
                #act_dist = np.array([(act_values.cpu().numpy()==x).sum() for x in range(args.n_prc_lev*2+2)])/sample_size

                message = "\n Epsilon {:2.3f} | RMSE Loss {:2.4f} | Average Reward {:2.4f}"\
                            .format(self.agent.epsilon, loss.cpu().item(), true_rwd.mean().cpu().item())
                self.logger.info(message)

                scalar_summary(self.writer, 'Train Loss', loss.cpu().item(), update_step)
                scalar_summary(self.writer, 'Train Reward', true_rwd.mean().cpu().item()*100, update_step)
                # Sample another dataset
                self.mkt.train_min1_agg, self.mkt.train_raw_q, self.mkt.cls_raw_q, self.mkt.cls_prc = select_data(self.train_data_list)
                update_step += 1

            if update_step % args.eval_interval == 0 and update_step !=0:
                result = self.evaluate(args, eval_step)
                eval_step += 1


    def evaluate(self, args, eval_step):
        test_exp = []
        if args.save_log:
            trd_hist_file = open(args.save_dir + '/trd_hist/' + self.model_id + '.txt', 'w')
            trd_hist_file.write(str(args) + '\r\n')
            trd_hist_file.flush()
        self.mkt.test_min1_agg, self.mkt.test_raw_q, self.mkt.te_cls_raw_q, self.mkt.te_cls_prc = select_data(self.test_data_list)

        self.logger.info("\n\n-----------------------Start evaluating an agent")

        episode_step = 0
        while episode_step < args.n_eval_episode:
            episode_exe_q = 0
            n_execute = 0
            episode_loss = []
            episode_reward = []
            cur_idx = random.randint(0+args.min_lookback, self.mkt.test_min1_agg.shape[0]-args.max_time)
            remain_q = args.order_amt
            remain_t = args.max_time
            while (remain_q and remain_t):
                dt, x1, x2, mid_prc = self.get_state(self.mkt.test_min1_agg, self.mkt.test_raw_q, cur_idx)
                self.agent.eval()
                act_value, agt_action = self.agent(x1.values)
                act_idx = agt_action.cpu().item()

                # calculating reward
                if act_idx != 0 : # if trade
                    prc_lev = act_idx - args.n_prc_lev
                    order_result = self.mkt.place_order(self.mkt.test_raw_q, dt, args.one_trd_amt, prc_lev, args.order_type)
                    trd_prc, exe_qu = order_result['qu_prc'], order_result['qu_exe']
                    episode_exe_q += exe_qu
                    true_rwd = self.get_reward(mid_prc, trd_prc)
                    remain_q -= exe_qu

                else: # should decide how to cal reward when not trade
                    true_rwd = self.get_reward(mid_prc, self.mkt.cls_prc)
                exp_rwd = act_value[agt_action].cpu().item()
                episode_reward.append(true_rwd)
                episode_loss.append(exp_rwd-true_rwd)

                remain_t -= 1
                n_execute += 1

            loss = np.sqrt(np.square(episode_loss).mean())[0]
            reward = np.mean(episode_reward)
            act_dist = np.array([(agt_action.cpu().numpy()==x).sum() for x in range(args.n_prc_lev*2+2)])/n_execute

            message = "\n Eval Step {} | Action Ratio {:2.3f} {:2.3f} {:2.3f} | "\
                        .format(eval_step, act_dist[0], act_dist[1], act_dist[2])
            message += "RMSE Loss {:2.4f} | Reward {:2.4f} | "\
                        .format(loss.cpu().item(), true_rwd.mean().cpu().item())
            message += "Executed order {:2.4f} "\
                        .format(episode_exe_q)
            message += "\n------------------------------------------- \n\n"

            self.logger.info(message)
            scalar_summary(self.writer, 'Eval Loss', loss, args.n_eval_episode*eval_step+episode_step)
            scalar_summary(self.writer, 'Eval Reward', reward, args.n_eval_episode*eval_step+episode_step)
        #return [(act_dist[0], act_dist[1], act_dist[2]), loss.cpu().item(), true_rwd.mean().cpu().item()*100]
        return

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
    elif 'debug'in args.mode:
        logger.info("Debug Mode Begins")
        solver.train(args)
    else:
        logger.info("Something Else")
