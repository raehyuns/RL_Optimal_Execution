import os
import time
import argparse
import random
import uuid
import pdb
import logging

from utils import *
from market import *
from agent import *

import torch
import torch.nn as nn
import torch.optim as optim


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
        self.model_id = model_id

        self.n_episode = args.n_episode
        self.console_display = args.console_display
        self.train_gap = args.train_gap
        self.sample_batch_size = args.sample_batch_size
        self.pu = args.pu
        self.state_pub_name = parse_state_pub(args.state_pub)

        self.mkt = Market(args, logger)
        self.exp_memory = ReplayMemory(args.max_memory)
        self.agent = Agent(args).cuda()

        self.mkt.train_set_raw_q, self.mkt.cls_raw_q = load_train_set(args)
        self.cls_prc = self.mkt.cls_raw_q.iloc[0].trd_prc
        self.train_set = agg_to_min1(self.mkt.train_set_raw_q)

        self.trans_hist = []

        self.logger.info("Solver object is initialized")

    def get_state(self, lookback):
        """
        Get state information data at current point
        [ Input ]
        time_limit : Maximum times agent have to make decision
        lookback   : How many minutes that we provide to agent (before init time)

        [ Output ]
        init_dt     : Randomly initialized starting point
        long_X      : Min aggregated information for lookback minutes (lookback, # features)
        short_X     : Latest 1 minute information aggregated by second (60, # features)
        """
        cur_idx = random.randint(0+lookback, self.train_set.shape[0])
        cur_dt = self.train_set.iloc[cur_idx, :]['dt_time']
        frt_dt = self.train_set.iloc[cur_idx-1, :]['dt_time']

        # Aggregated minute information to provide Longer term
        summary_cols = ['agg_vol', 'agg_trd_vol', 'pressure',
                        'ask_P1', 'bid_P1', 'ask_Qa', 'bid_Qa']
        lb = min(lookback, cur_idx)
        long_X = normalize_target_cols(self.train_set.iloc[cur_idx-lb:cur_idx][summary_cols])

        short_X = self.mkt.train_set_raw_q.loc[(self.mkt.train_set_raw_q.time_hr==frt_dt.hour)
                                        & (self.mkt.train_set_raw_q.time_min==frt_dt.minute)]
        short_X = normalize_target_cols(agg_to_sec1(short_X)[summary_cols])
        return cur_idx, cur_dt, long_X, short_X

    def get_reward(self, cls_prc, quote_prc):
        return (cls_prc-quote_prc)


    def train(self, args):
        def r2s(x): return(str(round(x,2)))
        def r4s(x): return(str(round(x,4)))
        def r4(x): return(round(x,4))

        trd_hist_file = open(args.save_dir + '/trd_hist/' + self.model_id + '.txt', 'w')
        trd_hist_file.write(str(args) + '\r\n')
        trd_hist_file.flush()
        self.logger.info("Start training an agent")

        step = 0
        while True:
            idx, dt, x1, x2 = self.get_state(args.min_lookback)
            self.agent.X = -100
            order_result = self.mkt.place_order(self.mkt.train_set_raw_q, dt, self.agent.X)
            state = (x1.values, x2.values)
            reward = self.get_reward(self.cls_prc, order_result['qu_prc'])
            self.exp_memory.push((state, reward))
            self.logger.info("Memory length {}".format(len(self.exp_memory.memory)))

            if step % args.update_interval == 0 and step!=0:
                sampled_exp = self.exp_memory.sample(self.sample_batch_size)
                x1, x2, r, x2_len = to_feed_format(sampled_exp)
                self.agent(x1,x2, x2_len)
            step += 1



if __name__ == "__main__":

    argp = argparse.ArgumentParser(description='Optimal Trade Execution',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Main Control
    argp.add_argument('--data_dir', action="store", default="./data")
    argp.add_argument('--save_dir', action="store", default="./save")
    argp.add_argument('--train_set_no', type=int, default=66)

    argp.add_argument('--save_log', action='store_true', default=False)


    argp.add_argument('--console_display', dest='console_display', action='store_false', default=True)
    argp.add_argument('--no_console_display', dest='console_display', action='store_false')
    argp.add_argument('--train_mode', action="store_true", default=True)
    argp.add_argument('--simulation_mode', action="store_true", default=False)
    argp.add_argument('--new_trans_hist', action="store_true", default=False)

    # Problem Setting (Parent Order)
    argp.add_argument('--n_episode', type=int, default=10)
    argp.add_argument('--time_limit', type=int, default=10, help='maximum minutes for decision')
    argp.add_argument('--min_lookback', type=int, default=10, help='how many minutes agent should look')
    argp.add_argument('--pu', type=float, default=0.05)
    argp.add_argument('--transaction_fee', type=float, default=0.0)
    argp.add_argument('--max_order', type=int, default=100)

    argp.add_argument('--max_prob_len', type=int, default=10)
    argp.add_argument('--min_prob_len', type=int, default=5)
    argp.add_argument('--max_order_size', type=int, default=300)
    argp.add_argument('--state_pub', action="store", default="1vol, 2trd_vol, 3pressure")

    # Agent
    argp.add_argument('--update_interval', type=int, default=5)
    argp.add_argument('--exploration_decay', type=float, default=0.999)
    argp.add_argument('--init_exploration_rate', type=float, default=0.9)
    argp.add_argument('--max_memory', type=int, default=700)
    argp.add_argument('--sample_batch_size', type=int, default=5)
    argp.add_argument('--train_gap', type=int, default=5)
    argp.add_argument('--deepQ_epoch', type=int, default=50)
    argp.add_argument('--deepQ_lr', type=int, default=0.01)

    argp.add_argument('--gru_h_dim', type=int, default=20)
    argp.add_argument('--bidirectional', default=True, action="store_false")
    argp.add_argument('--dropout', type=int, default=0.1)
    args = argp.parse_args()


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

    if args.train_mode:
        print("Train Mode Begins")
        solver.train(args)
    elif args.simulation_mode:
        print("Simulation Mode Begins")
        solver.simulate(args)
    else:
        print("Something Else")
