
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

        self.console_display = args.console_display
        self.sample_batch_size = args.sample_batch_size
        self.pu = args.pu
        self.state_pub_name = parse_state_pub(args.state_pub)
        self.qu = 10

        self.mkt = Market(args, logger)
        self.exp_memory = ReplayMemory(args.max_memory)
        self.agent = Agent(args).cuda()

        # store all train, test dataset in list
        # each element of list is tuple of (raw dataframe, close price)
        self.train_data_list, self.test_data_list = load_data(args)
        self.logger.info("All data loaded")

        self.train_set, self.mkt.train_set_raw_q, self.mkt.cls_raw_q, self.cls_prc = select_data(self.train_data_list)
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
        cur_idx = random.randint(0+lookback, self.train_set.shape[0]-1)
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

    def get_reward(self, cls_prc, ask_prc, bid_prc):
        return [(cls_prc-bid_prc)/bid_prc, 0., (ask_prc-cls_prc)/ask_prc]

    def train(self, args):
        if args.save_log:
            trd_hist_file = open(args.save_dir + '/trd_hist/' + self.model_id + '.txt', 'w')
            trd_hist_file.write(str(args) + '\r\n')
            trd_hist_file.flush()
        self.logger.info("-----------------------Start training an agent")

        step = 0
        while True:
            # Continuely add (state, reward tuple to memory)
            idx, dt, x1, x2 = self.get_state(args.min_lookback)
            # fixed amount is traded in first version
            bid_prc = self.mkt.place_order(self.mkt.train_set_raw_q, dt, self.qu)['qu_prc']
            ask_prc = self.mkt.place_order(self.mkt.train_set_raw_q, dt, -self.qu)['qu_prc']
            state = (x1.values, x2.values)
            reward = self.get_reward(self.cls_prc, bid_prc, ask_prc)
            self.exp_memory.push((state, reward))

            if step % args.update_interval == 0 and step!=0:
                # conduct update for every update interval
                self.agent.train()
                self.agent.opt.zero_grad()
                #self.logger.info("Memory length {}".format(len(self.exp_memory.memory)))
                sampled_exp = self.exp_memory.sample(self.sample_batch_size)
                x1, x2, rwd, x2_len = to_feed_format(sampled_exp)
                act_value, agt_action = self.agent(x1, x2, x2_len)
                eps_grd_act = torch.LongTensor(epsilon_greedy(agt_action, self.agent.epsilon)).cuda()

                exp_rwd = torch.gather(act_value, 1, eps_grd_act.unsqueeze(1))
                true_rwd = torch.gather(torch.FloatTensor(rwd).cuda(), 1, eps_grd_act.unsqueeze(1))

                self.agent.epsilon *= self.agent.eplr_decay
                loss = torch.sqrt(self.agent.criterion(exp_rwd, true_rwd))
                loss.backward()
                #nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_max_norm)
                self.agent.opt.step()
                act_dist = np.array([(eps_grd_act.cpu().numpy()==x).sum() for x in range(3)])/self.sample_batch_size

                message = "\n Epsilon {:2.3f} | Action Value {:2.3f} {:2.3f} {:2.3f} | "\
                            .format(self.agent.epsilon,
                                    act_dist[0], act_dist[1], act_dist[2])
                message += "RMSE Loss {:2.4f} | Average return {:2.4f} %"\
                            .format(loss.cpu().item(), true_rwd.mean().cpu().item()*100)

                self.logger.info(message)

                # Sample another dataset
                self.train_set, self.mkt.train_set_raw_q, self.mkt.cls_raw_q, self.cls_prc = select_data(self.train_data_list)

            if step % args.eval_interval == 0 and step!=0:

                self.evaluate(args)

            step += 1

    def evaluate(self, args):
        test_exp = []
        if args.save_log:
            trd_hist_file = open(args.save_dir + '/trd_hist/' + self.model_id + '.txt', 'w')
            trd_hist_file.write(str(args) + '\r\n')
            trd_hist_file.flush()
        self.test_set, self.mkt.test_set_raw_q, self.mkt.te_cls_raw_q, self.te_cls_prc = select_data(self.test_data_list)

        self.logger.info("-----------------------Start evaluating an agent")

        n_sample = 0
        while True:
            # Continuely add (state, reward tuple to memory)
            idx, dt, x1, x2 = self.get_state(args.min_lookback)
            # fixed amount is traded in first version
            bid_prc = self.mkt.place_order(self.mkt.test_set_raw_q, dt, self.qu)['qu_prc']
            ask_prc = self.mkt.place_order(self.mkt.test_set_raw_q, dt, -self.qu)['qu_prc']
            state = (x1.values, x2.values)
            reward = self.get_reward(self.te_cls_prc, bid_prc, ask_prc)
            test_memory.append((state, reward))
            n_sample += 1

            if n_sample == args.eval_batch_size:
                self.agent.eval()
                x1, x2, rwd, x2_len = to_feed_format(test_exp)
                act_value, agt_action = self.agent(x1, x2, x2_len)
                agt_action = torch.LongTensor(acg_action).cuda()

                exp_rwd = torch.gather(act_value, 1, eps_grd_act.unsqueeze(1))
                true_rwd = torch.gather(torch.FloatTensor(rwd).cuda(), 1, eps_grd_act.unsqueeze(1))

                loss = torch.sqrt(self.agent.criterion(exp_rwd, true_rwd))
                #nn.utils.clip_grad_norm_(model.parameters(), self.args.grad_max_norm)
                act_dist = np.array([(agt_action.cpu().numpy()==x).sum() for x in range(3)])/self.sample_batch_size

                message = "\nAction Value {:2.3f} {:2.3f} {:2.3f} | "\
                            .format(act_dist[0], act_dist[1], act_dist[2])
                message += "RMSE Loss {:2.4f} | Average return {:2.4f} %"\
                            .format(loss.cpu().item(), true_rwd.mean().cpu().item()*100)
                self.logger.info(message)

                # Sample another dataset
                self.train_set, self.mkt.train_set_raw_q, self.mkt.cls_raw_q, self.cls_prc = select_data(self.train_data_list)


if __name__ == "__main__":

    argp = argparse.ArgumentParser(description='Optimal Trade Execution',
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Main Control
    argp.add_argument('--data_dir', action="store", default="./data/samples")
    argp.add_argument('--save_dir', action="store", default="./save")
    argp.add_argument('--data_no_list', type=list, default=[n for n in range(58,73)])
    argp.add_argument('--save_log', action='store_true', default=False)
    argp.add_argument('--console_display', dest='console_display', action='store_true', default=False)
    argp.add_argument('--mode', default='train', choices=['train', 'simulation', 'debug'])
    argp.add_argument('--new_trans_hist', action="store_true", default=False)

    # Problem Setting
    argp.add_argument('--min_lookback', type=int, default=10, help='how many minutes agent should look')
    argp.add_argument('--pu', type=float, default=0.05)
    argp.add_argument('--transaction_fee', type=float, default=0.0)
    argp.add_argument('--max_order', type=int, default=100)
    argp.add_argument('--state_pub', action="store", default="1vol, 2trd_vol, 3pressure")

    #argp.add_argument('--time_limit', type=int, default=10, help='maximum minutes for decision')

    # Agent
    argp.add_argument('--update_interval', type=int, default=32)
    argp.add_argument('--eval_interval', type=int, default=500)
    argp.add_argument('--exploration_decay', type=float, default=0.999)
    argp.add_argument('--init_exploration_rate', type=float, default=0.9)
    argp.add_argument('--max_memory', type=int, default=200)
    argp.add_argument('--sample_batch_size', type=int, default=32)
    argp.add_argument('--eval_batch_size', type=int, default=100)
    ## model
    argp.add_argument('--gru_h_dim', type=int, default=20)
    argp.add_argument('--bidirectional', default=True, action="store_false")
    argp.add_argument('--dropout', type=int, default=0.1)
    ## optimizer
    argp.add_argument('--optimizer', type=str, default='Adam')
    argp.add_argument('--learning_rate', type=int, default=5e-3)
    argp.add_argument('--weight_decay', type=int, default=1e-4)
    argp.add_argument('--momentum', type=int, default=0.9)

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
