import random
import pdb
import numpy as np

from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.optim as optim
from torchsummary import summary

class Agent(nn.Module):

    def __init__(self, args):
        super(Agent, self).__init__()
        # input size is number of used variables (for now, simple model)
        # if use trainable preprocessing model
        input_size = args.min_lookback * 5
        self.epsilon = args.init_exploration_rate
        self.eplr_decay = args.exploration_decay
        self.min_eplr = args.min_exploration_rate
        args.fc_layer_size.insert(0, input_size)

        #self.min_gru = nn.GRU(input_size, args.gru_h_dim, bias=False,
        #                    bidirectional= args.bidirectional, batch_first=True)
        #self.sec_gru = nn.GRU(input_size, args.gru_h_dim, bias=False,
        #                    bidirectional= args.bidirectional, batch_first=True)

        self.criterion = nn.MSELoss()

        '''
        if args.bidirectional == True:
            h_out = args.gru_h_dim * 4
        else:
            h_out = args.gru_h_dim * 2
        '''

        fc = []
        for h_in, h_out in zip(args.fc_layer_size, args.fc_layer_size[1:]):
            fc.append(nn.Linear(h_in, h_out))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(args.dropout))
        self.fc_layer = nn.Sequential(*fc)
        self.pred_action = nn.Linear(args.fc_layer_size[-1], args.n_amt_lv*2+1)
        self.init_weight()
        self.select_optimizer(args)

    def init_weight(self):
        def seq_init(m):
            if type(m) == nn.Linear:
                nn.init.xavier_uniform(m.weight)
        self.fc_layer.apply(seq_init)
        nn.init.xavier_uniform(self.pred_action.weight)
        nn.init.zeros_(self.pred_action.bias)

    def select_optimizer(self, args):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        if(args.optimizer == 'Adam'):
            self.opt =  optim.Adam(parameters, lr=args.learning_rate,
                                        weight_decay=args.weight_decay)
        elif(args.optimizer == 'RMSprop'):
            self.opt =  optim.RMSprop(parameters, lr=args.learning_rate,
                                            weight_decay=args.weight_decay,
                                            momentum=args.momentum)
        elif(args.optimizer == 'SGD'):
            self.opt =  optim.SGD(parameters, lr=args.learning_rate,
                                        weight_decay=args.weight_decay,
                                        momentum=args.momentum)
        elif(args.optimizer == 'Adagrad'):
            self.opt =  optim.Adagrad(parameters, lr=args.learning_rate)
        elif(args.optimizer == 'Adadelta'):
            self.opt =  optim.Adadelta(parameters, lr=args.learning_rate)



    def forward(self, x1, x2=None, x2_len=None):
        """
        Get value function approximated by the agent
        [ Input ]
        x1      : Longer information aggregated every minute. Fixed (batch, lookback, n_feats) shape
        x2      : Shorter information aggregated every second. Variable lenth, padded as (batch, max_len, n_feats)
        x2_len  : unsorted length of x2 befored padding
        [ Output ]
        action_value    : [Bid, Netral, Ask] action value
        """
        '''
        def get_last_state(batch_h_seq, idx_list):
            last_h_list = []
            for h_seq, idx in zip(batch_h_seq, idx_list):
                last_h_list.append(h_seq[idx-1].unsqueeze(0))
            return torch.cat(last_h_list, 0)

        _, h1 = self.min_gru(torch.FloatTensor(x1).cuda())
        h1 = torch.cat([h1[0], h1[1]], 1)

        sorted_data, sorted_len, idx_unsort = sort_sequence(x2, x2_len)
        packed = pack_padded_sequence(sorted_data, sorted_len, batch_first=True)
        packed_h, _ = self.sec_gru(packed)
        padded_h, padded_length = pad_packed_sequence(packed_h, batch_first=True)
        h2 = unsort_sequence(get_last_state(padded_h, padded_length), idx_unsort)
        h_cat = torch.cat([h1, h2], 1)
        '''
        #state_rep = self.shared_layer(h_cat)
        state_rep = self.fc_layer(torch.FloatTensor(x1).cuda())
        #policy = F.softmax(self.pred_action(state_rep), dim=1)
        action_value = self.pred_action(state_rep)
        action = torch.max(action_value, 0)[1]
        #trade_amount = F.relu(self.pred_amount(state_rep))
        return action_value, action

    def predict():
        pass
