import random
import pdb
import numpy as np

from utils import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.optim as optim


class Agent(nn.Module):

    def __init__(self, args):
        super(Agent, self).__init__()
        # input size is number of used variables (for now, simple model)
        # if use trainable preprocessing model
        input_size = 7
        self.min_gru = nn.GRU(input_size, args.gru_h_dim, bias=False, dropout=args.dropout,
                            bidirectional= args.bidirectional, batch_first=True)
        self.sec_gru = nn.GRU(input_size, args.gru_h_dim, bias=False, dropout=args.dropout,
                            bidirectional= args.bidirectional, batch_first=True)

        if args.bidirectional == True:
            h_out = args.gru_h_dim * 4
        else:
            h_out = args.gru_h_dim * 2

        self.shared_layer = nn.Sequential(
                            nn.Linear(h_out, int(h_out/2)),
                            nn.ReLU(),
                            nn.Linear(int(h_out/2), 16),
                            nn.ReLU(),
                            nn.Linear(16, 8),
                            nn.ReLU(),
                            )
        self.pred_action = nn.Linear(8, 3)

    def forward(self, x1, x2, x2_len):
        """
        Get value function approximated by the agent
        [ Input ]
        x1      : Longer information aggregated every minute. Fixed (batch, lookback, n_feats) shape
        x2      : Shorter information aggregated every second. Variable lenth, padded as (batch, max_len, n_feats)
        x2_len  : unsorted length of x2 befored padding
        [ Output ]
        action_value    : [Bid, Netral, Ask] action value
        """
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

        state_rep = self.shared_layer(h_cat)
        action_value = F.softmax(self.pred_action(state_rep))
        #trade_amount = F.relu(self.pred_amount(state_rep))
        return action_value

    def predict():
        pass