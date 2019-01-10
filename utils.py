import datetime
import pdb
import numpy as np
import pandas as pd
import torch

def load_scaled_trans_hist_pub(args):
    if args.new_trans_hist:
        list_files = os.listdir(os.getcwd() + '/' + args.train_set_dir)
        trans_hist = pd.DataFrame()
        for file in list_files:
            rq = load_train_set(args, file)
            rq_min = agg_to_min1(rq)
            trans_hist = trans_hist.append(rq_min)
        trans_hist = trans_hist.sort_index()
        trans_hist.to_pickle("transition_history.pkl")

    trans_hist_pub = pd.read_pickle("transition_history.pkl")[state_pub_name]
    scale_max = trans_hist_pub.max()
    scale_min = trans_hist_pub.min()
    trans_hist_pub = (trans_hist_pub - scale_min)/(scale_max-scale_min)

    return trans_hist_pub, scale_max, scale_min

def load_train_set(args, file=None):
    '''
    load raw order book data & change column names
    '''

    if file is None:
        raw_file = args.data_dir+ '/' + 'trade2009_' + str(args.train_set_no) + '.csv'
    else:
        raw_file = args.data_dir + '/' + file
    rq = pd.read_csv(raw_file) # raw_tick # os.listdir(data_dir)
    print(raw_file, "is loaded")

    def preprocess(rq):
        rq['dt_time'] = [
            datetime.time(int(str(x)[:-7]), int(str(x)[-7:-5]), int(str(x)[-5:-3]), int(str(x)[-3:]) * 1000) for x in
            rq['time']]
        rq['time_hr'] = [x.hour for x in rq['dt_time']]
        rq['time_min'] = [x.minute for x in rq['dt_time']]
        rq['time_sec'] = [x.second for x in rq['dt_time']]

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

    # use all the data for training before final 10 minutes
    # final 10 minutes data is used for reward cal.
    # Closing time (15:15) should not be changed
    tr_rq = rq[rq.time < 151500000]
    cls_rq = rq[rq.time > 151500000]

    tr_rq = preprocess(tr_rq)
    cls_rq = preprocess(cls_rq)

    return tr_rq, cls_rq

def parse_state_pub(args_str):
    glossary = {'1vol': 'min1_vol', '2trd_vol': 'min1_trd_vol', '3pressure': 'pressure'}
    alias = args_str.split(', ')
    alias.sort()
    return [glossary[x] for x in alias]

def gen_vol(prc):
    vol = np.sqrt(sum(np.square(np.diff(np.log(prc))))) * 100
    return vol

def agg_to_min1(rq):
    ## Aggregate to 1 minute (volatility, trading_volume, orderbook_pressure)
    mkt_min1_vol = rq.groupby(['date', 'time_hr', 'time_min']).midprice.agg(gen_vol)
    mkt_min1_vol = pd.DataFrame(mkt_min1_vol)
    mkt_min1_vol.columns = ['agg_vol']
    mkt_min1_trd_vol = rq.groupby(['date', 'time_hr', 'time_min']).trd_vol.agg(np.sum)/10000
    mkt_min1_trd_vol = pd.DataFrame(mkt_min1_trd_vol)
    mkt_min1_trd_vol.columns = ['agg_trd_vol']
    mkt_agg = mkt_min1_vol.join(mkt_min1_trd_vol)

    ## Takes the last status
    mkt_last = rq.groupby(['date', 'time_hr', 'time_min']).last()
    mkt_last = mkt_last[['dt_time', 'midprice', 'trd_prc',
                         'bid_P1', 'bid_P2', 'bid_P3', 'bid_P4', 'bid_P5',
                         'ask_P1', 'ask_P2', 'ask_P3', 'ask_P4', 'ask_P5',
                         'bid_Qa', 'bid_Q1', 'bid_Q2', 'bid_Q3', 'bid_Q4', 'bid_Q5',
                         'ask_Qa', 'ask_Q1', 'ask_Q2', 'ask_Q3', 'ask_Q4', 'ask_Q5']]
    mkt_last['pressure'] = np.log(mkt_last.bid_Qa)-np.log(mkt_last.ask_Qa)
    mkt_min1 = mkt_agg.join(mkt_last)
    return mkt_min1

def agg_to_sec1(rq):
    ## Aggregate to 1 second (volatility, trading_volume, orderbook_pressure)
    mkt_sec1_vol = rq.groupby(['date', 'time_hr', 'time_min', 'time_sec']).midprice.agg(gen_vol)
    mkt_sec1_vol = pd.DataFrame(mkt_sec1_vol)
    mkt_sec1_vol.columns = ['agg_vol']
    mkt_sec1_trd_vol = rq.groupby(['date', 'time_hr', 'time_min', 'time_sec']).trd_vol.agg(np.sum)/10000
    mkt_sec1_trd_vol = pd.DataFrame(mkt_sec1_trd_vol)
    mkt_sec1_trd_vol.columns = ['agg_trd_vol']
    mkt_agg = mkt_sec1_vol.join(mkt_sec1_trd_vol)

    ## Takes the last status
    mkt_last = rq.groupby(['date', 'time_hr', 'time_min', 'time_sec']).last()
    '''
    mkt_last = mkt_last[['dt_time', 'midprice', 'trd_prc',
                         'bid_P1', 'bid_P2', 'bid_P3', 'bid_P4', 'bid_P5',
                         'ask_P1', 'ask_P2', 'ask_P3', 'ask_P4', 'ask_P5',
                         'bid_Qa', 'bid_Q1', 'bid_Q2', 'bid_Q3', 'bid_Q4', 'bid_Q5',
                         'ask_Qa', 'ask_Q1', 'ask_Q2', 'ask_Q3', 'ask_Q4', 'ask_Q5']]
    '''
    mkt_last = mkt_last[['midprice', 'trd_prc', 'bid_P1', 'ask_P1', 'bid_Qa', 'ask_Qa',]]
    mkt_last['pressure'] = np.log(mkt_last.bid_Qa)-np.log(mkt_last.ask_Qa)
    mkt_sec1 = mkt_agg.join(mkt_last)
    return mkt_sec1

def normalize_target_cols(all_df, ):
    non_target = ['agg_vol', 'agg_trd_vol', 'pressure']
    target_df = all_df.drop(non_target, 1)
    normed = min_max_normalize(target_df)
    normed_df = all_df[non_target].join(normed)
    return normed_df

def min_max_normalize(df):
    return (df-df.min())/(df.max()-df.min())

def zero_padding(mat):
    unsorted_length = []
    for vec in mat:
        unsorted_length.append(vec.shape[0])
    max_len = max(unsorted_length)
    padded_mat = np.zeros([len(mat), max_len, vec.shape[1]])
    for idx, vec in enumerate(mat):
        padded_mat[idx,:unsorted_length[idx]] = vec
    return padded_mat, unsorted_length


def to_feed_format(list_data):
    states_x1 = np.array([x[0][0] for x in list_data])
    states_x2 = np.array([x[0][1] for x in list_data])
    padded_x2, unsorted_length = zero_padding(states_x2)
    rewards = np.array([x[1] for x in list_data])
    return states_x1, padded_x2, rewards, unsorted_length


def sort_sequence(data, len_data):
    data = torch.FloatTensor(data).cuda()
    len_data = torch.LongTensor(len_data).cuda()

    _, idx_sort = torch.sort(len_data, dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)

    idx_sort = idx_sort.cuda()
    idx_unsort = idx_unsort.cuda()
    sorted_data = data.index_select(0, idx_sort)
    sorted_len = len_data.index_select(0, idx_sort)

    return sorted_data, sorted_len, idx_unsort

def unsort_sequence(data, idx_unsort):
    unsorted_data = data.index_select(0, idx_unsort)
    return unsorted_data