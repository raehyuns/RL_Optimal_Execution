import os
import datetime
import datetime as DT
import time
import random
import pdb
import numpy as np
import pandas as pd
import torch
import tensorflow as tf
import pymongo
from pymongo import MongoClient

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

def preprocess(rq):
    rq['dt_time'] = [
        datetime.time(int(str(x)[:-7]), int(str(x)[-7:-5]), int(str(x)[-5:-3]), int(str(x)[-3:]) * 1000) for x in
        rq['time']]
    rq['time_hr'] = [x.hour for x in rq['dt_time']]
    rq['time_min'] = [x.minute for x in rq['dt_time']]
    rq['time_sec'] = [x.second for x in rq['dt_time']]

    rq.rename(columns={
        'price': 'trd_prc', 'volume': 'trd_vol',
        'ASK_TOT_ORD_RQTY': 'ask_Qa', 'BID_TOT_ORD_RQTY': 'bid_Qa',
        'ASK_STEP1_BSTORD_PRC': 'ask_P1', 'ASK_STEP1_BSTORD_RQTY': 'ask_Q1',
        'BID_STEP1_BSTORD_PRC': 'bid_P1', 'BID_STEP1_BSTORD_RQTY': 'bid_Q1',
        'ASK_STEP2_BSTORD_PRC': 'ask_P2', 'ASK_STEP2_BSTORD_RQTY': 'ask_Q2',
        'BID_STEP2_BSTORD_PRC': 'bid_P2', 'BID_STEP2_BSTORD_RQTY': 'bid_Q2',
        'ASK_STEP3_BSTORD_PRC': 'ask_P3', 'ASK_STEP3_BSTORD_RQTY': 'ask_Q3',
        'BID_STEP3_BSTORD_PRC': 'bid_P3', 'BID_STEP3_BSTORD_RQTY': 'bid_Q3',
        'ASK_STEP4_BSTORD_PRC': 'ask_P4', 'ASK_STEP4_BSTORD_RQTY': 'ask_Q4',
        'BID_STEP4_BSTORD_PRC': 'bid_P4', 'BID_STEP4_BSTORD_RQTY': 'bid_Q4',
        'ASK_STEP5_BSTORD_PRC': 'ask_P5', 'ASK_STEP5_BSTORD_RQTY': 'ask_Q5',
        'BID_STEP5_BSTORD_PRC': 'bid_P5', 'BID_STEP5_BSTORD_RQTY': 'bid_Q5'}, inplace=True)
    return rq

def load_and_format(data_dir, sec_type, data_name, trade_assumption, file=None):
    '''
    load raw order book data & change column names
    mkt_data : real market transaction record
    sim_data : order --> transaction feedback from simulator
    '''
    symbol = data_name.split(sep='_')[0]
    mkt_raw_file = os.path.join(data_dir,'mkt_data', sec_type, symbol, data_name)
    rq = pd.read_csv(mkt_raw_file) # raw_tick
    print(mkt_raw_file, "is loaded")

    if trade_assumption == 'market_simulator':
        sim_raw_file = os.path.join(data_dir,'sim_data', sec_type, symbol, data_name)
        sim = pd.read_csv(sim_raw_file)
        sim['OrderTime'] = pd.to_datetime(sim['OrderTime'])
        sim['ExecTime'] = pd.to_datetime(sim['ExecTime'])
    else:
        sim = None

    # use all the data for training before final 10 minutes
    # final 10 minutes data is used for reward cal.
    #tr_rq = rq[(rq.time>90001000) & (rq.time < 151500000)]
    tr_rq = preprocess(rq)
    cls_prc = tr_rq.iloc[-1].trd_prc

    return tr_rq, sim, cls_prc

def load_data(sec_type, data_dir, target_name, n_test_day, trade_assumption='midprice', mode='debug', train_all=False):
    #  train with multiple securities & test with one target
    name2sym = {'삼성전자':'005930','삼성전자우':'005935','하이닉스':'000660',
                '현대차':'005380','LG화학':'051910','삼성바이오로직스':'207940',
                '셀트리온':'068270', 'POSCO':'005490', 'NAVER':'035420',
                '삼성물산':'028260','3월물':'101P3000','6월물':'101P6000',
                '9월물':'101P9000'}
    #dir_path = os.path.join(data_dir, sec_type)
    target_symbol = name2sym[target_name]
    if trade_assumption == 'midprice':
        if train_all:
            all_symbols = os.listdir(os.path.join(data_dir, 'mkt_data', sec_type))
            all_data_names = []
            for symb in all_symbols:
                all_data_names+=os.listdir(os.path.join(data_dir, 'mkt_data', sec_type, symb))
        # train and test with one target security
        else:
            all_data_names = os.listdir(os.path.join(data_dir, 'mkt_data', sec_type, target_symbol))
    else: # trade with market simulation
        if train_all:
            all_symbols = os.listdir(os.path.join(data_dir, 'sim_data', sec_type))
            all_data_names = []
            for symb in all_symbols:
                all_data_names+=os.listdir(os.path.join(data_dir, 'sim_data', sec_type, symb))
        # train and test with one target security
        else:
            all_data_names = os.listdir(os.path.join(data_dir, 'sim_data', sec_type, target_symbol))

    # randomly split train, test data and return preprocessed data list
    if mode=='debug':
        test_set_names = random.sample(all_data_names, 2)
        train_set_names = random.sample(list(set(all_data_names)-set(test_set_names)), 2)
    else:
        test_set_names = random.sample(all_data_names, n_test_day)
        train_set_names = list(set(all_data_names) - set(test_set_names))
        print('selected test set : {}'.format(test_set_names))

    train_data_list, test_data_list = [], []
    for data_name in train_set_names:
        if data_name[-3:] != 'csv':
            pass
        rawq_df, simtr_df, cls_prc = load_and_format(data_dir, sec_type, data_name, trade_assumption)
        min1_agg = agg_to_min1(rawq_df)
        train_data_list.append((min1_agg, rawq_df, simtr_df, cls_prc))
    for data_name in test_set_names:
        if data_name[-3:] != 'csv':
            pass
        rawq_df, simtr_df, cls_prc = load_and_format(data_dir, sec_type, data_name, trade_assumption)
        min1_agg = agg_to_min1(rawq_df)
        test_data_list.append((min1_agg, rawq_df, simtr_df, cls_prc))
    return train_data_list, test_data_list

def select_data(data_list):
    return random.choice(data_list)

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
    ## Takes the snapshot of last status
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

def select_and_normalize(all_df, ):
    #raw_cols = ['agg_vol', 'agg_trd_vol', 'pressure']
    raw_cols = ['agg_trd_vol', 'pressure']
    mm_cols = ['trd_prc']
    last_df = all_df.iloc[-1]
    snapshot_prc = ['bid_P1', 'bid_P2', 'bid_P3', 'bid_P4', 'bid_P5',
                'ask_P1', 'ask_P2', 'ask_P3', 'ask_P4', 'ask_P5']
    snapshot_bidq = ['bid_Q1', 'bid_Q2', 'bid_Q3', 'bid_Q4', 'bid_Q5']
    snapshot_askq = ['ask_Q1', 'ask_Q2', 'ask_Q3', 'ask_Q4', 'ask_Q5']

    mm_normed = min_max_normalize(all_df[mm_cols])
    prc_normed = (last_df[snapshot_prc] - last_df.midprice)/last_df.midprice * 100
    top5_q = last_df[snapshot_bidq].sum() + last_df[snapshot_askq].sum()
    bidq_normed = last_df[snapshot_bidq] / top5_q
    askq_normed = last_df[snapshot_askq] / top5_q

    normed_df = np.concatenate([all_df[raw_cols].values.flatten(), mm_normed.values.squeeze(),
                    prc_normed.values, bidq_normed.values, askq_normed.values])
    return normed_df

def min_max_normalize(df):
    return (df-df.min())/(df.max()-df.min())

def get_data_from_db(symbol, ):
    qara_fin = MongoClient('db.qara.kr')['fin']

    today_early = DT.datetime.now().replace(hour=8)
    t1 = time.time()
    pd.DataFrame(list(qara_fin.trading_lvl2.find({'symbol':symbol,
            "timestamp": {"$gt":time.mktime(today_early.timetuple())}},{'_id':1, 'ASK_STEP1_BSTORD_PRC':1, 'ASK_STEP2_BSTORD_RQTY':1})))
    #trans_df = pd.DataFrame(list(qara_fin.trading_lvl2.find({'symbol':symbol,
    #                        "timestamp": {"$gt":time.mktime(today_early.timetuple())}}),{'_id':1}))
    t2 = time.time()
    print(t2-t1)
    pdb.set_trace()
    trans_df = trans_df.sort_values(by=['local_timestamp'])
    trans_df = trans_df.reset_index(drop=True)
    dt_list = []
    t_list = []
    for idx, row in trans_df.iterrows():
        dt_obj = DT.datetime.fromtimestamp(row.local_timestamp)
        dt = int(dt_obj.strftime('%Y%m%d'))
        t = int(dt_obj.strftime('%H%M%S%f')[:-3])
        dt_list.append(dt)
        t_list.append(t)
    trans_df['date'] = dt_list
    trans_df['time'] = t_list
    trans_df['midprice'] = (trans_df.BID_STEP1_BSTORD_PRC + trans_df.ASK_STEP1_BSTORD_PRC) / 2
    raw_df = preprocess(trans_df)
    min1_agg = agg_to_min1(raw_df)
    return min1_agg, raw_df

def get_live_orders():
    order_db = MongoClient('db.qara.kr').order_request

    #cur_ts = time.mktime(DT.datetime.now().replace(second=0))
    cur_ts = time.mktime(DT.datetime.now().replace(hour=14,minute=13,second=0).timetuple())
    order_df = pd.DataFrame(list(order_db.order_request.find({'rem_amt':{'$gt':0}, 'order_start_ts':{'$lt':cur_ts}})))

    final_order = order_df.loc[order_df.order_end_ts<=cur_ts]
    valid_order = order_df.loc[order_df.order_end_ts>cur_ts]
    return final_order, valid_order


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

def randf(s, e):
    return (float(random.randrange(0, (e - s) * 9999)) / 10000) + s

def epsilon_greedy(greedy, epsilon):
    batch = greedy.shape[0]
    greedy = np.expand_dims(greedy, 0)
    random = np.expand_dims(np.random.randint(3, size=batch), 0)
    candidate = np.concatenate([random, greedy])
    prob = np.random.uniform(size=batch)
    return np.choose((prob>epsilon).astype(int), candidate)

#### tensor board functions

def summary(writer, loss, reward, step, mode, variables=None):
    if mode ==  'train':
        scalar_summary(writer, 'Train RMSE loss', loss, step)
        scalar_summary(writer, 'Train Reward', reward, step)
    else:
        scalar_summary(writer, 'Test RMSE loss', loss, step)
        scalar_summary(writer, 'Test Reward', reward, step)
    if variables:
        for i,v in enumerate(variables):
            variable_summary(*v, step, writer)

def variable_summary(name, var, step, writer):
    mean = torch.mean(var)
    add_scalar(name+'/mean', mean, step, writer)

    stddev = torch.sqrt(torch.mean(torch.pow((var - mean),2)))
    scalar_summary(name+'/stddev', stddev, step, writer)
    scalar_summary(name+'/max', torch.max(var), step, writer)
    scalar_summary(name+'/min', torch.min(var), step, writer)
    add_histogram(name+'/histogram', var,step,bins='doane')

def scalar_summary(writer, tag, value, step):
    """Log a scalar variable."""
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
    writer.add_summary(summary, step)

def histogram_summary(writer, name, values, step):
    hist = tf.summary.histogram(name, values)
    '''
    hist = tf.HistogramProto()
    hist.min = float(np.min(values))
    hist.max = float(np.max(values))
    hist.num = int(np.prod(values.shape))
    hist.sum = float(np.sum(values))
    hist.sum_squares = float(np.sum(values**2))
    '''
    summary = tf.Summary()
    summary.value.add(tag=name, histo=hist)
    writer.add_summary(summary, step)
    #summary = tf.Summary(value=[tf.Summary.Value(tag=name, histo=hist)])

def act_dist_summary(writer, act_dist, n_split, step, type):
    area = len(act_dist)//n_split
    for p in range(n_split):
        v = act_dist[p*area:(p+1)*area].sum()
        writer.add_scalar('{} Act Value Area {}'.format(type, p+1), v, step)
