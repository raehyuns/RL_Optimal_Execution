import argparse

def get_args():
    argp = argparse.ArgumentParser(description='Optimal Trade Execution',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Main Control
    argp.add_argument('--data_dir', action="store", default="./data/samples")
    argp.add_argument('--save_dir', action="store", default="./save")
    argp.add_argument('--data_no_list', type=list, default=[n for n in range(58,73)])
    argp.add_argument('--save_log', action='store_true', default=False)
    argp.add_argument('--tf_board', action='store_true', default=False)
    argp.add_argument('--console_display', dest='console_display', action='store_true', default=False)
    argp.add_argument('--mode', default='train', choices=['train', 'simulation', 'debug'])
    argp.add_argument('--new_trans_hist', action="store_true", default=False)
    argp.add_argument('--n_test_day', type=int, default=1)

    # Problem Setting
    argp.add_argument('--order_type', type=str, default='bid', choices=['bid', 'ask'])
    argp.add_argument('--benchmark_type', type=str, default='day_close', choices=['day_close', 'window_close'])
    argp.add_argument('--order_amt', type=int, default=100)
    argp.add_argument('--n_amt_lv', type=int, default=20)
    argp.add_argument('--trade_price_assumption', type=str, default='midprice', choices=['midprice', 'market-order'])
    argp.add_argument('--min_lookback', type=int, default=10, help='how many minutes agent should look')
    argp.add_argument('--max_time', type=int, default=30)
    argp.add_argument('--non_trade_penalty', type=float, default=5)
    argp.add_argument('--pu', type=float, default=0.05)
    argp.add_argument('--transaction_fee', type=float, default=0.0)
    argp.add_argument('--state_pub', action="store", default="1vol, 2trd_vol, 3pressure")

    #argp.add_argument('--time_limit', type=int, default=10, help='maximum minutes for decision')

    # Agent
    argp.add_argument('--update_interval', type=int, default=30)
    argp.add_argument('--eval_interval', type=int, default=50)
    argp.add_argument('--max_episode', type=int, default=50000)
    argp.add_argument('--n_eval_episode', type=int, default=10)
    argp.add_argument('--exploration_decay', type=float, default=0.995)
    argp.add_argument('--init_exploration_rate', type=float, default=0.9)
    argp.add_argument('--min_exploration_rate', type=float, default=0.05)
    argp.add_argument('--max_memory', type=int, default=1000)
    argp.add_argument('--sample_batch_size', type=int, default=64)
    argp.add_argument('--eval_batch_size', type=int, default=100)
    ## model
    argp.add_argument('--gru_h_dim', type=int, default=20)
    argp.add_argument('--fc_layer_size', type=list, default=[32, 16, 16, 8, 8])
    argp.add_argument('--bidirectional', default=True, action="store_false")
    argp.add_argument('--dropout', type=int, default=0.1)
    ## optimizer
    argp.add_argument('--optimizer', type=str, default='Adam')
    argp.add_argument('--learning_rate', type=int, default=5e-4)
    argp.add_argument('--weight_decay', type=int, default=1e-5)
    argp.add_argument('--momentum', type=int, default=0.9)
    argp.add_argument('--grad-max-norm', type=float, default=2)

    return argp.parse_args()
