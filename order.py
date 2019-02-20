import qaracs
from datetime import datetime,timedelta
#import time
import sched, time

'''
종목 코드
삼성전자 005930 | 삼성전자우 005935 | 하이닉스 000660 | 현대차 005380 | LG화학 051910
삼성바이오로직스 207940 | 셀트리온 068270 | POSCO 005490 | NAVER 035420 | 삼성물산 028260
선물 101P3000 | 101P6000 | 101P9000 | 101PC000
'''

def order(s_code, prc, qty=100, type='buy'):
    quote_prices = qaracs.get_quote_prices(s_code)
    for prc in quote_prices:
        res = qaracs.request_order(s_code, prc, qty, 'buy')
        if not res['success']:
            print(res)
        else:
            res = qaracs.request_market_price_all(proc_time)
#주문요청 파라미터: 종목/가격/개수/buy or sell
#res = qaracs.request_symbol_price('101P3000', 'futures')
#print(res)

'''
cur_qu_prc = res['prices']
for qt_lv in target_quote_lv:
    res = qaracs.request_order("005930", cur_qu_prc[qt_lv], 5, 'buy')
    print(res)
    proc_time = time.time() + timedelta(hours=0,minutes=30).total_seconds()
    res = qaracs.request_market_price_all(res['ordno'], proc_time)
    print(res)
'''
#시장가로 일괄수정 요청 파라미터: 예약시간 unix utc_timestamp
#proc_time = time.time() + timedelta(hours=0,minutes=30).total_seconds()
#res = qaracs.request_market_price_all(res['ordno'], proc_time)
#print(res)
'''
quote_lv_opt = ['bidho1', 'offerho1', 'bidho2', 'offerho2']
target_sym = '101P3000'
reserve_delay = 50
qt_amt = 100

starttime = time.time()
delay = 60.0
while True:
    start_dt = datetime.fromtimestamp(time.time()).replace(hour=9, minute=30)
    end_dt = datetime.fromtimestamp(time.time()).replace(hour=15, minute=15)
    # 호가 요청
    # res = qaracs.request_order("005935", 37600, 5, 'buy')
    now_dt = datetime.fromtimestamp(time.time())
    if (now_dt>=start_dt) and (now_dt<end_dt):
        prc_res = qaracs.request_symbol_price(target_sym, 'futures')
        cur_qu_prc = prc_res['prices']
        for ord_type in ['buy', 'sell']:
            for qt_lv in quote_lv_opt:
                ord_res = qaracs.request_order(target_sym, cur_qu_prc[qt_lv],
                                                    qt_amt, 'future', ord_type)
                print(ord_res)
                proc_time = time.time() + timedelta(hours=0,minutes=reserve_delay).total_seconds()
                resv_res = qaracs.request_market_price_all(ord_res['ordno'], proc_time)
                print(resv_res)

    time.sleep(delay - ((time.time() - starttime) % delay))
'''
