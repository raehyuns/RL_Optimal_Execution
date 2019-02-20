#qaracs 주문 요청을 위한 python sample code
#installing redis-py -> 터미널에서 sudo pip install redis실행
#order.py에서 qaracs.py를 import해서 사용하는 예제
#redis password 관련 c:\program files\redis\redis.windows-service.conf
#https://redislabs.com/lp/python-redis/

import redis,random,datetime,json
r = redis.Redis(host="kt.qara.ai",port=7379)
#r = redis.Redis(host="localhost")

p = r.pubsub()
p.subscribe("ResponseChannel")
rand = random.Random()

def make_id():
	return str(datetime.datetime.now().timestamp())+"."+str(rand.random())
def get_response(req_id):
    while True:
        msg = p.get_message(False,5)

        if(msg is None):#timeout
            return None
        try:
            msg_json = json.loads(msg['data'])
            if(msg_json['id']==req_id):
                return msg_json
        except:
            pass
    return None

def request_order(symbol,price,qty,symbol_type="stock",order='buy'):
    req_id = make_id()
    req_message={
		'req_type':"order",
        'symbol_type':symbol_type, #stock or future
        'id':req_id,
        'symbol':symbol,
        'price':price,
        'qty':qty,
        'order':order #'buy' / 'sell'
    }
    r.publish("RequestChannel",json.dumps(req_message))
    return get_response(req_id)

def request_market_price_all(ordno,utc_timestamp):
    req_id = make_id()
    req_message={
		'req_type':"market-price-all",
        'ordno':ordno,
        'reserve_time': utc_timestamp,
        'id':req_id
    }
    r.publish("RequestChannel",json.dumps(req_message))
    return get_response(req_id)

def request_symbol_price(symbol,symbol_type='stock'):
    req_id = make_id()
    req_message={
        'id':req_id,
        'req_type':'symbol-price-info',
        'symbol':symbol,
        'symbol_type':symbol_type
    }
    r.publish("RequestChannel",json.dumps(req_message))
    return get_response(req_id)
