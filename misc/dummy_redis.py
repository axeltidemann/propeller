# Copyright 2016 Telenor ASA, Author: Axel Tidemann

'''
Feeds data into the classify queue at the local redis instance. For debugging before
putting into the live API.

Author: Axel.Tidemann@telenor.com
'''

import redis
import time
import cPickle as pickle

red = redis.StrictRedis()

while True:
    if red.llen('classify') < 100:
        task = {'group': 'dummy', 'path': 'http://img.ekhanei.com/images/54/5467523091.jpg'}
        red.rpush('classify', pickle.dumps(task))
    else:
        time.sleep(1)
