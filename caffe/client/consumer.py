'''
Example client that consumes images to be processed by the Caffe framework.

python consumer.py server port

Author: Axel.Tidemann@telenor.com
'''

import sys

import redis
import matplotlib
matplotlib.use('Agg')
import caffe

sys.path.insert(0, '../web_demo')
from app import ImagenetClassifier

server = sys.argv[1]
port = sys.argv[2]

r_server = redis.StrictRedis(server, port)
pub = r_server.pubsub(ignore_subscribe_messages=True)
pub.subscribe('caffenet')
pub.subscribe('alexnet')
pub.subscribe('googlenet')

models = {}
ImagenetClassifier.default_args.update({'gpu_mode': True})
models['caffe'] = ImagenetClassifier(**ImagenetClassifier.default_args)

for model in models.itervalues():
    model.net.forward()

for message in pub.listen():
    print message
    
