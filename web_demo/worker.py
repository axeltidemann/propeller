'''
Client that consumes images to be processed by the Caffe framework.

Author: Axel.Tidemann@telenor.com
'''

import argparse
from collections import namedtuple
import cStringIO as StringIO
import urllib
import logging

import matplotlib
matplotlib.use('Agg')
import caffe
import redis

from app import ImagenetClassifier

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S')

ImagenetClassifier.default_args.update({'gpu_mode': True})
model = ImagenetClassifier(**ImagenetClassifier.default_args)
model.net.forward()

Task = namedtuple('Task', 'queue key')
Result = namedtuple('Result', 'OK maximally_accurate maximally_specific computation_time')

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '-s', '--server',
    help='the redis server address',
    default='localhost')
parser.add_argument(
    '-p', '--port',
    help='the redis port',
    default='6379')
parser.add_argument(
    '-q', '--queue',
    help='redis queue to read from',
    default='tasks')
args = parser.parse_args()

r_server = redis.StrictRedis(args.server, args.port)
r_server.config_set('notify-keyspace-events', 'Kh')

while True:
    task = Task(*r_server.brpop(args.queue))
    classify = r_server.hgetall(task.key)
    classify = namedtuple('Classify', classify.keys())(**classify)
    logging.info(classify)

    string_buffer = StringIO.StringIO(urllib.urlopen(classify.path).read())
    image = caffe.io.load_image(string_buffer)

    result = Result(*model.classify_image(image))
    
    r_server.hmset(task.key.replace('classify:','prediction:'), result.maximally_specific[0][0])

    # sorted sets, based on how secure you are
    r_server.hset('category:{}'.format(result.maximally_specific[0][0]), task.key.replace('classify:', ''), classify.path)
