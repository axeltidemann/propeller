'''
Client that publishes a list of images to be processed by the Caffe framework.

Author: Axel.Tidemann@telenor.com
'''

import argparse
from collections import namedtuple
import cPickle as pickle

import redis

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'img_list',
    help='the path to the image list file')
parser.add_argument(
    '-s', '--server',
    help='the redis server address',
    default='localhost')
parser.add_argument(
    '-p', '--port',
    help='the redis port',
    default='6379')
parser.add_argument(
    '-u', '--user',
    help='user for the images',
    default='web')
parser.add_argument(
    '-q', '--queue', 
    help='which task queue to post to',
    default='classify')
parser.add_argument(
    '-f', '--flush',
    help='number of images to flush to the redis server',
    default=10000)
args = parser.parse_args()

r_server = redis.StrictRedis(args.server, args.port)
pipe = r_server.pipeline()

with open(args.img_list) as f:
    i = 0
    for line in f:
        pipe.lpush(args.queue, pickle.dumps({'user': args.user, 'path': line.rstrip()}))
        i += 1
        if i % args.flush == 0:
            pipe.execute()
    pipe.execute()
