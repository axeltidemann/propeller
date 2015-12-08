'''
Listens to key changes.

Author: Axel.Tidemann@telenor.com
'''

import argparse
import logging
import cPickle as pickle

import redis

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%Y-%m-%d %I:%M:%S')

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
    '-k', '--key',
    help='key to listen to',
    default='classify:*')
parser.add_argument(
    '-q', '--queue',
    help='redis queue to post to',
    default='tasks')

args = parser.parse_args()

r_server = redis.StrictRedis(args.server, args.port)
pubsub = r_server.pubsub(ignore_subscribe_messages=True)
pubsub.psubscribe('__keyspace*__:{}'.format(args.key))

import sys
print 'currently not needed - will be in restful API'
sys.exit(0)

for msg in pubsub.listen():
    print msg
    # request = r_server.hgetall(args.key) 
    # r_server.lpush(args.queue, pickle.dumps(request))
    # logging.info(request)
