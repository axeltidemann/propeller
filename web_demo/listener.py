'''
Listens to key changes.

Author: Axel.Tidemann@telenor.com
'''

import argparse

import redis

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
args = parser.parse_args()

r_server = redis.StrictRedis(args.server, args.port)
pubsub = r_server.pubsub(ignore_subscribe_messages=True)
pubsub.psubscribe('__keyspace*__:{}'.format(args.key))

for msg in pubsub.listen():
    key = msg['channel']
    key = key[key.find(':')+1:]
    r_server.lpush('tasks', key)
