'''
Client that publishes images to be processed by the Caffe framework, blocks for the result.

Author: Axel.Tidemann@telenor.com
'''

import argparse
from collections import namedtuple

import redis

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'img_path',
    help='the path to the image')
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
    help='user for the image',
    default='web')
args = parser.parse_args()

r_server = redis.StrictRedis(args.server, args.port)
r_server.config_set('notify-keyspace-events', 'Kh')

r_server.hmset('classify', {'user': args.user, 'path': args.img_path})

key = 'prediction:{}:{}'.format(args.user, args.img_path)
pubsub = r_server.pubsub(ignore_subscribe_messages=True)
pubsub.psubscribe('__keyspace*__:{}'.format(key))

for result in pubsub.listen():
    description = r_server.hgetall(key)
    description = namedtuple('Description', description.keys())(**description)

    print description
    pubsub.unsubscribe()
    break
