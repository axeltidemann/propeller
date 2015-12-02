'''
Client that publishes images to be processed by the Caffe framework, blocks for the result.

Author: Axel.Tidemann@telenor.com
'''

import argparse
import uuid
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
    '-k', '--key',
    help='key for the image, randomly generated if not provided',
    default=uuid.uuid4())
args = parser.parse_args()

r_server = redis.StrictRedis(args.server, args.port)
r_server.config_set('notify-keyspace-events', 'Kh')

pubsub = r_server.pubsub(ignore_subscribe_messages=True)
pubsub.psubscribe('__keyspace*__:result:{}'.format(args.key))

r_server.hset('classify:{}'.format(args.key), 'path', args.img_path)

for result in pubsub.listen():
    key = result['channel']
    key = key[key.find(':')+1:]
    description = r_server.hgetall(key)
    description = namedtuple('Description', description.keys())(**description)
    print description
    pubsub.unsubscribe()
    break
