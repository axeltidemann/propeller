'''
Client that consumes images to be processed by the Caffe framework.

Author: Axel.Tidemann@telenor.com
'''

import argparse
from collections import namedtuple

import redis

Task = namedtuple('Task', 'queue key')

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
    description = r_server.hgetall(task.key)
    description = namedtuple('Description', description.keys())(**description)
    print description
    r_server.hmset(task.key.replace('classify:','result:'),
                   {'category': 'nonsense',
                    'probability': 0.5})
    
#import matplotlib
#matplotlib.use('Agg')
#import caffe

# sys.path.insert(0, '../web_demo')
# from app import ImagenetClassifier



    # pub = r_server.pubsub(ignore_subscribe_messages=True)
# pub.subscribe('classify_image')

# # models = {}
# # ImagenetClassifier.default_args.update({'gpu_mode': True})
# # models['caffe'] = ImagenetClassifier(**ImagenetClassifier.default_args)

# # for model in models.itervalues():
# #     model.net.forward()


# for message in pub.listen():
#     print message
    
