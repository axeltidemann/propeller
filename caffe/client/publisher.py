'''
Example client that publishes images to be processed by the Caffe framework.

python publisher.py server port channel message

Author: Axel.Tidemann@telenor.com
'''

import sys

import redis

server = sys.argv[1]
port = sys.argv[2]
channel = sys.argv[3]
message = sys.argv[4]

r_server = redis.StrictRedis(server, port)
r_server.publish(channel, message)
