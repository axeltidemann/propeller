# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import cStringIO as StringIO
import json
import base64

import flask
import redis
import requests
import boto3
from wand.image import Image

parser = argparse.ArgumentParser(description='''RESTful API service for image recognition. Note: this is a
minimal light-weight server that must not be used in production.''',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--port',
    help='Which port to serve content on',
    type=int, default=8080)
parser.add_argument(
    '--redis_server',
    help='Redis server address',
    default='localhost')
parser.add_argument(
    '--redis_port',
     help='Redis port',
    default='6379')
parser.add_argument(
    '--redis_prefix',
    help='Prefix name of redis list to write results in, the message UUID will be appended to this string',
    default='imgrec_')
parser.add_argument(
    '--queue',
    help='SQS queue to post image classification tasks to',
    default='classify')
parser.add_argument(
    '--timeout',
    help='How long to wait before failing to download in image',
    type=int,
    default=10)
args = parser.parse_args()

app = flask.Flask(__name__)

def resize_encode(data):
    with Image(file=StringIO.StringIO(data)) as img:
        img.resize(299,299) # Tensorflow defaults. Speed gains here.
        encoded = base64.b64encode(img.make_blob())

    return encoded

@app.route('/classify/<path:url>')
def classify(url):
    try:
        response = requests.get(url, timeout=args.timeout)
        jpeg_base64 = resize_encode(response.content)
        dispatch = queue.send_message(MessageBody=json.dumps({'image': {'data_uri': 'data:image/jpg;base64,{}'.format(jpeg_base64)}}))

        listname = '{}{}'.format(args.redis_prefix, dispatch['MessageId'])
        _, result = redis.blpop(listname)
        redis.delete(listname)
    except Exception as e:
        result = { 'status': 'error',
                   'message': str(e) }
    return result

redis = redis.StrictRedis(args.redis_server, args.redis_port)
sqs = boto3.resource('sqs')

try:
    queue = sqs.get_queue_by_name(QueueName=args.queue)
except:
    print 'The queue "{}" does not exist. It will be created.'.format(args.queue)
    queue = sqs.create_queue(QueueName=args.queue)

app.run(debug=True, host='0.0.0.0', port=args.port)
