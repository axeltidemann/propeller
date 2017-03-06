# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://www.tensorflow.org/versions/r0.7/tutorials/image_recognition/index.html

from __future__ import print_function
import re
import sys
import tarfile
import os
import time
import json
from binascii import a2b_base64
import argparse
import multiprocessing as mp

from six.moves import urllib
import tensorflow as tf
from tensorflow.python.platform import gfile
import numpy as np
import redis
import boto3

DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

def classify_images(cuda_device): 
    sqs = boto3.resource('sqs')
    queue = sqs.get_queue_by_name(QueueName=args.sqs_queue)
    r_server = redis.StrictRedis(args.redis_server, args.redis_port)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
    print('Using CUDA device {}'.format(cuda_device))

    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = args.memory_scale_relative_to_threads/args.threads

    with tf.Session(config=config) as sess: 
        load_graph(os.path.join(args.model_dir, 'classify_image_graph_def.pb'))
        inception_next_last_layer = sess.graph.get_tensor_by_name('pool_3:0')

        load_graph(args.classifier)
        transfer_predictor = sess.graph.get_tensor_by_name('output:0')

        while True:
            for message in queue.receive_messages():
                try:
                    blob = json.loads(message.body)
                    _, data = blob['image']['data_uri'].split(',')
                    image_data = a2b_base64(data)

                    starttime = time.time()
                    hidden_layer = sess.run(inception_next_last_layer,
                                            {'DecodeJpeg/contents:0': image_data})

                    hidden_layer_squeezed = np.squeeze(hidden_layer)
                    predictions = sess.run(transfer_predictor, {'input:0': np.atleast_2d(hidden_layer_squeezed) })
                    predictions = np.squeeze(predictions)
                    top_k = predictions.argsort()[-args.num_top_predictions:][::-1]
                    endtime = time.time()

                    result = { 'status': 'done',
                               'classification':
                               [ { 'category': mapping[str(node_id)], 'probability': float(predictions[node_id]) }
                                 for node_id in top_k ],
                               'computation_time': int(1000*(endtime-starttime)) }

                    if args.return_visual_features:
                        result['visual_features'] = hidden_layer_squeezed.tolist()

                except Exception as e:
                    result = { 'status': 'error',
                               'message': str(e) }
                

                result_json = json.dumps(result)
                r_server.lpush('{}{}'.format(args.redis_prefix, message.message_id), result_json)
                message.delete()
                
                if args.return_visual_features:
                    del result['visual_features'] # To avoid cluttering the output 
                print(result)

    
def load_graph(path):
    """"Creates a graph from saved GraphDef file and returns a saver."""
    with gfile.FastGFile(path, 'r') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')


def maybe_download_and_extract(model_dir):
    """Download and extract model tar file."""
    dest_directory = model_dir
    if not os.path.exists(dest_directory):
        os.makedirs(dest_directory)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_directory, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath,
                                                 reporthook=_progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Listens to a redis list, downloads
    the image and feeds it to the Inception model. Uses the next-to-last layer output as input
    to a classifier.''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'classifier',
        help='Path to the transfer learned model')
    parser.add_argument(
        'mapping',
        help='Path to a mapping from node IDs to readable text')
    parser.add_argument(
        '--model_dir',
        help='Path to Inception model, will be downloaded if not present.',
        default='/tmp/imagenet')
    parser.add_argument(
        '--num_top_predictions',
        help='Display this many predictions.',
        type=int,
        default=5)
    parser.add_argument(
        '--redis_server',
        default='localhost')
    parser.add_argument(
        '--redis_port',
        type=int,
        default=6379)
    parser.add_argument(
        '--redis_prefix',
        help='Prefix name of redis list to write results in, the message UUID will be appended to this string',
        default='imgrec_')
    parser.add_argument(
        '--sqs_queue',
        help='SQS queue to read images from',
        default='classify')
    parser.add_argument(
        '--gpus',
        help='How many GPUs to use',
        default=1,
        type=int)
    parser.add_argument(
        '--threads',
        help='How many threads to use pr GPU',
        default=3,
        type=int)
    parser.add_argument(
        '--memory_scale_relative_to_threads',
        help='How much relative memory to scale for each thread',
        default=.9,
        type=float)
    parser.add_argument(
        '--return_visual_features',
        help='Whether to return visual features, a 2048 floating point vector',
        action='store_true')

    args = parser.parse_args()

    print('TensorFlow version {}. Starting {} threads on {} GPU cores, total of {} workers.'.format(tf.__version__,
                                                                                                    args.threads,
                                                                                                    args.gpus,
                                                                                                    args.gpus*args.threads))

    maybe_download_and_extract(args.model_dir)
    with open(args.mapping) as f:
        mapping = json.load(f)

    for gpu in range(args.gpus):
        for _ in range(args.threads):
            mp.Process(target=classify_images, args=(gpu,)).start()
    
