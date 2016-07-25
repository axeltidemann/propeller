# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist

import os.path
import logging
import os
import glob
from random import shuffle
import time
import argparse

import tensorflow.python.platform
from six.moves import urllib
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.platform import gfile

from utils import load_graph, maybe_download_and_extract

print('TensorFlow version {}'.format(tf.__version__))

logging.getLogger().setLevel(logging.INFO)
    
def save_states(source, target, limit, mem_ratio, model_dir):
    load_graph(os.path.join(model_dir, 'classify_image_graph_def.pb'))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_ratio)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        next_last_layer = sess.graph.get_tensor_by_name('pool_3:0')

        images = glob.glob('{}/*.jpg'.format(source))
        shuffle(images)
        images = images[:limit]

        states = []

        t0 = time.time()

        for jpg in list(images):
            try:
                image_data = gfile.FastGFile(jpg).read()
                hidden_layer = sess.run(next_last_layer,
                                        {'DecodeJpeg/contents:0': image_data})
                hidden_layer = np.squeeze(hidden_layer)
                states.append(hidden_layer)
            except Exception as e:
                images.remove(jpg)
                print 'Something went wrong when processing {}: \n {}'.format(jpg, e) 

        print('Time spent collecting states: {}'.format(time.time() - t0))

        df = pd.DataFrame(data={'state': states}, index=images)
        df.index.name='filename'

        h5name = os.path.join(target, '{}.h5'.format(os.path.basename(source)))
        with pd.HDFStore(h5name, 'w') as store:
            store['data'] = df
          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Collects the next to last layer states from the Inception model.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'source',
        help='Folder with images.')
    parser.add_argument(
        'target',
        help='Where to put the states file, will have same name as images folder.')
    parser.add_argument(
        '--model_dir',
        help='Path to Inception files', 
        default='/tmp/imagenet')
    parser.add_argument(
        '--limit',
        help='Maximum amount of images to process',
        type=int,
        default=10000)
    parser.add_argument(
        '--mem_ratio',
        help='Ratio of memory to reserve on the GPU instance',
        type=float,
        default=.95)
    args = parser.parse_args()

    maybe_download_and_extract(args.model_dir)
    save_states(args.source, args.target, args.limit, args.mem_ratio, args.model_dir)
