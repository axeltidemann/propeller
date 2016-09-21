# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist

import os
import random
import time
import argparse
import multiprocessing as mp
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'../misc')))

import tensorflow.python.platform
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.python.platform import gfile

from utils import load_graph, maybe_download_and_extract, chunks

print('TensorFlow version {}'.format(tf.__version__))

KILL = 'POISON PILL'

def save_states(q, gpu, target, limit, mem_ratio, model_dir, seed=0, chunksize=1000):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print 'GPU {}'.format(gpu)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_ratio)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        
        load_graph(os.path.join(model_dir, 'classify_image_graph_def.pb'))
        next_last_layer = sess.graph.get_tensor_by_name('pool_3:0')
        
        while True:
            source = q.get()
            if source == KILL:
                break

            images = glob.glob('{}/*'.format(source))
            random.seed(seed)
            random.shuffle(images)
            if limit > 0:
                images = images[:limit]

            t0 = time.time()
            h5name = os.path.join(target, '{}.h5'.format(os.path.basename(os.path.normpath(source))))

            with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as store:
                for chunk in chunks(images, chunksize):

                    states = []
                    for jpg in list(chunk): # Creates a copy over which it is safe to iterate
                        try:
                            raw_data = gfile.FastGFile(jpg).read()
                            hidden_layer = sess.run(next_last_layer,
                                                    {'DecodeJpeg/contents:0': raw_data})
                            hidden_layer = np.squeeze(hidden_layer)
                            states.append(hidden_layer)

                        except Exception as e:
                            chunk.remove(jpg)
                            print 'Something went wrong when processing {}'.format(jpg)

                    X = np.vstack(states)
                    columns = [ 'f{}'.format(i) for i in range(X.shape[1]) ]
                    
                    df = pd.DataFrame(data=X, index=chunk, columns=columns)
                    df.index.name='filename'
                    store.append('data', df)

            print('Time spent collecting {} states: {}'.format(len(images), time.time() - t0))
          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Collects the next to last layer states from the Inception model.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'source',
        help='Folder(s) with images.',
        nargs='+')
    parser.add_argument(
        'target',
        help='Where to put the states file, will have same name as images folder.')
    parser.add_argument(
        '--model_dir',
        help='Path to Inception files', 
        default='/tmp/imagenet')
    parser.add_argument(
        '--limit',
        help='Maximum amount of images to process. 0 means no limit.',
        type=int,
        default=10000)
    parser.add_argument(
        '--gpus',
        help='How many GPUs to use',
        default=4,
        type=int)
    parser.add_argument(
        '--threads',
        help='How many threads to use pr GPU',
        default=2,
        type=int)

    args = parser.parse_args()

    maybe_download_and_extract(args.model_dir)

    q = mp.Queue()

    processes = 0
    for gpu in os.environ['CUDA_VISIBLE_DEVICES'].split(","):
        for _ in range(args.threads):
            if processes == len(args.source):
                break
            mp.Process(target=save_states, args=(q, gpu, args.target, args.limit,
                                                 .9/args.threads, args.model_dir)).start()
            processes += 1
                
    for folder in args.source:
        q.put(folder)

    for _ in range(args.gpus*args.threads):
        q.put(KILL)
