# Copyright 2016 Telenor ASA, Author: Axel Tidemann
# The software includes elements of example code. Copyright 2015 Google, Inc. Licensed under Apache License, Version 2.0.
# https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist

from __future__ import division
import os
import glob
from random import shuffle
import random
import time
import argparse
from functools import partial
import multiprocessing as mp
import shutil

import tensorflow.python.platform
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import pandas as pd
    
from utils import load_graph, maybe_download_and_extract

print('TensorFlow version {}'.format(tf.__version__))

# Inception default input dimensions
WIDTH=299
HEIGHT=299

KILL = 'POISON PILL'

def random_crop(image, min_scale=.5, max_scale=.95):
    shape = tf.to_float(tf.shape(image))
    factor = tf.constant(random.uniform(min_scale, max_scale))
    return tf.random_crop(image, [ tf.to_int32(shape[0]*factor),
                                   tf.to_int32(shape[1]*factor), 3 ])
    
# See the file image_adjustment_paramaters.py to see the effects of these parameters
brightness = partial(tf.image.random_brightness, max_delta=.3)
contrast = partial(tf.image.random_contrast, lower=.2, upper=2)
hue = partial(tf.image.random_hue, max_delta=.5)
saturation = partial(tf.image.random_saturation, lower=0, upper=5)

def transformations(image):
    flip = tf.image.random_flip_left_right(image)
    crop = random_crop(flip)
    resize = tf.image.resize_images(crop, WIDTH, HEIGHT)
    return tf.cast(resize, tf.uint8)
    
adjustments = [ brightness, contrast, hue, saturation ]

def augment_images(q, gpu, target, limit, mem_ratio, model_dir):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    print 'GPU {}'.format(gpu)
    
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=mem_ratio)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

        x = tf.placeholder('string')

        original = tf.image.decode_jpeg(x, channels=3)
        transformed = transformations(original)
        adjusted = random.choice(adjustments)(transformed)
        encoded = tf.image.encode_jpeg(adjusted)

        load_graph(os.path.join(model_dir, 'classify_image_graph_def.pb'))

        next_last_layer = sess.graph.get_tensor_by_name('pool_3:0')

        while True:
            h5 = q.get()
            if h5 == KILL:
                break

            data = pd.read_hdf(h5, 'data')
            images = data.index

            augmentation_repeat = max(limit/len(images) - 1, 0)
            rest = augmentation_repeat - int(augmentation_repeat)

            if augmentation_repeat:
                print('''{}: {} images, each image will be augmented {} times to achieve goal of {}'''
                      ''' images in total'''.format(h5, len(images), augmentation_repeat, limit))
            else:
                print('{}: {} images present. No augmentation to be done, copying over file.'.format(h5, len(images)))
                shutil.copy(h5, target)
                continue

            t0 = time.time()
            states = []

            for jpg in images:
                try:
                    raw_data = gfile.FastGFile(jpg).read()
                    iterations = int(augmentation_repeat+1) if random.random() < rest else int(augmentation_repeat)

                    for i in range(iterations):
                        augmented = sess.run(encoded, feed_dict={x: raw_data})
                        hidden_layer = sess.run(next_last_layer,
                                                {'DecodeJpeg/contents:0': augmented})
                        hidden_layer = np.squeeze(hidden_layer)
                        states.append(hidden_layer)

                except Exception as e:
                    print 'Something went wrong when augmenting {}: \n\t{}'.format(jpg, e) 

            print 'Time spent augmenting images in {}: {}'.format(h5, time.time() - t0)

            X = np.vstack(states)
            columns = [ 'f{}'.format(i) for i in range(X.shape[1]) ]

            df = pd.DataFrame(data=X, columns=columns)
            
            h5name = os.path.join(target, os.path.basename(h5))
            with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as store:
                store.append('data', df)

          
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Augments images by random transformations and image adjustments. Scales to Inception size.
    Uses multiple GPUs if available and if necessary.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'source',
        help='''HDF5 file(s) for categories. The index of the category HDF5 file will be used. This is to ensure the same
        files (in the same order) are being used for training and augmentation.''',
        nargs='+')
    parser.add_argument(
        'target',
        help='Where to put the HDF5 file, will have same name as the original HDF5 file.')
    parser.add_argument(
        '--limit',
        help='The sum of the images in the original folder and the augmentations.',
        type=int,
        default=8000)
    parser.add_argument(
        '--gpus',
        help='Which GPUs to use',
        default='0,1,2,3')
    parser.add_argument(
        '--threads',
        help='How many threads to use pr GPU',
        default=1,
        type=int)
    parser.add_argument(
        '--model_dir',
        help='Path to Inception files', 
        default='/tmp/imagenet')
    args = parser.parse_args()

    maybe_download_and_extract(args.model_dir)

    q = mp.Queue()
    
    processes = 0
    gpus = args.gpus.split(',')
    for gpu in gpus:
        for _ in range(args.threads):
            if processes == len(args.source):
                break
            mp.Process(target=augment_images, args=(q, gpu, args.target, args.limit,
                                                    .9/args.threads, args.model_dir)).start()
            processes += 1
                
    for h5 in args.source:
        q.put(h5)

    for _ in range(len(gpus)*args.threads):
        q.put(KILL)
