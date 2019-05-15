# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import json
from collections import Counter
import multiprocessing as mp

import numpy as np
import regex
import pandas as pd
import h5py
import tensorflow as tf
from keras.applications import Xception
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from scipy.misc import imresize

parser = argparse.ArgumentParser(description='''
    Reads HDF5 files with ad ids, titles, descriptions, price and image paths. Finds
    all unique graphemes and embeddings, stores them in the same file.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'data',
    help='HDF5 file with ads')
parser.add_argument(
    '--gpus',
    help='Number of GPUs to use',
    type=int,
    default=8)
parser.add_argument(
    '--cpus',
    help='Number of CPUs to use for image cropping and scaling',
    type=int,
    default=mp.cpu_count())
parser.add_argument(
    '--threads',
    help='How many threads to use pr GPU for image embedding',
    default=2,
    type=int)
args = parser.parse_args()

# Graphemes

def count(key):

    print('Opening', args.data, key)

    data = pd.read_hdf(args.data, key=key)

    raw_text = ''.join([ str(t).lower() for t in data.title + data.description ])
    graphemes = regex.findall(r'\X', raw_text, regex.U)

    return Counter(graphemes)

with h5py.File(args.data, 'r+', libver='latest') as h5_file:

    for content in ['graphemes', 'images']:
        if content in h5_file:
            del h5_file[content]
            print(content, 'already existed, deleting.')
    
    categories = [ 'categories/{}'.format(c) for c in list(h5_file['categories'].keys()) ]

pool = mp.Pool()
results = pool.map(count, categories)

counter = Counter()

for cntr in results:
    counter.update(cntr)

graphemes = pd.DataFrame(data=counter.most_common(), columns=['grapheme', 'n'])

graphemes.to_hdf(args.data, key='graphemes', mode='r+')

print('Graphemes found.')

# Image embeddings

STOP = 'stop'

def square(img):
    h = img.shape[0]
    w = img.shape[1]
    new_size = min(w,h)
    w_diff = (w - new_size)//2
    h_diff = (h - new_size)//2
    return img[ h_diff:new_size+h_diff, w_diff:new_size+w_diff ]

def crop_scale_image(task_q, embed_q, result_q):
    for ad_id, path in iter(task_q.get, STOP):
        squares = []
        if not pd.isnull(path) and os.path.exists(path):
            for _file in sorted(os.listdir(path)):
                try:
                    img = img_to_array(load_img(os.path.join(path,_file)))
                    img = square(img)
                    img = imresize(img, size=(299,299), interp='bicubic')
                    img = preprocess_input(img)
                    squares.append(img)
                except Exception as e:
                    print(ad_id, path, e)

        if len(squares):
            squares = np.stack(squares)
            embed_q.put((ad_id, squares))
        else:
            result_q.put((ad_id, np.zeros(0)))
    

def embed_image(embed_q, result_q, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    print('PID {} GPU {} starting'.format(os.getpid(), gpu))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as session:
        net = Xception(include_top=False, weights='imagenet', pooling='max')
        
        for ad_id, squares in iter(embed_q.get, STOP):
            embeddings = net.predict(squares)
            result_q.put((ad_id, embeddings))
                    
    print('PID {} GPU {} exiting'.format(os.getpid(), gpu))

task_q = mp.Queue()
embed_q = mp.Queue()
result_q = mp.Queue()

for _ in range(args.cpus):
    mp.Process(target=crop_scale_image, args=(task_q, embed_q, result_q)).start()

print(args.cpus, 'CPU processes launched for cropping and scaling.')

for gpu in range(args.gpus):
    for _ in range(args.threads):
        mp.Process(target=embed_image, args=(embed_q, result_q, gpu)).start()

print(args.gpus*args.threads, 'GPU processes launched to get image embeddings')

with h5py.File(args.data, 'r+', libver='latest') as h5_file:
    for c in categories:
        data = pd.read_hdf(args.data, key=c, columns=['images'])

        for ad_id, path in zip(data.index, data.images):
            task_q.put((ad_id, path))

        for _ in data.images:
            ad_id, embeddings = result_q.get()
            print(ad_id, embeddings.shape)
            h5_file.create_dataset('images/{}/{}'.format(c, ad_id), data=embeddings)
            
for _ in range(args.gpus*args.threads):
    embed_q.put(STOP)
    
for _ in range(args.cpus):
    task_q.put(STOP)
