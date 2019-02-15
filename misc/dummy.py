import os
import argparse
import multiprocessing as mp

import tensorflow as tf
from keras.applications import Xception
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import pandas as pd
from scipy.misc import imresize
import h5py

STOP = 'stop'

def square(img):
    h = img.shape[0]
    w = img.shape[1]
    new_size = min(w,h)
    w_diff = (w - new_size)//2
    h_diff = (h - new_size)//2
    return img[ h_diff:new_size+h_diff, w_diff:new_size+w_diff ]

def embed_image(task_q, result_q, gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

    print('PID {} GPU {} starting'.format(os.getpid(), gpu))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True

    with tf.Session(config=config) as session:
        net = Xception(include_top=False, weights='imagenet', pooling='max')

        for ad_id, path in iter(task_q.get, STOP):
            embeddings = np.empty(0)
            if not pd.isnull(path) and os.path.exists(path):
                squares = []
                for _file in sorted(os.listdir(path)):
                    try:
                        img = img_to_array(load_img(os.path.join(path,_file)))
                        img = square(img)
                        img = imresize(img, size=(299,299), interp='bicubic')
                        img = preprocess_input(img)
                        squares.append(img)
                    except Exception as e:
                        print(ad_id, path, e)

                squares = np.stack(squares)
                embeddings = net.predict(squares)
                
            result_q.put((ad_id, embeddings))
                    
    print('PID {} GPU {} exiting'.format(os.getpid(), gpu))


parser = argparse.ArgumentParser(description='''
Collects the next to last layer embeddings from the Xception model.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'data',
    help='HDF5 file with images column')
parser.add_argument(
    '--gpus',
    help='Number of GPUs to use',
    type=int,
    default=4)
parser.add_argument(
    '--threads',
    help='How many threads to use pr GPU',
    default=2,
    type=int)
args = parser.parse_args()

task_q = mp.Queue()
result_q = mp.Queue()

with pd.HDFStore(args.data, mode='r') as store:
    keys = store.keys()

data = []

for k in keys:
    data.append(pd.read_hdf(args.data, key=k, columns=['images']))

data = pd.concat(data)

print(len(data.images), 'folders where to find images to embed')

for gpu in range(args.gpus):
    for _ in range(args.threads):
        mp.Process(target=embed_image, args=(task_q, result_q, gpu)).start()
        
for ad_id, path in zip(data.index, data.images):
    task_q.put((ad_id, path))

with h5py.File('{}_embeddings'.format(args.data), 'w', libver='latest') as h5_file:
    for _ in data.images:
        ad_id, embeddings = result_q.get()
        print(ad_id, embeddings.shape)
        h5_file.create_dataset(ad_id, data=embeddings)

for _ in range(args.gpus*args.threads):
    task_q.put(STOP)
