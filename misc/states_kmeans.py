# Copyright 2016 Telenor ASA, Author: Axel Tidemann

"""
Performs a Barnes-Hut t-SNE visualization of the states in each category.
"""

from __future__ import print_function
import argparse
import glob
import os
import multiprocessing as mp
from functools import partial

from sklearn.cluster import MiniBatchKMeans
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'data_folder',
    help='Folder with Inception states')
parser.add_argument(
    '--png_folder',
    help='Where to put the resulting PNG files. If omitted, put in data_folder/../plots/',
    default=False)
parser.add_argument(
    '--all',
    help='Perform k-means on a global set of all the state files',
    action='store_true')
parser.add_argument(
    '--k',
    help='The max number of clusters (i.e. up to this k)',
    type=int,
    default=10)


args = parser.parse_args()

args.png_folder = args.png_folder if args.png_folder else '{}/plots/'.format(os.path.dirname(args.data_folder.rstrip('/')))

def analyze(path, max_k, h5_file):
    category = os.path.basename(h5_file.replace('.h5', ''))
    print('Processing category {}'.format(category))
    data = pd.read_hdf(h5_file, 'data')
    
    x = range(1,max_k)
    y = []
    for k in x:
        kmeans = MiniBatchKMeans(n_clusters=k)
        try:
            distances = kmeans.fit_transform(np.vstack(data.state))
            # transform() returns euclidean distance. The cost function of kmeans is the sum of all
            # squared distances.
            y.append(np.sum(np.min(distances, axis=1)**2)) 
        except:
            print('Category {} has only {} samples, skipping rest of kmeans.'.format(category, len(data)))
            break

    plt.clf()
    plt.plot(x[:len(y)],y)
    plt.title(category)
    plt.savefig('{}/kmeans_distances_from_centroids_{}.png'.format(path, category), dpi=300)

files = glob.glob('{}/*.h5'.format(args.data_folder))

if args.all:
    data = []
    for h5_file in files:
        _data = pd.read_hdf(h5_file, 'data')#.state[:args.n]
        data.extend(_data.state)

    y = []
    x = range(1, args.k)
    for k in x:
        kmeans = MiniBatchKMeans(n_clusters=k)
        distances = kmeans.fit_transform(np.vstack(data))
        y.append(np.mean(np.min(distances, axis=1)))

    plt.plot(x,y)
    plt.savefig('{}/kmeans_distances_from_centroids_global.png'.format(args.png_folder), dpi=300)
    
else:
    par_analyze = partial(analyze, args.png_folder, args.k)
    pool = mp.Pool()
    pool.map(par_analyze, files)


