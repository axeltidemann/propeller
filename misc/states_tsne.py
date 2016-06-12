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

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
from bhtsne import bh_tsne
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'data_folder',
    help='Folder with Inception states')
parser.add_argument(
    '--png_folder',
    help='Where to put the resulting PNG files',
    default=False)
parser.add_argument(
    '--all',
    help='Perform t-SNE on a global set of all the state files',
    action='store_true')
parser.add_argument(
    '--n',
    help='How many states from each category, when in global mode',
    type=int,
    default=1000)
parser.add_argument(
    '--components',
    help='How many components to do PCA reduction (for the global case). 0 means no PCA.',
    type=int,
    default=0)
args = parser.parse_args()

args.png_folder = args.png_folder if args.png_folder else '{}/plots/'.format(os.path.dirname(args.data_folder.rstrip('/')))

def analyze(path, h5_file):
    category = os.path.basename(h5_file.replace('.h5', ''))
    print('Processing category {}'.format(category))
    data = pd.read_hdf(h5_file, 'data')
    try:
        tsne = np.array([ y for y in bh_tsne(np.vstack(data.state)) ])
        plt.scatter(tsne[:,0], tsne[:,1])
        plt.title(category)
        plt.savefig('{}/{}.png'.format(path, category), dpi=300)
    except Exception as e:
        print(e)

files = glob.glob('{}/*.h5'.format(args.data_folder))

if args.all:
    data = []
    labels = []
    for i, h5_file in enumerate(files):
        _data = pd.read_hdf(h5_file, 'data').state[:args.n]
        data.extend(_data)
        labels.extend([i]*len(_data))

    if args.components:
        pca = PCA(n_components=args.components)
        X = pca.fit_transform(np.vstack(data))
    else:
        X = np.vstack(data)
        
    tsne = np.array([ y for y in bh_tsne(X) ])

    plt.scatter(tsne[:,0], tsne[:,1], c=labels)
    plt.savefig('{}/global_{}_components.png'.format(args.png_folder, args.components), dpi=300)
    
else:
    par_analyze = partial(analyze, args.png_folder)
    pool = mp.Pool()
    pool.map(par_analyze, files)
