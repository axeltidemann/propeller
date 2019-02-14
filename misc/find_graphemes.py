# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import os
import argparse
import json
from collections import Counter
import multiprocessing as mp

import numpy as np
import regex
import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='''
    Reads HDF5 files with ad ids, titles, descriptions, price and image paths. Finds
    all the unique graphemes in them, mean/std of price and how many images on average there are.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'hdf',
    help='HDF5 file with ads')
parser.add_argument('--target_folder',
                    help='Where to put the histograms',
                    default='/online_classifieds/chotot/plots/histograms')

args = parser.parse_args()

def traverse(key):

    print('Opening', args.hdf, key)

    data = pd.read_hdf(args.hdf, key, stop=1000)

    raw_text = ''.join([ str(t) for t in data.title + data.description ])
    graphemes = regex.findall(r'\X', raw_text, regex.U)

    for variable in zip(['title', 'description']):

        if variable == 'price':
            x = data.price
        else:
            x = data[variable].dropna().apply(lambda x: len(regex.findall(r'\X', x, regex.U)))

        N = np.format_float_scientific(len(x), precision=1, exp_digits=1)
        N = len(x)
        pct = np.around(100*np.mean(~pd.isnull(data[variable])), 1)
        
        fig, ax = plt.subplots()
        frq, edges = np.histogram(x, bins='doane') # doane best when data is not normally distributed
        ax.bar(edges[:-1], frq, width=np.diff(edges), align='edge')
        
        fig.axes[0].set_title('N = {} ({}%)'.format(N, pct))
        
        plt.grid(False)
        plt.xlabel('{}: $\mu = {}, \sigma={}$'.format(variable, np.mean(x), np.std(x)))
        plt.savefig('{}/{}_{}.png'.format(target_folder, key, variable), dpi=300)

    return Counter(graphemes)

with pd.HDFStore(args.hdf, mode='r') as store:
    keys = store.keys()
    
pool = mp.Pool()
results = pool.map(traverse, keys)

counter = Counter()

for cntr in results:
    counter.update(cntr)

file_name = '{}.json'.format(args.hdf)

with open(file_name, 'w') as _file:
    json.dump(counter, _file, sort_keys=True, indent=4)
