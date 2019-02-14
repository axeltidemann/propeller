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
from scipy import stats

parser = argparse.ArgumentParser(description='''
                                 Reads HDF5 files with ad ids, titles, descriptions, price and image paths. Finds
                                 all the unique graphemes in them, mean/std of price and how many images on average there are.
                                 ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('hdf',
                    help='HDF5 file with ads')
parser.add_argument('--test',
                    help='Run on a smaller part of the dataset',
                    dest='test', action='store_true')
parser.add_argument('--target_folder',
                    help='Where to put the histograms',
                    default='/online_classifieds/chotot/plots/histograms')
parser.add_argument('--image_prefix',
                    help='What to prefix the image paths with',
                    default='/online_classifieds/chotot/')

args = parser.parse_args()

QUANTILE_K2_THRESHOLD = 1000.0
QUANTILE_SIZE = 20
BOX_COX_MARGIN = 1e-4

stop = None
if args.test:
    stop = 10

def value_to_quantile(original_value, quantiles):
    if original_value <= quantiles[0]:
        return 0.0
    if original_value >= quantiles[-1]:
        return 1.0
    n_quantiles = float(len(quantiles) - 1)
    right = np.searchsorted(quantiles, original_value)
    left = right - 1

    interpolated = (left + ((original_value - quantiles[left])
                            / ((quantiles[right] + 1e-6) - quantiles[left]))) / n_quantiles
    return interpolated

def pretty_float(x):
    return np.format_float_scientific(x, precision=1, exp_digits=1)

def find_files(path):
    try:
        full_path = os.path.join(args.image_prefix, path)
        return len(os.listdir(full_path))
    except:
        return 0
    
def traverse(key):

    print('Opening', args.hdf, key)

    data = pd.read_hdf(args.hdf, key, stop=stop)
    data['images_length'] = data.images.apply(lambda x: find_files(x))
    
    raw_text = ''.join([ str(t) for t in data.title + data.description ])
    graphemes = regex.findall(r'\X', raw_text, regex.U)

    for variable in ['title', 'description', 'price', 'images_length']:

        if variable == 'price':

            x = data.price.dropna()

            fig, (ax1, ax2) = plt.subplots(2)

            frq, edges = np.histogram(x, bins='doane')
            ax1.bar(edges[:-1], frq, width=np.diff(edges), align='edge')
            ax1.set_ylabel('original')

            mean = np.mean(x)
            std = np.mean(x)
            filtered = False
            boxcox_lambda = None
            
            k2_original, p_original = stats.normaltest(x)

            # Data must be positive for Box-Cox transformation.
            boxcox_shift = float(np.min(x) * -1) + BOX_COX_MARGIN
            boxcox, lmbda = stats.boxcox(x + boxcox_shift)

            k2_boxcox, p_boxcox = stats.normaltest(boxcox)

            values, bins = np.histogram(boxcox, bins='doane')

            # We are plagued with skewed normal distributions with a huge first bin.
            if values[0] > values[1]:
                filtered_boxcox = boxcox[ boxcox >= bins[1] ]

                k2_boxcox, p_boxcox = stats.normaltest(filtered_boxcox)

                filtered = True

            print('{} stats. Original K2: {} P: {} Boxcox K2: {} P: {}'.format(
                variable, k2_original, p_original, k2_boxcox, p_boxcox))

            if lmbda < 0.9 or lmbda > 1.1:
                # Lambda is far enough from 1.0 to be worth doing boxcox
                if k2_original > k2_boxcox * 10 and k2_boxcox <= QUANTILE_K2_THRESHOLD:
                    # The boxcox output is significantly more normally distributed
                    # than the original data and is normal enough to apply
                    # effectively.
                    frq, edges = np.histogram(boxcox, bins='doane', density=True)
                    ax2.bar(edges[:-1], frq, width=np.diff(edges), align='edge')

                    mean = np.mean(boxcox)
                    std = np.std(boxcox)
                    i = np.linspace(mean - 3*std, mean + 3*std, 1000)
                    ax2.plot(i, stats.norm.pdf(i, mean, std), 'r--', label='Including first bin')

                    if filtered:
                        mean = np.mean(filtered_boxcox)
                        std = np.std(filtered_boxcox)
                        i = np.linspace(mean - 3*std, mean + 3*std, 1000)
                        ax2.plot(i, stats.norm.pdf(i, mean, std), 'm-', label='First bin filtered')
                        plt.legend()

                    l = np.around(lmbda, 2)
                    ax2.set_ylabel('boxcox $\lambda={}$'.format(l))
                    boxcox_lambda = lmbda

            if boxcox_lambda is None and k2_original > QUANTILE_K2_THRESHOLD:
                quantiles = np.unique(stats.mstats.mquantiles(x,
                                                              np.arange(QUANTILE_SIZE + 1, dtype=np.float64)
                                                              / float(QUANTILE_SIZE),
                                                              alphap=0.0,
                                                              betap=1.0,)).astype(float).tolist()

                frq, edges = np.histogram([ value_to_quantile(_x, quantiles) for _x in x ], bins='doane')
                ax2.bar(edges[:-1], frq, width=np.diff(edges), align='edge')
                ax2.set_ylabel('quantile')

                print('{} is non-normal, using quantiles: {}'.format(variable, quantiles))

            
        else:

            if variable == 'images_length':
                x = data[variable]
            else:
                x = data[variable].dropna().apply(lambda y: len(regex.findall(r'\X', y, regex.U)))
            fig, ax = plt.subplots()
            frq, edges = np.histogram(x, bins='doane') # doane best when data is not normally distributed
            ax.bar(edges[:-1], frq, width=np.diff(edges), align='edge')

        N = pretty_float(len(x))
        pct = np.around(100*np.mean(~pd.isnull(data[variable])), 1)
        
        fig.axes[0].set_title('N = {} ({}%)'.format(N, pct))
        plt.grid(False)
        plt.xlabel('{}: $\mu = {}, \sigma={}$'.format(variable, pretty_float(np.mean(x)), pretty_float(np.std(x))))
        plt.savefig('{}/{}_{}.png'.format(args.target_folder, key, variable), dpi=300)

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
