# Distributions of length title, description + price
# Also overall
# Plot grapheme sorted count with plotly

import os
import argparse
import json
from collections import Counter
import multiprocessing as mp

import numpy as np
import regex
import pandas as pd
import h5py
import plotly
import plotly.graph_objs as go
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='''
    Reads HDF5 files with ad ids, titles, descriptions, price and image paths. Finds
    all unique graphemes and embeddings, stores them in the same file.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'data',
    help='HDF5 file with ads')
args = parser.parse_args()

# Plot graphemes

graphemes = pd.read_hdf(args.data, key='graphemes')

x = graphemes.grapheme
y = graphemes.n
text = [ 'x={}, cumulative: {}%'.format(i, 100*sum(y[:i])/sum(y)) for i in range(len(y)) ]
data = [ go.Bar(x=x, y=y, text=text) ]

plotly.offline.plot(data, filename='{}_grapheme_plot.html'.format(os.path.basename(args.data)), auto_open=False)

# Distributions of grapheme length

def plot_distributions(title_len, description_len, image_len, price, fig_title):
    fig, axes = plt.subplots(2, 2, constrained_layout=True)

    fig.suptitle(fig_title)
    
    frq, edges = np.histogram(title_len, bins='doane')
    axes[0,0].bar(edges[:-1], frq, width=np.diff(edges), align='center')
    axes[0,0].set_xlabel('Title')

    frq, edges = np.histogram(description_len, bins='doane')
    axes[0,1].bar(edges[:-1], frq, width=np.diff(edges), align='center')
    axes[0,1].set_xlabel('Description')

    frq, edges = np.histogram(image_len, bins='doane')
    axes[1,0].bar(edges[:-1], frq, width=np.diff(edges), align='center')
    axes[1,0].set_xlabel('Images')

    frq, edges = np.histogram(price, bins='doane')
    axes[1,1].bar(edges[:-1], frq, width=np.diff(edges), align='center')
    axes[1,1].set_xlabel('Price')

    plt.savefig('{}.png'.format(fig_title), dpi=300)

def n_files(path):
    try:
        return len(os.listdir(path))
    except:
        return 0
    
def lengths(key):
    print('Reading', args.data, key)

    data = pd.read_hdf(args.data, key=key)

    title_len = data.title.apply(lambda x: 0 if pd.isnull(x) else len(x))
    desc_len = data.description.apply(lambda x: 0 if pd.isnull(x) else len(x))
    image_len = data.images.apply(n_files)
    price = data.price[ ~pd.isnull(data.price) ]
    
    plot_distributions(title_len, desc_len, image_len, price, os.path.basename(key))

    return (title_len, desc_len, image_len, price)

with h5py.File(args.data, 'r', libver='latest') as h5_file:
    categories = [ 'categories/{}'.format(c) for c in list(h5_file['categories'].keys()) ]

pool = mp.Pool()
results = pool.map(lengths, categories)

title_len = pd.Series()
desc_len = pd.Series()
image_len = pd.Series()
price = pd.Series()

for _title, _desc, _image, _price in results:
    title_len = title_len.append(_title)
    desc_len = desc_len.append(_desc)
    image_len = image_len.append(_image)
    price = price.append(_price)

plot_distributions(title_len, desc_len, image_len, price, '{}_overall'.format(os.path.basename(args.data)))
