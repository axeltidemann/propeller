# Copyright 2016 Telenor ASA, Author: Axel Tidemann

import argparse
import json
import os

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''Plots centroids of clusters.
    ''',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        'json',
        help='JSON file with clusters')
    parser.add_argument(
        'h5',
        help='Corresponding HDF5 file')
    parser.add_argument(
        '--filename',
        default=False,
        help='Filename for PNG file. If unspecified, will be JSON filename + .png')
    args = parser.parse_args()

    with open(args.json) as _json:
        clusters = json.load(_json)

    h5 = pd.read_hdf(args.h5)

    for seed, nodes in clusters.iteritems():
        states = [ h5.state[h5.index == node].values[0] for node in nodes ]

        X = np.vstack(states)

        mean = np.mean(X, axis=0)
        std = np.mean(X, axis=0)

        label = 'rejected' if seed == 'rejected' else None

        plt.errorbar(range(X.shape[1]), mean, std, label=label, alpha=.5)
        
    args.filename = args.filename if args.filename else '{}.png'.format(os.path.basename(args.json))

    plt.xlim([0,X.shape[1]])
    plt.legend()
    plt.savefig(args.filename, dpi=300)

    print 'Plot saved to {}'.format(args.filename)
