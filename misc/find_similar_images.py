'''
Finds similar images.

Author: Axel.Tidemann@telenor.com
'''

import argparse
import glob
import time

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'h5',
    help='HDF5 file with filenames and states')
parser.add_argument(
    '--file',
    help='Find the most similar image to this file. If not specified, all-to-all comparison is performed.',
    default=False)
parser.add_argument(
    '--threshold',
    help='Will find cosine similarity below this threshold.',
    type=float,
    default=1.0)
parser.add_argument(
    '--table',
    help='HDF5 table',
    default='data')
args = parser.parse_args()

t0 = time.time()
data = pd.read_hdf(args.h5, args.table)
print 'Loading {} ({} rows) in {} seconds.'.format(args.h5, len(data), time.time()-t0)

X = np.vstack(data.state)

if args.file:
    Y = data[data.index == args.file].state[0].reshape(1,-1)
    similarity = cosine_similarity(X,Y)
    for index in np.argsort(similarity, axis=None)[::-1]:
        if similarity[index] < args.threshold:
            print similarity[index][0], data.index[index]
            break
else:    
    similarity = np.tril(cosine_similarity(X), -1)
    for index in np.argsort(similarity, axis=None)[::-1]:
        img1, img2 = np.unravel_index(index, similarity.shape)

        if similarity[img1, img2] < args.threshold:
            print similarity[img1, img2], data.index[img1], data.index[img2]
            break
