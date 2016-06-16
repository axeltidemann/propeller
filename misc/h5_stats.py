# Copyright 2016 Telenor ASA, Author: Axel Tidemann

'''
Collects stats of categories.
'''

import argparse
import glob
import os
import shutil

import pandas as pd
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'data_folder',
    help='Folder with Inception states for training')
parser.add_argument(
    '--filename',
    help='Filename to save',
    default='counts.png')
parser.add_argument(
    '--limit',
    help='Minimum amount of data necessary',
    default=1000)
parser.add_argument(
    '--target',
    help='Where to put the file',
    default='/mnt/kaidee/curated/')
args = parser.parse_args()

files = sorted(glob.glob('{}/*.h5'.format(args.data_folder)))

counts = []
categories = []
for h5_file in files:
    length = len(pd.read_hdf(h5_file, 'data'))
    counts.append(length)
    category = os.path.basename(h5_file).replace('.h5','')
    categories.append(category)
    print '{}: {}'.format(category, length)

    if length > args.limit:
        shutil.copy(h5_file, args.target)
        print '--> copied to {}'.format(args.target)

sns.barplot(range(len(categories)), counts)
plt.xticks(range(len(categories)), categories)

plt.savefig(args.filename, dpi=300)
