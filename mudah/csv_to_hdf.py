import argparse
import os

import pandas as pd

parser = argparse.ArgumentParser(description='''
    Reads CSV files with ad ids, titles, descriptions, price and image path. Makes the necessary formatting.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'csv',
    help='CSV file(s).',
    nargs='+')
parser.add_argument('--test',
                    help='Run on smaller part of the dataset',
                    action='store_true')
args = parser.parse_args()

nrows = 1000 if args.test else None
names = ['ad_id', 'images', 'title', 'description', 'price' ]

with pd.HDFStore('mudah.h5', mode='w') as store:
    for csv_file in args.csv:

        raw = pd.read_csv(csv_file, header=None, names=names, encoding='latin-1', nrows=nrows, index_col=0, error_bad_lines=False, engine='python')

        print('Read', csv_file)
        
        filepath, csv_file = os.path.split(os.path.abspath(csv_file))
        csv_file = csv_file[:-4]

        raw.images = raw.images.apply(lambda x: os.path.join(filepath, x))

        store.append('categories/{}'.format(csv_file), raw, complib='blosc', complevel=9, data_columns=['ad_id'])
