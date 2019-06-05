import argparse
import os

import pandas as pd
import numpy as np

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

with pd.HDFStore('chotot.h5', mode='w') as store:
    for csv_file in args.csv:

        raw = pd.read_csv(csv_file, encoding='utf-8', nrows=nrows, index_col=0)

        #raw.images = raw.images.apply(lambda x: '/online_classifieds/download-images/' + x[2:])
        raw.images = [ '/online_classifieds/chotot_validation/images/{}/'.format(ad_id) for ad_id in raw.index ]

        print(raw)
        
        store.append('categories/{}'.format(os.path.basename(csv_file).replace('.csv', '')), raw, complib='blosc', complevel=9, data_columns=['ad_id'])
