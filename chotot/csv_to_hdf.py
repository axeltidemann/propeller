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

def strip(x):
    if pd.isnull(x):
        return x
    
    x = x.replace('s3://chotot-staging/content_moderation/','/online_classifieds/chotot/')
    k = x.rfind('/')
    return x[:k]

def valid(index, price):
    try:
        int(index)
        float(price)
        return True
    except:
        return False

nrows = 1000 if args.test else None
names = ['ad_id','title', 'description', 'price', 'images']

with pd.HDFStore('chotot.h5', mode='w') as store:
    for csv_file in args.csv:

        raw = pd.read_csv(csv_file, header=None, names=names, encoding='utf-8', nrows=nrows)
        filtered = raw.groupby(by='ad_id', sort=False).first()
        
        print('Read', csv_file, 'original length:', len(filtered))
        
        # Some errors have crept in. We remove those where the index is not an integer and the price is not a float.
        cleaned = pd.DataFrame(filtered[ [ valid(i,p) for i,p in zip(filtered.index, filtered.price) ] ])

        cleaned.price = cleaned.price.astype(float)
        cleaned.images = cleaned.images.apply(strip)

        store.append('categories/{}'.format(os.path.basename(csv_file)), cleaned, complib='blosc', complevel=9, data_columns=['ad_id'])
        print('Cleaned', csv_file, 'appended, new length:', len(cleaned),
              '({}%)'.format(np.around(100*len(cleaned)/len(filtered),1)))
