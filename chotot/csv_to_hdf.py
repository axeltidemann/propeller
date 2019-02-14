import argparse

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='''
    Reads CSV files with ad ids, titles, descriptions, price and image path. Makes the necessary formatting.
    ''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    'csv',
    help='CSV file(s).',
    nargs='+')
args = parser.parse_args()

def strip(x):
    if pd.isnull(x):
        return x
    
    x = x.replace('s3://chotot-staging/content_moderation/','')
    k = x.rfind('/')
    return x[:k]

def valid(index, price):
    try:
        int(index)
        float(price)
        return True
    except:
        return False
    

with pd.HDFStore('chotot.h5', mode='w') as store:
    for csv_file in args.csv:

        raw = pd.read_csv(csv_file, header=None, names=['ad_id','title', 'description', 'price', 'images'], encoding='utf-8')
        filtered = raw.groupby(by='ad_id', sort=False).first()
        
        print('Read', csv_file, 'original length:', len(filtered))
        
        # Some errors have crept in. We remove those where the index is not an integer and the price is not a float.
        cleaned = pd.DataFrame(filtered[ [ valid(i,p) for i,p in zip(filtered.index, filtered.price) ] ])

        cleaned.price = cleaned.price.astype(float)
        cleaned.images = cleaned.images.apply(strip)

        store.append(csv_file, cleaned, complib='blosc', complevel=9, data_columns=['ad_id'])
        print('Cleaned', csv_file, 'appended, new length:', len(cleaned),
              '({}%)'.format(np.around(100*len(cleaned)/len(filtered),1)))
