# Copyright 2016 Telenor ASA, Author: Axel Tidemann

from __future__ import division
import argparse
import glob
import time
from collections import defaultdict
import os

import pandas as pd
import numpy as np

parser = argparse.ArgumentParser(description='''Goes through
the authorative CSV file with mappings to subsubcategories (i.e. params).
Puts the corresponding visual features in a HDF5 file.
''', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument(
    '--main',
    help='Path to the authorative CSV file',
    default='/mnt/mudah/current_ads_info.csv')
parser.add_argument(
    '--category_csv_folder',
    help='Path to folder with CSV files',
    default='/mnt/mudah/csv/')
parser.add_argument(
    '--states_folder',
    help='Path to folder of visual features, stored in HDF5 format.',
    default='/mudahpix/all_states/')
parser.add_argument(
    '--target',
    help='Where to put the resulting HDF5 files.',
    default='/mudahpix/combined/states/')
parser.add_argument(
    '--limit',
    help='Limit of ads to process',
    default=10000,
    type=int)
args = parser.parse_args()

# We now will do more clustering basically, but done on actual data that says so.
# Exclude jobs and services (category 7*)

t0 = time.time()
ads = pd.read_csv(args.main, usecols=['ad_id', 'car_type', 'bag_type', 'category_level_one',
                                      'gender_type', 'shoe_type', 'tours_and_holidays_type',
                                      'property_type'])
print 'csv file loaded in {} seconds'.format(time.time()-t0)
ads_params = ads.columns

# 'beauty_item_type', 'category_level_two',

# Loop over the category_csv_folder as CSV
for csv in glob.glob('{}/*.csv'.format(args.category_csv_folder)):
    csv_data = pd.read_csv(csv)

    category = csv.split('/')[-1].strip('.csv')
    subcategories = defaultdict(list)
    counter = defaultdict(int)
    
    states_path = glob.glob('{}/*{}*'.format(args.states_folder, category))[0]
    states = pd.read_hdf(states_path)

    not_found = 0
    found = 0
    for ad_id, url in zip(csv_data.ad_id, csv_data.concat):

        if len(counter) and np.all([ count == args.limit for count in counter.values() ]):
            break
        
        sample = ads[ ads.ad_id == ad_id ]
        try:
            indices = ~pd.isnull(sample.values[0])
        except:
            not_found += 1
            continue
            
        subcategory = '_'.join([ '{}_{}'.format(param, value) for param, value in zip(ads_params[indices], sample.values[0][indices]) ][1:])
        subcategory = '{}_{}.h5'.format(category, subcategory)

        if counter[subcategory] == args.limit:
            continue

        filename = url.split('/')[-1]
        try:
            i = np.where([ filename in path for path in states.index ])[0][0]
        except:
            not_found += 1
            continue
        
        subcategories[subcategory].append(states.iloc[i])
        counter[subcategory] += 1

    for filename, series in subcategories.iteritems():
        h5name = os.path.join(args.target, filename)
        with pd.HDFStore(h5name, mode='w', complevel=9, complib='blosc') as store:

            X = np.vstack(series)
            columns = [ 'f{}'.format(i) for i in range(X.shape[1]) ]
            index = [s.name for s in series]
            
            df = pd.DataFrame(X, columns=columns, index=index)
            df.index.name='filename'
            
            store.append('data', df)

            print 'Saving {}, {} rows.'.format(filename, len(df))

    print '{} ads from {} were not found in {} or {}'.format(not_found, csv, args.main, states_path)
