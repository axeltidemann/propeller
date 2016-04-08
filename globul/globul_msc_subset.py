'''
Stores a subset of users based the number of datapoints given the range.

python globul_msc_subset.py /path/to/msc_data.h5 /path/to/msc_IMSI_count min max limit_users

Author: Axel.Tidemann@telenor.com
'''

import sys
from collections import defaultdict

import pandas as pd
from keras.utils import generic_utils

msc_path = sys.argv[1]
count_path = sys.argv[2]
least = int(sys.argv[3])
most = int(sys.argv[4])
limit_users = int(sys.argv[5])

with pd.get_store(count_path) as count_store:
    users = count_store['data']
    subset = users[ (users.datapoints >= least) & (users.datapoints <= most) ]

print '''Selecting users with datapoints in the range [{}-{}]:
{} users, {} rows. Limiting to {} users.'''.format(least, most, len(subset), sum(subset.datapoints), limit_users)

subset = subset[:limit_users]
subset_path = '{}_{}-{}_{}users'.format(msc_path, least, most, limit_users)

with pd.get_store(msc_path) as msc_store, \
     pd.HDFStore(subset_path, 'w', complevel=9, complib='blosc') as subset_store:

    progbar = generic_utils.Progbar(len(subset))
    for user in subset.index:
        if len(user):
            df = msc_store.select('data', where="callingSubscriberIMSI == user")
            subset_store.append('data', df, data_columns=True, min_itemsize=1)
        progbar.add(1)

print 'Subset stored in {}'.format(subset_path)
