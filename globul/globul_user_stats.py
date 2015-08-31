'''
Calculates the mean and std dev of the number of data points for the users in the globul dataset.

python globul_user_stats.py /path/to/ggsn.h5 /path/to/msc.h5

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd

ggsn_path = sys.argv[1]
msc_path = sys.argv[2]

with pd.get_store(ggsn_path) as ggsn_store, pd.get_store(msc_path) as msc_store:
    unique_ggsn = ggsn_store.select_column('ggsn', 'IMSI').unique()
    unique_msc = msc_store.select_column('msc', 'callingSubscriberIMSI').unique()

    unique_users = np.concatenate([unique_ggsn, unique_msc])
