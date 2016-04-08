'''
Calculates the mean and std dev of the number of data points for the users in the globul dataset, for MSC data only.

python globul_msc_stats.py /path/to/msc_data.h5

Author: Axel.Tidemann@telenor.com
'''

import sys
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

path = sys.argv[1]

with pd.get_store(path) as store:
    users = defaultdict(int)
    for chunk in store.select('data', chunksize=5e5):
        chunk['IMSI'] = chunk.callingSubscriberIMSI + chunk.calledSubscriberIMSI
        for user_id, count in zip(*np.unique(chunk.IMSI, return_counts=True)):
            users[user_id] += count

values = users.values()

with pd.HDFStore('{}_IMSI_count'.format(path), 'w', complevel=9, complib='blosc') as save:
    save.append('data', pd.DataFrame(values, columns=['datapoints'], index=users.keys()), data_columns=True)
    
    print '{} users, mean: {} std: {} number of datapoints per user.'.format(len(users), np.mean(values), np.std(values))
    sns.distplot(values, kde=False)
    plt.tight_layout()
    plt.savefig('user_datapoint_distribution.png', dpi=300)
