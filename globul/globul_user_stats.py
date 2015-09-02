'''
Calculates the mean and std dev of the number of data points for the users in the globul dataset.

python globul_user_stats.py /path/to/data.h5

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

with pd.get_store(sys.argv[1]) as store:
    users = defaultdict(int)
    for chunk in store.select('data', columns=['IMSI'], chunksize=50000):
        for user_id, count in zip(*np.unique(chunk.IMSI, return_counts=True)):
            users[user_id] += count

    values = users.values()

with pd.HDFStore('{}_IMSI_count'.format(sys.argv[1]), 'w', complevel=9, complib='blosc') as save:
    save.append('data', pd.DataFrame(values, columns=['count'], index=[ str(x) for x in users.keys()]), data_columns=True)
    
    print '{} users, mean: {} std: {} number of datapoints per user.'.format(len(users), np.mean(values), np.std(values))
    sns.distplot(values, kde=False)
    plt.tight_layout()
    plt.savefig('user_datapoint_distribution_nan_included.png', dpi=300)

    plt.clf()
    nan_number = users[np.nan]
    del users[np.nan]
    values = users.values()
    print '''The NaN group has {} entries. After removal the distribution is:
    mean: {} std: {} number of datapoints per non-nan user.'''.format(nan_number, np.mean(values), np.std(values))
    sns.distplot(values, kde=False)
    plt.tight_layout()
    plt.savefig('user_datapoint_distribution_nan_excluded.png', dpi=300)

