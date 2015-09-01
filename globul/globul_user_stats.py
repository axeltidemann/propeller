'''
Calculates the mean and std dev of the number of data points for the users in the globul dataset.

python globul_user_stats.py /path/to/data.h5

Author: Axel.Tidemann@telenor.com
'''

import sys
from collections import defaultdict

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

with pd.get_store(sys.argv[1]) as store:
    users = defaultdict(int)
    for chunk in store.select('data', columns=['IMSI'], chunksize=50000):
        for user_id, count in zip(*np.unique(chunk.IMSI, return_counts=True)):
            users[user_id] += count

    values = users.values()

    print '{} users, mean: {} std: {} number of datapoints per user.'.format(len(users), np.mean(values), np.std(values))
    sns.distplot(values, kde=False)
    plt.tight_layout()
    plt.savefig('user_datapoint_distribution.png', dpi=300)
