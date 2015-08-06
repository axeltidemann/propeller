'''
Calculates how many moves the user performs in the Wowbox app after getting free data.

python wowbox_analysis.py /path/to/data.h5 /path/to/figures/

Author: Axel.Tidemann@telenor.com
'''

import sys
import time
from functools import partial

# Necessary to run on joker without crashing when nohup'ing.
import matplotlib as mpl
mpl.use('Agg')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp

def count(users_index, path, free):
    users = pd.DataFrame(index=users_index, columns=['before', 'after'])
    store = pd.HDFStore(path, 'r')

    for user in users.index:
        u = str(user)
        # Find the times of when card_id was met with success.
        success = store.select('action_log', where="card_id == free.index and status == 'success' and user_id == u", columns=[])
        before = 0
        after = 0
        for index in success.index:
            early = index - pd.Timedelta(hours=1)
            before += len(store.select('action_log', 'index>early and index<index and user_id = u', columns=[]))
            late = index + pd.Timedelta(hours=1)
            after += len(store.select('action_log', 'index>index and index<late and user_id = u', columns=[]))
        users.ix[user] = [before, after]
    store.close()
    return users

store = pd.HDFStore(sys.argv[1], 'r')
cards = store['cards']
# There are others with the name 'Free' in them, find them like this:
# cards[ cards.name.str.contains('Free|free')==True ]
free = cards.query("name == 'Exclusive Offer 20MB Free'")
users_index = store.select('action_log', where='card_id == free.index', columns=['user_id']).user_id.unique()
store.close()

partial_count = partial(count, path=sys.argv[1], free=free)
pool = mp.Pool()
users = pd.concat(pool.map(partial_count, np.array_split(users_index, mp.cpu_count())))

sns.barplot(data=users)
plt.tight_layout()
plt.savefig('{}/before_after_free_data.png'.format(sys.argv[2]), dpi=300)
