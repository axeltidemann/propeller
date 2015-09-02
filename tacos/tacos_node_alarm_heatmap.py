'''
Plots a heat map of nodes versus alarms. 

python tacos_node_alarm_heatmap.py /path/to/data.h5

Author: Axel.Tidemann@telenor.com
'''

import sys
from collections import defaultdict, Counter

import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns

with pd.get_store(sys.argv[1]) as store:
    data = store.select('tacos', columns=['alarmtype', 'node'])

    alarmtypes = data.alarmtype.unique()
    alarmtypes_index = { alarm: i for i,alarm in enumerate(alarmtypes) }
    
    alarms = defaultdict(lambda : Counter())
    for name, group in data.groupby(['node', 'alarmtype'], sort=False):
        alarms[name[0]][name[1]] = len(group)

    heat = np.zeros((len(alarms), len(alarmtypes)))

    for row, node in enumerate(alarms.keys()):
        for alarmtype in alarms[node].keys():
            heat[row,alarmtypes_index[alarmtype]] = alarms[node][alarmtype]


    heat_norm = normalize(heat, axis=1, norm='l1')
    sns.heatmap(heat_norm, xticklabels=False, yticklabels=False)
    plt.xlabel('Alarm types')
    plt.ylabel('Nodes')
    plt.savefig('node alarmtypes heat map.png', dpi=300)
