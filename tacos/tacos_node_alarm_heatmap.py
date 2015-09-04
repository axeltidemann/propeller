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

def plot_alarms_distribution(alarms, title=''):
    alarmtypes = np.unique([ c for counter in alarms.values() for c in counter.keys() ])
    alarmtypes_index = { alarm: i for i,alarm in enumerate(alarmtypes) }
    
    heat = np.zeros((len(alarmtypes), len(alarms)))

    for col, node in enumerate(alarms.keys()):
        for alarmtype in alarms[node].keys():
            heat[alarmtypes_index[alarmtype], col] = alarms[node][alarmtype]

    heat_norm = normalize(heat, axis=0, norm='l1')

    sns.heatmap(heat_norm, xticklabels=False, yticklabels=False, cmap=plt.get_cmap('Greys'))
    plt.title(title)
    plt.ylabel('Alarm types')
    plt.xlabel('Nodes grouped by class' if len(title) == 0 else 'Nodes')
    plt.tight_layout()
    plt.savefig('figures/{} node alarmtypes heat map.png'.format(title), dpi=300)
    plt.clf()

with pd.get_store(sys.argv[1]) as store:
    data = store.select('tacos', columns=['alarmtype', 'node', 'networkclass'])
    alarms = defaultdict(Counter)
    for (networkclass, node, alarmtype), group in data.groupby(['networkclass', 'node', 'alarmtype']):
        alarms[node][alarmtype] = len(group)
    
    # Beware: some nodes are in several network classes.
    plot_alarms_distribution(alarms)

    for networkclass, group_1 in data.groupby(['networkclass']):
        alarms = defaultdict(Counter)
        for (node, alarmtype), group_2 in group_1.groupby(['node', 'alarmtype']):
            alarms[node][alarmtype] = len(group_2)

        try:
            plot_alarms_distribution(alarms, 'Class {}'.format(networkclass))
        except:
            print 'Error plotting for class {} failed'.format(networkclass)
