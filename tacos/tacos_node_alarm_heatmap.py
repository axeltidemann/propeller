'''
Plots a heat map of nodes versus alarms. 

python tacos_node_alarm_heatmap.py /path/to/data.h5

Author: Axel.Tidemann@telenor.com
'''

import sys
from collections import defaultdict, Counter

import pandas as pd
import numpy as np

with pd.get_store(sys.argv[1]) as store:
    data = store.select('tacos', columns=['alarmtype', 'node'])
    
    alarms = defaultdict(lambda : Counter())
    for name, group in data.groupby(['alarmtype', 'node'], sort=False):
        alarms[name[0]][name[1]] = len(group)
