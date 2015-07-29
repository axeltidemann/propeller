'''
Writes the FHS dataframes into the proper format for PySpike. 

In its current version, it exports national level priorities and
county level priorities.

We skip priority errors that occur on the same time (see: np.unique),
but this is probably beneficial - closer investigation of the Excel
source file reveal that there are many duplicate entries. See the
columns NE_type and NE_name.

python tacos_hdf_to_spikes.py /path/to/data.h5 /path/to/spike_trains/

Author: Axel.Tidemann@telenor.com
'''

from __future__ import print_function
import sys

import pandas as pd
import numpy as np

def write_spikes(df, t_start, t_end, file):
    edges = (0, (t_end - t_start).total_seconds())
    print('# {}'.format(edges), file=file)
    for priority in np.unique(df.priority):
        if not pd.isnull(priority):
            print('# {}'.format(priority), file=file)
            time_stamps = [ (i - t_start).total_seconds() for i in df[ df.priority == priority ].index ]
            print(' '.join([ str(t) for t in np.unique(time_stamps)]), file=file) 

fhs = pd.read_hdf(sys.argv[1], 'fhs')

first = min(fhs.index)
last = max(fhs.index)

t_start = first - pd.Timedelta(seconds=1)
t_end = last + pd.Timedelta(seconds=1)

P1 = fhs[ fhs.priority == 'P1' ].shift(-1, pd.Timedelta(hours=1))
P2 = fhs[ fhs.priority == 'P2' ].shift(-1, pd.Timedelta(hours=1))

P_others = fhs.query(" priority != 'P1' and priority != 'P2' ")

fhs = pd.concat([ P1, P2, P_others ])

with open('{}/spike_trains_Norway.txt'.format(sys.argv[2]), 'w') as file:
    write_spikes(fhs, t_start, t_end, file)

for county in np.unique(fhs.county):
    with open('{}/spike_trains_{}.txt'.format(sys.argv[2], county), 'w') as file:
        write_spikes(fhs[ fhs.county == county ], t_start, t_end, file)
