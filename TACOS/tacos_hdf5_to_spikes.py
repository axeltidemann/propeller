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

import sys

import pandas as pd
import numpy as np
import pyspike as spk
import matplotlib.pyplot as plt
import seaborn as sns
import ipdb

def save_matrix_plot(df, title, path):
    plt.figure()
    sns.heatmap(df, annot=True, fmt='3.2f')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.clf()

def df_to_spike_train(df, t_start, edges):
    return spk.SpikeTrain(np.unique([ (i - t_start).total_seconds() for i in df.index ]), edges)

fhs = pd.read_hdf(sys.argv[1], 'fhs')
labels = filter(lambda x: not pd.isnull(x), np.unique(fhs.priority))

first = min(fhs.index)
last = max(fhs.index)

t_start = first - pd.Timedelta(seconds=60*60+1)
t_end = last + pd.Timedelta(seconds=1)

edges = (0, (t_end - t_start).total_seconds())

spike_trains = [ df_to_spike_train(fhs.query('priority == @priority'), t_start, edges) for priority in labels ]

# Plot simple relationship within errors within an hourly window.
normal = pd.DataFrame(spk.spike_sync_matrix(spike_trains, max_tau=60*60), index=labels, columns=labels)
save_matrix_plot(normal, 'Priority errors, Norway', '{}/Norway_errors.png'.format(sys.argv[2]))

# First simple experiment: shift every priority one hour. Somewhat arbirtrarily.
shifted = pd.DataFrame(index=labels, columns=labels)
for P in shifted.index:
    P_shifted = df_to_spike_train(fhs[ fhs.priority == P ].shift(-1, pd.Timedelta(hours=1)), t_start, edges)
    shifted[P] = [ spk.spike_sync(P_shifted, s_t, max_tau=60*60) for s_t in spike_trains ]
shifted.index = [ '$\mathregular{'+x+'_{-1h}}$' for x in labels ]
save_matrix_plot(shifted, 'Priority errors shifted, Norway', '{}/Norway_errors_shifted.png'.format(sys.argv[2]))

P1 = fhs.query("priority == 'P1'")
P2 = fhs.query("priority == 'P2'")

# More complex experiment: find which time shift led to the best synchronous spike correlation.
results = pd.DataFrame(index = np.linspace(0,60,13,dtype=int), columns = np.unique(fhs.county))

for delay in results.index:
    for county in results.columns:
        spike_trains = [ df_to_spike_train(fhs.query('county == @county and priority == @priority'), t_start, edges)
                         for priority in labels ]
        
        P1_shift = df_to_spike_train(P1.query('county == @county').shift(-1, pd.Timedelta(minutes=delay)), t_start, edges)
        P1_sync = [ spk.spike_sync(P1_shift, s_t, max_tau=delay*60) for s_t in spike_trains ]
        P1_sync[0] = 0 # We just compared the shifted version with itself.

        P2_shift = df_to_spike_train(P2.query('county == @county').shift(-1, pd.Timedelta(minutes=delay)), t_start, edges)
        P2_sync = [ spk.spike_sync(P2_shift, s_t, max_tau=delay*60) for s_t in spike_trains ]
        P2_sync[1] = 0 
        results[county][delay] = np.max(P1_sync + P2_sync)

save_matrix_plot(results, 'Maximum gain of time shifts', '{}/Maximum_gains_of_time_shifts_per_county.png'.format(sys.argv[2])
