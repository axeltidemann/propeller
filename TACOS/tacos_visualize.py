'''
Create plots from the nicely formatted TACOS data, both hourly and weekly
for all counties in Norway.

python tacos_visualize.py /path/to/data.h5 /path/to/figures/

Author: Axel.Tidemann@telenor.com
'''

import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_averages(df, name, path):

    priorities = {'H': [], 'D': []}
    description = {'H': 'hourly', 'D': 'daily'}

    for period in priorities.keys():
        for P in np.unique(df.priority):
            resampled = df[ df.priority == P ].resample(period, how=len)
            series = pd.Series(resampled.priority, index=resampled.index, name=P)
            priorities[period].append(series/series.max())

        P_df = pd.concat(priorities[period], axis=1)
        plt.figure()
        plt.title('{} {} average'.format(name, description[period]))
        sns.heatmap(P_df.T, xticklabels=False)

        indices = np.linspace(0,len(P_df)-1,4).astype(int)
        dates = P_df.index[indices]
        plt.gca().set_xticks(indices)
        plt.gca().set_xticklabels([ d.strftime('%b %d') for d in dates ])
        plt.gca().set_xlabel('')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig('{}/{}_{}.png'.format(path,name,period), dpi=300)
        plt.clf()

fhs = pd.read_hdf(sys.argv[1], 'fhs')

# The dataset is supposed to be 2014-11-01 -> 2015-02-28, so we limit them to this range. However, there
# are some outliers since we now define the start time as the "outage start", and some are from before this
# period.

limited = fhs.ix['2014-11-01':'2015-03-01']

plot_averages(limited, 'Norway', sys.argv[2])

for county in np.unique(limited.county):
    plot_averages(limited[ limited.county == county ], county, sys.argv[2])

    
