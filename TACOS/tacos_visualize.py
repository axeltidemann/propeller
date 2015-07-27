'''
Create plots from the nicely formatted TACOS data, both hourly and weekly
for all counties in Norway.

First argument is where to find the HDF5 file, second where to put the png files.

Author: Axel.Tidemann@telenor.com
'''

import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

fhs = pd.read_hdf(sys.argv[1], 'fhs')

description = {'H': 'hourly', 'D': 'daily'}

for county in np.unique(fhs.county):
    county_df = fhs[ fhs.county == county ]

    priorities = {'H': [], 'D': []}
    for period in priorities.keys():
        for P in np.unique(county_df.priority):
            resampled = county_df[ county_df.priority == P ].resample(period, how=len)
            series = pd.Series(resampled.priority, index=resampled.index, name=P)
            priorities[period].append(series/series.max())

        df = pd.concat(priorities[period], axis=1)
        plt.figure()
        plt.title('{} {} average'.format(county, description[period]))
        sns.heatmap(df.T, xticklabels=False)

        indices = np.linspace(0,len(df)-1,4).astype(int)
        dates = df.index[indices]
        plt.gca().set_xticks(indices)
        plt.gca().set_xticklabels([ d.strftime('%b %d') for d in dates ])
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.savefig('{}/{}_{}.png'.format(sys.argv[2],county,period))
        plt.clf()
