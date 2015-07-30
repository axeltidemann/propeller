'''
Various plots of the appear.in data for Min Xie, as well as Excel file in a formatted way.

python /path/to/appear.in/data.h5 /path/to/figures/

Author: Axel.Tidemann@telenor.com
'''

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('ggplot')

df = pd.read_hdf(sys.argv[1], 'questionnaire')
df.sort(inplace=True)

month = df.index[0].month
start = 0
start_index = df.index[0]

with pd.ExcelWriter('{}/min.xlsx'.format(sys.argv[2])) as xl:
    for i, index in enumerate(df.index):
        if index.month != month:
            month = index.month
            df.ix[start:i].to_excel(xl, sheet_name=start_index.strftime('%Y %B'))
            start = i
            start_index = index

room_unique, room_counts = np.unique(df.roomName, return_counts=True)
room_sizes = [ 10, 20, 50, 100, 150, 200, 250 ]

for questionId in np.unique(df.questionId):
    filtered = df.query('rating == 0 and questionId == @questionId')
    everything = df.query('questionId == @questionId')
    daily = filtered.resample('D', how=len)
    # We offset the timestamps so that the aggregated values are at the end of each
    # week and month, instead of at the first day. Makes more sense when you read it,
    # since the sums are typically computed at the end of each week/month.
    weekly = filtered.resample('W', how=len, loffset='1W')
    monthly = filtered.resample('M', how=len, loffset='1M')

    plt.figure()
    daily.rating.plot(label='Daily')
    weekly.rating.plot(label='Weekly')
    monthly.rating.plot(label='Monthly')
    text = 'Average number of 0 ratings, {} rooms'.format(questionId)
    plt.title(text)
    plt.xlabel('')
    plt.xlim([ daily.ix[0].name, daily.ix[-1].name ]) # Account for week/month offset
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(sys.argv[2], text), dpi=300)
    plt.clf()

    daily_ratio = daily/everything.resample('D', how=len)
    weekly_ratio = weekly/everything.resample('W', how=len, loffset='1W')
    monthly_ratio = monthly/everything.resample('M', how=len, loffset='1M')
    
    plt.figure()
    daily_ratio.rating.plot(label='Daily')
    weekly_ratio.rating.plot(label='Weekly')
    monthly_ratio.rating.plot(label='Monthly')
    text = 'Ratio of 0 ratings, {} rooms'.format(questionId)
    plt.title(text)
    plt.xlabel('')
    plt.xlim([ daily.ix[0].name, daily.ix[-1].name ]) # Account for week/month offset
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig('{}/{}.png'.format(sys.argv[2], text), dpi=300)
    plt.clf()
    
    for i in range(len(room_sizes)-1):
        plt.figure()
        rooms = room_unique[ (room_sizes[i] <= room_counts) & (room_counts < room_sizes[i+1]) ]

        for room in rooms:
            try:
                df[ (df.roomName == room) & (df.questionId == questionId) ].rating.plot(style='+', label=room)
            except:
                print '{} did not have {} data'.format(room, questionId)
        
        text = '{} rooms with data points in range [{}-{})'.format(questionId, room_sizes[i], room_sizes[i+1])
        plt.title(text)

        plt.ylim([-10, 110])
        if len(rooms) < 20:
            plt.legend(loc='center left', ncol=2)
        plt.xlabel('')
        plt.ylabel('rating')
        plt.savefig('{}/{}.png'.format(sys.argv[2], text), dpi=300)
        plt.tight_layout()
        plt.clf()
