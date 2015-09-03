'''
Import CSV data to pandas. This is the shortened version, where only callingSubscriberIMSI and cell_ID are loaded.

python globul_msc_to_pandas.py /path/to/data.h5 /path/to/msc.csv

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd
import numpy as np

hdf5_file = sys.argv[1]
input_file = sys.argv[2]
print 'Reading only callingSubscriberIMSI and cell_ID from {}'.format(input_file)

with pd.HDFStore(hdf5_file, 'w', complevel=9, complib='blosc') as store:
    csv = pd.read_csv(input_file, 
                      parse_dates={ 'timestamp': ['startDateCharge','startTimeCharge'] },
                      date_parser=lambda x: pd.to_datetime(x, coerce=True),
                      converters={'startTimeCharge': str, 'callingSubscriberIMSI': str, 'cell_ID': str},
                      index_col='timestamp',
                      usecols=['callingSubscriberIMSI', 'cell_ID', 'startDateCharge','startTimeCharge'],
                      chunksize=50000,
                      error_bad_lines=False)

    dropped = []
    for chunk in csv:
        dropped.append(np.mean(pd.isnull(chunk.index)))
        chunk.drop(chunk.index[pd.isnull(chunk.index)], inplace=True) # NaT as index
        store.append('msc', chunk, data_columns=True)

    print '{} stored in {}. {}% was dropped since NaT was used as an index.'.format(input_file, hdf5_file, 100*np.mean(dropped))
