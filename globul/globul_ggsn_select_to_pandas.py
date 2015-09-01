'''
Import GGSN CSV data to pandas. This is the shortened version, where only IMSI and cell_id are loaded.

python globul_ggsn_select_to_pandas.py /path/to/data.h5 /path/to/ggsn.csv

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd
import numpy as np

input_file = sys.argv[2]
print 'Reading only IMSI and cell_id from {}'.format(input_file)
     
with pd.HDFStore(sys.argv[1], 'w', complevel=9, complib='blosc') as store:
     csv = pd.read_csv(input_file,
                       parse_dates={ 'timestamp': ['recordOpeningDate', 'recordOpeningTime'] },
                       date_parser=lambda x: pd.to_datetime(x, coerce=True),
                       converters={'IMSI': str, 'cell_ID': str},
                       index_col='timestamp',
                       usecols=['IMSI', 'cell_ID', 'recordOpeningDate', 'recordOpeningTime'],
                       chunksize=50000,
                       error_bad_lines=False)

     dropped = []
     for chunk in csv:
          dropped.append(np.mean(pd.isnull(chunk.index)))
          chunk.drop(chunk.index[pd.isnull(chunk.index)], inplace=True) # NaT as index
          store.append('ggsn', chunk, data_columns=True)

     print '{} stored in HDF5. {}% was dropped since NaT was used as an index.'.format(input_file, 100*np.mean(dropped))
