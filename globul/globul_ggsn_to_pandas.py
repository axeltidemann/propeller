'''
Import GGSN CSV data to pandas.

python bulgaria_ggsn_to_pandas.py /path/to/data.h5 /path/to/ggsn*csv

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd
import numpy as np

from utils import determine_dtypes

with pd.HDFStore(sys.argv[1], 'a', complevel=9, complib='blosc') as store:
     for input_file in sys.argv[2:]:
     
          kwargs = {'parse_dates': { 'timestamp': ['recordOpeningDate', 'recordOpeningTime'], 
                                     'report_date': ['dateOfReport', 'timeOfReport'],
                                     'change_date': ['changeDate', 'changeTime'],
                                     'load_date': ['loadDate'] },
                    'date_parser': lambda x: pd.to_datetime(x, coerce=True),
                    'index_col': 'timestamp',
                    'chunksize': 50000}

          # For those cases with many columns, it might be worth it to scan through the entire file and 
          # determine the dtypes automatically when reading in chunks instead of specifying them on import.
          # If this breaks in case of multiple file input, you'd better remove the headers and concatenate them
          # all into one big file.
          kwargs['dtype'] = determine_dtypes(input_file, **kwargs)

          csv = pd.read_csv(input_file, **kwargs)

          dropped = []
          for chunk in csv:
               dropped.append(np.mean(pd.isnull(chunk.index)))
               chunk.drop(chunk.index[pd.isnull(chunk.index)], inplace=True) # NaT as index
               chunk.dynamicAddrFlag = chunk.dynamicAddrFlag.astype(pd.np.bool) # Somehow escapes the dtype settings
               store.append('ggsn', chunk, data_columns=True)

          print '{} stored in HDF5. {}% was dropped since NaT was used as an index.'.format(input_file, 100*np.mean(dropped))