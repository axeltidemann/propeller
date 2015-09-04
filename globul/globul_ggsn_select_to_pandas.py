'''
Import selected columns of the GGSN CSV data to pandas.

python globul_ggsn_select_to_pandas.py /path/to/ggsn.h5 /path/to/ggsn.csv

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd

from globul_to_pandas import to_hdf5

usecols = ['IMSI', 'cell_ID', 'recordType', 'recordOpeningDate', 'recordOpeningTime']

csv_kwargs = {'parse_dates': { 'timestamp': ['recordOpeningDate', 'recordOpeningTime'] },
              'date_parser': lambda x: pd.to_datetime(x, coerce=True),
              'converters': { col: str for col in usecols },
              'index_col': 'timestamp',
              'usecols': usecols,
              'chunksize': 50000,
              'error_bad_lines': False}

to_hdf5(sys.argv[1], sys.argv[2], csv_kwargs)
