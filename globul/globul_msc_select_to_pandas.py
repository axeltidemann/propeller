'''
Import selected columns of the MSC CSV data to pandas.

python globul_msc_to_pandas.py /path/to/msc.h5 /path/to/msc.csv

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd

from globul_to_pandas import to_hdf5

usecols = ['callingSubscriberIMSI', 'calledSubscriberIMSI', 'cell_ID', 'startDateCharge','startTimeCharge', 'typeCode']

csv_kwargs = {'parse_dates': { 'timestamp': ['startDateCharge','startTimeCharge'] },
              'date_parser': lambda x: pd.to_datetime(x, coerce=True),
              'converters': { col: str for col in usecols },
              'index_col': 'timestamp',
              'usecols': usecols,
              'chunksize': 50000,
              'error_bad_lines': False}

to_hdf5(sys.argv[1], sys.argv[2], csv_kwargs)
