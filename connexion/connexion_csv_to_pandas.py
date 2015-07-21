'''
Import CSV data to pandas.

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd

with pd.HDFStore('data.h5', 'w', complevel=9, complib='blosc') as store:
    for input_file in sys.argv[1:]:
        csv = pd.read_csv(input_file, index_col=0, chunksize=50000)
        for chunk in csv:
            store.append('connexion', chunk, data_columns=True)
        print '{} stored in HDF5.'.format(input_file)
