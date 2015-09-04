'''
The imports for both MSC and GGSN data differ only in the **kwargs to pandas.csv_read, so the functionality is
provided here.

Author: Axel.Tidemann@telenor.com
'''

import pandas as pd
import numpy as np

def to_hdf5(hdf5_file, input_file, csv_kwargs):
    with pd.HDFStore(hdf5_file, 'w', complevel=9, complib='blosc') as store:
        csv = pd.read_csv(input_file, **csv_kwargs)

        dropped = []
        for chunk in csv:
            dropped.append(np.mean(pd.isnull(chunk.index)))
            chunk.drop(chunk.index[pd.isnull(chunk.index)], inplace=True) # NaT as index
            store.append('data', chunk, data_columns=True)

        print '{} stored in {}. {}% was dropped due to NaT indices.'.format(input_file, hdf5_file, 100*np.mean(dropped))
