'''
Import CSV data to pandas.

python connexion_deviceanalyzer_to_pandas.py /path/to/data.h5 /path/to/csv*

Author: Axel.Tidemann@telenor.com
'''


import sys

import pandas as pd

with pd.HDFStore(sys.argv[1], 'w', complevel=9, complib='blosc') as store:
     for input_file in sys.argv[2:]:
         csv = pd.read_csv(input_file, header=None, names=['x', 'y', 'description', 'value', 'bool'],
                           chunksize=50000, index_col=0, delimiter=';')
         
         for chunk in csv:
             store.append('csv', chunk, data_columns=True)

         print '{} stored in HDF5.'.format(input_file)
         
