'''
Import Excel data files to pandas.

python tacos_xlsx_to_pandas.py /path/do/data.h5 /path/to/excel_file*

First argument is path/to/data.h5, second is to the Excel file.

Author: Axel.Tidemann@telenor.com
'''

import sys
import unicodedata

import pandas as pd

# A thorough 'ASCIIfication' is necessary, because the HDF5 storage does not like unicode in python 2.x,
# and there are ints, floats (i.e. NaNs) and datetimes stored in the same columns here. (NE_name is the worst.)
to_ascii = lambda x: unicodedata.normalize('NFKD', x).encode('ascii', 'ignore').strip() if type(x) is unicode else str(x)

converters = { key: to_ascii for key in [ 'category', 'priority', 'problem_area', 'problem_type', 'consequence',
                                          'municipality', 'county', 'corrected_by', 'fault_cause', 'closure_note',
                                          'NE_type', 'NE_name' ] }

with pd.HDFStore(sys.argv[1], 'a', complevel=9, complib='blosc') as store:
    for input_file in sys.argv[2:]:
        xl = pd.read_excel(input_file, sheetname=0, index_col=9, # Index on outage_start
                           converters = converters)
        store.append('fhs', xl, data_columns=True)
        print '{} stored in HDF5.'.format(input_file)
        
