'''
Importing CSV data to pandas.

python tacos_csv_to_pandas.py /path/to/data.h5 /path/to/csv*

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd

dateparse = lambda x: pd.NaT if pd.isnull(x) else pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')

with pd.HDFStore(sys.argv[1], 'a', complevel=9, complib='blosc') as store:
    for input_file in sys.argv[2:]:
        csv = pd.read_csv(input_file,
                          index_col=0,
                          parse_dates=[0,1,2],
                          date_parser=dateparse,
                          names=['emsfirsttime','emsclosedtime','actiontime','identifier','node','alarmtype',
                                 'kpafylke','kpakommune','fhsseverity','severity','networkclass','fhsproblemarea','summary'],
                          header=0,
                          converters={'kpafylke': str, 'kpakommune': str,'fhsseverity': str,
                                      'severity': str, 'class': str}
                          chunksize=50000)

        for chunk in csv:
            store.append('tacos', chunk, data_columns=True, 
                         min_itemsize={ 'identifier': 178, 
                                        'kpakommune': 15,
                                        'node': 56 })

        print '{} stored in HDF5.'.format(input_file)
