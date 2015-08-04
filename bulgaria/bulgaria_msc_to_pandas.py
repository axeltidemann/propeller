'''
Import CSV data to pandas. 

python bulgaria_msc_to_pandas.py /path/to/data.h5 /path/to/msc*.csv

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd

force_int = lambda x: -1 if pd.isnull(x) else x

with pd.HDFStore(sys.argv[1], 'a', complevel=9, complib='blosc') as store:
    for input_file in sys.argv[2:]:
        
        csv = pd.read_csv(input_file, 
                          parse_dates={ 'timestamp': ['startDateCharge','startTimeCharge'], 'load_date': ['loadDate']}, 
                          index_col='timestamp', 
                          chunksize=50000,
                          dtype={ key: pd.np.object for key in
                                  ['callingNumber', 'calledNumber', 'translatedNumber', 'smstOriginalCallingNumber',
                                   'rawCalledPartyNumber', 'callIDNumber', 'recordSeqNum', 'reservedField4' ]})

        for chunk in csv:
            store.append('msc', chunk, data_columns=True,
                         min_itemsize={'reservedField4': 3, 
                                       'smstOriginalCallingNumber': 19, 
                                       'outgoingAssgnRoute': 7,
                                       'rawCalledPartyNumber': 24,
                                       'translatedNumber': 21})

        print '{} stored in HDF5.'.format(input_file)
