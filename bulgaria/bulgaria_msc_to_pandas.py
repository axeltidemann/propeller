'''
Import CSV data to pandas. 

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd

from utils import determine_dtypes

with pd.HDFStore('data.h5', 'a', complevel=9, complib='blosc') as store:
    for input_file in sys.argv[1:]:
        
        csv = pd.read_csv(input_file, 
                          parse_dates={ 'timestamp': ['startDateCharge','startTimeCharge'], 'load_date': ['loadDate']}, 
                          index_col='timestamp', 
                          chunksize=50000, 
                          dtype={ 'calledNumber': pd.np.float64,
                                  'reservedField4': pd.np.float64,
                                  'translatedNumber': pd.np.float64,
                                  'smstOriginalCallingNumber': pd.np.object,
                                  'rawCalledPartyNumber': pd.np.object,
                                  'callIDNumber': pd.np.float64,
                                  'recordSeqNum': pd.np.float64 })
        
        for chunk in csv:
            store.append('msc', chunk, data_columns=True,
                         min_itemsize={ 'smstOriginalCallingNumber': 19, 
                                        'outgoingAssgnRoute': 7,
                                        'rawCalledPartyNumber': 24 })

        print '{} stored in HDF5.'.format(input_file)
