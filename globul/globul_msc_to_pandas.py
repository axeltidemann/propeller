'''
Import CSV data to pandas. The third input parameter is the output of `wc -l ggsn.csv`, this is
needed to print out the progress.

python bulgaria_msc_to_pandas.py /path/to/data.h5 /path/to/msc*.csv /path/to/line_count.txt

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd

with open(sys.argv[3], 'r') as line_count_file:
     line = line_count_file.readline()
     num_lines = int(line.split()[0])

input_file = sys.argv[2]
print 'Reading {}, contains {} lines.'.format(input_file, num_lines)

with pd.HDFStore(sys.argv[1], 'a', complevel=9, complib='blosc') as store:
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
