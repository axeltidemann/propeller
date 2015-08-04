'''
Import appear.in json data to pandas.

python appear.in_json_to_pandas.py /path/to/data.h5 /path/to/json_file*

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd

force_ascii = lambda x: x if pd.isnull(x) else str(x.encode('ascii', 'ignore'))

with pd.HDFStore(sys.argv[1], 'w', complevel=9, complib='blosc') as store:
     for input_file in sys.argv[2:]:
         json = pd.read_json(input_file, orient='index', convert_dates=['timestamp'],
                             dtype={ 'questionId': str, 'id': str })
         json.index = json.timestamp
         del json['timestamp']
         json.roomName = json.roomName.apply(force_ascii)
         json.text = json.text.apply(force_ascii)
         store.append('questionnaire', json, data_columns=True)

         print '{} stored in HDF5.'.format(input_file)
