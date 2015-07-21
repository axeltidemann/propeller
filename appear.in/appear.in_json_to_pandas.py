import sys

import pandas as pd

force_ascii = lambda x: x if pd.isnull(x) else str(x.encode('ascii', 'ignore'))

with pd.HDFStore('data.h5', 'w', complevel=9, complib='blosc') as store:
     for input_file in sys.argv[1:]:
         json = pd.read_json(input_file, orient='index', convert_dates=['timestamp'],
                             dtype={ 'questionId': str, 'id': str })
         json.index = json.timestamp
         del json['timestamp']
         json.roomName = json.roomName.apply(force_ascii)
         json.text = json.text.apply(force_ascii)
         store.append('questionnaire', json, data_columns=True)

         print '{} stored in HDF5.'.format(input_file)
