import sys

import pandas as pd

dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d %H:%M:%S.%f')

with pd.HDFStore('data.h5', 'a', complevel=9, complib='blosc') as store:
    for input_file in sys.argv[1:]:
        csv = pd.read_csv(input_file, index_col=0,
                          parse_dates=[0,1,2], date_parser=dateparse,
                          dtype={ 'kpafylke': pd.np.object,
                                  'kpakommune': pd.np.object },
                          chunksize=50000)

        for chunk in csv:
            store.append('tacos', chunk, data_columns=True, 
                         min_itemsize={ 'identifier': 178, 
                                        'kpakommune': 15,
                                        'node': 56 })

        print '{} stored in HDF5.'.format(input_file)
