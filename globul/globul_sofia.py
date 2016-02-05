'''
Extracts outgoing call/SMS data for Sofia.

python globul_sofia.py /path/to/meta.h5 /path/to/msc.h5

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd
import numpy as np
from keras.utils import generic_utils

meta_path = sys.argv[1]
msc_path = sys.argv[2]
sofia_path = '{}sofia.h5'.format(meta_path[:meta_path.rfind('/')+1] if meta_path.rfind('/') > -1 else '')

with pd.get_store(meta_path) as meta_store, \
     pd.get_store(msc_path) as msc_store, \
     pd.HDFStore(sofia_path, 'w', complevel=9, complib='blosc') as sofia_store:

    chunksize = int(5e5)

    site_info = meta_store['site_info']
    sofia = site_info.query("Region == 'SOFIA_CITY'")

    progbar = generic_utils.Progbar(msc_store.get_storer('data').nrows)

    print 'Extracting outgoing calls/SMS from Sofia'
    
    for chunk in msc_store.select('data', chunksize=chunksize):
        chunk['outgoing'] = chunk.typeCode.apply(lambda x: x in ['01', '05'])
        chunk = pd.DataFrame(chunk.query('outgoing == True')) # Cannot do assignments on slices later on

        cell_ID_truncated = chunk.cell_ID.str[:-1]
        in_sofia = cell_ID_truncated.apply(lambda x: len(x) and (int(x) in sofia.index))
        chunk['outgoing_sofia'] = chunk.outgoing == in_sofia
        chunk = chunk.query('outgoing_sofia == True')
        del chunk['outgoing']
        del chunk['outgoing_sofia']
        del chunk['calledSubscriberIMSI']
        sofia_store.append('data', chunk, data_columns=True)

        progbar.add(chunksize)
        
    print 'Sofia users stored in {}'.format(sofia_path)
