import datetime

import pandas as pd

def save_df(name, df):
    df.to_hdf('{}_{}.h5'.format(name, datetime.datetime.now().isoformat()),
              key='result', mode='w', format='table', complib='blosc', complevel=9)
        

def chunks(chunkable, n):
    """ Yield successive n-sized chunks from l. """
    for i in xrange(0, len(chunkable), n):
        yield chunkable[i:i+n]

def safe_filename(filename):
    import base64 # With multiprocessing, this needs to be imported here.
    """ Base64 encodes the string, so you can safely use is as a filename. """
    return base64.urlsafe_b64encode(filename)
