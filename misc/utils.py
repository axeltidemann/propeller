import datetime

import pandas as pd

def save_df(name, df):
    df.to_hdf('{}_{}.h5'.format(name, datetime.datetime.now().isoformat()),
              key='result', mode='w', format='table', complib='blosc', complevel=9)
        

