'''
Various utilities. 

Author: Axel.Tidemann@telenor.com
'''

import pandas as pd

def determine_dtypes(file_name, **kwargs):
    csv = pd.read_csv(file_name, **kwargs)

    types = {}
    for chunk in csv:
        for col in chunk.columns:
            if col in types:
                if chunk[col].dtype in types[col]:
                    types[col][chunk[col].dtype] += 1
                else:
                    types[col][chunk[col].dtype] = 1
            else:
                types[col] = { chunk[col].dtype: 1 }
        
    clean = {}
    for key in types.keys():
        if len(types[key]) == 1:
            clean[key] = types[key].keys()[0]
        else:
            if pd.np.int64 in types[key].keys() and pd.np.float64 in types[key].keys():
                clean[key] = pd.np.float64
            else:
                clean[key] = max(types[key], key=lambda x: types[key][x])

    return clean
