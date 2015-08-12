'''
Analyise the TACOS/NOC log files using word2vec/GloVe text mining tools.

python tacos_corpus_analysis.py /path/to/data.h5

Author: Axel.Tidemann@telenor.com
'''

import sys

import pandas as pd

with pd.get_store(sys.argv[1]) as store:

    columns = [ 'identifier', 'node', 'alarmtype', 'kpafylke', 'kpakommune', 'fhsseverity', 'severity', \
                'class', 'fhsproblemarea', 'summary' ]

    data = []
    for col in columns:
        column = store.select_column('tacos', col)
        unique = len(column.unique())
        nan = sum(pd.isnull(column))
        data.append([ unique, 100.0*unique/store.get_storer('tacos').nrows, nan, 100.0*nan/store.get_storer('tacos').nrows ])

    stats = pd.DataFrame(data, index = columns, columns = ['unique', 'unique_ratio', 'nan', 'nan_ratio'])
        
print stats
    

