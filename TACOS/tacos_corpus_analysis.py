'''
Analyise the TACOS/NOC log files using word2vec/GloVe text mining tools.

python tacos_corpus_analysis.py /path/to/data.h5

Author: Axel.Tidemann@telenor.com
'''

import sys
import multiprocessing as mp
from functools import partial

import pandas as pd
import gensim
import logging
import ipdb

#logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def frequencies(store):
    
    columns = [ 'identifier', 'node', 'alarmtype', 'kpafylke', 'kpakommune', 'fhsseverity', 'severity', \
                'class', 'fhsproblemarea', 'summary' ]
    data = []
    for col in columns:
        column = store.select_column('tacos', col)
        unique = len(column.unique())
        nan = sum(pd.isnull(column))
        data.append([ unique, 100.0*unique/store.get_storer('tacos').nrows, nan, 100.0*nan/store.get_storer('tacos').nrows ])

    return pd.DataFrame(data, index = columns, columns = ['unique', 'unique_ratio', 'nan', 'nan_ratio'])
    

class DFSentences:
    def __init__(self, store, column):
        self.store = store
        self.column = column

    def __iter__(self):
        corpus = self.store.select('tacos', columns=[self.column], chunksize=50000)
        for chunk in corpus:
            yield chunk.fillna('')[self.column]

def save_word_model(column, path):
    with pd.get_store(path) as store:
        sentences = DFSentences(store, column)
        model = gensim.models.Word2Vec(sentences) #, workers=mp.cpu_count())
        model.save('{}.word2vec.{}'.format(path, column))

partial_save = partial(save_word_model, path=sys.argv[1])

pool = mp.Pool()
pool.map(partial_save, ['identifier', 'node', 'alarmtype',  'fhsproblemara' ])
