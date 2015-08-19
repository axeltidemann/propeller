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

def frequency(column, path):
    with pd.get_store(path) as store:
        result = store.select_column('tacos', column)
        unique = len(result.unique())
        nan = sum(pd.isnull(result))
        
    return [ column, [ unique, 100.0*unique/store.get_storer('tacos').nrows, nan, 100.0*nan/store.get_storer('tacos').nrows ] ]

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


partial_freq = partial(frequency, path=sys.argv[1])

pool = mp.Pool()
columns, data = zip(*pool.map(partial_freq, [ 'identifier', 'node', 'alarmtype', 'kpafylke', 'kpakommune',
                                              'fhsseverity', 'severity', 'class', 'fhsproblemarea', 'summary' ]))
print pd.DataFrame(data, index=columns, columns=['unique', 'unique_ratio', 'nan', 'nan_ratio'])

sys.exit()
    
partial_save = partial(save_word_model, path=sys.argv[1])
pool.map(partial_save, ['identifier', 'node', 'alarmtype',  'fhsproblemarea' ])
