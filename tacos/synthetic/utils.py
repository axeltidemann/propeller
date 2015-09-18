import math
import numpy as np
import pandas as pd
import random
import faker
from keras.utils import generic_utils
from keras.layers.core import Layer
from datetime import datetime
import theano.tensor as T
import faker
import pdb


def scale_to_unit_circle(index, scale):
    return [math.sin((2*math.pi)/scale*index), math.cos((2*math.pi)/scale*index)]

def unit_circle_to_scale(circ_coords, scale):
    return (scale*(np.arctan2(circ_coords[0], circ_coords[1])/(2*math.pi)))%scale

def sec_of_week(dti):
    return dti.dayofweek * 86400 + dti.hour*3600 + dti.minute*60 + dti.second


class UnitVector(Layer):
    '''
        Divide the layer by its magnitude.
    '''
    def __init__(self):
        super(UnitVector, self).__init__()

    def get_output(self, train=False):
        X = self.get_input(train)
        return X/ T.sqrt(T.sum(T.sqr(X), 1, keepdims=True))

    def get_config(self):
        return {"name": self.__class__.__name__}

def cosine_similarity(y_true, y_pred):
    norm_y_true = T.sqrt(T.sum(T.sqr(y_true), 1, keepdims=True))
    norm_y_pred = T.sqrt(T.sum(T.sqr(y_pred), 1, keepdims=True))
    dot = T.tensordot(y_true, y_pred, axes=[1,1])
    cossim = dot / (norm_y_true * norm_y_pred)
    objective = 1-cossim
    return objective.mean(axis=-1)

def get_random_periodic_time_series(len_seq, params):
    fake = faker.Faker()
    while True:  #because of bug in pandas
        try:
            rand_date = fake.date_time()
            f = random.sample(params["freqs"].keys(), 1)[0]
            m = random.sample(range(params["mult"]),1)[0]+1
            #m1 = random.sample(range(params["mult"]),1)[0]+1
            #m2 = random.sample(range(params["mult"]),1)[0]+1
            #freq = "'" + str(m1) + "T," + str(m2)+ "S'"
            freq = "'" + str(m) + f + "'"
            seq = pd.date_range(rand_date, periods=len_seq+1, freq=freq) # +1 for prediction
            if params["add_noise"]:
                noise = pd.to_timedelta(np.random.normal(scale=0.01, size=len_seq+1)*(m2*1), unit='s')
                seq += noise
            return freq, seq
        except Exception,e: 
            print str(e)
            return
            #pass

def get_random_periodic_time_series_as_blob(num_seqs, len_seq, params):
    data = []
    freqs = []

    print 'Generating', num_seqs , 'random time series...'
    progbar = generic_utils.Progbar(num_seqs)
    for i in range(num_seqs):
        freq, seq = get_random_periodic_time_series(len_seq, params)
        for s in seq:
            data.append(s)
            freqs.append(freq)
        progbar.add(1)
    return freqs, data


def get_random_batch(X, y, params):

    num_seqs = X.shape[0]
    len_seq = X.shape[1]
    num_rand_seqs = 1 + int(math.ceil(float(num_seqs)/len_seq))
    freqs, data = get_random_periodic_time_series_as_blob(num_rand_seqs, len_seq, params)
    ptr=0
    for i in range(num_seqs):
        for j in range(len_seq):
            X[i, j, :] = params["time_repr"](data[ptr+j])
        #y[i, params["label_repr"](data[ptr+len_seq])] = 1
        params["label_repr"](y, i, data[ptr+len_seq])
        ptr += 1
    return freqs, data


# helper function to sample an index from a probability array
def sample(a, temperature=1.0):
    a = np.log(a)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1,a,1))
