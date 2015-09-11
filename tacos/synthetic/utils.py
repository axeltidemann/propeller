import math
import numpy as np
import pandas as pd
import random
import faker
from keras.utils import generic_utils
from datetime import datetime
import theano.tensor as T
import faker

def scale_to_unit_circle(index, scale):
    return [math.sin((2*math.pi)/scale*index), math.cos((2*math.pi)/scale*index)]

def unit_circle_to_scale(circ_coords, scale):
    return (scale*(np.arctan2(circ_coords[0], circ_coords[1])/(2*math.pi)))%scale

def sec_of_week(dti):
    return dti.dayofweek * 86400 + dti.hour*3600 + dti.minute*60 + dti.second

len_circ_repr = 2
def circular_time_representation(dti):
    repres=[]
    pointer = dti.second
    repres += scale_to_unit_circle(pointer, 60)
#    
#    pointer += dti.minute*60
#    repres += scale_to_unit_circle(pointer, 60*60)
#
#    pointer += dti.hour*3600
#    repres += scale_to_unit_circle(pointer, 60*60*24)
#
#    pointer += dti.dayofweek*86400
#    repres += scale_to_unit_circle(pointer, 60*60*24*7)

    return repres


def cosine_similarity(y_true, y_pred):
    norm_y_true = T.sqrt(T.sum(T.sqr(y_true), 1))
    norm_y_pred = T.sqrt(T.sum(T.sqr(y_pred), 1))
    dot = T.tensordot(y_true, y_pred, axes=[1,1])
    cossim = dot / (norm_y_true * norm_y_pred)
    objective = -1.0*cossim
    return objective.mean(axis=-1)

def get_random_periodic_time_series(seq_size, params):
    # 'day' 'hour' 'minute' 'second' 'business day'
    fake = faker.Faker()

    while True:  #because of bug in pandas
        try:
            rand_date = fake.date_time()
            #rand_date = datetime.today()
            f = random.sample(params["freqs"].keys(), 1)[0]
            m = random.sample(range(params["mult"]),1)[0]+1
            freq = "'" + str(m)+f + "'"
            seq = pd.date_range(rand_date, periods=seq_size, freq=freq)
            if params["add_noise"]:
                noise = pd.to_timedelta(np.random.normal(scale=0.01, size=seq_size)*m*params["freqs"][f], unit='s')
                seq += noise
            return seq
        except:
            pass


def get_random_batch(X, y, params):
    print 'Generating', len(X), 'samples...'
    progbar = generic_utils.Progbar(X.shape[0])

    seq_size = X.shape[1]+1  # +1 bcs including the prediction y
    freqs = []
    for i in range(len(X)):
        seq = get_random_periodic_time_series(seq_size, params)
        freqs.append(seq.freq)
        for j in range(seq_size-1):
            X[i, j, :] = circular_time_representation(seq[j])
        y[i,int(seq[-1].second)] = 1
        #y[i, char_indices[next_chars[i]]] = 1
        progbar.add(1)
    return freqs


# helper function to sample an index from a probability array
def sample(a, temperature=1.0):
    a = np.log(a)/temperature
    a = np.exp(a)/np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1,a,1))
