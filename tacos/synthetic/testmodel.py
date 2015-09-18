from keras.models import Sequential, Graph
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM, GRU

import sys, calendar, datetime, math
import numpy as np

from utils import get_random_batch, scale_to_unit_circle, sample, UnitVector, unit_circle_to_scale

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy import linalg as LA

model_path = sys.argv[1]
test_size = 100
seq_size = 15


len_circ_repr = 6
def time_representation(dti):
    repres=[]
    repres += scale_to_unit_circle(dti.second, 60)
    repres += scale_to_unit_circle(dti.minute, 60)
    repres += scale_to_unit_circle(dti.hour, 24)

    return repres


X_test = np.zeros((test_size, seq_size, len_circ_repr), dtype=np.float)
y_test = np.zeros((test_size, len_circ_repr), dtype=np.float)

params = {
#    "freqs":{'D':86400,'H':3600,'T':60,'S':1, 'B':86400, 'W-SUN':86400*7 },
    "freqs":{'S':1, 'T':60},
    "add_noise": False,
    "mult": 59,
    "time_repr":time_representation
}

freqs = get_random_batch(X_test, y_test, params)



## build the model: 
print('Build model...')
graph = Graph()
graph.add_input(name='input', ndim=3)
graph.add_node(GRU(len_circ_repr, 128, return_sequences=True), name='gru1', input='input')
graph.add_node(GRU(128, 128, return_sequences=False), name='gru2', input='gru1')
graph.add_node(Dense(128, 2, activation='tanh'), name='split1', input='gru2')
graph.add_node(Dense(128, 2, activation='tanh'), name='split2', input='gru2')
graph.add_node(Dense(128, 2, activation='tanh'), name='split3', input='gru2')
#graph.add_node(GRU(32, 2, return_sequences=False), name='split1', input='tdd')
#graph.add_node(TimeDistributedDense(32, 2, activation='tanh'), name='split2', input='gru')
#graph.add_node(TimeDistributedDense(32, 2, activation='tanh'), name='split3', input='gru')
#graph.add_node(TimeDistributedDense(32, 2, activation='tanh'), name='split4', input='gru')
graph.add_node(UnitVector(), name='uv1', input='split1')
graph.add_node(UnitVector(), name='uv2', input='split2')
graph.add_node(UnitVector(), name='uv3', input='split3')
#graph.add_node(UnitVector(), name='uv4', input='split4')
graph.add_output(name='out1', input='uv1')
graph.add_output(name='out2', input='uv2')
graph.add_output(name='out3', input='uv3')
#graph.add_output(name='out4', input='uv4')

print('Compile model...')

#graph.compile('rmsprop', {'out1':'mse', 'out2':'mse', 'out3':'mse', 'out4':'mse'})
graph.compile('rmsprop', {'out1':'mse', 'out2':'mse', 'out3':'mse'})

print("Loading model ...")
graph.load_weights(model_path)

print("Generating predictions ...")
y_pred = graph.predict({'input':X_test, 'out1':y_test[:,:2], 'out2':y_test[:,2:4], 'out2':y_test[:,4:6]}, verbose=0)

for i in range(len(X_test)):
    print freqs[i]
    
    secs = [round(unit_circle_to_scale(c, 60)) for c in X_test[i,:,:2]]
    print secs, round(unit_circle_to_scale(y_test[i,:2], 60)), round(unit_circle_to_scale(y_pred['out1'][i,:], 60))
    mins = [round(unit_circle_to_scale(c, 60)) for c in X_test[i,:,2:4]]
    print mins, round(unit_circle_to_scale(y_test[i,2:4], 60)), round(unit_circle_to_scale(y_pred['out2'][i,:], 60))
    hours = [round(unit_circle_to_scale(c, 24)) for c in X_test[i,:,4:6]]
    print hours, round(unit_circle_to_scale(y_test[i,4:6], 24)), round(unit_circle_to_scale(y_pred['out3'][i,:], 24))


    x1 = X_test[i,:,0]
    y1 = X_test[i,:,1]
    x2 = y_test[i,0] 
    y2 = y_test[i,1] 
    x3 = y_pred['out1'][i,0] 
    y3 = y_pred['out1'][i,1] 

    plt.axis((-1.5,1.5,-1.5,1.5))
    plt.scatter(x1, y1, c='w')
    plt.scatter(x2, y2, c='g')
    plt.scatter(x3, y3, c='r')
    plt.plot((0,x2),(0, y2))
    plt.show()

    x1 = X_test[i,:,2]
    y1 = X_test[i,:,3]

    x2 = y_test[i,2] 
    y2 = y_test[i,3] 

    x3 = y_pred['out2'][i,0] 
    y3 = y_pred['out2'][i,1] 

    plt.axis((-1.5,1.5,-1.5,1.5))
    plt.scatter(x1, y1, c='w')
    plt.scatter(x2, y2, c='g')
    plt.scatter(x3, y3, c='r')
    plt.plot((0,x2),(0, y2))
    plt.show()


#
#pred_dates = np.array([int(unit_circle_to_scale(coords, 604800)) for coords in y_pred])
#true_dates = np.array([int(unit_circle_to_scale(coords, 604800)) for coords in y_test])
#
#for a, b, f in zip(true_dates, pred_dates, freqs):
#    print a, b, (a-b), f




