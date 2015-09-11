from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM

import sys, calendar, datetime, math
import numpy as np

from utils import get_random_batch, len_circ_repr, scale_to_unit_circle, sample

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy import linalg as LA

model_path = sys.argv[1]
test_size = 100
seq_size = 10

X_test = np.zeros((test_size, seq_size, len_circ_repr), dtype=np.float)
y_test = np.zeros((test_size, 60), dtype=np.bool)

params = {
#    "freqs":{'D':86400,'H':3600,'T':60,'S':1, 'B':86400, 'W-SUN':86400*7 },
    "freqs":{'S':1},
    "add_noise": False,
    "mult": 3
}

freqs = get_random_batch(X_test, y_test, params)

## build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(Dense(len_circ_repr, 10), activation="sigm")
model.add(Dense(10, 10), activation="sigm")
model.add(LSTM(len_circ_repr, 32, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(32, 32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, 60))
model.add(Activation('softmax'))

print('Compile model...')
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print("Loading model ...")
model.load_weights(model_path)

print("Generating predictions ...")
y_pred = model.predict(X_test, verbose=0)

for i in range(len(X_test)):
    x1 = X_test[i,:,0]
    y1 = X_test[i,:,1]

    ind_t = np.argmax(y_test[i,:])
    true_s=scale_to_unit_circle(ind_t, 60)
    x2 = true_s[0] 
    y2 = true_s[1] 
    
    ind_p = np.argmax(y_pred[i,:])

    print ind_t, ind_p

    pred_s=scale_to_unit_circle(ind_p, 60)
    x3 = pred_s[0] 
    y3 = pred_s[1] 

    print freqs[i]


    plt.axis((-1.5,1.5,-1.5,1.5))
    plt.scatter(x1, y1, c='w')
    plt.scatter(x2, y2, c='g')
    plt.scatter(x3, y3, c='r')
    plt.plot((0,x3),(0, y3))
    #plt.plot(np.cos(t), np.sin(t), linewidth=1)
    plt.show()


#
#pred_dates = np.array([int(unit_circle_to_scale(coords, 604800)) for coords in y_pred])
#true_dates = np.array([int(unit_circle_to_scale(coords, 604800)) for coords in y_test])
#
#for a, b, f in zip(true_dates, pred_dates, freqs):
#    print a, b, (a-b), f




