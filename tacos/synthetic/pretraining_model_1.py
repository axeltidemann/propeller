import sys, os.path

from monitoring import LossHistory
from utils import get_random_batch, len_circ_repr, cosine_similarity
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping

import numpy as np

save_model_path = "./pretrained_models/" + sys.argv[0] + ".h5"

nb_iter = 100
nb_epochs = 10000
seq_size=5
train_size=50000
val_size=200

X_train = np.zeros((train_size, seq_size, len_circ_repr), dtype=np.float)
y_train = np.zeros((train_size, 60), dtype=np.bool)
X_val = np.zeros((val_size, seq_size, len_circ_repr), dtype=np.float)
y_val = np.zeros((val_size, 60), dtype=np.bool)

# random seqs parameters
seq_params = {
    "freqs":{'S':1},
    "add_noise": False,
    "mult": 1
}

## build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
#model.add(LSTM(len_circ_repr, 32, return_sequences=True))
model.add(GRU(len_circ_repr, 32, return_sequences=True))
model.add(Dropout(0.2))
model.add(TimeDistributedDense(32, 32))
#model.add(LSTM(32, 32, return_sequences=False))
model.add(GRU(32, 32, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(32, 60))
model.add(Activation('softmax'))

print('Compile model...')
#model.compile(loss='mse', optimizer='adagrad')
#model.compile(loss=cosine_similarity, optimizer='adagrad')
model.compile(loss='categorical_crossentropy', optimizer='adagrad')

history = LossHistory()
checkpointer = ModelCheckpoint(filepath=save_model_path, verbose=1, save_best_only=True)

for e in range(nb_iter):
    print('-'*40)
    print('Iteration', e)
    print('-'*40)

    if os.path.isfile(save_model_path):
        print("Loading best model so far...")
        model.load_weights(save_model_path)

    print("Generating training data...")
    get_random_batch(X_train, y_train, seq_params)
    print("Generating validation data...")
    get_random_batch(X_val, y_val, seq_params)
    print("Fitting data...")

    earlystopper = EarlyStopping(monitor='val_loss', patience=25, verbose=2)
    model.fit(X_train, y_train, batch_size=128, nb_epoch=nb_epochs, validation_data=(X_val, y_val), callbacks=[checkpointer, earlystopper, history])
    
