import matplotlib.pyplot as plt
from keras.callbacks import History, Callback
import numpy as np

plt.ion()
fig = plt.figure()   
ax = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
line1, = ax.plot([], [], 'r-')
line2, = ax2.plot([], [], 'r-')

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_batch_end(self, batch, logs={}):
        if (logs.get('batch') % 50 != 0):
            return
        self.losses.append(logs.get('loss'))
        ax.set_ylim(ymax = np.max(self.losses)*1.1, ymin = np.min(self.losses)*0.9)
        ax.set_xlim(xmax = len(self.losses)+1, xmin = 0)
        line1.set_ydata(self.losses)
        line1.set_xdata(range(len(self.losses)))
        fig.canvas.draw()

    def on_epoch_end(self, epoch, logs={}):
        self.val_losses.append(logs.get('val_loss'))
        ax2.set_ylim(ymax = np.max(self.val_losses)*1.1, ymin = np.min(self.val_losses)*0.9)
        ax2.set_xlim(xmax = len(self.val_losses)+1, xmin = 0)
        line2.set_ydata(self.val_losses)
        line2.set_xdata(range(len(self.val_losses)))
        fig.canvas.draw()

