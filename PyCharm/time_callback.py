import tensorflow as tf
import time

class Timing_Callback(tf.keras.callbacks.Callback):
    """Custom keras callback to meaure the traintime.
    on_train_begin() and on_train_end() are called by the callback itself.

    Initialize without any parameter.
    Get traintime with Timing_Callback.times.
    """
    def __init__(self, logs=None):  # initialize
        self.times = []

    def on_train_begin(self, logs={}):  # called when training starts
        self.starttime = time.time()

    def on_train_end(self, logs={}):  # called when training ends
        self.times.append(time.time() - self.starttime)
