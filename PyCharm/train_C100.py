import models
import history as ht
from time_callback import Timing_Callback
import numpy as np
import tensorflow as tf
import os
import sys


''' 
This script shows the training for the CNN model "C100" (see comparison and evaluation in the thesis).
For the training and comparison of the thesis other network settings (model_name and filters) were used.
'''

''' 
MirrorStrategy for computing on multiple GPUs since multi_gpu_model is deprecated,
devices have to be declared manually.
More than 2 GPUs only works with HierarchicalCopyAllReduce.
First VM used 4 GPUs, Last VM used 2 GUPs.
'''
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"]) 
# use this command when the first strategy doesn't work on the VM
# strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"],
#                                           cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())

# CNN Filters of the evaluation chapter
# filters = [(40), (80), (120), (160)]  # 0.3 mio parameters
# filters = [(32), (64), (96), (128)]  # 0.2 mio parameters
filters = [(23), (46), (69), (92)]  # 0.1 mio parameters

# Filename for saving
script_name = os.path.basename(sys.argv[0])
filename = script_name.split('.')[0]
# Train settings
start = 0
end = 30
model_name = 'build_cnn_4c_1f'
epochs = 80
batch_size = 24
# Learning Rate settings
lr = 0.001
min_lr = 0.0001
# Dropout setting
dropouts = [0.0, 0.2]


def get_compiled_model(num_rows, num_columns):
    """Returns the compiled model.
    Builds the model for the given string.

    Parameters
    ----------
    num_rows : int
        inputs.shape[1] of input data, here 96 mel bands
    num_columns : int
        inputs.shape[2] of input data, here 1360 timeframes

    Returns
    -------
    model : compiled keras model
    """
    num_channels = 1
    # Create model
    model = None
    if model_name == 'build_crnn_4c_2r':
        model = models.build_crnn_4c_2r(input_shape=(num_rows, num_columns, num_channels),
                                        filters=filters, dropouts=dropouts)
    elif model_name == 'build_cnn_4c_1f':
        model = models.build_cnn_4c_1f(input_shape=(num_rows, num_columns, num_channels),
                                       filters=filters, dropouts=dropouts)
    elif model_name == 'build_cnn_5c_1f':
        model = models.build_cnn_5c_1f(input_shape=(num_rows, num_columns, num_channels),
                                       filters=filters)
    elif model_name == 'build_crnn_3c_2r':
        model = models.build_crnn_3c_2r(input_shape=(num_rows, num_columns, num_channels),
                                        filters=filters)
    elif model_name == 'build_crnn_4c_lstm':
        model = models.build_crnn_4c_lstm(input_shape=(num_rows, num_columns, num_channels),
                                          filters=filters)
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.Precision(name='precision')],
                  optimizer=tf.keras.optimizers.Adam(lr=lr))
    return model


def train_model():
    ''' Trains the model:
    loads subdatasets, runs get_compiled_model(),
    creates n models and fit them, saves the history of every model with ht.save_history(),
    evaluate with test and validation set after training

    Returns
    -------
    histories : list of keras history objects
    '''
    histories = []
    # load labels and inputs
    x_train = np.load('inputs/x_train.npy', allow_pickle=True)
    y_train = np.load('inputs/y_train.npy', allow_pickle=True)
    x_val = np.load('inputs/x_val.npy', allow_pickle=True)
    y_val = np.load('inputs/y_val.npy', allow_pickle=True)
    x_test = np.load('inputs/x_test.npy', allow_pickle=True)
    y_test = np.load('inputs/y_test.npy', allow_pickle=True)
    num_rows = x_train.shape[1]
    num_columns = x_train.shape[2]
    
    # build in every loop a new model and trains it
    for i in range(start, end):
        weights_path_loss = 'weights/loss/' + filename + '_' + str(i)
        
        # initialize callbacks
        checkpoint_loss_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=weights_path_loss,
            save_weights_only=True,
            monitor='val_loss',
            mode='min',
            verbose=0,
            save_best_only=True)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=min_lr)
        time_callback = Timing_Callback()
        
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
        # build models in the MirroredStrategy
        with strategy.scope():
            model = get_compiled_model(num_rows, num_columns)
            
        # train the model
        history = model.fit(
            x_train, y_train,
            verbose=2,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            callbacks=[time_callback, checkpoint_loss_callback, reduce_lr])  # checkpoint_loss_callback
        history.history['traintime'] = time_callback.times
        model.load_weights(weights_path_loss)
       
        # evaluate
        eval_val = model.evaluate(x_val, y_val, batch_size=batch_size)
        eval_test = model.evaluate(x_test, y_test, batch_size=batch_size)
        history.history['eval_val'] = eval_val
        history.history['eval_test'] = eval_test
        ht.save_history(history, filename + '_' + str(i))  # saves the history object of the trained model
        histories.append(history)
    return histories


# training
train_model()
