from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, ELU, Reshape, Dense, Conv2D, \
    Dropout, Flatten, MaxPooling2D, BatchNormalization

'''
This script contains all used keras models.
'''

# The input shape, the number of classes, padding and kernel size are the same for every model.
input_shape = (96, 1360, 1)  # shape of the mel-spectrograms
n_classes = 10  # 10 music genres
padding = 'same'
kernel_size = (3, 3)

'''
Every network-function expects a tuble list as pooling and a scalar list for filters. The Length depends on 
the model architecture. All attributes have a default value.

All methods return a keras model.
'''


def build_cnn_5c_1f(input_shape=input_shape,
                    n_classes=n_classes,
                    poolings=[(2, 4), (2, 4), (2, 4), (3, 5), (4, 4)],
                    filters=[(47), (95), (95), (142), (190)]):
    # initialize model
    model = Sequential(name='CNN_5C')
    # Block 1
    model.add(Conv2D(input_shape=input_shape, filters=filters[0], kernel_size=kernel_size,
                     padding=padding, name='conv1'))
    model.add(BatchNormalization(name='norm1', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[0], name='pooling1'))
    # Block 2
    model.add(Conv2D(filters=filters[1], kernel_size=kernel_size, padding=padding, name='conv2'))
    model.add(BatchNormalization(name='norm2', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[1], name='pooling2'))
    # Block 3
    model.add(Conv2D(filters=filters[2], kernel_size=kernel_size, padding=padding, name='conv3'))
    model.add(BatchNormalization(name='norm3', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[2], name='pooling3'))
    # Block 4
    model.add(Conv2D(filters=filters[3], kernel_size=kernel_size, padding=padding, name='conv4'))
    model.add(BatchNormalization(name='norm4', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[3], name='pooling4'))
    # Block 5
    model.add(Conv2D(filters=filters[4], kernel_size=kernel_size, padding=padding, name='conv5'))
    model.add(BatchNormalization(name='norm5', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[4], name='pooling5'))
    # Flatten
    model.add(Flatten(name='flatten'))
    # Softmax output
    model.add(Dense(units=n_classes, name='softmax_output', activation='softmax'))
    return model


def build_cnn_4c_1f(input_shape=input_shape,
                    n_classes=n_classes,
                    poolings=[(2, 4), (2, 4), (3, 5), (4, 4)],
                    filters=[(47), (95), (95), (142)],
                    dropouts=[0.0, 0,0]):
    # initialize model
    model = Sequential(name='CNN_4C')
    # Block 1
    model.add(Conv2D(input_shape=input_shape, filters=filters[0], kernel_size=kernel_size,
                     padding=padding, name='conv1'))
    model.add(BatchNormalization(name='norm1', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[0], name='pooling1'))
    if dropouts[0] > 0:
        model.add(Dropout(dropouts[0]))
    # Block 2
    model.add(Conv2D(filters=filters[1], kernel_size=kernel_size, padding=padding, name='conv2'))
    model.add(BatchNormalization(name='norm2', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[1], name='pooling2'))
    if dropouts[0] > 0:
        model.add(Dropout(dropouts[0]))
    # Block 3
    model.add(Conv2D(filters=filters[2], kernel_size=kernel_size, padding=padding, name='conv3'))
    model.add(BatchNormalization(name='norm3', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[2], name='pooling3'))
    if dropouts[0] > 0:
        model.add(Dropout(dropouts[0]))
    # Block 4
    model.add(Conv2D(filters=filters[3], kernel_size=kernel_size, padding=padding, name='conv4'))
    model.add(BatchNormalization(name='norm4', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[3], name='pooling4'))
    if dropouts[1] > 0:
        model.add(Dropout(dropouts[1]))
    # Flatten
    model.add(Flatten(name='flatten'))
    # Softmax output
    model.add(Dense(units=n_classes, name='softmax_output', activation='softmax'))
    return model


def build_crnn_3c_2r(input_shape=input_shape,
                     n_classes=n_classes,
                     poolings=[(3, 3), (4, 4), (8, 8)],
                     filters=[(68), (137), (137), (68), (68)]):
    # Model initialisieren
    model = Sequential(name='CRNN_3C')
    # Block 1
    model.add(Conv2D(input_shape=input_shape, filters=filters[0], kernel_size=kernel_size,
                     padding=padding, name='conv1'))
    model.add(BatchNormalization(name='norm1', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[0], name='pooling1'))
    # Block 2
    model.add(Conv2D(filters=filters[1], kernel_size=kernel_size, padding=padding, name='conv2'))
    model.add(BatchNormalization(name='norm2', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[1], name='pooling2'))
    # Block 3
    model.add(Conv2D(filters=filters[2], kernel_size=kernel_size, padding=padding, name='conv3'))
    model.add(BatchNormalization(name='norm3', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[2], name='pooling3'))
    model.add(Reshape((14, filters[2])))
    # RNN
    model.add(GRU(filters[3], return_sequences=True, name='gru1'))
    model.add(GRU(filters[4], return_sequences=False, name='gru2'))
    model.add(Dense(units=n_classes, name='softmax_output', activation='softmax'))
    return model


def build_crnn_4c_2r(input_shape=input_shape,
                     n_classes=n_classes,
                     poolings=[(2, 2), (3, 3), (4, 4), (4, 4)],
                     filters=[(68), (137), (137), (137), (68), (68)],
                     dropouts=[0.0, 0,0, 0,0]):
    # Model initialisieren
    model = Sequential(name='CRNN_4C')
    # Block 1
    model.add(Conv2D(input_shape=input_shape, filters=filters[0], kernel_size=kernel_size,
                     padding=padding, name='conv1'))
    model.add(BatchNormalization(name='norm1', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[0], name='pooling1'))
    if dropouts[0] > 0:
        model.add(Dropout(dropouts[0], name='dropout1'))
    # Block 2
    model.add(Conv2D(filters=filters[1], kernel_size=kernel_size, padding=padding, name='conv2'))
    model.add(BatchNormalization(name='norm2', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[1], name='pooling2'))
    if dropouts[0] > 0:
        model.add(Dropout(dropouts[0], name='dropout2'))
    # Block 3
    model.add(Conv2D(filters=filters[2], kernel_size=kernel_size, padding=padding, name='conv3'))
    model.add(BatchNormalization(name='norm3', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[2], name='pooling3'))
    if dropouts[0] > 0:
        model.add(Dropout(dropouts[0], name='dropout2'))
    # Block 4
    model.add(Conv2D(filters=filters[3], kernel_size=kernel_size, padding=padding, name='conv4'))
    model.add(BatchNormalization(name='norm4', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[3], name='pooling4'))
    if dropouts[1] > 0:
        model.add(Dropout(dropouts[1], name='dropout4'))
    # Reshaping
    model.add(Reshape((14, filters[3])))
    # RNN
    model.add(GRU(filters[4], return_sequences=True, name='gru1'))
    model.add(GRU(filters[5], return_sequences=False, name='gru2'))
    if dropouts[2] > 0:
        model.add(Dropout(dropouts[2]))
    # Softmax output
    model.add(Dense(units=n_classes, name='softmax_output', activation='softmax'))
    return model


def build_crnn_4c_lstm(input_shape=input_shape,
                     n_classes=n_classes,
                     poolings=[(2, 2), (3, 3), (4, 4), (4, 4)],
                     filters=[(68), (137), (137), (137), (68), (68)]):
    # Model initialisieren
    model = Sequential(name='CRNN_LSTM')
    # Block 1
    model.add(Conv2D(input_shape=input_shape, filters=filters[0], kernel_size=kernel_size,
                     padding=padding, name='conv1'))
    model.add(BatchNormalization(name='norm1', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[0], name='pooling1'))
    # Block 2
    model.add(Conv2D(filters=filters[1], kernel_size=kernel_size, padding=padding, name='conv2'))
    model.add(BatchNormalization(name='norm2', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[1], name='pooling2'))
    # Block 3
    model.add(Conv2D(filters=filters[2], kernel_size=kernel_size, padding=padding, name='conv3'))
    model.add(BatchNormalization(name='norm3', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[2], name='pooling3'))
    # Block 4
    model.add(Conv2D(filters=filters[3], kernel_size=kernel_size, padding=padding, name='conv4'))
    model.add(BatchNormalization(name='norm4', axis=1))
    model.add(ELU())
    model.add(MaxPooling2D(pool_size=poolings[3], name='pooling4'))
    # Reshaping
    model.add(Reshape((14, filters[3])))
    # RNN
    model.add(LSTM(filters[4], return_sequences=True, name='lstm1'))
    model.add(LSTM(filters[5], return_sequences=False, name='lstm2'))
    # Softmax output
    model.add(Dense(units=n_classes, name='softmax_output', activation='softmax'))
    return model


