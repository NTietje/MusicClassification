import numpy as np
import models_dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import tensorflow as tf

# folder of saved weights
folder = 'loss'

''' best CNN model configurations '''
filename = 'gpu3_cnn_drop_best_200_12'
title = 'Confusion-Matrix: Best C200'
model_name = 'build_cnn_4c_1f'
lr = 0.001
min_lr = 0.0001
dropouts = [0.0, 0.2]

''' best CRNN model configurations '''
# filename = 'gpu3_cRnn_drop_best_3_100_7'
# title = 'Confusion-Matrix: Best CR100'
# model_name = 'build_crnn_4c_2r'
# lr = 0.00025
# min_lr = 0.0001
# dropouts = [0.0, 0.3, 0.0]


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
    # Create model
    model = None
    num_channels = 1
    if model_name == 'build_cnn_4c_1f':  # CNN model
        filters = [(32), (64), (96), (128)]  # 0.2 mio parameters
        model = models_dropout.build_cnn_4c_1f(input_shape=(num_rows, num_columns, num_channels),
                                                   filters=filters, dropouts=dropouts)
    elif model_name == 'build_crnn_4c_2r':  # CRNN model
        filters = [(30), (60), (60), (60), (30), (30)]  # 0.1 mio parameters
        model = models_dropout.build_crnn_4c_2r(input_shape=(num_rows, num_columns, num_channels),
                                                    filters=filters, dropouts=dropouts)
    # Compile model
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.Precision(name='precision')],
                  optimizer=tf.keras.optimizers.Adam(lr=lr))
    return model


def predict(filename):
    """Returns the prediction and the labels.
    Prediction is calculated with keras model.predict().

    Parameters
    ----------
    filename : string
        filename of the model weights

    Returns
    -------
    groundtruth : float array
        labels as floats
    prediction : float array
        predictions as floats
    """
    weights_path = 'weights/' + folder + '/' + filename
    model = get_compiled_model(num_rows, num_columns)
    model.load_weights(weights_path)
    prediction = model.predict(inputs)
    groundtruth = one_hot.argmax(axis=1)
    prediction = prediction.argmax(axis=1)
    return groundtruth, prediction


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None):
    """
    This function returns the confusion matrix plot object.
    Normalization can be applied by setting `normalize=True`.

    Parameters
    ----------
    y_true : float array
        labels
    y_pred : float array
        predictions
    classes : str list
        classnames shown in the confusion matrix
    normalize : bool
        normalize the values (percent)
    title : str
        title of the confusion matrix, shown on top

    Returns
    -------
    ax : matplotlib axes object
        contains the confusion matrix values, graphic plot
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    cm = confusion_matrix(y_true, y_pred)  # Compute confusion matrix with function of scikit library
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",  # Rotate the tick labels and set their alignment.
             rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'  # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# Classnames to show in the confusion matrix
class_names = ['Blues', 'Classical', 'Country', 'Disco', 'Hip-Hop', 'Jazz', 'Metal', 'Pop', 'Reggae', 'Rock']
np.set_printoptions(precision=2)  # two decimal places

# Load inputs and labels
x_val = np.load('inputs/x_val.npy', allow_pickle=True)
y_val = np.load('inputs/y_val.npy', allow_pickle=True)
x_test = np.load('inputs/x_test.npy', allow_pickle=True)
y_test = np.load('inputs/y_test.npy', allow_pickle=True)

# merge both datasets
inputs = np.append(x_val, x_test, axis=0)
one_hot = np.append(y_val, y_test, axis=0)

# Dimensions for model building
num_rows = inputs.shape[1]
num_columns = inputs.shape[2]
y_groundtruth, y_prediction = predict(filename)

# Plot normalized confusion matrix
plot_confusion_matrix(y_groundtruth, y_prediction, classes=class_names, normalize=True, title=title)
plt.savefig('matrix/' + filename + '-confu.png')  # Save confusion matrix as PNG
plt.show()
