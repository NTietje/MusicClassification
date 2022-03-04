import numpy as np
from sklearn.model_selection import train_test_split

'''standardize the spectrograms to std, norm[-1,1] and norm[0,1]
for the comparison with no standardization (none)
'''

# loads inouts and labels
y = np.load('groundtruth/one_hot_gtzan.npy', allow_pickle=True)
x = np.load('inputs/gtzan_29s_12khz_ref1_512_256_96.npy', allow_pickle=True)

# global data subsets
x_train = None
y_train = None
x_val = None
y_val = None
x_test = None
y_test = None


# Split dataset in train, validation and test subset
def split():
    global x_train, x_test, x_val, y_train, y_test, y_val
    # Split in test validation and train set
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=476)
    # stratify=destination-matrix to balance genres in the subsets
    # Second split to get test- and trainset out of y_train and y_test
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=0.5, stratify=y_test, random_state=21)


# Standarize all specs in an array to mean 0 and variance 1
def std(arrays, norm_mean, norm_std):
    standarized = (arrays - norm_mean) / norm_std
    return standarized


# Standarize all x subsets
def std_all():
    global x_train
    global x_val
    global x_test
    norm_mean = x_train.mean()
    norm_std = x_train.std()
    x_train = std(x_train, norm_mean, norm_std)
    x_val = std(x_val, norm_mean, norm_std)
    x_test = std(x_test, norm_mean, norm_std)


# Normalize all subsets
def norm_all(lo_out, hi_out):
    global x_train
    global x_val
    global x_test
    datasets = [x_train, x_val, x_test]
    # scaling based on train subset
    lo_in = np.amin(x_train)  # minimum
    hi_in = np.amax(x_train)  # maximum
    range_out = hi_out - lo_out
    range_in = hi_in - lo_in
    # scale values between lo_out and hi_out, for each subset
    for i in range(len(datasets)):
        datasets[i] = ((datasets[i] - lo_in) / range_in) * range_out + lo_out
    # save in global datasets
    x_train = datasets[0]
    x_val = datasets[1]
    x_test = datasets[2]


# Reshape input sizes to the expected dimensions for the network
def reshape():
    global x_train
    global x_val
    global x_test
    global num_rows
    global num_columns
    global num_channels
    num_rows = x.shape[1]
    num_columns = x.shape[2]
    x_train = x_train.reshape(x_train.shape[0], num_rows, num_columns, num_channels)
    x_val = x_val.reshape(x_val.shape[0], num_rows, num_columns, num_channels)
    x_test = x_test.reshape(x_test.shape[0], num_rows, num_columns, num_channels)


# arrays to float, expected by the networks
def to_float():
    global x_train
    global x_val
    global x_test
    global y_train
    global y_val
    global y_test

    x_train = np.asarray(x_train).astype('float32')
    y_train = np.asarray(y_train).astype('float32')
    x_val = np.asarray(x_val).astype('float32')
    y_val = np.asarray(y_val).astype('float32')
    x_test = np.asarray(x_test).astype('float32')
    y_test = np.asarray(y_test).astype('float32')


'''std'''
# split()
# std_all()
# reshape()
# to_float()

'''norm[-1,1]'''
# split()
# norm_all(-1, 1)
# reshape()
# to_float()

'''norm[0,1]'''
split()
norm_all(0, 1)
reshape()
to_float()

'''none'''
# split()
# reshape()
# to_float()


