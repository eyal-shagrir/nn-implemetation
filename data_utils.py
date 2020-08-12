import numpy as np
import scipy.io as sio

DATA_FOLDER = 'NNdata'
DATA_SETS = ['GMM', 'Peaks', 'SwissRoll']
DATA_FILES = [''.join([data_set, 'Data.mat']) for data_set in DATA_SETS]


def get_data(data_set):
    if data_set not in DATA_SETS:
        return None
    data_set_path = '/'.join([DATA_FOLDER, ''.join([data_set, 'Data.mat'])])
    data = sio.loadmat(data_set_path)
    X_tr = data['Yt'].astype('float')
    y_tr = data['Ct'].T.astype('float')
    X_te = data['Yv'].astype('float')
    y_te = data['Cv'].T.astype('float')
    return X_tr, y_tr, X_te, y_te


def normalize_data(X_tr, X_te):
    """
    :param X_tr: training data matrix of size n×m1
    :param X_te: testing data matrix of size n×m2
    :return: the original data minus the mean data unit
    """
    mean = np.mean(X_tr, axis=1).reshape(X_tr.shape[0], 1)
    X_tr -= mean
    X_te -= mean


def get_as_vec(M, order='F'):
    flattened = M.flatten(order=order)
    vec = flattened.reshape(flattened.shape[0], 1)
    return vec


def shuffle_data(X, y):
    n, m = X.shape
    l = y.shape[1]
    random_indices = np.arange(m).reshape(m, 1)
    np.random.shuffle(random_indices)
    new_X = X[np.arange(n), random_indices].T
    new_y = y[random_indices, np.arange(l)]
    return new_X, new_y


def get_mini_batches(X, y, mb_num):
    X, y = shuffle_data(X, y)
    mbs = np.array_split(X, mb_num, axis=1)
    mbs_labels = np.array_split(y, mb_num, axis=0)
    return mbs, mbs_labels