import pickle
import numpy as np

def unpickle(file):
    # unpickles cifar10 dataset
    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


def getCIFAR10(direc, filename, batches):
    # Converts the data in batches to a full training set
    for j in range(1, batches + 1):
        file = direc + filename + str(j)
        dic = unpickle(file)
        if j == 1:
            X_train = dic[b'data']
            y_train = dic[b'labels']
        else:
            temp_X = dic[b'data']
            temp_y = dic[b'labels']
            X_train = np.concatenate((X_train, temp_X))
            y_train = np.concatenate((y_train, temp_y))
    return X_train, y_train

