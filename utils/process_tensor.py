import numpy as np

def process_tensor(X):
    X[np.isnan(X)] = 0.0
    X[np.isposinf(X)] = 3.4e37
    X[np.isneginf(X)] = -1.2e-37
    return X

def padding(X, padding):
    pad_list = [(padding[0],),(padding[1],),(0,),(0,)]
    return np.pad(X,pad_list,'constant')