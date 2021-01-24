import numpy as np

def process_tensor(X):
    # X[np.isnan(X)] = 0.0
    # X[np.isposinf(X)] = 3.4e37
    # X[np.isneginf(X)] = -1.2e-37

    return X