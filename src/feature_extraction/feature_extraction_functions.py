import numpy as np


def log_variance(x, w):
    z = np.dot(x, w)
    return np.log(np.divide(np.var(z, axis=0), np.sum(np.var(z, axis=0))))


def log_band_power(x, w):
    z = np.dot(x, w)
    return np.log(1/z.shape[1] * np.sum(np.abs(z)**2, axis=0))
