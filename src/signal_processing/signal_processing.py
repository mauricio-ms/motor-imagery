import numpy as np
from scipy import signal

FS = 250
NYQUIST = .5 * FS

LOW_FREQS = range(4, 41, 4)
QUANTITY_BANDS = len(LOW_FREQS)-1


def filter_in_all_frequency_bands(trial):
    filtered_signals = np.zeros((QUANTITY_BANDS, *trial.shape))
    for n_low_freq in range(0, len(LOW_FREQS)):
        low_freq = LOW_FREQS[n_low_freq]
        if low_freq == LOW_FREQS[-1]:
            break

        high_freq = LOW_FREQS[n_low_freq+1]

        # Create a 5 order Chebyshev Type 2 filter to the specific band (low_freq - high_freq)
        b, a = signal.cheby2(5, 0.5, [low_freq, high_freq], btype="bandpass", fs=250)

        filtered_signals[n_low_freq, :, :] = signal.filtfilt(b, a, trial, axis=0)

    return filtered_signals


def filter_bank(eeg):
    filtered_signals = np.zeros((QUANTITY_BANDS, *eeg.shape))
    for n_trial in range(eeg.shape[0]):
        trial = eeg[n_trial, :, :]
        filtered_signals[:, n_trial, :, :] = filter_in_all_frequency_bands(trial)

    return filtered_signals


def apply_bandpass_filter(eeg):
    eeg.left_data = bandpass_filter(eeg.left_data)
    eeg.right_data = bandpass_filter(eeg.right_data)


def bandpass_filter(eeg_signal):
    b, a = signal.butter(4, [8, 30], btype="bandpass", fs=250)
    return signal.filtfilt(b, a, eeg_signal, axis=1)

import time
def cross_correlation_sequence(x, y):
    start_time = time.time()
    signal_len = len(x)
    r = [cross_correlation(x, y, m)
            for m in range(-(signal_len - 1), signal_len)]
    # print("--- %s seconds ---" % (time.time() - start_time))
    return r

import pandas as pd
def cross_correlation(x, y, m):
    """
    Parameters
    ----------
    x: time-samples array
        Signal x
    y: time-samples array
        Signal y
    m: int
        The lag parameter
    Returns
    -------
    The cross-correlation between the signals x and y
    """
    # return np.sum([x[i]*y[i-m] for i in range(0, len(x) - np.abs(m))])
    m = np.abs(m)
    cc_len = len(x) - m
    # return np.sum(np.dot(x[0:cc_len], np.roll(y, -m)[0:cc_len]))
    # return x.corr(y.shift(m))
    return np.sum(np.dot(x[0:cc_len], shift(y, -m)[0:cc_len]))


def shift(arr, num, fill_value=np.nan):
    result = np.empty_like(arr)
    if num > 0:
        # result[:num] = fill_value
        result[num:] = arr[:-num]
    elif num < 0:
        # result[num:] = fill_value
        result[:num] = arr[-num:]
    else:
        result[:] = arr
    return result

