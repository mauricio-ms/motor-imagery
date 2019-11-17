from main import ROOT_DIR
from EEG import EEG

import numpy as np
import pandas as pd


def read_eeg_files(path_files, time_length, time_window, epoch_size=None):
    left_data = None
    right_data = None
    for left_data_file, right_data_file in path_files:
        next_data = __read_eeg_file(left_data_file, right_data_file, time_length, time_window, epoch_size)

        if left_data is None:
            left_data, right_data = next_data
        else:
            next_left_data, next_right_data = next_data
            left_data = np.concatenate((left_data, next_left_data))
            right_data = np.concatenate((right_data, next_right_data))

    return EEG(left_data, right_data)


def read_eeg_file(left_data_file, right_data_file, time_length, time_window, training=True, epoch_size=None):
    return EEG(*(__read_eeg_file(left_data_file, right_data_file, time_length, time_window, epoch_size)), training)


def __read_eeg_file(left_data_file, right_data_file, time_length, time_window, epoch_size=None):
    # Read the data
    left_data = __extract_single_trial(__load_csv(left_data_file), time_length, time_window)
    right_data = __extract_single_trial(__load_csv(right_data_file), time_length, time_window)

    # Read the epoch data
    if epoch_size is not None:
        left_data = __epoch(left_data, epoch_size)
        right_data = __epoch(right_data, epoch_size)

    return left_data, right_data


def __load_csv(file_path):
    return pd.read_csv(ROOT_DIR + "/" + file_path, header=None)


def __epoch(eeg, size):
    data = None
    for trial in eeg:
        single_epoch = __window_apply(pd.DataFrame(trial), __identity, size, size//2)
        if data is None:
            data = single_epoch
        else:
            data = np.concatenate((data, single_epoch))

    return data


def __extract_single_trial(eeg, trial_length, trial_length_to_extract=None):
    if trial_length_to_extract is None:
        trial_length_to_extract = trial_length
    return __window_apply(eeg, __identity, trial_length_to_extract, trial_length)


def __window_apply(df, mapper, window_size, step):
    results = []

    for x in range(0, df.shape[0], step):
        end_index_window = x + window_size - 1 if x+window_size-1 <= df.shape[0] else df.shape[0]
        window = df[x:end_index_window+1]
        if window.shape[0] == window_size:
            results = results + [mapper(window.values)]

    return np.squeeze(results)


def __identity(x):
    return x
