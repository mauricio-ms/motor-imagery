import numpy as np
import pandas as pd

from main import ROOT_DIR


def identity(x):
    return x


def load_csv(file_path):
    return pd.read_csv(ROOT_DIR + "/" + file_path, header=None)


def window_apply(df, mapper, window_size, step):
    results = []

    for x in range(0, df.shape[0], step):
        end_index_window = x + window_size - 1 if x+window_size-1 <= df.shape[0] else df.shape[0]
        window = df[x:end_index_window+1]
        if window.shape[0] == window_size:
            results = results + [mapper(window.values)]

    return np.squeeze(results)


def extract_single_trial(eeg, trial_length):
    return window_apply(eeg, identity, trial_length, trial_length)
