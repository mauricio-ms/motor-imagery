from data_preparation import load_csv, epoch, extract_single_trial

import numpy as np


class EEG:
    def __init__(self, left_data_file, right_data_file, time_window, training=True, epoch_size=None):
        if epoch_size is not None:
            self.left_data = epoch(extract_single_trial(load_csv(left_data_file), time_window), epoch_size)
            self.right_data = epoch(extract_single_trial(load_csv(right_data_file), time_window), epoch_size)
        else:
            self.left_data = extract_single_trial(load_csv(left_data_file), time_window)
            self.right_data = extract_single_trial(load_csv(right_data_file), time_window)

        self.training = training
        self.labels = np.concatenate((np.zeros(self.left_data.shape[0]),
                                      np.ones(self.right_data.shape[0])))

    def __init__(self, left_data, right_data, training=True):
        self.left_data = left_data
        self.right_data = right_data
        self.training = training
        self.labels = np.concatenate((np.zeros(self.left_data.shape[0]),
                                      np.ones(self.right_data.shape[0])))
