from data_preparation import load_csv, extract_single_trial

import numpy as np


class EEG:
    def __init__(self, left_data_file, right_data_file, time_window, training=True):
        self.left_data = extract_single_trial(load_csv(left_data_file), time_window)
        self.right_data = extract_single_trial(load_csv(right_data_file), time_window)
        self.training = training
        self.labels = np.concatenate((np.zeros(self.left_data.shape[0]),
                                      np.ones(self.right_data.shape[0])))
