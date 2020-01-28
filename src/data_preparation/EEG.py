import numpy as np


class EEG:
    def __init__(self, left_data, right_data, training=True):
        self.left_data = left_data
        self.right_data = right_data
        self.X = np.concatenate((self.left_data, self.right_data))
        self.training = training
        self.n_channels = self.left_data.shape[2]
        self.n_left_trials = self.left_data.shape[0]
        self.n_right_trials = self.right_data.shape[0]
        self.labels = np.concatenate((np.zeros(self.n_left_trials), np.ones(self.n_right_trials)))
        self.n_trials = len(self.labels)
