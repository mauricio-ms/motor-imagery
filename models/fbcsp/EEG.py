import numpy as np


class EEG:
    def __init__(self, left_data, right_data, training=True):
        self.left_data = left_data
        self.right_data = right_data
        self.training = training
        self.labels = np.concatenate((np.zeros(self.left_data.shape[0]), np.ones(self.right_data.shape[0])))
