import numpy as np
from scipy import linalg


class CSP:
    def __init__(self, left_data, right_data, m=1):
        self.left_data = left_data
        self.right_data = right_data
        self.m = m
        self.n_components = m*2
        self.W_b = self.compute_transformation_matrix()
        self.W_b_t = np.transpose(self.W_b)

    def compute_transformation_matrix(self):
        # TODO: test with param average_trial_covariance

        # To compute the covariance matrix, is necessary 2-D matrices (channels, observations)
        # So, here we concatenate the time and trials
        s1 = np.cov(np.transpose(np.concatenate(self.left_data, axis=0)))
        s2 = np.cov(np.transpose(np.concatenate(self.right_data, axis=0)))

        w, v = linalg.eigh(s1, s1+s2)
        return select_cols(v, self.m)

    def compute_features(self, eeg):
        return [self.compute_features_to_single_trial(trial)
                for trial in eeg]

    def compute_features_to_single_trial(self, trial):
        product = np.dot(np.dot(self.W_b_t, np.transpose(trial)), np.dot(trial, self.W_b))
        return np.log(np.divide(np.diag(product), np.sum(product)))


# TODO move this function to an helper appropriated
def select_cols(w, m):
    n_cols = w.shape[1]
    return w[:, [*range(0, m), *range(n_cols-m, w.shape[1])]]
