import numpy as np
from scipy import linalg


class CSP:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.__m = n_components//2
        self.fitted = False
        self.left_data = None
        self.right_data = None
        self.W_b = None
        self.W_b_t = None

    def fit(self, left_data, right_data):
        self.left_data = left_data
        self.right_data = right_data
        self.W_b = self.__compute_transformation_matrix()
        self.W_b_t = np.transpose(self.W_b)
        self.fitted = True

    def __compute_transformation_matrix(self):
        # TODO: test with param average_trial_covariance

        # To compute the covariance matrix, is necessary 2-D matrices (channels, observations)
        # So, here we concatenate the time and trials
        s1 = np.cov(np.transpose(np.concatenate(self.left_data, axis=0)))
        s2 = np.cov(np.transpose(np.concatenate(self.right_data, axis=0)))

        w, v = linalg.eigh(s1, s1+s2)
        return select_cols(v, self.__m)

    def compute_features(self, eeg):
        if not self.fitted:
            raise Exception("The model has not yet been fit.")
        return [self.__compute_features_to_single_trial(trial)
                for trial in eeg]

    def __compute_features_to_single_trial(self, trial):
        product = np.dot(np.dot(self.W_b_t, np.transpose(trial)), np.dot(trial, self.W_b))
        return np.log(np.divide(np.diag(product), np.sum(product)))


# TODO move this function to an helper appropriated
def select_cols(w, m):
    n_cols = w.shape[1]
    return w[:, [*range(0, m), *range(n_cols-m, w.shape[1])]]
