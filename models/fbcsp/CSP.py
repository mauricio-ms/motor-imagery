import numpy as np
from scipy import linalg


class CSP:
    def __init__(self, average_trial_covariance=False, n_components=2):
        self.average_trial_covariance = average_trial_covariance
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
        if self.average_trial_covariance:
            s1 = np.mean([np.cov(np.transpose(trial)) for trial in self.left_data], axis=0)
            s2 = np.mean([np.cov(np.transpose(trial)) for trial in self.right_data], axis=0)
        else:
            # To compute the covariance matrix, is necessary 2-D matrices (channels, observations)
            # So, here we concatenate the time and trials
            s1 = np.cov(np.transpose(np.concatenate(self.left_data, axis=0)))
            s2 = np.cov(np.transpose(np.concatenate(self.right_data, axis=0)))

        w, v = linalg.eigh(s1, s1+s2)

        # CSP requires the eigenvalues and the eig-vectors be sorted in descending order
        order_mask = np.argsort(w)
        order_mask = order_mask[::-1]

        v = v[:, order_mask]

        return select_cols(v, self.__m)

    def compute_features(self, eeg):
        if not self.fitted:
            raise Exception("The model has not yet been fit.")
        return [self.__compute_features_to_single_trial(trial)
                for trial in eeg]

    def __compute_features_to_single_trial(self, trial):
        z = np.dot(trial, self.W_b)
        return np.log(np.var(z, axis=0))


# TODO move this function to an helper appropriated
def select_cols(w, m):
    n_cols = w.shape[1]
    return w[:, [*range(0, m), *range(n_cols-m, w.shape[1])]]
