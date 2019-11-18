import numpy as np
from scipy import linalg
from models.utils.array_helper import select_cols
from models.feature_extraction.feature_extraction_functions import log_variance


class CSP:
    def __init__(self, average_trial_covariance=False, n_components=2, compute_features_to_single_trial=log_variance):
        self.average_trial_covariance = average_trial_covariance
        self.n_components = n_components
        self.__m = n_components//2
        self.compute_features_to_single_trial = compute_features_to_single_trial
        self.fitted = False
        self.left_data = None
        self.right_data = None
        self.W = None

    def fit(self, left_data, right_data):
        self.left_data = left_data
        self.right_data = right_data
        self.W = self.__compute_transformation_matrix()
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

        w, v = linalg.eigh(s2, s1+s2)

        # CSP requires the eigenvalues and the eig-vectors be sorted in descending order
        order_mask = np.argsort(w)
        order_mask = order_mask[::-1]

        v = v[:, order_mask]

        return select_cols(v, self.__m)

    def project(self, trial):
        if not self.fitted:
            raise Exception("The model has not yet been fit.")
        return np.dot(trial, self.W)

    def compute_features(self, eeg):
        if not self.fitted:
            raise Exception("The model has not yet been fit.")
        return [self.compute_features_to_single_trial(trial, self.W)
                for trial in eeg]
