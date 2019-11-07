from signal_processing import filter_bank, QUANTITY_BANDS
from CSP import CSP

import numpy as np


class FilterBankCSPFeatureExtraction:
    """
    Class responsible to extract the features based on the raw EEG data
    using the algorithm filter bank CSP
    Attributes
    ----------
    n_components : int
        The number of components to use as relevant features in the CSP algorithm
    training_features : n_features-d array
        The training features extracted by the filter-bank CSP algorithm
    test_features : n_features-d array
        The test features extracted by the filter-bank CSP algorithm
    n_features : int
        The number of features extracted by the filter-bank CSP algorithm
    """
    def __init__(self, eeg_training, eeg_test, n_components=2):
        """
        Parameters
        ----------
        n_components : int
            The number of components to use as relevant features in the CSP algorithm
        eeg_training : Eeg
            The Eeg object that contains the raw training eeg data separated by class (left and right)
        eeg_test : Eeg
            The Eeg object that contains the raw test eeg data separated by class (left and right)
        """
        self.n_components = n_components
        self.__bands = range(QUANTITY_BANDS)
        self.__csp_by_band = [CSP(average_trial_covariance=False, n_components=self.n_components)
                              for _ in self.__bands]
        self.training_labels = eeg_training.labels
        self.test_labels = eeg_test.labels
        self.training_features = self.extract_features(eeg_training)
        self.test_features = self.extract_features(eeg_test)
        self.n_features = self.training_features.shape[1]
        if self.n_features != self.test_features.shape[1]:
            raise Exception("The number of features extracted from the training and test dataset's should be equal")

    def extract_features(self, eeg):
        left_bands = filter_bank(eeg.left_data)
        right_bands = filter_bank(eeg.right_data)

        features = None
        for n_band in self.__bands:
            left_band_training = left_bands[n_band]
            right_band_training = right_bands[n_band]

            x = np.concatenate((left_band_training, right_band_training))
            csp = self.__csp_by_band[n_band]
            if eeg.training:
                csp.fit(left_band_training, right_band_training)

            if n_band == 0:
                features = csp.compute_features(x)
            else:
                features = np.concatenate((features, csp.compute_features(x)), axis=1)

        return features
