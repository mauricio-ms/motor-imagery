from mne.decoding import CSP
from signal_processing import filter_bank

import numpy as np


class FilterBankCSPFeatureExtraction:
    """
    Class responsible to extract the features based on the raw EEG data
    using the algorithm filter bank CSP

    Attributes
    ----------
    training : bool
        If the data is training
    y : 1-d array
        The labels of the raw data (only when training is True)
    csp : CSP
        The MNE Common Spatial Pattern object
    m : int
        The dimension of a single feature, in this case, is the number of relevant CSP features used
    features : n_features-d array
        The features extracted by the filter-bank CSP algorithm
    n_features : int
        The number of features extracted by the filter-bank CSP algorithm
    """
    def __init__(self, eeg, n_components=2, csp_by_band=None):
        """
        Parameters
        ----------
        n_components : int
            The number of components to use as relevant features in the CSP algorithm
        eeg : Eeg
            The Eeg object that contains the raw eeg data separated by class (left and right)
        """
        left_bands_data = filter_bank(eeg.left_data)
        right_bands_data = filter_bank(eeg.right_data)

        self.n_components = n_components
        self.__bands = range(left_bands_data.shape[0])
        if csp_by_band is None:
            self.csp_by_band = [CSP(n_components=self.n_components, reg=None, log=True, norm_trace=False)
                                for _ in self.__bands]
        else:
            self.csp_by_band = csp_by_band

        self.training = eeg.training
        self.y = eeg.labels
        self.features = self.extract_features(left_bands_data, right_bands_data)
        self.n_features = self.features.shape[1]

    def extract_features(self, left_bands, right_bands):
        print("Extracting features ...")
        features = None
        for n_band in self.__bands:
            print("Band ", n_band + 1)
            left_band_training = left_bands[n_band]
            right_band_training = right_bands[n_band]

            x = np.concatenate((left_band_training, right_band_training))

            # Reshape to the format expected by MNE Library
            x = np.transpose(x, [0, 2, 1])
            csp = self.csp_by_band[n_band]

            if n_band == 0:
                features = self.compute_features(csp, x)
            else:
                features = np.concatenate((features, self.compute_features(csp, x)), axis=1)

        return features

    def compute_features(self, csp, x):
        return csp.fit_transform(x, self.y) if self.training else csp.transform(x)
