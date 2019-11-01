from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing

import numpy as np


class MIBIFFeatureSelection2:
    """
    Class responsible to select the best features
    using the MIBIF algorithm

    Attributes
    ----------
    feature_extraction : FeatureExtraction
        The feature_extraction parameter received
    k : int
        The best k features selected
    features : (k x feature_extraction.n_features)-d array
        The array with the selected features
    """
    def __init__(self, training_features_extraction, test_features_extraction, k=4, scale=True):
        """
        Parameters
        ----------
        feature_extraction : FeatureExtraction
            The FeatureExtraction object computed
        k : int
            The best k features to select
        scale: bool
            If should scale or not the selected features
        """
        select_K = SelectKBest(mutual_info_classif, k=k)\
            .fit(training_features_extraction.features, training_features_extraction.y)

        print(training_features_extraction.features.shape)
        print(test_features_extraction.features.shape)

        self.training_features = select_K.transform(training_features_extraction.features)
        self.test_features = select_K.transform(test_features_extraction.features)

        print(self.training_features.shape)
        print(self.test_features.shape)

        if scale:
            scaler = preprocessing.StandardScaler()
            self.training_features = scaler.fit_transform(self.training_features, training_features_extraction.y)
            self.test_features = scaler.fit_transform(self.test_features)

