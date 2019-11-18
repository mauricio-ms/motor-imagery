from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing

import numpy as np


class MIBIFFeatureSelection:
    """
    Class responsible to select the best features
    using the MIBIF algorithm

    Attributes
    ----------
    k : int
        The best k features selected
    training_features : (k x features.training_features.n_features)-d array
        The array with the selected training features
    test_features : (k x features.test_features.n_features)-d array
        The array with the selected test features
    """
    def __init__(self, features, k=4, scale=True):
        """
        Parameters
        ----------
        features : FilterBankCSPFeatureExtraction
            The FilterBankCSPFeatureExtraction object with the training and test features computed
        k : int
            The best k features to select
        scale: bool
            If should scale or not the selected features
        """
        self.k = k
        indexes_selected_features = self.get_indexes_selected_features(features)
        print("Selected features: ", indexes_selected_features)
        self.training_features = features.training_features[:, indexes_selected_features]
        self.test_features = features.test_features[:, indexes_selected_features]

        if scale:
            scaler = preprocessing.StandardScaler()
            self.training_features = scaler.fit_transform(self.training_features, features.training_labels)
            self.test_features = scaler.transform(self.test_features)

    def get_indexes_selected_features(self, features):
        mutual_info = mutual_info_classif(features.training_features, features.training_labels)
        mutual_info_indexes = np.argsort(mutual_info)[::-1]

        start_features = range(0, features.n_features, features.n_components)
        indexes_selected_features = None
        for selected_feature in mutual_info_indexes[0:self.k]:
            start_feature = MIBIFFeatureSelection.get_start_feature(start_features, selected_feature,
                                                                    features.n_components)
            end_feature = min(start_feature + features.n_components - 1, features.n_features)
            next_features = np.asarray(range(start_feature, end_feature + 1))
            if indexes_selected_features is None:
                indexes_selected_features = next_features
            else:
                indexes_selected_features = np.concatenate((indexes_selected_features, next_features))

        return np.unique(np.sort(indexes_selected_features))

    @staticmethod
    def get_start_feature(start_features, value, interval):
        for i in range(len(start_features)):
            if value - start_features[i] < interval:
                return start_features[i]

        return None
