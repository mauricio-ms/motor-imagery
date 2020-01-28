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
    def __init__(self, training_features, test_features, training_labels, n_components, k=4, scale=True):
        """
        Parameters
        ----------
        training_features : array
            The training features
        test_features : array
            The test features
        k : int
            The best k features to select
        scale: bool
            If should scale or not the selected features
        """
        self.n_components = n_components
        self.n_features = training_features.shape[1]
        self.k = k
        indexes_selected_features = self.get_indexes_selected_features(training_features, training_labels)
        self.training_features = training_features[:, indexes_selected_features]
        self.test_features = test_features[:, indexes_selected_features]

        if scale:
            scaler = preprocessing.StandardScaler()
            self.training_features = scaler.fit_transform(training_features, training_labels)
            self.test_features = scaler.transform(test_features)

    def get_indexes_selected_features(self, training_features, training_labels):
        mutual_info = mutual_info_classif(training_features, training_labels)
        mutual_info_indexes = np.argsort(mutual_info)[::-1]

        start_features = range(0, self.n_features, self.n_components)
        indexes_selected_features = None
        for selected_feature in mutual_info_indexes[0:self.k]:
            start_feature = MIBIFFeatureSelection.get_start_feature(start_features, selected_feature, self.n_components)
            end_feature = min(start_feature + self.n_components - 1, self.n_features)
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
