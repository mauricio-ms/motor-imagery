from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing

import numpy as np


class MIBIFFeatureSelection:
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
        self.k = k
        indexes_selected_features = self.get_indexes_selected_features(training_features_extraction)
        print("Selected features: ", indexes_selected_features)
        self.training_features = training_features_extraction.features[:, indexes_selected_features]
        self.test_features = test_features_extraction.features[:, indexes_selected_features]

        if scale:
            scaler = preprocessing.StandardScaler()
            self.training_features = scaler.fit_transform(self.training_features, training_features_extraction.y)
            self.test_features = scaler.fit_transform(self.test_features)

    def get_indexes_selected_features(self, training_features_extraction):
        mutual_info = mutual_info_classif(training_features_extraction.features, training_features_extraction.y)
        mutual_info_indexes = np.argsort(mutual_info)[::-1]

        start_features = range(0, training_features_extraction.n_features, training_features_extraction.n_components)
        indexes_selected_features = None
        for selected_feature in mutual_info_indexes[0:self.k]:
            start_feature = MIBIFFeatureSelection.get_start_feature(start_features, selected_feature,
                                                                    training_features_extraction.n_components)
            end_feature = min(start_feature + training_features_extraction.n_components - 1,
                              training_features_extraction.n_features)
            features = np.asarray(range(start_feature, end_feature + 1))
            if indexes_selected_features is None:
                indexes_selected_features = features
            else:
                indexes_selected_features = np.concatenate((indexes_selected_features, features))

        return np.unique(np.sort(indexes_selected_features))

    @staticmethod
    def get_start_feature(start_features, value, interval):
        for i in range(len(start_features)):
            if value - start_features[i] < interval:
                return start_features[i]

        return None
