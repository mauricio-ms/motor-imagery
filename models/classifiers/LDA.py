from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

import numpy as np


class LDA:
    def __init__(self, training_features, y_training, test_features, y_test, threshold=None):
        self.training_features = training_features
        self.y_training = y_training
        self.test_features = test_features
        self.y_test = y_test
        self.threshold = threshold

    def get_accuracy(self):
        clf = LinearDiscriminantAnalysis()
        clf.fit(self.training_features, self.y_training)

        if self.threshold:
            all_predictions = np.asarray(list(map(lambda x: self.__classify(clf, x), self.test_features)))
            known = all_predictions != -1
            predictions = all_predictions[known]
            y_known = self.y_test[known]
            percent_unknown = 100 * (np.count_nonzero(all_predictions == -1)) / len(all_predictions)
            print(f"Unknown: {percent_unknown} %")
            return accuracy_score(y_known, predictions)

        predictions = clf.predict(self.test_features)
        return accuracy_score(self.y_test, predictions)

    def __classify(self, clf, x):
        probability = clf.predict_proba([x])[0]
        if probability[0] > self.threshold + 0.1:
            return 0
        elif probability[1] > self.threshold + 0.1:
            return 1
        else:
            return -1
