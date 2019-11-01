from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


class Lda:
    # TODO Adjust the features as a object
    def __init__(self, training_features, y_training, test_features, y_test):
        self.training_features = training_features
        self.y_training = y_training
        self.y_test = y_test
        self.test_features = test_features

    def get_accuracy(self):
        clf = LinearDiscriminantAnalysis()
        clf.fit(self.training_features, self.y_training)
        predicted = clf.predict(self.test_features)
        return accuracy_score(self.y_test, predicted)
