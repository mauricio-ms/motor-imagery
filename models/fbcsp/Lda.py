from sklearn import svm
from sklearn.metrics import accuracy_score


class Svm:
    # TODO Adjust the features as a object
    def __init__(self, kernel, penalty, scale, training_features, y_training, test_features, y_test):
        self.kernel = kernel
        self.penalty = penalty
        self.scale = scale
        self.training_features = training_features
        self.y_training = y_training
        self.y_test = y_test
        self.test_features = test_features

    def get_accuracy(self):
        clf = svm.SVC(C=self.penalty, kernel=self.kernel) if self.scale \
            else svm.SVC(C=self.penalty, gamma="scale", kernel=self.kernel)
        clf.fit(self.training_features, self.y_training)
        predicted = clf.predict(self.test_features)
        return accuracy_score(self.y_test, predicted)
