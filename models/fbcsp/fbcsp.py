from Eeg import Eeg
from FilterBankCSPFeatureExtraction import FilterBankCSPFeatureExtraction
from MIBIFFeatureSelection import MIBIFFeatureSelection
from MIBIFFeatureSelection2 import MIBIFFeatureSelection2
from mne.decoding import CSP

from sklearn import svm
from sklearn.metrics import accuracy_score

import numpy as np


TIME_WINDOW = 750
CSP_RELEVANT_FEATURES = 2
K = 15
subjects = range(1, 10)
accuracies = {}
for k in range(1, K+1):
    accuracies[k] = np.zeros(len(subjects))

for subject in subjects:
    print("Subject: ", subject)

    # Load training data
    print("Loading training data ...")
    training_data = Eeg(f"data/bnci/by-subject-complete/lefthand-training-subject-{subject}.csv",
                        f"data/bnci/by-subject-complete/righthand-training-subject-{subject}.csv", TIME_WINDOW)

    # Training feature extraction
    print("Extracting training features ...")
    csp = CSP(n_components=CSP_RELEVANT_FEATURES, reg=None, log=True, norm_trace=False)
    training_features = FilterBankCSPFeatureExtraction(csp, training_data)

    # Load test data
    print("Loading test data ...")
    test_data = Eeg(f"data/bnci/by-subject-complete/lefthand-test-subject-{subject}.csv",
                    f"data/bnci/by-subject-complete/righthand-test-subject-{subject}.csv", TIME_WINDOW, False)

    # Test feature extraction
    print("Extracting test features ...")
    test_features = FilterBankCSPFeatureExtraction(csp, test_data)

    # Feature selection
    for k in range(1, K+1):
        scale = True
        fs = MIBIFFeatureSelection(training_features, test_features, k, scale)

        selected_training_features = fs.training_features
        selected_test_features = fs.test_features

        # SVM classifier
        kernel = "linear"
        C = 0.8
        clf = svm.SVC(C=C, kernel=kernel) if scale else svm.SVC(C=C, gamma="scale", kernel=kernel)
        clf.fit(selected_training_features, training_features.y)
        y_pred = clf.predict(selected_test_features)
        acc = accuracy_score(test_features.y, y_pred)
        print(acc)
        accuracies[k][subject-1] = acc

print("Accuracies")
for k in range(1, K+1):
    print("================== k: ", k)
    for subject in subjects:
        print("Subject %s: %s" % (subject, accuracies[k][subject-1]))
    print("Mean accuracy: %s\n\n" % (np.mean(accuracies[k])))

print("Mean accuracies")
for k in range(1, K+1):
    print("k %s: %s" % (k, np.mean(accuracies[k])))
