from Eeg import Eeg
from FilterBankCSPFeatureExtraction import FilterBankCSPFeatureExtraction
from MIBIFFeatureSelection import MIBIFFeatureSelection
from mne.decoding import CSP

from sklearn import svm
from sklearn.metrics import accuracy_score

import numpy as np


TIME_WINDOW = 750
CSP_RELEVANT_FEATURES = 2

subjects = range(1, 10)
accuracies = np.zeros(len(subjects))
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
    # MIBIF algorithm
    k = 4
    scale = False
    fs = MIBIFFeatureSelection(training_features, test_features, k, scale)
    training_features.features = fs.training_features
    test_features.features = fs.test_features

    # ss = preprocessing.StandardScaler()
    # training_features.features = ss.fit_transform(training_features.features[:, selected_features], training_features.y)
    # test_features.features = ss.fit_transform(test_features.features[:, selected_features])

    # select_K = SelectKBest(mutual_info_classif, k=10).fit(training_features.features, training_features.y)
    #
    # print(training_features.features.shape)
    # print(test_features.features.shape)
    #
    # New_train = select_K.transform(training_features.features)
    # New_test = select_K.transform(test_features.features)
    #
    # print(New_train.shape)
    # print(New_test.shape)

    # ss = preprocessing.StandardScaler()
    # X_select_train = ss.fit_transform(New_train, training_features.y)
    # X_select_test = ss.fit_transform(New_test)

    # SVM classifier
    clf = svm.SVC(C=0.8, gamma="scale", kernel="rbf")
    clf.fit(training_features.features, training_features.y)
    y_pred = clf.predict(test_features.features)
    acc = accuracy_score(test_features.y, y_pred)
    print(acc)
    accuracies[subject-1] = acc

print("Accuracies")
for subject in subjects:
    print("Subject %s: %s" % (subject, accuracies[subject-1]))

print("Mean accuracy: ", np.mean(accuracies))