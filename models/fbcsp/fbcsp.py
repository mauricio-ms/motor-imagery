from Eeg import Eeg
from FeatureExtraction import FeatureExtraction
from mne.decoding import CSP

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score

import numpy as np

TIME_WINDOW = 750

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
    csp = CSP(n_components=2, reg=None, log=True, norm_trace=False)
    training_features = FeatureExtraction(csp, training_data)

    # Load test data
    print("Loading test data ...")
    test_data = Eeg(f"data/bnci/by-subject-complete/lefthand-test-subject-{subject}.csv",
                    f"data/bnci/by-subject-complete/righthand-test-subject-{subject}.csv", TIME_WINDOW)

    # Test feature extraction
    print("Extracting test features ...")
    test_features = FeatureExtraction(csp, test_data)

    # Feature selection
    # MIBIF algorithm
    select_K = SelectKBest(mutual_info_classif, k=10).fit(training_features.features, training_features.y)

    print(training_features.features.shape)
    print(test_features.features.shape)

    New_train = select_K.transform(training_features.features)
    New_test = select_K.transform(test_features.features)

    print(New_train.shape)
    print(New_test.shape)

    ss = preprocessing.StandardScaler()
    X_select_train = ss.fit_transform(New_train, training_features.y)
    X_select_test = ss.fit_transform(New_test)

    #calssify
    clf = svm.SVC(C=0.8, kernel="rbf")
    clf.fit(X_select_train, training_features.y)
    y_pred = clf.predict(X_select_test)
    print(test_features.y)
    print(y_pred)
    acc = accuracy_score(test_features.y, y_pred)
    print(acc)
    accuracies[subject-1] = acc

print("Accuracies")
print(accuracies)
