from Eeg import Eeg
from FeatureExtraction import FeatureExtraction
from mne.decoding import CSP

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn import preprocessing
from sklearn import svm
from sklearn.metrics import accuracy_score

import numpy as np


def get_start_feature(start_features, value):
	for i in range(len(start_features)):
		if value - start_features[i] < 3:
			return start_features[i]

	return None

TIME_WINDOW = 750
CSP_RELEVANT_FEATURES = 3

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
    k = 5
    mutual_info = mutual_info_classif(training_features.features, training_features.y)
    mutual_info_indexes = np.argsort(mutual_info)[::-1]

    start_features = range(0, training_features.n_features, CSP_RELEVANT_FEATURES)
    selected_features = None
    for selected_feature in mutual_info_indexes[1:k]:
        start_feature = get_start_feature(start_features, selected_feature)
        end_feature = min(start_feature + CSP_RELEVANT_FEATURES - 1, training_features.n_features)
        if selected_features is None:
            selected_features = np.asarray(range(start_feature, end_feature+1))
        else:
            selected_features = np.concatenate((selected_features, np.asarray(range(start_feature, end_feature+1))))
    # np.sort(selected_features)

    print("Features selected: ", selected_features)

    training_features.features = training_features.features[:, selected_features]
    test_features.features = test_features.features[:, selected_features]

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