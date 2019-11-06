"""
Script to analyze the features and test optimizations
"""
import matplotlib.pyplot as plt
import numpy as np
import itertools

from EEG import EEG
from mne.decoding import CSP
from FilterBankCSPFeatureExtraction import FilterBankCSPFeatureExtraction
from signal_processing import bandpass_filter
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score


TIME_WINDOW = 750
CSP_RELEVANT_FEATURES = 3

subject = 3

# Load training data
training_data = EEG(f"data/bnci/by-subject-complete/lefthand-training-subject-{subject}.csv",
                    f"data/bnci/by-subject-complete/righthand-training-subject-{subject}.csv", TIME_WINDOW)
bandpass_filter(training_data)

# Training feature extraction
csp = CSP(n_components=CSP_RELEVANT_FEATURES, reg=None, log=True, norm_trace=False)
training_features = FilterBankCSPFeatureExtraction(csp, training_data)

# Load test data
test_data = EEG(f"data/bnci/by-subject-complete/lefthand-test-subject-{subject}.csv",
                f"data/bnci/by-subject-complete/righthand-test-subject-{subject}.csv", TIME_WINDOW, False)
bandpass_filter(test_data)

# Test feature extraction
test_features = FilterBankCSPFeatureExtraction(csp, test_data)

bands = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [8, 9],
    [10, 11],
    [12, 13],
    [14, 15],
    [16, 17]
]

for i in range(1, len(bands)+1):
    print(i)
    mask_features_combinations = list(itertools.combinations(bands, i))
    for mask_features in mask_features_combinations:
        mask_features = list(itertools.chain.from_iterable(mask_features))

        print(mask_features)

        selected_training_features = training_features.features[:, mask_features]
        selected_test_features = test_features.features[:, mask_features]

        scaler = preprocessing.StandardScaler()
        selected_training_features = scaler.fit_transform(selected_training_features, training_features.y)
        selected_test_features = scaler.fit_transform(selected_test_features)

        clf = LinearDiscriminantAnalysis()
        clf.fit(selected_training_features, training_features.y)
        # X_lda = clf.fit_transform(selected_training_features, training_features.y)
        # print(clf.explained_variance_ratio_)

        predicted = clf.predict(selected_test_features)
        lda_accuracy = accuracy_score(test_features.y, predicted)
        print("LDA accuracy: ", lda_accuracy)

# plt.scatter(range(0, len(selected_test_features)),
#             selected_test_features[:, 0])
# plt.scatter(range(0, len(selected_test_features)),
#             selected_test_features[:, 1])
# plt.show()
