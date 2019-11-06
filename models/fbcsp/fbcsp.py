"""
Implementation mainly based on the paper:
    Filter bank common spatial pattern algorithm on BCI competition IV Datasets 2a and 2b
"""

from EEG import EEG
from signal_processing import bandpass_filter
from FilterBankCSPFeatureExtraction import FilterBankCSPFeatureExtraction
from MIBIFFeatureSelection import MIBIFFeatureSelection
# from MIBIFFeatureSelection2 import MIBIFFeatureSelection2 as MIBIFFeatureSelection
from Svm import Svm
from Lda import Lda
from mne.decoding import CSP
from evaluation import print_accuracies, print_mean_accuracies

import numpy as np


TIME_WINDOW = 750
CSP_RELEVANT_FEATURES = 2

subjects = range(1, 10)
accuracies = {
    "svm": {},
    "lda": {}
}

for subject in subjects:
    print("Subject: ", subject)

    # Load training data
    print("Loading training data ...")
    training_data = EEG(f"data/bnci/by-subject-complete/lefthand-training-subject-{subject}.csv",
                        f"data/bnci/by-subject-complete/righthand-training-subject-{subject}.csv", TIME_WINDOW)
    bandpass_filter(training_data)

    # Training feature extraction
    print("Extracting training features ...")
    csp = CSP(n_components=CSP_RELEVANT_FEATURES, reg=None, log=True, norm_trace=False)
    training_features = FilterBankCSPFeatureExtraction(csp, training_data)

    # Load test data
    print("Loading test data ...")
    test_data = EEG(f"data/bnci/by-subject-complete/lefthand-test-subject-{subject}.csv",
                    f"data/bnci/by-subject-complete/righthand-test-subject-{subject}.csv", TIME_WINDOW, False)
    bandpass_filter(test_data)

    # Test feature extraction
    print("Extracting test features ...")
    test_features = FilterBankCSPFeatureExtraction(csp, test_data)

    # Feature selection
    for k in range(1, training_features.n_features+1):
        if k not in accuracies["svm"]:
            accuracies["svm"][k] = np.zeros(len(subjects))
        if k not in accuracies["lda"]:
            accuracies["lda"][k] = np.zeros(len(subjects))

        scale = True
        fs = MIBIFFeatureSelection(training_features, test_features, k, scale)

        selected_training_features = fs.training_features
        selected_test_features = fs.test_features

        # SVM classifier
        svm_accuracy = Svm("linear", 0.8, not scale,
                           selected_training_features, training_features.y,
                           selected_test_features, test_features.y).get_accuracy()
        print("SVM accuracy:", svm_accuracy)
        accuracies["svm"][k][subject-1] = svm_accuracy

        # LDA classifier
        lda_accuracy = Lda(selected_training_features, training_features.y,
                           selected_test_features, test_features.y).get_accuracy()
        print("LDA accuracy:", lda_accuracy)
        accuracies["lda"][k][subject-1] = lda_accuracy

print("== SVM ACCURACIES ==")
print_accuracies(accuracies["svm"], subjects)

print("== LDA ACCURACIES ==")
print_accuracies(accuracies["lda"], subjects)

print("== SVM MEAN ACCURACIES ==")
print_mean_accuracies(accuracies["svm"])

print()

print("== LDA MEAN ACCURACIES ==")
print_mean_accuracies(accuracies["lda"])

svm_mean_accuracies = [np.mean(accuracies["svm"][subject]) for subject in accuracies["svm"]]
lda_mean_accuracies = [np.mean(accuracies["lda"][subject]) for subject in accuracies["lda"]]
print("Best SVM (%s - %s): " % (np.argmax(svm_mean_accuracies)+1, np.max(svm_mean_accuracies)))
print("Best LDA (%s - %s): " % (np.argmax(lda_mean_accuracies)+1, np.max(lda_mean_accuracies)))
