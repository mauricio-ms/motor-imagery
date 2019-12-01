"""
Implementation mainly based on the paper:
    Filter bank common spatial pattern algorithm on BCI competition IV Datasets 2a and 2b
"""
from src.data_preparation.data_preparation import read_eeg_file
from src.algorithms.fbcsp.FilterBankCSPFeatureExtraction import FilterBankCSPFeatureExtraction
from src.algorithms.fbcsp.MIBIFFeatureSelection import MIBIFFeatureSelection
from src.classifiers.SVM import SVM
from src.classifiers.LDA import LDA
from src.evaluation.evaluation import plot_accuracies_by_subjects, print_mean_accuracies

import numpy as np

TIME_LENGTH = 750
TIME_WINDOW = 750
EPOCH_SIZE = 500
CSP_RELEVANT_FEATURES = 2
# DATA_FOLDER = "data/bnci/by-subject-data-with-feedback-to-user"
DATA_FOLDER = "data/bci-iv-a/subject-dependent"

subjects = range(1, 6)
accuracies = {
    "SVM": np.zeros(len(subjects)),
    "LDA": np.zeros(len(subjects))
}

for subject in subjects:
    print("========= Subject: ", subject)

    # Load training data
    print("Loading training data ...")
    left_data_file = f"{DATA_FOLDER}/left-hand-training-subject-{subject}.csv"
    right_data_file = f"{DATA_FOLDER}/right-hand-training-subject-{subject}.csv"
    training_data = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE)

    # Load test data
    print("Loading test data ...")
    left_data_file = f"{DATA_FOLDER}/left-hand-test-subject-{subject}.csv"
    right_data_file = f"{DATA_FOLDER}/right-hand-test-subject-{subject}.csv"
    test_data = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE, False)

    # Feature extraction
    print("Extracting features ...")
    features = FilterBankCSPFeatureExtraction(training_data, test_data)

    scale = True
    k = 6
    fs = MIBIFFeatureSelection(features, k, scale)

    selected_training_features = fs.training_features
    selected_test_features = fs.test_features

    # Classification
    print("Classifying features ...")
    accuracy_index = subject - 1

    # SVM classifier
    svm_accuracy = SVM("rbf", 0.8, not scale,
                       selected_training_features, features.training_labels,
                       selected_test_features, features.test_labels).get_accuracy()
    print(f"SVM accuracy: {svm_accuracy:.4f}")
    accuracies["SVM"][accuracy_index] = svm_accuracy

    # LDA classifier
    lda_accuracy = LDA(selected_training_features, features.training_labels,
                       selected_test_features, features.test_labels).get_accuracy()
    print(f"LDA accuracy: {lda_accuracy:.4f}")
    accuracies["LDA"][accuracy_index] = lda_accuracy

    print()

# Evaluation
plot_accuracies_by_subjects(subjects, accuracies)
print_mean_accuracies(accuracies)
