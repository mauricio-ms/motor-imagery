from models.data_preparation.data_preparation import read_eeg_file, read_eeg_files
from models.fbcsp.FilterBankCSPFeatureExtraction import FilterBankCSPFeatureExtraction
from models.fbcsp.MIBIFFeatureSelection import MIBIFFeatureSelection
from models.classifiers.SVM import SVM
from models.classifiers.LDA import LDA
from models.evaluation.evaluation import plot_accuracies_by_subjects, print_mean_accuracies

import numpy as np


TIME_LENGTH = 750
TIME_WINDOW = 750
EPOCH_SIZE = 500
CSP_RELEVANT_FEATURES = 2

subjects = range(1, 10)
subjects_set = set(subjects)
accuracies = {
    "SVM": np.zeros(len(subjects)),
    "LDA": np.zeros(len(subjects))
}

for test_subject in subjects:
    print("Test subject: ", test_subject)
    training_subjects = list(subjects_set - {test_subject})
    print("Training subjects: ", training_subjects)

    # Load training data
    print("Loading training data ...")
    path_files = [(f"data/bnci/by-subject/lefthand-subject-{training_subject}.csv",
                   f"data/bnci/by-subject/righthand-subject-{training_subject}.csv")
                  for training_subject in training_subjects]
    training_data = read_eeg_files(path_files, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE)

    # Load test data
    print("Loading test data ...")
    left_data_file = f"data/bnci/by-subject/lefthand-subject-{test_subject}.csv"
    right_data_file = f"data/bnci/by-subject/righthand-subject-{test_subject}.csv"
    test_data = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE, False)

    # Feature extraction
    print("Extracting features ...")
    features = FilterBankCSPFeatureExtraction(training_data, test_data)

    # Feature Selection
    scale = True
    k = 6
    fs = MIBIFFeatureSelection(features, k, scale)

    selected_training_features = fs.training_features
    selected_test_features = fs.test_features

    # Classification
    print("Classifying features ...")
    accuracy_index = test_subject-1

    # SVM classifier
    svm_accuracy = SVM("rbf", 0.8, not scale,
                       selected_training_features, features.training_labels,
                       selected_test_features, features.test_labels).get_accuracy()
    print("SVM accuracy:", svm_accuracy)
    accuracies["SVM"][accuracy_index] = svm_accuracy

    # LDA classifier
    lda_accuracy = LDA(selected_training_features, features.training_labels,
                       selected_test_features, features.test_labels).get_accuracy()
    print("LDA accuracy:", lda_accuracy)
    accuracies["LDA"][accuracy_index] = lda_accuracy

    print()

# Evaluation
plot_accuracies_by_subjects(subjects, accuracies)
print_mean_accuracies(accuracies)
