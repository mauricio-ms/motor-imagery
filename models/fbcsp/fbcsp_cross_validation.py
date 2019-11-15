"""
Implementation mainly based on the paper:
    Filter bank common spatial pattern algorithm on BCI competition IV Datasets 2a and 2b
"""

from data_preparation import load_csv, epoch, extract_single_trial
from signal_processing import bandpass_filter
from EEG import EEG
from FilterBankCSPFeatureExtraction import FilterBankCSPFeatureExtraction
from Svm import Svm
from Lda import Lda
from evaluation import plot_accuracies_by_subjects

from sklearn import preprocessing
import numpy as np


TIME_LENGTH = 750
TIME_WINDOW = 250
EPOCH_SIZE = 64
CSP_RELEVANT_FEATURES = 2

subjects = range(1, 10)
subjects_set = set(subjects)
accuracies = {
    "SVM": np.zeros(len(subjects)),
    "LDA": np.zeros(len(subjects))
}


def read_data(subject):
    left_data_file = f"data/bnci/by-subject/lefthand-subject-{subject}.csv"
    right_data_file = f"data/bnci/by-subject/righthand-subject-{subject}.csv"

    # Read the data
    left_data = extract_single_trial(load_csv(left_data_file), TIME_LENGTH, TIME_WINDOW)
    right_data = extract_single_trial(load_csv(right_data_file), TIME_LENGTH, TIME_WINDOW)

    # Read the epoch data
    if EPOCH_SIZE is not None:
        left_data = epoch(left_data, EPOCH_SIZE)
        right_data = epoch(right_data, EPOCH_SIZE)

    # Bandpass filter the data
    left_data = bandpass_filter(left_data)
    right_data = bandpass_filter(right_data)

    return left_data, right_data


for test_subject in subjects:
    print("Test subject: ", test_subject)
    training_subjects = list(subjects_set - {test_subject})
    print("Training subjects: ", training_subjects)

    # Load training data
    print("Loading training data ...")
    left_training_data = None
    right_training_data = None
    for training_subject in training_subjects:
        next_left_training_data, next_right_training_data = read_data(training_subject)

        if left_training_data is None:
            left_training_data = next_left_training_data
            right_training_data = next_right_training_data
        else:
            left_training_data = np.concatenate((left_training_data, next_left_training_data))
            right_training_data = np.concatenate((right_training_data, next_right_training_data))

    training_data = EEG(left_training_data, right_training_data)

    # Load test data
    print("Loading test data ...")
    test_data = EEG(*read_data(test_subject))

    # Feature extraction
    print("Extracting features ...")
    features = FilterBankCSPFeatureExtraction(training_data, test_data)

    # # Features scaling
    # scaler = preprocessing.StandardScaler()
    # training_features = scaler.fit_transform(features.training_features, features.training_labels)
    # test_features = scaler.transform(features.test_features)

    # Classification
    print("Classifying features ...")
    accuracy_index = test_subject-1

    # SVM classifier
    svm_accuracy = Svm("linear", 0.8, True,
                       features.training_features, features.training_labels,
                       features.test_features, features.test_labels).get_accuracy()
    print("SVM accuracy:", svm_accuracy)
    accuracies["SVM"][accuracy_index] = svm_accuracy

    # LDA classifier
    lda_accuracy = Lda(features.training_features, features.training_labels,
                       features.test_features, features.test_labels, .5).get_accuracy()
    print("LDA accuracy:", lda_accuracy)
    accuracies["LDA"][accuracy_index] = lda_accuracy

    print()

# Evaluation
plot_accuracies_by_subjects(subjects, accuracies)

for classifier in accuracies.keys():
    print(f"{classifier} - Mean accuracy: ", np.mean(accuracies[classifier]))
