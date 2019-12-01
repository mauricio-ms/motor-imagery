"""
Implementation mainly based on the paper:
    Separable Common Spatio-Spectral Patterns for Motor Imagery BCI Systems
"""
from src.data_preparation.data_preparation import read_eeg_file
from src.signal_processing.FilterBank import FilterBank
from src.algorithms.scssp.SpatioSpectralCovariancesEstimation import SpatioSpectralCovariancesEstimation
from src.algorithms.scssp.EigenvaluesProblems import EigenvaluesProblems
from src.classifiers.SVM import SVM
from src.classifiers.LDA import LDA
from src.evaluation.evaluation import plot_accuracies_by_subjects, print_mean_accuracies

import numpy as np

TIME_LENGTH = 750
TIME_WINDOW = 750
EPOCH_SIZE = 500
D = 12
# DATA_FOLDER = "data/bnci/by-subject-data-with-feedback-to-user"
DATA_FOLDER = "data/bci-iv-a/subject-dependent"


def compute_feature_vector(eeg):
    lambda_k = eigenvalues.lambda_k

    z_k_left = np.zeros((eeg.left_data.shape[1], len(lambda_k)))
    z_k_right = np.zeros((eeg.right_data.shape[1], len(lambda_k)))

    for n_epoch in range(0, eeg.left_data.shape[1]):
        z_k_left[n_epoch, :] = compute_features(eeg.left_data[:, n_epoch, :, :])

    for n_epoch in range(0, eeg.right_data.shape[1]):
        z_k_right[n_epoch, :] = compute_features(eeg.right_data[:, n_epoch, :, :])

    z = np.concatenate((z_k_left, z_k_right))
    y = np.concatenate((np.zeros(z_k_left.shape[0]), np.ones(z_k_right.shape[0])))

    return z, y


def compute_features(eeg_data):
    lambda_k = eigenvalues.lambda_k

    y_k = np.zeros((lambda_k.shape[0], eeg_data.shape[1]))

    for k in range(0, lambda_k.shape[0]):
        W = eigenvalues.compute_eigenvector(k)

        for t in range(0, eeg_data.shape[1]):
            X = eeg_data[:, t, :]
            x = np.concatenate(X)
            y_k[k, t] = np.dot(W, x)

    return np.log(np.divide(np.var(y_k, axis=1), np.sum(np.var(y_k, axis=1))))


print("D: ", D)

subjects = range(1, 6)
accuracies = {
    "SVM": np.zeros(len(subjects)),
    "LDA": np.zeros(len(subjects))
}

# Create the Filter Bank
filter_bank = FilterBank(4, 40, 4)

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

    # Apply the filter bank to the training data
    training_data.left_data = filter_bank.apply(training_data.left_data)
    training_data.right_data = filter_bank.apply(training_data.right_data)

    # Compute the spatial and spectral covariances
    spatio_spectral_covariances = SpatioSpectralCovariancesEstimation(training_data)

    eigenvalues = EigenvaluesProblems(spatio_spectral_covariances, d=D)

    # Compute the training features
    z_training, y_training = compute_feature_vector(training_data)

    # Apply the filter bank to the test data
    test_data.left_data = filter_bank.apply(test_data.left_data)
    test_data.right_data = filter_bank.apply(test_data.right_data)

    # Compute the test features
    z_test, y_test = compute_feature_vector(test_data)

    # Classification
    print("Classifying features ...")
    accuracy_index = subject - 1

    # SVM classifier
    svm_accuracy = SVM("rbf", 0.8, True, z_training, y_training, z_test, y_test).get_accuracy()
    print(f"SVM accuracy: {svm_accuracy:.4f}")
    accuracies["SVM"][accuracy_index] = svm_accuracy

    # LDA classifier
    lda_accuracy = LDA(z_training, y_training, z_test, y_test).get_accuracy()
    print(f"LDA accuracy: {lda_accuracy:.4f}")
    accuracies["LDA"][accuracy_index] = lda_accuracy

    print()

plot_accuracies_by_subjects(subjects, accuracies)
print_mean_accuracies(accuracies)
