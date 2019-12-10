"""
Implementation mainly based on the paper:
    A New Design of Mental State Classification for Subject Independent BCI Systems
"""
from src.data_preparation.data_preparation import read_eeg_file, read_eeg_files
from scipy import signal
# from src.algorithms.csp.CSP import CSP
from mne.decoding import CSP
from src.classifiers.SVM import SVM
from src.classifiers.LDA import LDA
from src.evaluation.evaluation import plot_accuracies_by_subjects, print_mean_accuracies
import matplotlib.pyplot as plt

import numpy as np


def katz_fd(x):
    x = np.array(x)
    dists = np.abs(np.ediff1d(x))
    ll = dists.sum()
    ln = np.log10(np.divide(ll, dists.mean()))
    aux_d = x - x[0]
    d = np.max(np.abs(aux_d[1:]))
    return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))


TIME_LENGTH = 250
TIME_WINDOW = 250
EPOCH_SIZE = None
DATA_FOLDER = "data/bci-iv-a/subject-independent/100hz"
CSP_COMPONENTS = 4

subjects = range(1, 6)
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
    path_files = [(f"{DATA_FOLDER}/left-hand-subject-{training_subject}.csv",
                   f"{DATA_FOLDER}/right-hand-subject-{training_subject}.csv")
                  for training_subject in training_subjects]
    training_data = read_eeg_files(path_files, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE)

    print(f"Left trials: {training_data.left_data.shape[0]}")
    print(f"Right trials: {training_data.right_data.shape[0]}")

    # Load test data
    print("Loading test data ...")
    left_data_file = f"{DATA_FOLDER}/left-hand-subject-{test_subject}.csv"
    right_data_file = f"{DATA_FOLDER}/right-hand-subject-{test_subject}.csv"
    test_data = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE, False)

    print(f"Left trials: {test_data.left_data.shape[0]}")
    print(f"Right trials: {test_data.right_data.shape[0]}")

    # Pre-processing
    print("Pre-processing ...")
    print("Applying 5º order Butterworth bandpass filter (8-30 Hz)")
    b, a = signal.butter(5, [8, 30], btype="bandpass", fs=100)

    training_data.left_data = signal.filtfilt(b, a, training_data.left_data, axis=1)
    training_data.right_data = signal.filtfilt(b, a, training_data.right_data, axis=1)

    # plt.plot(training_data.left_data[1, :, 1])
    # plt.show()

    test_data.left_data = signal.filtfilt(b, a, test_data.left_data, axis=1)
    test_data.right_data = signal.filtfilt(b, a, test_data.right_data, axis=1)

    training_data.X = np.concatenate((training_data.left_data, training_data.right_data))
    test_data.X = np.concatenate((test_data.left_data, test_data.right_data))

    # Reshape to the format expected by MNE Library
    X = np.transpose(training_data.X, [0, 2, 1])
    csp = CSP(n_components=CSP_COMPONENTS, reg=None, log=None, norm_trace=False, transform_into="csp_space")
    csp.fit(X, training_data.labels)
    training_data.Z = np.transpose(csp.transform(X), [0, 2, 1])

    # csp = CSP(average_trial_covariance=False, n_components=CSP_COMPONENTS)
    # csp.fit(training_data.left_data, training_data.right_data)
    # training_data.Z = np.array([csp.project(x) for x in training_data.X])

    # Feature extraction
    print("Extracting features ...")
    training_data.csp_log_var_features = np.zeros((training_data.Z.shape[0], CSP_COMPONENTS))
    training_data.katz_fractal_features = np.zeros((training_data.Z.shape[0], CSP_COMPONENTS))
    for n_epoch in range(0, training_data.Z.shape[0]):
        trial = training_data.Z[n_epoch, :, :]
        training_data.csp_log_var_features[n_epoch, :] = np.log(np.var(trial, axis=0)/np.sum(np.var(trial, axis=0)))
        for n_channel in range(0, trial.shape[1]):
            training_data.katz_fractal_features[n_epoch, n_channel] = katz_fd(trial[:, n_channel])

    # training_data.features = np.concatenate((training_data.csp_log_var_features, training_data.katz_fractal_features), axis=1)
    # training_data.features = training_data.katz_fractal_features
    training_data.features = training_data.csp_log_var_features

    # csp = CSP(average_trial_covariance=False, n_components=CSP_COMPONENTS)
    # csp.fit(test_data.left_data, test_data.right_data)
    #
    # test_data.Z = np.array([csp.project(x) for x in test_data.X])

    X = np.transpose(test_data.X, [0, 2, 1])
    # csp = CSP(n_components=CSP_COMPONENTS, reg=None, log=None, norm_trace=False, transform_into="csp_space")
    # csp.fit(X, test_data.labels)
    test_data.Z = np.transpose(csp.transform(X), [0, 2, 1])

    test_data.csp_log_var_features = np.zeros((test_data.Z.shape[0], CSP_COMPONENTS))
    test_data.katz_fractal_features = np.zeros((test_data.Z.shape[0], CSP_COMPONENTS))
    for n_epoch in range(0, test_data.Z.shape[0]):
        trial = test_data.Z[n_epoch, :, :]
        test_data.csp_log_var_features[n_epoch, :] = np.log(np.var(trial, axis=0)/np.sum(np.var(trial, axis=0)))
        for n_channel in range(0, trial.shape[1]):
            test_data.katz_fractal_features[n_epoch, n_channel] = katz_fd(trial[:, n_channel])

    # test_data.features = np.concatenate((test_data.csp_log_var_features, test_data.katz_fractal_features), axis=1)
    # test_data.features = test_data.katz_fractal_features
    test_data.features = test_data.csp_log_var_features

    # Classification
    print("Classifying features ...")
    accuracy_index = test_subject - 1

    # SVM classifier
    svm_accuracy = SVM("rbf", 0.8, True,
                       training_data.features, training_data.labels,
                       test_data.features, test_data.labels).get_accuracy()
    print(f"SVM accuracy: {svm_accuracy:.4f}")
    accuracies["SVM"][accuracy_index] = svm_accuracy

    # LDA classifier
    lda_accuracy = LDA(training_data.features, training_data.labels,
                       test_data.features, test_data.labels).get_accuracy()
    print(f"LDA accuracy: {lda_accuracy:.4f}")
    accuracies["LDA"][accuracy_index] = lda_accuracy

    print()

# TODO TO CROSS-VALIDATE ROTATE THE TEST DATA

# Evaluation
plot_accuracies_by_subjects(subjects, accuracies)
print_mean_accuracies(accuracies)

for algorithm in accuracies.keys():
    print(algorithm)
    for accuracy in accuracies.get(algorithm):
        print(accuracy)
