from src.data_preparation.data_preparation import read_eeg_file, read_eeg_files
from scipy import signal
from src.classifiers.SVM import SVM
from src.classifiers.LDA import LDA
from src.evaluation.evaluation import plot_accuracies_by_subjects, print_mean_accuracies

import numpy as np

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
    # TODO SETAR ESSA INFO NA LEITURA (CRIAR LEITURA POR DATASET)
    b, a = signal.butter(5, [8, 30], btype="bandpass", fs=100)

    training_data.left_data = signal.filtfilt(b, a, training_data.left_data, axis=1)
    training_data.right_data = signal.filtfilt(b, a, training_data.right_data, axis=1)

    test_data.left_data = signal.filtfilt(b, a, test_data.left_data, axis=1)
    test_data.right_data = signal.filtfilt(b, a, test_data.right_data, axis=1)

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