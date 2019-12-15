from src.data_preparation.data_preparation import read_eeg_file, read_eeg_files
from src.signal_processing.signal_processing import cross_correlation_sequence
from src.classifiers.SVM import SVM
from src.classifiers.LDA import LDA
import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import skew, kurtosis

TIME_LENGTH = 250
TIME_WINDOW = 250
EPOCH_SIZE = None
DATA_FOLDER = "data/bci-iv-a/subject-independent/100hz"
ref_channel = 51

n_channels = 118
channels_mask = np.ones(n_channels, bool)
channels_mask[ref_channel] = False
channels_indexes = [ch for ch, valid in enumerate(channels_mask) if valid]

subjects = range(1, 6)
subjects_set = set(subjects)
accuracies = {
    "SVM": np.zeros(len(subjects)),
    "LDA": np.zeros(len(subjects))
}


def extract_feature_vector(cr_sequence):
    return [
        np.mean(cr_sequence),
        np.std(cr_sequence),
        skew(cr_sequence),
        kurtosis(cr_sequence),
        np.min(cr_sequence),
        np.max(cr_sequence)]


for test_subject in subjects:
    print("Test subject: ", test_subject)
    training_subjects = list(subjects_set - {test_subject})
    print("Training subjects: ", training_subjects)

    # Load training data
    print("Loading training data ...")
    path_files = [(f"{DATA_FOLDER}/left-hand-subject-{training_subject}.csv",
                   f"{DATA_FOLDER}/right-hand-subject-{training_subject}.csv")
                  for training_subject in training_subjects]
    training_data = read_eeg_files(path_files)
    training_data.left_data = np.array(training_data.left_data)
    training_data.right_data = np.array(training_data.right_data)

    print(f"Left trials: {training_data.left_data.shape[0]}")
    print(f"Right trials: {training_data.right_data.shape[0]}")

    # Load test data
    print("Loading test data ...")
    left_data_file = f"{DATA_FOLDER}/left-hand-subject-{test_subject}.csv"
    right_data_file = f"{DATA_FOLDER}/right-hand-subject-{test_subject}.csv"
    test_data = read_eeg_file(left_data_file, right_data_file, training=False)
    test_data.left_data = np.array(test_data.left_data)
    test_data.right_data = np.array(test_data.right_data)

    # Pre-processing
    print("Calculating cross correlations ...")
    training_left_cr = np.zeros((training_data.left_data.shape[1] - 1, 2 * training_data.left_data.shape[0] - 1))
    training_right_cr = np.zeros((training_data.right_data.shape[1], 2 * training_data.right_data.shape[0] - 1))

    test_left_cr = np.zeros((test_data.left_data.shape[1] - 1, 2 * test_data.left_data.shape[0] - 1))
    test_right_cr = np.zeros((test_data.right_data.shape[1], 2 * test_data.right_data.shape[0] - 1))

    training_ref_data = training_data.right_data[:, ref_channel]
    test_ref_data = test_data.right_data[:, ref_channel]

    for (i, n_channel) in enumerate(channels_indexes):
        print(n_channel)
        training_left_cr[i, :] = cross_correlation_sequence(training_ref_data, training_data.left_data[:, n_channel])
        test_left_cr[i, :] = cross_correlation_sequence(test_ref_data, test_data.left_data[:, n_channel])

    for n_channel in range(0, n_channels):
        print(n_channel)
        training_right_cr[n_channel, :] = cross_correlation_sequence(training_ref_data, training_data.right_data[:, n_channel])
        test_right_cr[n_channel, :] = cross_correlation_sequence(test_ref_data, test_data.right_data[:, n_channel])

    # Extraction Features
    print("Extracting features ...")
    training_left_features = np.zeros((training_left_cr.shape[0], 6))
    training_right_features = np.zeros((training_right_cr.shape[0], 6))

    test_left_features = np.zeros((test_left_cr.shape[0], 6))
    test_right_features = np.zeros((test_right_cr.shape[0], 6))

    for n_channel in range(0, training_left_features.shape[0]):
        training_left_features[n_channel, :] = extract_feature_vector(training_left_cr[n_channel, :])
        test_left_features[n_channel, :] = extract_feature_vector(test_left_cr[n_channel, :])

    for n_channel in range(0, test_right_features.shape[0]):
        training_right_features[n_channel, :] = extract_feature_vector(training_right_cr[n_channel, :])
        test_right_features[n_channel, :] = extract_feature_vector(test_right_cr[n_channel, :])

    training_labels = np.concatenate((np.zeros(training_left_features.shape[0]),
                                      np.ones(training_right_features.shape[0])))
    training_features = np.concatenate((training_left_features, training_right_features), axis=0)
    test_labels = np.concatenate((np.zeros(test_left_features.shape[0]),
                                  np.ones(test_right_features.shape[0])))
    test_features = np.concatenate((test_left_features, test_right_features), axis=0)

    # Classification
    print("Classifying features ...")
    accuracy_index = test_subject - 1

    # SVM classifier
    svm_accuracy = SVM("rbf", 0.8, True,
                       training_features, training_labels,
                       test_features, test_labels).get_accuracy()
    print(f"SVM accuracy: {svm_accuracy:.4f}")
    accuracies["SVM"][accuracy_index] = svm_accuracy

    # LDA classifier
    lda_accuracy = LDA(training_features, training_labels,
                       test_features, test_labels).get_accuracy()
    print(f"LDA accuracy: {lda_accuracy:.4f}")
    accuracies["LDA"][accuracy_index] = lda_accuracy

# n_epoch = 1
# ref_data = test_data.right_data[n_epoch, :, ref_channel]
# left_epoch = test_data.left_data[n_epoch, :, :]
# n_channel = 50
# channel_data = left_epoch[:, n_channel]
#
# m = 2
# x = ref_data
# y = channel_data
# l = len(x) - np.abs(m)
# R1 = np.sum([x[i]*y[i-m] for i in range(0, len(x) - np.abs(m))])
# R2 = np.sum(np.dot(x[0:l], np.roll(y, m)[0:l]))
# R3 = pd.Series(x).corr(pd.Series(y).shift(m))
# print(R1)
# print(R2)
# print(R3)
#
#
# R = cross_correlation_sequence(ref_data, channel_data)
#
# plt.plot(R)
# plt.show()
# print("End")
