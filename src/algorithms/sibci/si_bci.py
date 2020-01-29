from src.data_preparation.data_preparation import read_eeg_file
from scipy import signal
from scipy import linalg
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
import pyriemann.utils.mean as rie_mean
import numpy as np


def compute_spatial_filters(left_data, right_data):
    n_channels = left_data.shape[2]
    cov_shape = (n_channels, n_channels)

    # Estimate the covariance matrix of every trial
    n_left_trials = left_data.shape[0]
    cov = np.zeros((n_left_trials, *cov_shape))
    for n_trial in range(n_left_trials):
        #trial = signal.sosfilt(sos, left_data[n_trial], axis=0)
        trial = left_data[n_trial]
        cov[n_trial] = np.cov(np.transpose(trial))

    # calculate average of covariance matrix
    cov_1 = rie_mean.mean_covariance(cov, metric="logeuclid")

    # Estimate the covariance matrix of every trial
    n_right_trials = right_data.shape[0]
    cov = np.zeros((n_right_trials, *cov_shape))
    for n_trial in range(n_right_trials):
        #trial = signal.sosfilt(sos, right_data[n_trial], axis=0)
        trial = right_data[n_trial]
        cov[n_trial] = np.cov(np.transpose(trial))

    # calculate average of covariance matrix
    cov_2 = rie_mean.mean_covariance(cov, metric="logeuclid")

    # Solve the generalized eigenvalue problem
    n_pairs = CSP_COMPONENTS // 2
    w, vr = linalg.eig(cov_1, cov_2, right=True)
    w = np.abs(w)
    sorted_indexes = np.argsort(w)
    chosen_indexes = np.zeros(2 * n_pairs).astype(int)
    chosen_indexes[0:n_pairs] = sorted_indexes[0:n_pairs]
    chosen_indexes[n_pairs:2 * n_pairs] = sorted_indexes[-n_pairs:]

    return vr[:, chosen_indexes]


def extract_features(left_data, right_data):
    W = compute_spatial_filters(left_data, right_data)

    X = np.concatenate((left_data, right_data))
    n_trials = X.shape[0]
    features = {
        "CSP": np.zeros((n_trials, CSP_COMPONENTS)),
        "KATZ_FRACTAL": np.zeros((n_trials, CSP_COMPONENTS))
    }

    for n_trial in range(n_trials):
        x = X[n_trial]
        #x = signal.sosfilt(sos, x, axis=0)
        z = np.dot(np.transpose(W), np.transpose(x))
        #z = signal.sosfilt(sos, z, axis=1)
        features["CSP"][n_trial] = np.log(np.divide(np.var(z, axis=1), np.sum(np.var(z, axis=1))))
        for n_component in range(CSP_COMPONENTS):
            features["KATZ_FRACTAL"][n_trial, n_component] = katz_fractal(z[n_component])

    return features


def katz_fractal(x):
    n = len(x) - 1

    # Calculate the total length L of the signal
    L = 0
    for n_i in range(n):
        # Use the Euclidean distance formula to obtain the distance between the consecutive points
        x_distance = 1
        y_distance = (x[n_i] - x[n_i + 1]) ** 2
        L = L + np.sqrt(x_distance + y_distance)

    # Calculate the diameter of the signal, that is the farthest distance between the starting point to any other point
    d = np.zeros(n)
    for n_i in range(n):
        # Use the Euclidean distance formula to obtain the distance between the points to the origin
        x_distance = n_i ** 2
        y_distance = (x[0] - x[n_i + 1]) ** 2
        d[n_i] = np.sqrt(x_distance + y_distance)
    d = np.max(d)

    ln = np.log10(n)
    return ln / (np.log10(d / L) + ln)


def classify(X_train, Y_train, X_test, Y_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)
    predictions = lda.predict(X_test)
    return accuracy_score(Y_test, predictions)


FS = 100
TIME_LENGTH = int(FS * 2.5)
TIME_WINDOW = int(FS * 2.5)
DATA_FOLDER = "data/si-bci/bci-iii-dataset-iv-a"
CSP_COMPONENTS = 8

subjects = range(1, 6)
subjects_set = set(subjects)
accuracies = {
    "CSP": np.zeros(len(subjects)),
    "KATZ_FRACTAL": np.zeros(len(subjects))
}

sos = signal.butter(5, [8, 30], analog=False, btype="band", output="sos", fs=FS)

for test_subject in subjects:
    print("Subject: ", test_subject)
    train_subjects = list(subjects_set - {test_subject})

    # Load training data
    path_train_files = [(f"{DATA_FOLDER}/left-hand-subject-{train_subject}.csv",
                        f"{DATA_FOLDER}/right-hand-subject-{train_subject}.csv")
                        for train_subject in train_subjects]
    train_data_by_subject = [read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW)
                             for left_data_file, right_data_file in path_train_files]

    # Load test data
    left_data_file = f"{DATA_FOLDER}/left-hand-subject-{test_subject}.csv"
    right_data_file = f"{DATA_FOLDER}/right-hand-subject-{test_subject}.csv"
    test_data = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, training=False)

    test_data.left_data = signal.sosfilt(sos, test_data.left_data, axis=1)
    test_data.right_data = signal.sosfilt(sos, test_data.right_data, axis=1)

    train_data = {
        "Y": [],
        "CSP": None,
        "KATZ_FRACTAL": None
    }
    for (i, train_data_subject) in enumerate(train_data_by_subject):
        train_data_subject.left_data = signal.sosfilt(sos, train_data_subject.left_data, axis=1)
        train_data_subject.right_data = signal.sosfilt(sos, train_data_subject.right_data, axis=1)

        train_data["Y"] = np.concatenate((train_data["Y"], train_data_subject.labels))
        train_features = extract_features(train_data_subject.left_data, train_data_subject.right_data)
        if i == 0:
            train_data["CSP"] = train_features["CSP"]
            train_data["KATZ_FRACTAL"] = train_features["KATZ_FRACTAL"]
        else:
            train_data["CSP"] = np.concatenate((train_data["CSP"], train_features["CSP"]))
            train_data["KATZ_FRACTAL"] = np.concatenate((train_data["KATZ_FRACTAL"], train_features["KATZ_FRACTAL"]))

    test_data.F = extract_features(test_data.left_data, test_data.right_data)

    # Normalize the features
    train_data["CSP"] = stats.zscore(train_data["CSP"], axis=0)
    train_data["KATZ_FRACTAL"] = stats.zscore(train_data["KATZ_FRACTAL"], axis=0)
    test_data.F["CSP"] = stats.zscore(test_data.F["CSP"], axis=0)
    test_data.F["KATZ_FRACTAL"] = stats.zscore(test_data.F["KATZ_FRACTAL"], axis=0)

    accuracy_index = test_subject - 1
    accuracies["CSP"][accuracy_index] = classify(train_data["CSP"], train_data["Y"],
                                                 test_data.F["CSP"], test_data.labels)
    accuracies["KATZ_FRACTAL"][accuracy_index] = classify(train_data["KATZ_FRACTAL"], train_data["Y"],
                                                          test_data.F["KATZ_FRACTAL"], test_data.labels)

# Evaluation
print()
for feature_extraction_method in accuracies:
    print(feature_extraction_method)
    for subject, cv_accuracies in enumerate(accuracies[feature_extraction_method]):
        acc_mean = np.mean(cv_accuracies) * 100
        acc_std = np.std(cv_accuracies) * 100
        print(f"\tSubject {subject + 1} average accuracy: {acc_mean:.4f} +/- {acc_std:.4f}")
    average_acc_mean = np.mean(accuracies[feature_extraction_method]) * 100
    average_acc_std = np.std(accuracies[feature_extraction_method]) * 100
    print(f"\tAverage accuracy: {average_acc_mean:.4f} +/- {average_acc_std:.4f}")
