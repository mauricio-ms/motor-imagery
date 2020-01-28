from src.data_preparation.data_preparation import read_eeg_file, read_eeg_files
from scipy import signal
from scipy import linalg
from sklearn.model_selection import StratifiedKFold
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
        trial = signal.sosfilt(sos, left_data[n_trial], axis=0)
        cov[n_trial] = np.cov(np.transpose(trial))

    # calculate average of covariance matrix
    cov_1 = rie_mean.mean_covariance(cov, metric="riemann")

    # Estimate the covariance matrix of every trial
    n_right_trials = right_data.shape[0]
    cov = np.zeros((n_right_trials, *cov_shape))
    for n_trial in range(n_right_trials):
        trial = signal.sosfilt(sos, right_data[n_trial], axis=0)
        cov[n_trial] = np.cov(np.transpose(trial))

    # calculate average of covariance matrix
    cov_2 = rie_mean.mean_covariance(cov, metric="riemann")

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
    X = np.concatenate((left_data, right_data))
    W = compute_spatial_filters(left_data, right_data)

    n_trials = X.shape[0]
    features = {
        "CSP": np.zeros((n_trials, CSP_COMPONENTS)),
        "KATZ_FRACTAL": np.zeros((n_trials, CSP_COMPONENTS))
    }

    for n_trial in range(n_trials):
        x = X[n_trial]
        x = signal.sosfilt(sos, x, axis=0)
        z = np.dot(np.transpose(W), np.transpose(x))
        #z = signal.sosfilt(sos, z, axis=1)
        features["CSP"][n_trial] = np.log(np.divide(np.var(z, axis=1), np.sum(np.var(z, axis=1))))
        for n_component in range(CSP_COMPONENTS):
            features["KATZ_FRACTAL"][n_trial, n_component] = katz_fd(z[n_component])

    return features


def katz_fd(x):
    dists = np.abs(np.ediff1d(x))
    ll = dists.sum()
    ln = np.log10(np.divide(ll, dists.mean()))
    aux_d = x - x[0]
    d = np.max(np.abs(aux_d[1:]))
    return np.divide(ln, np.add(ln, np.log10(np.divide(d, ll))))


def classify(X_train, Y_train, X_test, Y_test):
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, Y_train)
    predictions = lda.predict(X_test)
    return accuracy_score(Y_test, predictions)


FS = 100
TIME_LENGTH = 250
TIME_WINDOW = 250
DATA_FOLDER = "data/si-bci/bci-iii-dataset-iv-a"
CSP_COMPONENTS = 8
K_FOLD = 10

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
    path_files = [(f"{DATA_FOLDER}/left-hand-subject-{train_subject}.csv",
                   f"{DATA_FOLDER}/right-hand-subject-{train_subject}.csv")
                  for train_subject in train_subjects]
    train_data = read_eeg_files(path_files, TIME_LENGTH, TIME_WINDOW)

    # Load test data
    left_data_file = f"{DATA_FOLDER}/left-hand-subject-{test_subject}.csv"
    right_data_file = f"{DATA_FOLDER}/right-hand-subject-{test_subject}.csv"
    test_data = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, training=False)
    test_data.X = np.concatenate((test_data.left_data, test_data.right_data))

    train_data.F = extract_features(train_data.left_data, train_data.right_data)

    accuracy_index = test_subject - 1
    fold_accuracies = {
        "CSP": np.zeros(K_FOLD),
        "KATZ_FRACTAL": np.zeros(K_FOLD)
    }
    cv = StratifiedKFold(n_splits=K_FOLD, shuffle=True)
    for (k, (validation_index, test_index)) in enumerate(cv.split(test_data.X, test_data.labels)):
        X_test = test_data.X[validation_index]
        Y_test = test_data.labels[validation_index]
        test_data.F = extract_features(X_test[Y_test == 0], X_test[Y_test == 1])

        fold_accuracies["CSP"][k] = classify(train_data.F["CSP"], train_data.labels,
                                             test_data.F["CSP"], Y_test)
        acc = classify(train_data.F["KATZ_FRACTAL"], train_data.labels,
                       test_data.F["KATZ_FRACTAL"], Y_test)
        print(acc)
        fold_accuracies["KATZ_FRACTAL"][k] = acc

    accuracies["CSP"][accuracy_index] = np.mean(fold_accuracies["CSP"])
    accuracies["KATZ_FRACTAL"][accuracy_index] = np.mean(fold_accuracies["KATZ_FRACTAL"])

# Evaluation
for feature_extraction_method in accuracies:
    print(feature_extraction_method)
    for subject, cv_accuracies in enumerate(accuracies[feature_extraction_method]):
        acc_mean = np.mean(cv_accuracies) * 100
        acc_std = np.std(cv_accuracies) * 100
        print(f"\tSubject {subject + 1} average accuracy: {acc_mean:.4f} +/- {acc_std:.4f}")
    average_acc_mean = np.mean(accuracies[feature_extraction_method]) * 100
    average_acc_std = np.std(accuracies[feature_extraction_method]) * 100
    print(f"\tAverage accuracy: {average_acc_mean:.4f} +/- {average_acc_std:.4f}\n")
