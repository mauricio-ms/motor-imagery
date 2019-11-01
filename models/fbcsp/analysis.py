"""
Script to analyze the features and test optimizations
"""
import matplotlib.pyplot as plt

from Eeg import Eeg
from mne.decoding import CSP
from FilterBankCSPFeatureExtraction import FilterBankCSPFeatureExtraction
from MIBIFFeatureSelection import MIBIFFeatureSelection
from Svm import Svm


TIME_WINDOW = 750
CSP_RELEVANT_FEATURES = 2

subject = 2

# TODO Analisar os dados desse subject
# TODO TESTAR APLICAR UM FILTRO BANDPASS NOS DADOS ANTES

# Load training data
training_data = Eeg(f"data/bnci/by-subject-complete/lefthand-training-subject-{4}.csv",
                    f"data/bnci/by-subject-complete/righthand-training-subject-{4}.csv", TIME_WINDOW)

# Training feature extraction
csp = CSP(n_components=CSP_RELEVANT_FEATURES, reg=None, log=True, norm_trace=False)
training_features = FilterBankCSPFeatureExtraction(csp, training_data)

# Load test data
test_data = Eeg(f"data/bnci/by-subject-complete/lefthand-test-subject-{subject}.csv",
                f"data/bnci/by-subject-complete/righthand-test-subject-{subject}.csv", TIME_WINDOW, False)

# Test feature extraction
test_features = FilterBankCSPFeatureExtraction(csp, test_data)

# Feature selection
k = 1
scale = True
fs = MIBIFFeatureSelection(training_features, test_features, k, scale)

selected_training_features = fs.training_features
selected_test_features = fs.test_features

plt.scatter(range(0, len(selected_test_features)),
            selected_test_features[:, 0])
plt.scatter(range(0, len(selected_test_features)),
            selected_test_features[:, 1])
plt.show()


# SVM classifier
svm_accuracy = Svm("linear", 2, not scale,
                   selected_training_features, training_features.y,
                   selected_test_features, test_features.y).get_accuracy()
print("SVM accuracy:", svm_accuracy)
