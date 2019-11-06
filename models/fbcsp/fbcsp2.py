"""
Implementation mainly based on the paper:
    Filter bank common spatial pattern algorithm on BCI competition IV Datasets 2a and 2b
"""

from EEG import EEG
from CSP import CSP
from FilterBankCSPFeatureExtraction import FilterBankCSPFeatureExtraction
from MIBIFFeatureSelection import MIBIFFeatureSelection
from Svm import Svm
from Lda import Lda


TIME_WINDOW = 750
CSP_RELEVANT_FEATURES = 2

subject = 4

# Load training data
print("Loading training data ...")
training_data = EEG(f"data/bnci/by-subject-complete/lefthand-training-subject-{subject}.csv",
                    f"data/bnci/by-subject-complete/righthand-training-subject-{subject}.csv", TIME_WINDOW)
# bandpass_filter(training_data)

# Training feature extraction
print("Extracting training features ...")
csp = CSP(training_data.left_data, training_data.right_data)

training_features = FilterBankCSPFeatureExtraction(csp, training_data)

# Load test data
print("Loading test data ...")
test_data = EEG(f"data/bnci/by-subject-complete/lefthand-test-subject-{subject}.csv",
                f"data/bnci/by-subject-complete/righthand-test-subject-{subject}.csv", TIME_WINDOW, False)
# bandpass_filter(test_data)

# Test feature extraction
print("Extracting test features ...")
test_features = FilterBankCSPFeatureExtraction(csp, test_data)

# # Feature selection
# for k in range(1, training_features.n_features+1):
# if k not in accuracies["svm"]:
#     accuracies["svm"][k] = np.zeros(len(subjects))
# if k not in accuracies["lda"]:
#     accuracies["lda"][k] = np.zeros(len(subjects))
#
k = 4
scale = True
fs = MIBIFFeatureSelection(training_features, test_features, k, scale)

selected_training_features = fs.training_features
selected_test_features = fs.test_features

# SVM classifier
svm_accuracy = Svm("linear", 0.8, True,
                   selected_training_features, training_features.y,
                   selected_test_features, test_features.y).get_accuracy()
print("SVM accuracy:", svm_accuracy)

# LDA classifier
lda_accuracy = Lda(selected_training_features, training_features.y,
                   selected_test_features, test_features.y).get_accuracy()
print("LDA accuracy:", lda_accuracy)
