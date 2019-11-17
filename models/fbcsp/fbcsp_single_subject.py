"""
Implementation mainly based on the paper:
    Filter bank common spatial pattern algorithm on BCI competition IV Datasets 2a and 2b
"""

from EEG import EEG
from FilterBankCSPFeatureExtraction import FilterBankCSPFeatureExtraction
from MIBIFFeatureSelection import MIBIFFeatureSelection
from Svm import Svm
from Lda import Lda


TIME_WINDOW = 750
EPOCH_SIZE = None
CSP_RELEVANT_FEATURES = 2

subject = 1

# Load training data
print("Loading training data ...")
training_data = EEG(f"data/bnci/by-subject-complete/lefthand-training-subject-{subject}.csv",
                    f"data/bnci/by-subject-complete/righthand-training-subject-{subject}.csv",
                    TIME_WINDOW, epoch_size=EPOCH_SIZE)
# bandpass_filter(training_data)

print(training_data.left_data.shape)

# Load test data
print("Loading test data ...")
test_data = EEG(f"data/bnci/by-subject-complete/lefthand-test-subject-{subject}.csv",
                f"data/bnci/by-subject-complete/righthand-test-subject-{subject}.csv",
                TIME_WINDOW, False, EPOCH_SIZE)
# bandpass_filter(test_data)

# Feature extraction
features = FilterBankCSPFeatureExtraction(training_data, test_data)

# SVM classifier
svm_accuracy = Svm("linear", 0.8, True,
                   features.training_features, features.training_labels,
                   features.test_features, features.test_labels).get_accuracy()
print("SVM accuracy:", svm_accuracy)

# LDA classifier
lda_accuracy = Lda(features.training_features, features.training_labels,
                   features.test_features, features.test_labels).get_accuracy()
print("LDA accuracy:", lda_accuracy)
