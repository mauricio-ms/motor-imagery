from data_preparation import read_eeg_file
from FilterBankCSPFeatureExtraction import FilterBankCSPFeatureExtraction
from MIBIFFeatureSelection import MIBIFFeatureSelection
from Svm import Svm
from Lda import Lda


TIME_LENGTH = 750
TIME_WINDOW = 750
EPOCH_SIZE = None
CSP_RELEVANT_FEATURES = 2

subject = 3

# Load training data
print("Loading training data ...")
left_data_file = f"data/bnci/by-subject-complete/lefthand-training-subject-{subject}.csv"
right_data_file = f"data/bnci/by-subject-complete/righthand-training-subject-{subject}.csv"
training_data = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE)

# Load test data
print("Loading test data ...")
left_data_file = f"data/bnci/by-subject-complete/lefthand-test-subject-{subject}.csv"
right_data_file = f"data/bnci/by-subject-complete/righthand-test-subject-{subject}.csv"
test_data = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE, False)

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
