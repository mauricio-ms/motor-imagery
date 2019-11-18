from models.data_preparation.data_preparation import read_eeg_file
from models.fbcsp.FilterBankCSPFeatureExtraction import FilterBankCSPFeatureExtraction
from models.fbcsp.MIBIFFeatureSelection import MIBIFFeatureSelection
from models.classifiers.SVM import SVM
from models.classifiers.LDA import LDA


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
svm_accuracy = SVM("linear", 0.8, True,
                   features.training_features, features.training_labels,
                   features.test_features, features.test_labels).get_accuracy()
print("SVM accuracy:", svm_accuracy)

# LDA classifier
lda_accuracy = LDA(features.training_features, features.training_labels,
                   features.test_features, features.test_labels).get_accuracy()
print("LDA accuracy:", lda_accuracy)
