from data_preparation import load_csv, extract_single_trial
from signal_processing import filter_bank

TIME_WINDOW = 750

# Load training data
left_hand_training_data = load_csv("data/bnci/by-subject-complete/lefthand-training-subject-2.csv")
right_hand_training_data = load_csv("data/bnci/by-subject-complete/righthand-training-subject-2.csv")

# Epoch data
left_training = extract_single_trial(left_hand_training_data, TIME_WINDOW)
right_training = extract_single_trial(right_hand_training_data, TIME_WINDOW)

# Filter Bank
left_bands_training = filter_bank(left_training)
right_bands_training = filter_bank(right_training)
