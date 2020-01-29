import numpy as np
import mne
from src.data_preparation.data_preparation import read_eeg_file
from src.algorithms.csp.CSP import CSP
from src.signal_processing.signal_processing import apply_bandpass_filter

TIME_LENGTH = 250
TIME_WINDOW = 250
EPOCH_SIZE = None
DATA_FOLDER = "data/bci-iv-a/subject-independent"

subject = 2


def plot_topomap(ch_names, epoch):
    standard_montage = mne.channels.make_standard_montage("standard_1020")
    info = mne.create_info(ch_names=ch_names,
                           sfreq=100, ch_types="eeg", montage=standard_montage)

    raw = mne.EvokedArray(epoch.T, info, 0)
    raw.plot_topomap(cmap="Spectral_r", contours=4,
                     outlines="skirt", show_names=True)

# Load training data
print("Loading data ...")
left_data_file = f"{DATA_FOLDER}/left-hand-subject-{subject}.csv"
right_data_file = f"{DATA_FOLDER}/right-hand-subject-{subject}.csv"
eeg = read_eeg_file(left_data_file, right_data_file, TIME_LENGTH, TIME_WINDOW, EPOCH_SIZE)
apply_bandpass_filter(eeg)

n_epoch = 25

left_epoch = eeg.left_data[n_epoch, :, :]
right_epoch = eeg.right_data[n_epoch, :, :]

ch_names = ["FC1", "FC2", "C1", "C2", "C3", "C4", "CP1", "CP2"]
#
# plot_topomap(ch_names, left_epoch)
# plot_topomap(ch_names, right_epoch)

csp = CSP(average_trial_covariance=True, n_components=8)
csp.fit(eeg.left_data, eeg.right_data)

left_epoch_projected = csp.project(left_epoch)
right_epoch_projected = csp.project(right_epoch)

plot_topomap(ch_names, left_epoch_projected)
plot_topomap(ch_names, right_epoch_projected)
