import numpy as np
from scipy import signal

FS = 250
NYQUIST = 0.5 * FS

LOW_FREQS = range(4, 41, 4)
QUANTITY_BANDS = LOW_FREQS[-1]//LOW_FREQS[0]-1


def filter_in_all_frequency_bands(trial):
    n_channels = trial.shape[1]
    filtered_signals = np.zeros((QUANTITY_BANDS, *trial.shape))
    for n_low_freq in range(0, len(LOW_FREQS)):
        low_freq = LOW_FREQS[n_low_freq]
        if low_freq == LOW_FREQS[-1]:
            break

        high_freq = LOW_FREQS[n_low_freq+1]

        # Create a 5 order butter-worth filter to the specific band (low_freq - high_freq)
        b, a = signal.butter(5, [low_freq/NYQUIST, high_freq/NYQUIST], btype="bandpass")

        for n_channel in range(n_channels):
            filtered_signals[n_low_freq, :, n_channel] = signal.filtfilt(b, a, trial[:, n_channel])

    return filtered_signals


def filter_bank(eeg):
    filtered_signals = np.zeros((QUANTITY_BANDS, *eeg.shape))
    for n_trial in range(eeg.shape[0]):
        trial = eeg[n_trial, :, :]
        filtered_signals[:, n_trial, :, :] = filter_in_all_frequency_bands(trial)

    return filtered_signals
