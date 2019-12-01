import numpy as np
from scipy import signal


class FilterBank:
    def __init__(self, low_freq, high_freq, step, overlap=0):
        if (high_freq-low_freq) % step != 0:
            raise Exception("The frequencies cannot be generated with the given step")
        self.low_freq = low_freq
        self.high_freq = high_freq
        self.step = step
        self.overlap = overlap
        self.bands = [self.__obtain_band_frequencies(n_band, freq)
                      for (n_band, freq) in enumerate(range(low_freq, high_freq, step))]

    def __obtain_band_frequencies(self, n_band, freq):
        low_freq = freq if n_band == 0 else freq-self.overlap
        return low_freq, low_freq+self.step

    def apply(self, eeg_data):
        return self.__compute_bands(eeg_data)

    def __compute_bands(self, eeg_data):
        filter_bank = np.zeros((len(self.bands), *eeg_data.shape))
        for n_trial in range(eeg_data.shape[0]):
            trial = eeg_data[n_trial, :, :]
            filter_bank[:, n_trial, :, :] = self.__compute_bands_for_trial(trial)

        return filter_bank

    def __compute_bands_for_trial(self, trial):
        trial_filter_bank = np.zeros((len(self.bands), *trial.shape))
        for (n_band, band) in enumerate(self.bands):
            low_freq, high_freq = band

            # Create a 5 order Chebyshev Type 2 filter to the specific band (low_freq - high_freq)
            b, a = signal.cheby2(6, 0.5, [low_freq, high_freq], btype="bandpass", fs=250)

            trial_filter_bank[n_band, :, :] = signal.filtfilt(b, a, trial, axis=0)

        return trial_filter_bank

