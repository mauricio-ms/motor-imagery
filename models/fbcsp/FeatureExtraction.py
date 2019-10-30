from signal_processing import filter_bank

import numpy as np


class FeatureExtraction:
    def __init__(self, csp, eeg):
        left_bands_data = filter_bank(eeg.left_data)
        right_bands_data = filter_bank(eeg.right_data)

        self.y = eeg.labels
        self.features = self.extract_features(csp, left_bands_data, right_bands_data)

    def extract_features(self, csp, left_bands, right_bands):
        features = None
        for n_band in range(0, left_bands.shape[0]):
            print("Band ", n_band + 1)
            left_band_training = left_bands[n_band]
            right_band_training = right_bands[n_band]

            x = np.concatenate((left_band_training, right_band_training))

            # Reshape to the format expected by MNE Library
            x = np.transpose(x, [0, 2, 1])

            if n_band == 0:
                features = csp.fit_transform(x, self.y)
            else:
                features = np.concatenate((features, csp.fit_transform(x, self.y)), axis=1)

        return features
