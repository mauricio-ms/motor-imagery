from src.algorithms.scssp.SpatioSpectralCovariances import SpatioSpectralCovariances
import numpy as np


class SpatioSpectralCovariancesEstimation:
    def __init__(self, eeg):
        self.left_covariances = self.__compute_covariances(eeg.left_data)
        self.right_covariances = self.__compute_covariances(eeg.right_data)

    def __compute_covariances(self, eeg_data):
        x = self.__concatenate_epochs(eeg_data)

        n_f = x.shape[0]
        n_i = x.shape[1]
        n_ch = x.shape[2]

        spatial_covariance = np.zeros((n_ch, n_ch))
        for i in range(0, n_i):
            spatial_covariance = spatial_covariance + np.dot(x[:, i, :].T, x[:, i, :])
        spatial_covariance = np.divide(spatial_covariance, n_f * n_i)

        spectral_covariance = np.zeros((n_f, n_f))
        for i in range(0, n_i):
            spectral_covariance = spectral_covariance + np.dot(x[:, i, :], x[:, i, :].T)
        spectral_covariance = np.divide(spectral_covariance, n_ch * n_i)

        return SpatioSpectralCovariances(spatial_covariance, spectral_covariance)

    @staticmethod
    def __concatenate_epochs(eeg_data):
        n_epochs = eeg_data.shape[1]
        n_ch = eeg_data.shape[3]

        n_f = eeg_data.shape[0]
        n_t = eeg_data.shape[2]

        n_i = n_t * n_epochs

        return np.reshape(eeg_data, (n_f, n_i, n_ch))