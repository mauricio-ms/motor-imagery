from src.utils.array_helper import flat
from scipy import linalg

import numpy as np


class EigenvaluesProblems:
    def __init__(self, spatio_spectral_covariances, d=2):
        left_covariances = spatio_spectral_covariances.left_covariances
        right_covariances = spatio_spectral_covariances.right_covariances

        v_l, self.W_L = linalg.eigh(left_covariances.spectral, left_covariances.spectral + right_covariances.spectral)
        v_r, self.W_R = linalg.eigh(left_covariances.spatial, left_covariances.spatial + right_covariances.spatial)

        self.d = d
        self.__d_indexes = flat([[i, -i-1] for i in range(0, d)])
        self.lambda_k = self.__compute_eigenvalues_indexes(v_l, v_r)

    def __compute_eigenvalues_indexes(self, v_l, v_r):
        i = 0
        size_eigenvalues = len(v_l) * len(v_r)
        eigenvalues = np.zeros(size_eigenvalues)
        lambda_k = np.zeros((size_eigenvalues, 2))
        for p in range(0, len(v_l)):
            for q in range(0, len(v_r)):
                eigenvalues_product = v_l[p] * v_r[q]
                eigenvalues[i] = eigenvalues_product / (eigenvalues_product + (1 - v_l[p]) * (1 - v_r[q]))
                lambda_k[i, :] = [p, q]
                i = i + 1

        # Sort the eigenvalues in descending way
        desc_idx = np.argsort(-eigenvalues)
        lambda_k = lambda_k[desc_idx]

        return lambda_k[self.__d_indexes, :]

    def compute_eigenvector(self, k):
        p, q = self.lambda_k[k]
        return np.kron(self.W_R[:, int(q)], self.W_L[:, int(p)])
