import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg

class_1 = np.array([
    [144, 221],
    [217, 316],
    [292, 413],
    [418, 560]
])

class_2 = np.array([
    [144, 557],
    [217, 469],
    [292, 348],
    [418, 217]
])

plt.scatter(class_1[:, 0], class_1[:, 1], c="r")
plt.scatter(class_2[:, 0], class_2[:, 1], c="b")

plt.title("Before CSP filtering")
plt.show()

# Compute the csp params
s1 = np.cov(class_1.T)
s2 = np.cov(class_2.T)

w, v = linalg.eigh(s2, s1+s2)

# CSP requires the eigenvalues and the eig-vectors be sorted in descending order
order_mask = np.argsort(w)
order_mask = order_mask[::-1]
w = w[order_mask]
v = v[:, order_mask]

# f = v[:, [0, -1]]
f = v

class_1_filtered = np.dot(class_1, f)
class_2_filtered = np.dot(class_2, f)

plt.scatter(class_1_filtered[:, 0], class_1_filtered[:, 1], c="r")
plt.scatter(class_2_filtered[:, 0], class_2_filtered[:, 1], c="b")

plt.title("After CSP filtering")
plt.show()
