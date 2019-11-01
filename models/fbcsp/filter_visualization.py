"""
Script to optimize the parameters of the filter used in FBCSP algorithm
"""
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt

low_freq = 20
high_freq = 30
b, a = signal.cheby2(5, 0.5, [low_freq, high_freq], btype="bandpass", fs=250)

w, h = signal.freqs(b, a)

plt.semilogx(w, 20 * np.log10(abs(h)))
plt.xlabel('Frequency')
plt.ylabel('Amplitude response [dB]')
plt.grid()
plt.show()
