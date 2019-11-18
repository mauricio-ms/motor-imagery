"""
Script to optimize the parameters of the filter used in FBCSP algorithm
"""
from scipy import signal

import numpy as np
import matplotlib.pyplot as plt

low_freq = 4
high_freq = 40
fs = 250
# b, a = signal.iirfilter(8, [low_freq, high_freq], btype="bandpass", fs=fs)
# b, a = signal.butter(9, [low_freq, high_freq], btype="bandpass", fs=fs)
b, a = signal.cheby2(5, 50, [low_freq, high_freq], btype="bandpass", fs=fs)
w, h = signal.freqz(b, a)

# bands = (0, 2, 4, 40, 45, 60, 80, 125)
# desired = (0, 0, 1, 1, 0, 0, 0, 0)
# fir_firls = signal.firls(37, bands, desired, fs=fs)
# w, h = signal.freqz(fir_firls)

plt.plot((fs * 0.5 / np.pi) * w, abs(h))
# plt.semilogx(w, 20 * np.log10(abs(h)))
plt.xlabel('Frequency')
plt.ylabel('Amplitude response [dB]')
plt.grid()
plt.show()
