from scipy import signal


# TODO This can be a class to decouple the filter used by the Filterbank class
def cheby2(order, rs, low_freq, high_freq, fs):
    b, a = signal.cheby2(order, rs, [low_freq, high_freq], btype="bandpass", fs=fs)
    return lambda data: __filtfilt(b, a, data)


def __filtfilt(b, a, data):
    return signal.filtfilt(b, a, data, axis=0)
