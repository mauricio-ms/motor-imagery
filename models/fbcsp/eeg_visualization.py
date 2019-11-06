import mne
from EEG import EEG
from mne.datasets import sample
from scipy import signal

import numpy as np

print(__doc__)

# data_path = sample.data_path()
# fname = data_path + '/MEG/sample/sample_audvis_raw.fif'

# raw = mne.io.read_raw_fif(fname)
# raw.plot()

subject = 1
fname = f"data/bnci/by-subject-complete/lefthand-test-subject-{subject}.csv"

# Definition of channel types and names.
ch_types = ["eeg", "eeg", "eeg"]
ch_names = ["1", "2", "3"]

info = mne.create_info(ch_names=ch_names, sfreq=250, ch_types=ch_types)

data = EEG(f"data/bnci/by-subject-complete/lefthand-test-subject-{subject}.csv",
           f"data/bnci/by-subject-complete/righthand-test-subject-{subject}.csv", 750, False)

left_data = np.transpose(data.left_data[0], [1, 0])

b, a = signal.cheby2(5, 0.5, [4, 20], btype="bandpass", fs=250)
left_data = signal.filtfilt(b, a, left_data, axis=1)

raw = mne.io.RawArray(left_data[:, 0:100], info)

scalings = {
    "eeg": 2
}
raw.plot(scalings="auto")
