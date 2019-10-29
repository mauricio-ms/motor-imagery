import mne

fname = "/home/mauricio/development/projects/motor-imagery/julia/motor-imagery/data/bnci/gdf/B0101T.gdf"
raw = mne.io.read_raw_gdf(fname, eog=['EOG:ch01', 'EOG:ch02', 'EOG:ch03'])

print(raw.info)

eeg_indices = mne.pick_channels(raw.info['ch_names'], include=['EEG:C3', 'EEG:Cz', 'EEG:C4'])
print(eeg_indices)

info = mne.pick_info(raw.info, eeg_indices)

mne.viz.plot_topomap(raw[0:3, 0:750][0], info)

print("End")
