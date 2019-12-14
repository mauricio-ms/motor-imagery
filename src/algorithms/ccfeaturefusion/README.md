Implementation based on article:

> A performance based feature selection technique for subject independent MI based BCI

## Cross‑correlation
In signal processing, cross-correlation (CC) is used to measure the degree of similarity of two signals.

The main reason for applying cross correlation techniques in the
feature extraction of motor imagery signal is that multichannel 
EEG signals are highly correlated and typically cannot provide 
independent information about brain activity. 
Additionally different channels from different locations of the brain 
do not contain equal amounts of discriminative information for any particular task.

## Feature Extraction
- Time domain features contains the time domain information.
- Frequency domain features captures the frequency information.
- Wavelet based features contain both time and frequency information.
- Fractal dimension features measure the degree of roughness of the signal.
- Statistical features indicate the characteristics of the original signal without overlapping of information.





---
The basic principle of cross-correlation is to measure the similarity in shape
between two signals which makes it suitable for EEG analysis

The cross-correlation technique
offers low computational complexity and minimizes the
effect of random noise present in EEG signals