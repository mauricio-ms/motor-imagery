# Dataset
- Data of 120 repetitions of each MI class were available for each person in total.
- http://bnci-horizon-2020.eu/database/data-sets (4. Two class motor imagery (004-2014))

# Drawbacks
- FBCSP has a high computational cost at the training phase since
it requires a separate feature extractor for each spectral band,
each of which requires calculation of generalized eigenvectors
for spatial covariance matrices.

- Since each spectral band is treated independently, possible correlations 
between different EEG rhythms are ignored by the FBCSP method, 
which in turn causes redundancy in the extracted feature set.

- FBCSP does not provide any measure for comparing discriminant 
power of the features obtained from different spectral bands. 
Although the CSP features within each band are sorted based on 
their discriminant power, it is not possible to sort the features
across different bands.