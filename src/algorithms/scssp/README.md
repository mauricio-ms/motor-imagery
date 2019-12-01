The method was proposed in the article: Separable Common Spatio-Spectral Patterns for Motor Imagery BCI Systems

# Dataset
- Data of 120 repetitions of each MI class were available for each person in total.
- http://bnci-horizon-2020.eu/database/data-sets (4. Two class motor imagery (004-2014))

# Advantages compared to the FBCSP method
- Has significantly less computational cost for training since it requires
training of only two CSP-type modules instead of Nf modules in FBCSP.

- The features are extracted based on joint
analysis of both spatial and spectral characteristics of the signal.

- A measure is provided to rank the discriminatory power of 
extracted spatio-spectral features, which enables us to directly 
perform dimensionality reduction without any need to deploy 
a separate subsequent feature extraction/selection module.