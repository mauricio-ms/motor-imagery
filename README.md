# Motor Imagery
Project to analyze the accuracy of multiple algorithms published in papers about EEG binary motor imagery problem.

The motor imagery (MI) based BCI is capable of translating the Subject’s movement intention 
to controls the external devices. Imagining a movement or performing an action mentally is
known as MI. In MI tasks, subjects are instructed to imagine themselves performing a specific
motor action (e.g. hand, foot) without overt motor output and each task is treated as a MI class.

# Notes
* Was noted that in the covariance matrices estimation in the CSP algorithm, is better to use metric 
logeuclid or euclid instead riemman, because the accuracy is practically the same, and the riemman is much slower.

# Results

## FBCSP
> BCI competition III dataset IV-a

    > Accuracies obtained in reproduction of the article in fbcsp folder
    GNB: 87.3366 +/- 9.3793
    SVM: 87.1405 +/- 9.9846
    LDA: 86.3889 +/- 10.1286

    > Accuracies obtained applying manual selection of 8 electrodes in motor cortex area
    GNB: 80.8725 +/- 12.0977
    SVM: 81.9150 +/- 10.2631
    LDA: 82.0229 +/- 10.4780

## SI-BCI
> BCI competition III dataset IV-a

    > Accuracies obtained in reproduction of the article in si_bci folder
    SVM
        CSP: 88.1429 +/- 11.1767
        KATZ_FRACTAL: 90.2143 +/- 8.3989
    LDA
        CSP: 87.5714 +/- 10.7033
        KATZ_FRACTAL: 90.8571 +/- 7.6884

    > Accuracies obtained applying manual selection of 8 electrodes in motor cortex area
    SVM
        CSP: 81.2857 +/- 11.9493
        KATZ_FRACTAL: 81.4286 +/- 11.9352
    LDA
        CSP: 80.2857 +/- 11.1584
        KATZ_FRACTAL: 81.6429 +/- 9.3503
        
## SPECIFIC-BAND-CSP-FEATURES
> BCI competition III dataset IV-a

    > Accuracies obtained in reproduction of the article in specific_band_csp_features folder
    GNB: 81.9286 +/- 11.4377
    SVM: 87.9286 +/- 8.0143
    LDA: 86.6429 +/- 9.8408

    > Accuracies obtained applying manual selection of 8 electrodes in motor cortex area
    GNB: 76.1429 +/- 11.8330
    SVM: 81.7143 +/- 14.7365
    LDA: 83.0714 +/- 12.5318
