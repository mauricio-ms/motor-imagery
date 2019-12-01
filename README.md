## Motor Imagery
Project to test the accuracy of multiple algorithms published in articles to the EEG binary motor imagery problem.

The motor imagery (MI) based BCI is capable of translating the Subject’s movement intention 
to controls the external devices. Imagining a movement or performing an action mentally is
known as MI. In MI tasks, subjects are instructed to imagine themselves performing a specific
motor action (e.g. hand, foot) without overt motor output and each task is treated as a MI class.

# Pre-processing
A frequency band of 0.5 - 100 Hz is most commonly used during EEG
recording to obtain a wide frequency spectrum for further
analysis and investigation.

# Time Segmentation
- Article 12 says:
* After the feature selection procedure, the average KL-divergence of 
the selected features in each time segment is calculated, and the one 
with the largest value is chosen as the optimal time segment, which will be 
further clarified below. 
To accurately estimate the conditional probability within KL-divergence 
for each single feature, a Parzen window is employed without requiring 
permutation over all combination of features.
* When feature number is determined, we can select the
optimal feature subset and optimal time segment simultaneously.

# Features
- In BCI, frequency band power features and time domain features represent EEG signals. Band power features represent the power of EEG signals
for a given frequency band averaged over a time window and time
domain features are the combination of EEG signals from all channels. MI
BCI extensively uses band power features.

# Feature Selection
- Embedded approaches select the feature subset based on the intrinsic
properties of parameter optimization, but they are unsuitable where 
there are significant interactions between relevant feature.
- In filter approaches, a score that indicates the
importance of a feature is assigned to each individual feature based on an
independent evaluation criterion. Then the filter approaches select the
top ranked features with high scores and discard the rest without the
application of any classifier.
- The wrapper approaches greedily search for discriminative features by classification 
results obtained by a predetermined classifier. However, much more computational expense
as well as poor general ability makes it impractical.
- The article 12 presents a different from the classical filter approaches, 
the proposed method selects those features that offer maximum class separability 
instead of maximum information in terms of Kullback-Leibler (KL) divergence measure.

# Possible optimization parameters
- Filter parameters of the bandpass filters.

# Classifiers
- The classifiers based on deep neural networks have been used in MI BCI
research to improve the accuracy of multi-class signal analysis.

- Regularized Fisher LDA, an enhancement of LDA has also been used for right and left-hand
motor imagery that uses decision boundary or hyperplane in feature
space for classifying features in distinct classes. Fisher LDA obtains better
generalization capabilities and gives better results than LDA.

- Logarithmic band power (LBP) rather than variance. (Article 10)

# Evaluation
- Accuracy:
The accuracy of a measurement system is the degree of closeness
of measurements of a quantity to its actual (true) value.

- Kappa:
Defined as the proportion of correctly classified samples 
after accounting for the probability of chance agreement:

Kappa = (Pr(A) - Pr(E)) / (1 - Pr(E))

where Pr(A) represents the actual observed agreement and Pr(E) represents the probability of expected
agreement by chance.
A κ value might range between 1 and −1, which corresponds to a perfect and a completely incorrect
classification, respectively. On the other hand, a κ with value 0 indicates that the performance is equal to a
random guess.

- 10-fold cross validation:
In this cross validation technique, the dataset
is randomly divided into 10 equal subsets where one of
the subsets is used for the test while rests 9 are used to
the training. The cross validation is repeated 10 times,
and then the results of 10 times are averaged to yield
a single classification rate. The execution time is estimated 
by summing up the expended time during spatial
filtering (training) and classification (testing).

This validation procedure mixes the dataset randomly and divides
into ten equally sized distinct partitions. Each partition is then
used for testing, while other partitions are used for training the
model. This results in ten different error rates or accuracy, which
are averaged. This is the error of tenfold cross validation. To
further improve the estimate, the procedure is repeated ten times
and all error rates over these ten runs are again averaged [3].
The average accuracy or error rate over ten runs obtained for the
test data is taken as the performance evaluation criteria, which
is named as validation accuracy or validation error rate of one
subject.

- Average Power Spectral Density Plot:
The average power spectral density of two-class
EEG signals by frequencies can be generated to
analyse if the band selection algorithm is working
appropriately, the bands with good discrimination
should have different values 

# BCI
- The MobileSim is a software that can be used as a robot simulator, the article 11 used this approach.
- Uma abordagem legal seria conectar a BCI para jogar o jogo do dinossauro do Google Chrome.
- Since motor-imagery BCI systems are mostly designed for
long-term utilization by the user, it is usually assumed that
the BCI algorithm has access to a training dataset with long
enough trials, which are collected over at least two different
recording sessions.

## BCI can be designed in two ways: 
- Subject-dependent (SD) based BCI and Subject-independent based BCI. 
In the SD based BCI, the training part in the MI classification requires 
a significant number of EEG signal features for recognizing MI patterns 
of a specific subject  with an acceptable performance.
The SD based BCI system requires long recording sessions and afterwards 
several training sessions for the Subject to be able to use the system.
- In the SI based BCI, the systems are trained with the data of a group 
of subjects instead of a single subject.

# Possible next implementations
- Article 10
- Article 13