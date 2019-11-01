import numpy as np


def print_accuracies(accuracies, subjects):
    for k in accuracies.keys():
        print("================== k: ", k)
        for subject in subjects:
            print("Subject %s: %s" % (subject, accuracies[k][subject - 1]))
        print("Mean accuracy: %s\n\n" % (np.mean(accuracies[k])))


def print_mean_accuracies(accuracies):
    for k in accuracies.keys():
        print("k %s: %s" % (k, np.mean(accuracies[k])))
