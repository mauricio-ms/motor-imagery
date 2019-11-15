import numpy as np
import matplotlib.pyplot as plt


def print_accuracies(accuracies, subjects):
    for k in accuracies.keys():
        print("================== k: ", k)
        for subject in subjects:
            print("Subject %s: %s" % (subject, accuracies[k][subject - 1]))
        print("Mean accuracy: %s\n\n" % (np.mean(accuracies[k])))


def print_mean_accuracies(accuracies):
    for k in accuracies.keys():
        print("k %s: %s" % (k, np.mean(accuracies[k])))


def plot_accuracies_by_subjects(subjects, accuracies):
    x = np.arange(len(subjects))
    bar_width = .35

    fig, ax = plt.subplots()
    for (i, classifier) in enumerate(accuracies.keys()):
        position = x - bar_width / 2 if i % 2 == 0 else x + bar_width / 2
        ax.bar(position, accuracies[classifier], bar_width, label=classifier)

    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracies by subjects")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend()

    fig.tight_layout()

    plt.show()
