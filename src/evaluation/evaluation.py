import numpy as np
import matplotlib.pyplot as plt


def print_mean_accuracies(accuracies):
    for classifier in accuracies.keys():
        acc = accuracies[classifier]
        acc_mean = np.mean(acc)
        std_mean = np.std(accuracies[classifier])
        print(f"{classifier} - Mean accuracy: {acc_mean:.4f} +/- {std_mean:.4f}")


def plot_accuracies_by_subjects(subjects, accuracies):
    x = np.arange(len(subjects))
    bar_width = .30

    fig, ax = plt.subplots()
    for (i, classifier) in enumerate(accuracies.keys()):
        i = i + 1
        position = x if i % 3 == 0 else x - bar_width * i
        ax.bar(position, accuracies[classifier], bar_width, label=classifier)

    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracies by subjects")
    ax.set_xticks(x)
    ax.set_xticklabels(subjects)
    ax.legend()

    fig.tight_layout()

    plt.show()
