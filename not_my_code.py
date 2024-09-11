"""
This code was copied from https://medium.com/analytics-vidhya/generation-of-a-concatenated-confusion-matrix-in-cross-validation-912485c4a972
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from os.path import join

from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_test_list, predicted_labels_list, class_names, save_dir: str | None = None
):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)

    save_figures = False
    if save_dir:
        save_figures = True
        Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Plot non-normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(
        cnf_matrix,
        classes=class_names,
        title=None if save_figures else "Confusion matrix, without normalization",
    )
    if save_figures:
        plt.savefig(join(save_dir, f"confusion_matrix.pdf"))
        # save figures without showing
        plt.close()
    else:
        plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    generate_confusion_matrix(
        cnf_matrix,
        classes=class_names,
        normalize=True,
        title=None if save_figures else "Normalized confusion matrix",
    )
    if save_figures:
        plt.savefig(join(save_dir, f"confusion_matrix_normalized.pdf"))
        plt.close()
    else:
        plt.show()


def generate_confusion_matrix(
    cnf_matrix, classes, normalize=False, title: str | None = "Confusion matrix"
):
    if normalize:
        cnf_matrix = cnf_matrix.astype("float") / cnf_matrix.sum(axis=1)[:, np.newaxis]
        print("Showing or saving normalized confusion matrix")
    else:
        print("Showing or saving confusion matrix, without normalization")

    plt.imshow(cnf_matrix, interpolation="nearest", cmap=plt.get_cmap("Blues"))
    if title:
        plt.title(title)
    plt.colorbar()

    # Move x-axis tick marks and labels to top, but hid tick marks (last 3
    # options)
    plt.tick_params(
        top=True,
        labeltop=True,
        bottom=False,
        labelbottom=False,
        axis="both",
        which="both",
        length=0,
    )

    # Move x-axis label to top (alternate way that results in wonky figure)
    # ax.xaxis.set_ticks_position('top')

    # Move x label to top
    ax = plt.gca()
    ax.xaxis.set_label_position("top")

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes, rotation=0)

    fmt = ".2f" if normalize else "d"
    thresh = cnf_matrix.max() / 2.0

    for i, j in itertools.product(
        range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])
    ):
        plt.text(
            j,
            i,
            format(cnf_matrix[i, j], fmt),
            horizontalalignment="center",
            color="white" if cnf_matrix[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    return cnf_matrix


# Copied from https://towardsdatascience.com/multi-class-classification-extracting-performance-metrics-from-the-confusion-matrix-b379b427a872
def get_performance_metrics(cnf_matrix):
    np.set_printoptions(precision=2)

    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)

    # Specificity or true negative rate
    TNR = TN / (TN + FP)

    # Precision or positive predictive value
    PPV = TP / (TP + FP)

    # Negative predictive value
    NPV = TN / (TN + FN)

    # Fall out or false positive rate
    FPR = FP / (FP + TN)

    # False negative rate
    FNR = FN / (TP + FN)

    # False discovery rate
    FDR = FP / (TP + FP)

    # Overall accuracy for each class
    ACC = (TP + TN) / (TP + FP + FN + TN)

    # Balanced accuracy
    BA = (TPR + TNR) / 2

    # F1 score
    F1 = 2 * TP / (2 * TP + FP + FN)

    return {
        "tp": TP,
        "tn": TN,
        "fp": FP,
        "fn": FN,
        "tpr": TPR,
        "recall": TPR,
        "tnr": TNR,
        "specificity": TNR,
        "ppv": PPV,
        "precision": PPV,
        "npv": NPV,
        "fpr": FPR,
        "fnr": FNR,
        "fdr": FDR,
        "accuracy": ACC,
        "balanced_accuracy": BA,
        "f1_score": F1,
    }
