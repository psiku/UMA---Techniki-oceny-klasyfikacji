from sklearn.metrics import confusion_matrix
from .confusion_matrix import ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def compare_confusion_matrices(y_true, y_pred, labels=None, normalize=False):
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    our_cm = ConfusionMatrix(y_true, y_pred, labels=labels)
    our_matrix = our_cm.get()
    if normalize:
        our_matrix_plot = our_matrix.astype(np.float64)
        our_matrix_plot /= our_matrix_plot.sum(axis=1, keepdims=True)
    else:
        our_matrix_plot = our_matrix

    sk_matrix = confusion_matrix(y_true, y_pred, labels=labels)
    if normalize:
        sk_matrix_plot = sk_matrix.astype(np.float64)
        sk_matrix_plot /= sk_matrix_plot.sum(axis=1, keepdims=True)
    else:
        sk_matrix_plot = sk_matrix

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(our_matrix_plot, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=labels, yticklabels=labels, cmap='Blues', ax=axes[0])
    axes[0].set_title('Our Confusion Matrix' + (' (Normalized)' if normalize else ''))
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')

    sns.heatmap(sk_matrix_plot, annot=True, fmt='.2f' if normalize else 'd',
                xticklabels=labels, yticklabels=labels, cmap='Greens', ax=axes[1])
    axes[1].set_title('Sklearn Confusion Matrix' + (' (Normalized)' if normalize else ''))
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')

    plt.tight_layout()
    plt.show()
