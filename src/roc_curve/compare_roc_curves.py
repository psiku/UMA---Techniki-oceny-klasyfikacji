from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, roc_auc_score
from .roc_curve import ROC
import matplotlib.pyplot as plt
import numpy as np


def compare_roc_curves(y_true, y_proba, class_names=None):
    classes = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=classes)

    if class_names is None:
        class_names = [str(cls) for cls in classes]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for i, cls in enumerate(classes):
        roc = ROC(y_true_bin[:, i], y_proba[:, i])
        roc.compute()
        axes[0].plot(roc.fpr, roc.tpr, label=f'{class_names[i]} (AUC = {roc.auc:.2f})')

    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[0].set_title("Our ROC Curves")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend()
    axes[0].grid(True)

    for i, cls in enumerate(classes):
        fpr_sk, tpr_sk, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        auc_sk = roc_auc_score(y_true_bin[:, i], y_proba[:, i])
        axes[1].plot(fpr_sk, tpr_sk, label=f'{class_names[i]} (AUC = {auc_sk:.2f})')

    axes[1].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[1].set_title("Sklearn ROC Curves")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()
