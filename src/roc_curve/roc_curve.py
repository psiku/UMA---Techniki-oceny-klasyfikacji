import numpy as np
import matplotlib.pyplot as plt


class ROC:
    def __init__(self, y_true, y_scores):
        self.y_true = np.array(y_true)
        self.y_scores = np.array(y_scores)
        self.fpr = None
        self.tpr = None
        self.thresholds = None
        self.auc = None

    def compute(self):
        desc_score_indices = np.argsort(-self.y_scores)
        y_scores_sorted = self.y_scores[desc_score_indices]
        y_true_sorted = self.y_true[desc_score_indices]

        P = sum(y_true_sorted)
        N = len(y_true_sorted) - P

        tpr = []
        fpr = []
        tp = 0
        fp = 0

        thresholds = []

        for i in range(len(y_scores_sorted)):
            if y_true_sorted[i] == 1:
                tp += 1
            else:
                fp += 1

            tpr.append(tp / P if P else 0)
            fpr.append(fp / N if N else 0)
            thresholds.append(y_scores_sorted[i])

        self.tpr = np.array(tpr)
        self.fpr = np.array(fpr)
        self.thresholds = np.array(thresholds)
        self.auc = np.trapz(self.tpr, self.fpr)  # pole pod krzywą

    def plot(self):
        if self.fpr is None or self.tpr is None:
            self.compute()

        plt.figure(figsize=(6, 6))
        plt.plot(self.fpr, self.tpr, label=f'ROC curve (AUC = {self.auc:.2f})')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.show()