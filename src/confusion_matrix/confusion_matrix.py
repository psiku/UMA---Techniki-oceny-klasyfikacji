import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ConfusionMatrix:
    def __init__(self, y_true, y_pred, labels=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels or sorted(set(y_true) | set(y_pred))
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
        self.matrix = None

    def compute(self):
        n = len(self.labels)
        self.matrix = np.zeros((n, n), dtype=int)
        for t, p in zip(self.y_true, self.y_pred):
            i = self.label_to_index[t]
            j = self.label_to_index[p]
            self.matrix[i, j] += 1
        return self.matrix

    def get(self):
        if self.matrix is None:
            self.compute()
        return self.matrix

    def print_matrix(self):
        if self.matrix is None:
            self.compute()
        print("Confusion Matrix:")
        print("Labels:", self.labels)
        print(self.matrix)

    def plot(self, normalize=False, cmap='Blues'):
        if self.matrix is None:
            self.compute()
        matrix = self.matrix.astype(np.float64)
        if normalize:
            matrix = matrix / matrix.sum(axis=1, keepdims=True)
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, fmt='.2f' if normalize else '.0f',
                    xticklabels=self.labels, yticklabels=self.labels, cmap=cmap)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
        plt.show()

    # TP FN
    # FP TN
    def get_class_stats(self, class_label):
        if self.matrix is None:
            self.compute()
        idx = self.label_to_index[class_label]
        TP = self.matrix[idx, idx]
        FP = self.matrix[:, idx].sum() - TP
        FN = self.matrix[idx, :].sum() - TP
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        return {"precision": precision, "recall": recall, "f1": f1}