import numpy as np

class ConfusionMatrix:
    def __init__(self, y_true, y_pred, labels=None):
        self.y_true = y_true
        self.y_pred = y_pred
        self.labels = labels or sorted(set(y_true) | set(y_pred))
        self.matrix = None

    def compute(self):

        pass

    def get(self):
        pass

    def print_matrix(self):

        pass

    def plot(self, normalize=False, cmap='Blues'):

        pass

    def get_class_stats(self, class_label):
        pass