import numpy as np

class PerformanceMatrix():
    def __init__(self, predicted, actual):
        self.predicted = predicted
        self.actual = actual

    def accuracy(self):
        ...

    def precision(self):
        ...

    def recall(self):
        ...

    def f1_score(self):
        ...

    def calculate_score(self, metric: str):
        ...

