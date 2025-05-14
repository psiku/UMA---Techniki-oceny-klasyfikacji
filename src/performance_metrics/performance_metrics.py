import numpy as np
from collections import Counter


class PerformanceMetrics():
    def __init__(self, predicted, actual):
        self.predicted = np.array(predicted)
        self.actual = np.array(actual)
        self.labels = np.unique(np.concatenate((self.actual, self.predicted)))

        assert len(self.actual) == len(self.predicted)
        self.class_count = Counter(self.actual)

    def accuracy(self, average: str = None):
        per_class_accuracy = {}
        class_counts = self.class_count

        for label in self.labels:
            mask = self.actual == label
            correct = (self.predicted[mask] == self.actual[mask]).sum()
            total = mask.sum()
            per_class_accuracy[label] = correct / total if total > 0 else 0.0

        if average is None or average == 'micro':
            correct = (self.predicted == self.actual).sum()
            return correct / len(self.predicted)

        elif average == 'macro':
            return np.mean(list(per_class_accuracy.values()))

        elif average == 'weighted':
            total_samples = len(self.actual)
            weighted_sum = sum(per_class_accuracy[label] * class_counts[label] for label in self.labels)
            return weighted_sum / total_samples

        elif average == 'per_class':
            return per_class_accuracy

        else:
            raise ValueError(f"Unsupported average type: {average}")

    def precision(self, average: str):
        per_class_tp = {}
        per_class_fp = {}
        class_counts = self.class_count

        for label in self.labels:
            per_class_tp[label] = np.sum((self.predicted == label) & (self.actual == label))
            per_class_fp[label] = np.sum((self.predicted == label) & (self.actual != label))

        per_class_precision = {}
        for label in self.labels:
            tp = per_class_tp[label]
            fp = per_class_fp[label]
            precision = tp / (tp + fp)
            per_class_precision[label] = precision

        if average == 'macro':
            return np.mean(list(per_class_precision.values()))

        elif average == 'weighted':
            total_samples = len(self.actual)
            weighted_sum = sum(per_class_precision[label] * class_counts[label] for label in self.labels)
            return weighted_sum / total_samples

        elif average == 'micro':
            tp = np.sum(list(per_class_tp.values()))
            fp = np.sum(list(per_class_fp.values()))
            return tp / (tp + fp)

        elif average == 'per_class':
            return per_class_precision

        else:
            raise ValueError(f"Unsupported average type: {average}")

    def recall(self, average: str = 'micro'):
        per_class_tp = {}
        per_class_fn = {}
        class_counts = self.class_count

        for label in self.labels:
            per_class_tp[label] = np.sum((self.predicted == label) & (self.actual == label))
            per_class_fn[label] = np.sum((self.predicted != label) & (self.actual == label))

        per_class_recall = {}
        for label in self.labels:
            tp = per_class_tp[label]
            fn = per_class_fn[label]
            recall = tp / (tp + fn)
            per_class_recall[label] = recall

        if average == 'macro':
            return np.mean(list(per_class_recall.values()))

        elif average == 'weighted':
            total_samples = len(self.actual)
            weighted_sum = sum(per_class_recall[label] * class_counts[label] for label in self.labels)
            return weighted_sum / total_samples

        elif average == 'micro':
            tp = np.sum(list(per_class_tp.values()))
            fn = np.sum(list(per_class_fn.values()))
            return tp / (tp + fn)

        elif average == 'per_class':
            return per_class_recall

        else:
            raise ValueError(f"Unsupported average type: {average}")

    def f1_score(self, average: str = 'micro'):
        per_class_tp = {label: np.sum((self.predicted==label)&(self.actual==label))
                        for label in self.labels}
        per_class_fp = {label: np.sum((self.predicted==label)&(self.actual!=label))
                        for label in self.labels}
        per_class_fn = {label: np.sum((self.predicted!=label)&(self.actual==label))
                        for label in self.labels}

        per_class_f1 = {}
        for label in self.labels:
            tp = per_class_tp[label]
            fp = per_class_fp[label]
            fn = per_class_fn[label]
            denom = 2*tp + fp + fn
            per_class_f1[label] = (2*tp / denom) if denom>0 else 0.0

        if average == 'per_class':
            return per_class_f1

        elif average == 'macro':
            return np.mean(list(per_class_f1.values()))

        elif average == 'weighted':
            total = len(self.actual)
            return sum(per_class_f1[label] * self.class_count[label]
                    for label in self.labels) / total

        elif average == 'micro':
            tp = sum(per_class_tp.values())
            fp = sum(per_class_fp.values())
            fn = sum(per_class_fn.values())
            denom = 2*tp + fp + fn
            return (2*tp / denom) if denom>0 else 0.0

        else:
            raise ValueError(f"Unsupported average type: {average}")

    def calculate_metric(self, metric: str, average: str = 'micro'):
        if metric == 'accuracy':
            return self.accuracy(average=average)
        elif metric == 'precision':
            return self.precision(average=average)
        elif metric == 'recall':
            return self.recall(average=average)
        elif metric == 'f1_score':
            return self.f1_score(average=average)
        else:
            raise ValueError(f"Unsupported metric: {metric}")