import numpy as np
from collections import Counter
from typing import Dict, Union, Optional


class PerformanceMetrics():
    def __init__(self, predicted, actual):
        self.predicted = np.array(predicted)
        self.actual = np.array(actual)

        if self.predicted.shape != self.actual.shape:
            raise ValueError("Predicted and actual arrays must have the same shape.")

        self.labels = np.unique(np.concatenate((self.actual, self.predicted)))
        self.counts = Counter(self.actual)

    def _per_class(self):
        per_class = {}
        total = len(self.actual)

        for label in self.labels:
            tp = np.sum((self.predicted == label) & (self.actual == label))
            fp = np.sum((self.predicted == label) & (self.actual != label))
            fn = np.sum((self.predicted != label) & (self.actual == label))
            tn = total - tp - fp - fn
            per_class[label] = {
                'tp': tp,
                'fp': fp,
                'fn': fn,
                'tn': tn,
            }
        return per_class

    def _aggregate(self, per_class_vals: Dict[str, float], average: Optional[str]) -> Union[float, Dict[str, float]]:
        if average == 'per_class':
            return per_class_vals
        if average == 'macro':
            return np.mean(list(per_class_vals.values()))
        if average == 'weighted':
            total = len(self.actual)
            return sum(val * self.counts[label]
                       for label, val in per_class_vals.items()) / total
        if average is None or average == 'micro':
            return None
        raise ValueError(f"Unsupported metric: {average}")

    def accuracy(self, average: Optional[str] = None) -> Union[float, Dict[str, float]]:
        per = self._per_class()
        per_acc = {
            lbl: (v['tp'] / (v['tp'] + v['fn']))  if (v['tp'] + v['fn']) > 0 else 0.0 for lbl, v in per.items()
        }
        if average in (None, 'micro'):
            return np.mean(self.predicted == self.actual)
        return self._aggregate(per_acc, average)

    def precision(self, average: str = 'micro') -> Union[float, Dict[str, float]]:
        per = self._per_class()
        per_prec = {lbl: (v['tp'] / (v['tp'] + v['fp']))
                    if (v['tp'] + v['fp']) > 0 else 0.0
                    for lbl, v in per.items()}

        if average == 'micro':
            total_tp = sum(v['tp'] for v in per.values())
            total_fp = sum(v['fp'] for v in per.values())
            return total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        return self._aggregate(per_prec, average)

    def recall(self, average: str = 'micro') -> Union[float, Dict[str, float]]:
        per = self._per_class()
        per_rec = {lbl: (v['tp'] / (v['tp'] + v['fn']))
                   if (v['tp'] + v['fn']) > 0 else 0.0
                   for lbl, v in per.items()}

        if average == 'micro':
            total_tp = sum(v['tp'] for v in per.values())
            total_fn = sum(v['fn'] for v in per.values())
            return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        return self._aggregate(per_rec, average)

    def f1_score(self, average: str = 'micro') -> Union[float, Dict[str, float]]:
        per = self._per_class()
        per_f1 = {}
        for lbl, v in per.items():
            denom = 2 * v['tp'] + v['fp'] + v['fn']
            per_f1[lbl] = (2 * v['tp'] / denom) if denom > 0 else 0.0

        if average == 'micro':
            tot_tp = sum(v['tp'] for v in per.values())
            tot_fp = sum(v['fp'] for v in per.values())
            tot_fn = sum(v['fn'] for v in per.values())
            denom = 2 * tot_tp + tot_fp + tot_fn
            return (2 * tot_tp / denom) if denom > 0 else 0.0

        return self._aggregate(per_f1, average)

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
