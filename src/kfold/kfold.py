import numpy as np
import pandas as pd
from typing import Any, List, Optional, Union
from collections import defaultdict
from src.performance_metrics.performance_metrics import PerformanceMetrics
from src.models.abstract_classifier import AbstractClassifier


class ManualKFoldValidator:
    def __init__(
        self,
        model: Any,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: Optional[int] = None,
        stratified: bool = False,

    ):
        self.model = model
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.stratified = stratified

    def _make_folds(self, y: Union[np.ndarray, pd.Series]) -> List[np.ndarray]:
        n = len(y)
        idx = np.arange(n)
        rng = np.random.RandomState(self.random_state)
        if self.shuffle:
            rng.shuffle(idx)
        return np.array_split(idx, self.n_splits)

    def _make_stratified_folds(self, y: Union[np.ndarray, pd.Series]) -> List[np.ndarray]:
        y_arr = np.asarray(y)
        labels = np.unique(y_arr)
        buckets = {lab: np.where(y_arr == lab)[0] for lab in labels}
        rng = np.random.RandomState(self.random_state)
        class_splits = {}
        for lab, inds in buckets.items():
            if self.shuffle:
                rng.shuffle(inds)
            class_splits[lab] = np.array_split(inds, self.n_splits)

        folds = []
        for j in range(self.n_splits):
            fold_j = np.concatenate([class_splits[lab][j] for lab in labels])
            folds.append(fold_j)
        return folds

    def cross_validate(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: Union[np.ndarray, pd.Series],
        metrics: List[str] = ['accuracy','precision','recall','f1_score'],
        averages: List[str] = ['micro'],
    ) -> pd.DataFrame:
        if self.stratified:
            folds = self._make_stratified_folds(y)
        else:
            folds = self._make_folds(y)

        records = []
        for fold_idx, test_idx in enumerate(folds, start=1):
            train_idx = np.setdiff1d(np.arange(len(y)), test_idx)

            if hasattr(X, 'iloc'):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            else:
                X_train, X_test = X[train_idx], X[test_idx]

            if hasattr(y, 'iloc'):
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            else:
                y_train, y_test = y[train_idx], y[test_idx]

            clf = self.model
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            pm = PerformanceMetrics(y_pred, y_test)

            for metric in metrics:
                for avg in averages:
                    score = pm.calculate_metric(metric, average=avg)
                    records.append({
                        'fold':    fold_idx,
                        'metric':  metric,
                        'average': avg,
                        'score':   score
                    })

        return pd.DataFrame.from_records(records)
