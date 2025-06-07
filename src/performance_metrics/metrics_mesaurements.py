import time
from typing import Callable, Union, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .performance_metrics import PerformanceMetrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def compare_function_time(
    func, *args, n: int = 10, **kwargs
) -> Tuple[float, float]:
    start = time.perf_counter()
    for _ in range(n):
        func(*args, **kwargs)
    total = time.perf_counter() - start
    return total / n, total


def calculate_init_plus_calculation_time(
    predictions: Union[list, np.ndarray],
    actual: Union[list, np.ndarray],
    metric_name: str,
    average: str = 'micro',
    n: int = 10
) -> Tuple[float, float]:
    wrapper = lambda: getattr(
        PerformanceMetrics(predictions, actual), metric_name
    )(average)
    return compare_function_time(wrapper, n=n)


def plot_execution_times(
    metric_name: str,
    sklearn_fn: Callable,
    predictions: Union[List, np.ndarray],
    actual: Union[List, np.ndarray],
    ns: List[int],
    average: str = 'micro'
):
    our_avgs, our_totals = [], []
    sk_avgs, sk_totals = [], []

    labels = PerformanceMetrics(predictions, actual).labels.tolist()

    for n in ns:
        our_avg, our_total = calculate_init_plus_calculation_time(
            predictions, actual,
            metric_name=metric_name,
            average=average,
            n=n
        )
        our_avgs.append(our_avg)
        our_totals.append(our_total)

        if metric_name == 'accuracy':
            sk_wrapper = lambda: sklearn_fn(actual, predictions)
        else:
            sk_wrapper = lambda: sklearn_fn(
                actual, predictions,
                average=None if average == 'per_class' else average,
                labels=labels
            )
        sk_avg, sk_total = compare_function_time(sk_wrapper, n=n)
        sk_avgs.append(sk_avg)
        sk_totals.append(sk_total)

    plt.figure(figsize=(12, 5))
    plt.plot(ns, our_totals, label='Our Total Time', color='blue')
    plt.plot(ns, our_avgs, '--', label='Our Avg Time', color='blue')
    plt.plot(ns, sk_totals, label='Sklearn Total Time', color='orange')
    plt.plot(ns, sk_avgs, '--', label='Sklearn Avg Time', color='orange')
    plt.xlabel('Number of Iterations (n)')
    plt.ylabel('Time (seconds)')
    plt.title(f"Timing '{metric_name}' (avg='{average}')")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_metrics_df(
    perf: PerformanceMetrics,
    y_true: Union[list, np.ndarray],
    y_pred: Union[list, np.ndarray],
    average: str = 'micro'
) -> pd.DataFrame:
    # our metrics
    my_acc, my_prec, my_rec, my_f1 = (
        perf.accuracy(average),
        perf.precision(average),
        perf.recall(average),
        perf.f1_score(average)
    )
    # sklearn metrics
    sk_acc = accuracy_score(y_true, y_pred)
    avg_arg = None if average == 'per_class' else average
    labels = perf.labels.tolist()
    sk_prec = precision_score(y_true, y_pred, average=avg_arg, labels=labels)
    sk_rec = recall_score(y_true, y_pred, average=avg_arg, labels=labels)
    sk_f1 = f1_score(y_true, y_pred, average=avg_arg, labels=labels)

    if average != 'per_class':
        return pd.DataFrame({
            'PerformanceMetrics': [my_acc, my_prec, my_rec, my_f1],
            'sklearn': [sk_acc, sk_prec, sk_rec, sk_f1]
        }, index=['accuracy', 'precision', 'recall', 'f1'])

    # per-class indexing
    data = {
        ('precision', 'PerformanceMetrics'): [perf.precision('per_class')[l] for l in labels],
        ('precision', 'sklearn'): sk_prec,
        ('recall', 'PerformanceMetrics'): [perf.recall('per_class')[l] for l in labels],
        ('recall', 'sklearn'): sk_rec,
        ('f1', 'PerformanceMetrics'): [perf.f1_score('per_class')[l] for l in labels],
        ('f1', 'sklearn'): sk_f1,
    }
    df = pd.DataFrame(data, index=labels)
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=['metric', 'source'])
    return df
