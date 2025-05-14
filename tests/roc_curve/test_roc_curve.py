import pytest
from src.roc_curve.roc_curve import ROC


@pytest.fixture
def example_data():
    y_true = [0, 0, 1, 1, 0, 1, 0, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8, 0.2, 0.9, 0.6, 0.7]
    return y_true, y_scores


def test_compute_auc(example_data):
    y_true, y_scores = example_data
    roc = ROC(y_true, y_scores)
    roc.compute()
    assert roc.auc is not None
    assert 0 <= roc.auc <= 1


def test_fpr_tpr_lengths(example_data):
    y_true, y_scores = example_data
    roc = ROC(y_true, y_scores)
    roc.compute()
    assert len(roc.fpr) == len(y_scores)
    assert len(roc.tpr) == len(y_scores)


def test_tpr_monotonic(example_data):
    y_true, y_scores = example_data
    roc = ROC(y_true, y_scores)
    roc.compute()
    assert all(x <= y for x, y in zip(roc.tpr, roc.tpr[1:]))


def test_auc_known_value():
    y_true = [0, 0, 1, 1]
    y_scores = [0.1, 0.4, 0.35, 0.8]
    roc = ROC(y_true, y_scores)
    roc.compute()
    expected_auc = 0.75
    assert abs(roc.auc - expected_auc) < 1e-2
