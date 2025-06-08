import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from src.kfold.kfold import ManualKFoldValidator


@pytest.fixture
def sample_data():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
    return pd.DataFrame(X), pd.Series(y)


@pytest.fixture
def simple_model():
    return DecisionTreeClassifier(random_state=0)


def test_kfold_validator_basic_run(sample_data, simple_model):
    X, y = sample_data
    validator = ManualKFoldValidator(model=simple_model, n_splits=5, shuffle=True, random_state=0)
    results = validator.cross_validate(X, y)

    assert isinstance(results, pd.DataFrame)
    assert set(results.columns) == {'fold', 'metric', 'average', 'score'}
    assert len(results) == 5 * 4


def test_kfold_validator_stratified_vs_non_stratified(sample_data, simple_model):
    X, y = sample_data

    non_strat = ManualKFoldValidator(model=simple_model, stratified=False, random_state=0)
    strat = ManualKFoldValidator(model=simple_model, stratified=True, random_state=0)

    df_non_strat = non_strat.cross_validate(X, y)
    df_strat = strat.cross_validate(X, y)

    assert df_strat.shape == df_non_strat.shape
    assert all(df_strat['metric'].isin(['accuracy', 'precision', 'recall', 'f1_score']))


def test_folds_are_disjoint(sample_data, simple_model):
    X, y = sample_data
    validator = ManualKFoldValidator(model=simple_model, n_splits=5, shuffle=True, random_state=0)
    if validator.stratified:
        folds = validator._make_stratified_folds(y)
    else:
        folds = validator._make_folds(y)

    flat_indices = np.concatenate(folds)
    unique_indices = np.unique(flat_indices)
    assert len(flat_indices) == len(unique_indices), "Folds overlap or have duplicates"
    assert set(unique_indices) == set(range(len(y))), "Missing or extra samples in folds"


def test_kfold_different_metrics(sample_data, simple_model):
    X, y = sample_data
    validator = ManualKFoldValidator(model=simple_model, n_splits=3, random_state=0)
    results = validator.cross_validate(
        X, y,
        metrics=['accuracy', 'f1_score'],
        averages=['micro', 'macro']
    )

    expected_rows = 3 * 2 * 2  # 3 folds × 2 metrics × 2 averages
    assert results.shape[0] == expected_rows
    assert set(results['average']) == {'micro', 'macro'}
    assert set(results['metric']) == {'accuracy', 'f1_score'}
