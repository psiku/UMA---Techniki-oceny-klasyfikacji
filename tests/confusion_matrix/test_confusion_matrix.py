import numpy as np
import pytest
from src.confusion_matrix.confusion_matrix import ConfusionMatrix


@pytest.fixture
def cm():
    y_true = ['cat', 'dog', 'cat', 'cat', 'dog', 'fish']
    y_pred = ['dog', 'dog', 'cat', 'cat', 'dog', 'fish']
    return ConfusionMatrix(y_true, y_pred)


def test_compute_matrix(cm):
    expected = np.array([
        [2, 1, 0],  # cat
        [0, 2, 0],  # dog
        [0, 0, 1]   # fish
    ])
    result = cm.compute()
    np.testing.assert_array_equal(result, expected)


def test_get_class_stats(cm):
    cm.compute()

    stats_cat = cm.get_class_stats('cat')
    stats_dog = cm.get_class_stats('dog')
    stats_fish = cm.get_class_stats('fish')

    assert abs(stats_cat['precision'] - 1.0) < 1e-2
    assert abs(stats_cat['recall'] - (2 / 3)) < 1e-2
    assert abs(stats_cat['f1'] - 0.8) < 1e-2

    assert abs(stats_dog['precision'] - (2 / 3)) < 1e-2
    assert abs(stats_dog['recall'] - 1.0) < 1e-2
    assert abs(stats_dog['f1'] - 0.8) < 1e-2

    assert abs(stats_fish['precision'] - 1.0) < 1e-2
    assert abs(stats_fish['recall'] - 1.0) < 1e-2
    assert abs(stats_fish['f1'] - 1.0) < 1e-2


def test_get_method(cm):
    matrix = cm.get()
    assert matrix is not None
    assert matrix.shape == (3, 3)
