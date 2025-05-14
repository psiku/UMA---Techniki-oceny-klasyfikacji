import pytest

from src.performance_metrics.performance_metrics import PerformanceMetrics


def test_performance_metrics_initialization():

    performance_metrics = PerformanceMetrics([], [])

    assert isinstance(performance_metrics, PerformanceMetrics)


def test_performance_metrics_initialization_different_lengths():
    try:
        PerformanceMetrics([1,2], [1,2,3])
    except AssertionError:
        assert True
    else:
        assert False


def test_accuracy_micro():
    classes = [0, 1, 1, 0, 0, 1, 0, 0, 1]
    predicted = [1, 1, 1, 0, 1, 0, 0, 1, 0]

    performance_metrics = PerformanceMetrics(predicted, classes)
    acc = performance_metrics.accuracy('micro')

    real_acc = 4 / len(classes)

    assert acc == real_acc


def test_accuracy_per_class_with_two_classes():
    classes = [0, 1, 1, 0, 0, 1, 0, 0, 1]
    predicted = [1, 1, 1, 0, 1, 0, 0, 1, 0]

    performance_metrics = PerformanceMetrics(predicted, classes)
    result = performance_metrics.accuracy('per_class')

    zero_acc = 2 / 5
    one_acc = 2 / 4

    assert zero_acc == result[0]
    assert one_acc == result[1]


def test_accuracy_per_class_with_two_three_classes():
    classes = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    predicted = [0, 1, 1, 1, 0, 2, 2, 2, 2]

    performance_metrics = PerformanceMetrics(predicted, classes)
    result = performance_metrics.accuracy('per_class')

    zero_acc = 1 / 2
    one_acc = 2 / 3
    two_acc = 4 / 4

    assert zero_acc == result[0]
    assert one_acc == result[1]
    assert two_acc == result[2]


def test_accuracy_macro():
    classes = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    predicted = [0, 1, 1, 1, 0, 2, 2, 2, 2]

    performance_metrics = PerformanceMetrics(predicted, classes)
    result = performance_metrics.accuracy('macro')

    zero_acc = 1 / 2
    one_acc = 2 / 3
    two_acc = 4 / 4

    macro_avg = (zero_acc + one_acc + two_acc) / 3

    assert macro_avg == result


def test_accuracy_weighted():
    classes = [0, 0, 1, 1, 1, 2, 2, 2, 2]
    predicted = [0, 1, 1, 1, 0, 2, 2, 2, 2]

    performance_metrics = PerformanceMetrics(predicted, classes)
    result = performance_metrics.accuracy('weighted')

    zero_acc = 1 / 2
    one_acc = 2 / 3
    two_acc = 4 / 4

    weighted_acc = (zero_acc * 2 + one_acc * 3 + two_acc * 4) / 9

    assert weighted_acc == result


def test_accuracy_wrong_average_type():
    performance_metrics = PerformanceMetrics([1, 2], [1, 2])

    try:
        performance_metrics.accuracy('wrong')
    except ValueError as e:
        assert str(e) == "Unsupported average type: wrong"
    else:
        assert False


def test_precision_macro():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)
    precision = metrics.precision(average='macro')

    zero_prec = 2 / 3
    one_prec = 2 / 2

    expected = (zero_prec + one_prec) / 2

    assert precision == pytest.approx(expected, rel=1e-6)


def test_precision_micro():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)
    precision = metrics.precision(average='micro')

    expected = 4/5

    assert precision == pytest.approx(expected, rel=1e-6)


def test_precision_per_class():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)
    precision = metrics.precision(average='per_class')

    zero_prec = 2 / 3
    one_prec = 2 / 2

    assert precision[0] == zero_prec
    assert precision[1] == one_prec


def test_precision_weighted():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)
    precision = metrics.precision(average='weighted')

    zero_prec = 2 / 3
    one_prec = 2 / 2

    expected = (zero_prec * 2 + one_prec * 3) / 5

    assert precision == expected


def test_precision_wrong_average():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)

    try:
        metrics.precision(average='wrong')
    except ValueError as e:
        assert str(e) == "Unsupported average type: wrong"
    else:
        assert False


def test_recall_macro():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)
    recall = metrics.recall(average='macro')

    zero_rec = 2 / 2
    one_rec = 2 / 3

    expected = (zero_rec + one_rec) / 2

    assert recall == pytest.approx(expected, rel=1e-6)


def test_recall_micro():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)
    recall = metrics.recall(average='micro')

    expected = 4/5

    assert recall == pytest.approx(expected, rel=1e-6)


def test_recall_per_class():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)
    recall = metrics.recall(average='per_class')

    zero_rec = 2 / 2
    one_rec = 2 / 3

    assert recall[0] == zero_rec
    assert recall[1] == one_rec


def test_recall_weighted():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)
    recall = metrics.recall(average='weighted')

    zero_rec = 2 / 2
    one_rec = 2 / 3

    expected = (zero_rec * 2 + one_rec * 3) / 5

    assert recall == expected


def test_recall_wrong_average():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)

    try:
        metrics.recall(average='wrong')
    except ValueError as e:
        assert str(e) == "Unsupported average type: wrong"
    else:
        assert False


def test_f1_score_macro():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted, actual)
    f1 = metrics.f1_score(average='macro')

    expected = 4 / 5
    assert f1 == pytest.approx(expected, rel=1e-6)

def test_f1_score_micro_two_class():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)

    expected = 0.8
    assert metrics.f1_score(average='micro') == pytest.approx(expected, rel=1e-6)

def test_f1_score_micro_three_class():
    actual = [0, 1, 2, 2]
    predicted = [0, 2, 2, 2]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)

    expected = 0.75
    assert metrics.f1_score(average='micro') == pytest.approx(expected, rel=1e-6)


def test_f1_score_per_class():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)
    f1 = metrics.f1_score(average='per_class')

    zero_f1 = 4 / 5
    one_f1 = 4 / 5

    assert f1[0] == zero_f1
    assert f1[1] == one_f1


def test_f1_score_per_class_three_classes():
    actual = [0, 1, 2, 2]
    predicted = [0, 2, 2, 2]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)
    f1 = metrics.f1_score(average='per_class')

    zero_f1 = 1
    one_f1 = 0
    two_f1 = 4 / 5

    assert f1[0] == zero_f1
    assert f1[1] == one_f1
    assert f1[2] == two_f1


def test_f1_score_wrong_average():
    actual = [0, 1, 1, 0, 1]
    predicted = [0, 1, 0, 0, 1]
    metrics = PerformanceMetrics(predicted=predicted, actual=actual)

    try:
        metrics.f1_score(average='wrong')
    except ValueError as e:
        assert str(e) == "Unsupported average type: wrong"
    else:
        assert False