import numpy as np

from medusa.harness import metrics


def test_smape_zero_on_match():
    y = np.array([1.0, 10.0, 100.0])
    assert metrics.smape(y, y) == 0.0


def test_smape_symmetric():
    a = np.array([1.0, 5.0, 9.0])
    b = np.array([2.0, 4.0, 12.0])
    assert metrics.smape(a, b) == metrics.smape(b, a)


def test_mase_scales_by_naive_step():
    train = np.array([1.0, 2.0, 3.0, 4.0])  # naive step MAE = 1
    y_true = np.array([5.0, 6.0])
    y_pred = np.array([5.5, 5.5])  # MAE = 0.5
    assert metrics.mase(y_true, y_pred, train) == 0.5


def test_plausibility_inside_and_outside():
    assert metrics.plausibility(1.0, 0.5, 2.0) == 1.0
    assert 0.0 < metrics.plausibility(3.0, 0.5, 2.0) < 1.0
    assert metrics.plausibility(float("inf"), 0.5, 2.0) == 0.0


def test_implied_doubling_time_on_pure_exponential():
    # N(t) = N0 * 2^(t / td) with td = 0.5 h
    def predict(_params, t_s):
        return 10.0 * 2.0 ** ((t_s / 3600.0) / 0.5)

    td = metrics.implied_doubling_time_h(predict, {})
    assert abs(td - 0.5) < 0.02
