"""Pins the seed helper library. The meta-loop MUST NOT edit this file."""

import numpy as np

from medusa import toolkit


def _load(name):
    ns = {}
    exec(compile(toolkit.SNIPPETS[name], name, "exec"), ns)  # noqa: S102
    return ns[name]


def test_every_snippet_execs():
    for name, code in toolkit.SNIPPETS.items():
        ns = {}
        exec(compile(code, name, "exec"), ns)  # noqa: S102
        assert callable(ns[name]), name


def test_robust_log_fit_recovers_a_line():
    fn = _load("robust_log_fit")
    t = np.linspace(0, 5, 40)
    truth = np.array([2.0, 0.8])
    y = truth[0] * np.exp(truth[1] * t)

    def resid(p):
        return np.log(np.clip(p[0] * np.exp(p[1] * t), 1e-9, None)) - np.log(y)

    p = fn(resid, [1.0, 0.1], np.array([0.1, 0.0]), np.array([10.0, 3.0]))
    assert np.allclose(p, truth, rtol=0.05)


def test_logistic_and_baranyi_are_monotone_and_saturating():
    lg = _load("logistic_curve")
    ba = _load("baranyi_lag")
    t = np.linspace(0, 20, 200)
    for y in (lg(t, 1, 0.8, 1e4), ba(t, 1, 0.8, 1.0, 1e4)):
        assert np.all(np.diff(y) >= -1e-6)
        assert y[-1] > y[0] * 100
        assert y[-1] <= 1e4 * 1.01


def test_close_the_books_satisfies_the_cash_identity():
    fn = _load("close_the_books")
    n = 12
    mrr = np.linspace(1e4, 5e4, n)
    hc = np.full(n, 5.0)
    cap = np.zeros(n)
    cap[3] = 1e6
    ni, cash = fn(mrr, hc, 1e4, 2e4, np.full(n, 8e3), cap, 5e5)
    bridge = cash[1:] - cash[:-1] - ni[1:] - cap[1:]
    assert np.abs(bridge).max() < 1e-6


def test_integrate_ode_runs():
    fn = _load("integrate_ode")
    y = fn(lambda t, s: [-0.1 * s[0]], [100.0], np.array([0.0, 3600.0, 7200.0]))
    assert y.shape[1] == 3 and y[0, -1] < y[0, 0]


def test_render_library_lists_all_snippets():
    doc = toolkit.render_library()
    for name in toolkit.SNIPPETS:
        assert f"`{name}`" in doc
