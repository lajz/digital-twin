from medusa.agent import canned
from medusa.contract.checker import check_twin_object, check_twin_source

GOOD = canned.LOGISTIC


def test_canned_sources_pass_static_checks():
    for src in canned.CANNED_SEQUENCE:
        assert check_twin_source(src).ok, src[:60]


def test_forbidden_import_rejected():
    src = GOOD.replace("import numpy as np", "import numpy as np\nimport torch")
    res = check_twin_source(src)
    assert not res.ok
    assert any("torch" in e for e in res.errors)


def test_non_allowlisted_import_rejected():
    src = GOOD.replace("import numpy as np", "import numpy as np\nimport pandas as pd")
    assert not check_twin_source(src).ok


def test_missing_twin_class_rejected():
    assert not check_twin_source("x = 1\n").ok


def test_forbidden_call_rejected():
    src = GOOD.replace("def fit(self, obs):", "def fit(self, obs):\n        eval('1')")
    assert not check_twin_source(src).ok


def test_object_check_catches_bad_params_and_bounds():
    class BadTriple:
        FAMILY = "x"
        PARAMS = {"a": (1.0, 2.0)}  # not a triple
        def fit(self, obs):  # noqa: D401
            return {}
        def predict(self, params, t):
            return t

    assert not check_twin_object(BadTriple()).ok

    class Ok:
        FAMILY = "x"
        PARAMS = {"a": (1.0, 2.0, "u")}
        def fit(self, obs):
            return {"a": 1.5}
        def predict(self, params, t):
            return t

    assert check_twin_object(Ok(), params={"a": 1.5}).ok
    assert not check_twin_object(Ok(), params={"a": 9.0}).ok
    assert not check_twin_object(Ok(), params={"b": 1.5}).ok
