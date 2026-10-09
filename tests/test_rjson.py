"""The JSON interface for host languages: arguments in, results out, state restored."""

import json
import math
import os
from fractions import Fraction

import numpy as np

from symident import rjson

SPEC = {"f": {"x": "-k*x"}, "g": {"y": "s*x"}, "reconstruct": True}


def test_kwargs_of_maps_host_spellings():
    kw = rjson.kwargs_of({"f": {"x": "-k*x"}, "trafo": None, "gauge_preference": "AUTO",
                          "control": {"timeout": "Inf", "term_cap": 7, "degree_cap": None},
                          "unknown": 1})
    assert set(kw) == {"f", "gauge_preference", "control"}
    assert kw["gauge_preference"] is None
    assert kw["control"]["timeout"] == math.inf and kw["control"]["term_cap"] == 7
    assert rjson.kwargs_of({"gauge_preference": ["k"]})["gauge_preference"] == ["k"]


def test_dumps_cleans_values_json_cannot_hold():
    out = json.loads(rjson._dumps({"a": float("inf"), 1: (1, 2), "f": len, "n": np.int64(3),
                                   "r": np.float64(0.5), "v": np.arange(2), "s": {5}, "q": Fraction(1, 2)}))
    assert out == {"a": "inf", "1": [1, 2], "n": 3, "r": 0.5, "v": [0, 1], "s": [5],
                   "q": "1/2"}


def test_detect_and_reduce_round_trip(monkeypatch):
    # the scaling of x against s, found and removed; the environment is restored after the call
    monkeypatch.delenv("SYMIDENT_PROBE_ONLY", raising=False)
    out = rjson.detect_json(json.dumps(SPEC), env=json.dumps({"SYMIDENT_PROBE_ONLY": "1"}),
                            options=json.dumps({"cores": None, "nested": {"a": 1}}))
    res = json.loads(out)
    assert "SYMIDENT_PROBE_ONLY" not in os.environ
    assert (res["rank"], res["dim"], res["identifiable"]) == (2, 3, False)
    assert res["info"]["kernel"] is None and res["warnings"] == []
    assert [d["weights"] for d in res["symmetries"]] == [{"s": "1", "x": "-1"}]
    red = json.loads(rjson.reduce_json(out, json.dumps({"timeout": "Inf"})))
    assert (red["removed"], red["remaining"], red["warnings"]) == (["X₁"], [], [])


def test_context_restores_a_preset_variable(monkeypatch):
    monkeypatch.setenv("SYMIDENT_PROBE_ONLY", "keep")
    with rjson._context(json.dumps({"SYMIDENT_PROBE_ONLY": "other"}), None):
        assert os.environ["SYMIDENT_PROBE_ONLY"] == "other"
    assert os.environ["SYMIDENT_PROBE_ONLY"] == "keep"
