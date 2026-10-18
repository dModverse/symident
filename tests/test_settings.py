"""Switches and the settings a result records."""

import re
import warnings
from pathlib import Path

import pytest

import symident as si
from symident import state


def test_diag_turns_every_diagnostic_switch_on():
    with si.switches(DIAG=True):
        assert state.env("LIEDIAG") and state.env("SOLVEDIAG")
        assert not state.env("NOFIELDS")
    assert not state.env("LIEDIAG")


def test_unknown_switch_warns(monkeypatch):
    monkeypatch.setenv("SYMIDENT_NOSUCHROUTE", "1")
    with pytest.warns(UserWarning, match="SYMIDENT_NOSUCHROUTE"):
        si.detect(si.Model({"x": "-k*x"}, {"y": "s*x"}))


def test_result_records_control_switches_and_primes():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with si.switches(NOJET=True):
            r = si.detect(si.Model({"x": "-k*x"}, {"y": "s*x"}),
                          control=si.reconst_control(degree_cap=3, seed=5))
    st = r.info["settings"]
    assert st["control"]["degree_cap"] == 3 and st["control"]["seed"] == 5
    assert st["switches"]["NOJET"] == "1"
    assert len(st["primes"]) == 8 and st["version"] == si.__version__


def test_route_errors_are_recorded_or_raised(monkeypatch):
    from symident.detection.observability import reconstruct

    def broken(*a, **k):
        raise RuntimeError("broken field setup")
    monkeypatch.setattr(reconstruct, "field_setup", broken)
    model = si.Model({"R": "-kR*R", "S": "kact*R/(1 + kinh*I) - kd*S", "I": "b + ks*S - dI*I"},
                     {"z": "s2*S"}, events=[{"var": "R", "time": 0, "value": "d0", "method": "add"}],
                     conditions={"rows": ["a", "b"], "cols": {"d0": [1, 2]}})
    with si.switches(DEBUG="0"):
        r = si.detect(model, reconstruct=True, equilibrate=True)
    assert any("broken field setup" in e for e in r.info["errors"])
    with si.switches(DEBUG="1"), pytest.raises(RuntimeError, match="broken field setup"):
        si.detect(model, reconstruct=True, equilibrate=True)


def test_api_reference_lists_every_switch():
    api = (Path(__file__).parents[1] / "docs" / "api.rst").read_text(encoding="utf-8")
    listed = set(re.findall(r"^``([A-Z0-9_]+)``$", api, re.M))
    assert set(state.SWITCHES) <= listed


def test_a_retired_route_switch_is_unknown():
    import pytest

    import symident as si
    from symident import state
    assert "NOFORK" not in state.SWITCHES and "NOFWDSEED" in state.SWITCHES
    with si.switches(NOFORK=True), pytest.warns(UserWarning, match="NOFORK"):
        si.detect(si.Model({"x": "-k*x"}, {"y": "x"}))
