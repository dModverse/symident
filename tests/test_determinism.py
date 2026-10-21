"""The same input gives the same result in every process, whatever the string hashing and
the number of cores."""

import json
import os
import subprocess
import sys

import pytest

RUN = r"""
import json, sys
import symident as si

def result(m, **kw):
    d = si.detect(m, reconstruct=True, cores=int(sys.argv[1]), **kw).to_dict()
    d["info"].pop("elapsed", None)
    d["info"].pop("timing", None)
    return d

ev = [{"var": "R", "time": 0, "value": "d0", "method": "add"},
      {"var": "R", "time": 1, "value": "d1", "method": "add"}]
inh = si.Model({"R": "-kR*R", "S": "kact*R/(1 + kinh*I) - kd*S", "I": "b + ks*S - dI*I"},
               {"z": "s2*S"}, events=ev,
               conditions={"rows": ["a", "b"], "cols": {"d0": [1, 2], "d1": [1, 1]}})
hidden = si.Model({"x": "-k*x", "z": "piecewise(-a*z, z > c, -b*z)"}, {"y": "s*x"})
bist = si.Model({"x": "c + a*x - b*x^3 + kin*u - d*x*z", "z": "e*x - g*z", "u": "0"}, {"y": "s*x"},
                inputs=["u"], events=[{"var": "u", "time": 0, "value": "1", "method": "replace"}])
ab = si.Model({"A": "-k1*A + k2*B", "B": "k1*A - k2*B"}, {"y": "s*A"},
              conditions={"rows": ["a", "b"], "cols": {"k2": [1, "k3"]}})
out = [result(inh, equilibrate=True),
       result(inh, equilibrate=True, control=si.reconst_control(degree_cap=1, degree_cap_max=1, homotopy=False)),
       result(hidden)]
# the msolve path of the resting states and the stacked path
with si.switches(NOFWDSEED=True):
    out.append(result(bist, equilibrate=True))
with si.state.options(stacked_min=0):
    out.append(result(ab))
print(json.dumps(out, sort_keys=True, default=str))
"""


def _runs(settings):
    procs = []
    for hashseed, cores in settings:
        env = dict(os.environ, PYTHONHASHSEED=str(hashseed))
        procs.append(subprocess.Popen([sys.executable, "-c", RUN, str(cores)], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    outs = []
    for p in procs:
        out, err = p.communicate(timeout=600)
        assert p.returncode == 0, err
        outs.append(json.loads(out))
    return outs


@pytest.mark.parametrize("settings", [[(1, 1), (2, 1), (1, 4)]])
def test_same_result_in_every_process(settings):
    outs = _runs(settings)
    general = [s for s in outs[0][0]["symmetries"] if s["type"] == "general"]
    assert general and general[0]["explicit"]
    assert not [s for s in outs[0][1]["symmetries"] if s["type"] == "general"][0]["explicit"]
    for o in outs[1:]:
        assert o == outs[0]


def test_session_seeds_depend_on_seed_model_and_stage_only():
    from symident.session import Session
    s = Session(0, "m")
    assert s.seed("point", 1) == 7274323681973122755
    assert s.rng("x", "a", 2).random() == s.rng("x", "a", 2).random()
    assert len({Session(0, "m").seed("p"), Session(1, "m").seed("p"), Session(0, "n").seed("p"),
                s.seed("q")}) == 4


INH = dict(odes={"R": "-kR*R", "S": "kact*R/(1 + kinh*I) - kd*S", "I": "b + ks*S - dI*I"},
           observables={"z": "s2*S"},
           events=[{"var": "R", "time": 0, "value": "d0", "method": "add"},
                   {"var": "R", "time": 1, "value": "d1", "method": "add"}],
           conditions={"rows": ["a", "b"], "cols": {"d0": [1, 2], "d1": [1, 1]}})


def _detect(**kw):
    import symident as si
    d = si.detect(si.Model(**INH), reconstruct=True, equilibrate=True, **kw).to_dict()
    d["info"].pop("elapsed", None)
    d["info"].pop("timing", None)
    return d


def test_a_time_limit_that_does_not_fire_changes_nothing():
    import symident as si
    plain = _detect()
    limited = _detect(control=si.reconst_control(timeout=1e6))
    assert plain["info"]["limits"] == limited["info"]["limits"] == []
    assert plain["symmetries"] == limited["symmetries"]


def test_a_time_limit_that_fires_is_reported():
    import symident as si
    d = _detect(control=si.reconst_control(timeout=1e-9))
    assert any(lim.startswith("timeout") for lim in d["info"]["limits"])


def test_the_jet_route_stops_at_its_work_limit():
    from symident.sym.jets import jet_generator
    cond = {"f": ["x = -k*x + a*y", "y = -m*y"], "g": ["x"], "ic": ["x = x0", "y = y0"]}
    assert jet_generator([cond], ["a", "y0"], "a")["ok"]
    assert jet_generator([cond], ["a", "y0"], "a", work_limit=1)["reason"] == "work limit"
