"""Curve samples of the resting conditions: forced curve mode closes the same directions as direct resting solves,
peels the exact scalings, and solves no resting condition directly."""

import traceback

import pytest
import sympy as spy

import symident as si
from symident.detection import work
from symident.detection.observability import rest
from symident.sym.msolve import _msolve_binary


def _has_msolve():
    try:
        _msolve_binary()
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _has_msolve(), reason="needs msolve")

EVENTS = [{"var": "L", "time": 0, "value": "dose", "method": "add"}]
CONDS = {"rows": ["c0", "c1", "o1"], "cols": {"dose": [0, 1, 1], "kt": ["kt", "kt", "ktO"]}}

# a parallel combination of two rates, a translation, and a resting core of three states that only msolve solves
SERIES = dict(
    odes={"Rm": "kt - kd*Rm", "x": "kp*Rm*(1 + L) - b*c/(b + c)*x", "L": "-kq*L"},
    observables={"y1": "s*x", "y2": "Rm"},
)
CORE = dict(
    odes={
        "Rm": "kt - kd*Rm",
        "A": "ka*Rm + kl*L - k1*A^2*B^2 - (da1 + da2)*A",
        "B": "kb - k2*B^2*C^2 - db*B",
        "C": "kc - k3*C^2*A^2 - dc*C",
        "L": "-kq*L",
    },
    observables={"y1": "s1*A", "y2": "s2*B", "y3": "Rm", "y4": "C"},
)


def _canon(gen):
    """The generator up to a factor: every entry divided by the first."""
    names = sorted(gen)
    e = {k: spy.sympify(str(v).replace("^", "**")) for k, v in gen.items()}
    return tuple((k, str(spy.simplify(e[k] / e[names[0]]))) for k in names)


def _run(model, monkeypatch, curve, control=None):
    monkeypatch.setenv("SYMIDENT_RESTCURVE" if curve else "SYMIDENT_NORESTCURVE", "1")
    seen = {"direct": 0, "msolve": 0}
    solve_direct, run_msolve = rest.RestingStates.solve_direct, work.run_msolve

    def counted(self, *a):
        seen["direct"] += 1
        return solve_direct(self, *a)

    def msolve(cmd, text=True):
        seen["msolve"] += any(f.name == "solve_raw" for f in traceback.extract_stack())
        return run_msolve(cmd, text)

    monkeypatch.setattr(rest.RestingStates, "solve_direct", counted)
    monkeypatch.setattr(work, "run_msolve", msolve)
    m = si.Model(**model, events=EVENTS, conditions=CONDS, inputs=["L"])
    r = si.detect(m, equilibrate=True, reconstruct=True, cores=2, control=control)
    return r, seen


@pytest.mark.parametrize("model", [SERIES, CORE], ids=["series", "core"])
def test_curves_close_the_directions_of_direct_solves(model, monkeypatch):
    with monkeypatch.context() as mp:
        direct, seen_d = _run(model, mp, curve=False)
    curve, seen_c = _run(model, monkeypatch, curve=True)
    assert {_canon(s.generator) for s in curve.symmetries} == {_canon(s.generator) for s in direct.symmetries}
    assert all(s.level == "exact" for s in curve.symmetries)

    def peeled(r):
        return {_canon(s.generator) for s in r.symmetries if s.kind == "scaling" and s.route is None}

    assert peeled(direct) and peeled(curve) == peeled(direct)
    assert seen_d["direct"] > 0
    assert seen_c == {"direct": 0, "msolve": 0}


def test_direct_core_needs_msolve(monkeypatch):
    _, seen = _run(CORE, monkeypatch, curve=False)
    assert seen["msolve"] > 0


def test_a_rational_direction_of_degree_two(monkeypatch):
    r, _ = _run(SERIES, monkeypatch, curve=True)
    gens = [_canon(s.generator) for s in r.symmetries]
    assert (("b", "1"), ("c", "-c**2/b**2")) in gens


def test_a_constant_translation(monkeypatch):
    r, _ = _run(CORE, monkeypatch, curve=True)
    tr = [s for s in r.symmetries if s.kind == "translation"]
    assert [dict(s.generator) for s in tr] == [{"da1": "1", "da2": "-1"}]


def test_curve_samples_record_no_route_errors(monkeypatch):
    r, _ = _run(SERIES, monkeypatch, curve=True)
    assert not r.info.get("errors")


def test_rank_profiles_skip_continued_conditions(monkeypatch):
    from symident.detection.observability import joint

    profiled = []
    block_profiles = joint.JointSystem.block_profiles

    def watched(self):
        fns = block_profiles(self)
        if fns is None:
            return None

        def wrap(mi, f):
            def g(point, p, Nt):
                out = f(point, p, Nt)
                sol = self.solve_cond(point, p, self.equil_conds[mi])
                profiled.append(bool(out.get("ok")) and bool(sol and sol.get("cont")))
                return out

            return g

        return [wrap(mi, f) for mi, f in enumerate(fns)]

    monkeypatch.setattr(joint.JointSystem, "block_profiles", watched)
    _run(SERIES, monkeypatch, curve=True)
    assert profiled and not any(profiled)


def test_short_curves_raise_their_order(monkeypatch):
    with monkeypatch.context() as mp:
        direct, _ = _run(SERIES, mp, curve=False)
    curve, _ = _run(SERIES, monkeypatch, curve=True, control=si.reconst_control(gap_order_cap=2))
    assert {_canon(s.generator) for s in curve.symmetries} == {_canon(s.generator) for s in direct.symmetries}
    assert all(s.level == "exact" for s in curve.symmetries)


def test_continued_conditions_certify_the_lie_order(monkeypatch):
    with monkeypatch.context() as mp:
        direct, _ = _run(CORE, mp, curve=False)
    curve, seen = _run(CORE, monkeypatch, curve=True)
    for r in (direct, curve):
        assert r.info["lie_certified"] is True
        assert "drawn forward" in r.info["order_bound_why"]
    assert curve.info["lie_block_orders"] == direct.info["lie_block_orders"] == [2, 5, 5]
    assert seen == {"direct": 0, "msolve": 0}
