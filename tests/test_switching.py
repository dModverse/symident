"""Switches in the right-hand side: piecewise(), comparisons, logic, steps, sign, abs,
max and min, and the hybrid analysis of switches on states.

The hybrid cases are checked against ranks and generators derived by hand for each small
model, compared exactly as subspaces at a rational point."""

import random

import pytest
import sympy as spy

import symident as si
from symident.sym.switching import Switches


def det(f, g, **kw):
    kw.setdefault("cores", 1)
    model = si.Model(f, g, trafo=kw.pop("trafo", None), fixed=kw.pop("fixed", ()),
                     events=kw.pop("events", None), inputs=kw.pop("inputs", ()),
                     conditions=kw.pop("conditions", None), regimes=kw.pop("regimes", None))
    return si.detect(model, **kw)


def value(sw, e, point):
    # the parsed expression at a point, each indicator set by the sign of its function
    pt = {spy.Symbol(k): spy.Rational(v) for k, v in point.items()}
    ind = {h: int(bool(q.xreplace(pt) > 0)) for h, q in zip(sw.ind, sw.funcs)}
    return e.xreplace(ind).xreplace(pt)


def same_span(res, expected):
    """The directions of `res` span exactly the generators in `expected`."""
    coords = res.info["coordinates"]
    rng = random.Random(3)
    pt = {spy.Symbol(z): spy.Rational(rng.randint(2, 60), rng.randint(2, 60)) for z in coords}

    def col(g):
        return spy.Matrix([spy.sympify(str(g.get(z, 0)).replace("^", "**")).xreplace(pt)
                           for z in coords])
    got = [col(s.sympy()) for s in res.symmetries]
    want = [col(g) for g in expected]
    assert len(got) == len(want) == res.dim - res.rank
    if want:
        A, B = spy.Matrix.hstack(*got), spy.Matrix.hstack(*want)
        assert A.rank() == B.rank() == spy.Matrix.hstack(A, B).rank() == len(want)


@pytest.mark.parametrize("text, point, expect", [
    ("piecewise(a, x > t1 && x < t2, b)", {"x": "1.5", "t1": "1", "t2": "2"}, "a"),
    ("piecewise(a, x > t1 && x < t2, b)", {"x": "3", "t1": "1", "t2": "2"}, "b"),
    ("piecewise(a, x < t1 || x > t2, b)", {"x": "3", "t1": "1", "t2": "2"}, "a"),
    ("piecewise(a, x < t1, b, x < t2)", {"x": "1.5", "t1": "1", "t2": "2"}, "b"),
    ("piecewise(a, x < t1, b, x < t2)", {"x": "3", "t1": "1", "t2": "2"}, "0"),
    ("ifelse(!(x > t1), a, b)", {"x": "3", "t1": "1"}, "b"),
    ("Heaviside(x - t1)*a", {"x": "3", "t1": "1"}, "a"),
    ("sign(t1 - x)*a", {"x": "3", "t1": "1"}, "-a"),
    ("abs(t1 - x)", {"x": "3", "t1": "1"}, "2"),
    ("max(x, t1, t2)", {"x": "1.5", "t1": "1", "t2": "2"}, "2"),
    ("min(x, t1, t2)", {"x": "1.5", "t1": "1", "t2": "2"}, "1"),
    ("-x^2 + (x >= t1)", {"x": "3", "t1": "1"}, "-8"),
])
def test_parse(text, point, expect):
    sw = Switches(True)
    e = sw.parse(text)
    got = value(sw, e, point)
    assert spy.simplify(got - spy.sympify(expect).xreplace(
        {spy.Symbol(k): spy.Rational(v) for k, v in point.items()})) == 0


@pytest.mark.parametrize("text, expect", [
    ("log(x, 2)", "log(x)/log(2)"), ("log10(x)", "log(x)/log(10)"), ("log2(x)", "log(x)/log(2)"),
    ("exp10(x)", "10**x"), ("sin(x)", "sin(x)"), ("foo(x)", "foo(x)"),
    ("ln(x)", "log(x)"), ("exp2(x)", "2**x"), ("acot(x)", "atan(1/x)"),
    ("pow(x, 3) + cbrt(x)", "x**3 + x**(1/3)"), ("cot(x) + erf(x)", "cot(x) + erf(x)"),
])
def test_parse_functions_without_switch(text, expect):
    sw = Switches(True)
    assert sw.parse(text) == spy.sympify(expect)
    assert not sw.funcs


def test_parse_decided():
    # the declared signs settle these conditions, so no switch is left
    sw = Switches(True)
    assert sw.parse("piecewise(a, x > 0, b)") == spy.Symbol("a")
    assert sw.parse("abs(x)*Heaviside(-k)") == 0
    assert sw.parse("max(x, -1)") == spy.Symbol("x")
    assert not sw.funcs
    # a surface shared by two conditions is one switch
    sw.parse("piecewise(a, x > thr, b) + Heaviside(thr - x)")
    assert len(sw.funcs) == 1
    # equality off an identity holds on a set of measure zero
    assert sw.parse("piecewise(a, x == thr, b)") == spy.Symbol("b")
    assert sw.parse("k*(x != thr) + c*(x == x)") == spy.Symbol("k") + spy.Symbol("c")


def test_pi_and_euler_number_in_a_switched_model():
    r = det({"x": "piecewise(pi*b, x > c, exp(1)*b) - k*x"}, {"y": "s*x"}, reconstruct=True)
    assert [(s.support, s.level) for s in r.symmetries] == [(["b", "c", "s", "x"], "exact")]
    assert r.info["switching"]["surfaces"] == ["c - x = 0"]


def test_a_switch_on_parameters_only_needs_its_side():
    f = {"x": "min(p, 2) - k*x"}
    with pytest.raises(ValueError, match="parameters only"):
        det(f, {"y": "s*x"})
    assert [s.support for s in det(f, {"y": "s*x"}, regimes=["p > 2"]).symmetries] == [["p"]]
    assert [s.support for s in det(f, {"y": "s*x"}, regimes=["p < 2"]).symmetries] == [["p", "s", "x"]]


def test_one_crossing_free_initial_state():
    # (i) a scaling of every regime that maps the switching surface to itself survives
    f = {"x": "piecewise(-k1*x, x > thr, -k2*x)"}
    r = det(f, {"y": "c*x"}, reconstruct=True)
    assert (r.rank, r.dim) == (4, 5)
    assert r.symmetries[0]["weights"] == {"c": "-1", "thr": "1", "x": "1"}
    assert r.info["rank_proven"] == 1
    # a direction on parameters alone that keeps every observable comes on top
    r = det(f, {"y": "(s1 + s2)*x"}, reconstruct=True)
    assert (r.rank, r.dim) == (4, 6)
    same_span(r, [{"s1": "s1", "s2": "s2", "x": "-x", "thr": "-thr"}, {"s1": "1", "s2": "-1"}])


def test_hidden_state_across_thresholds():
    # (ii) a parameter that acts only on a state hidden in its own regime is identifiable
    # once another regime observes that state: a field commuting with both regimes keeps
    # their mixed Lie derivatives
    f = {"x": "-k*x + piecewise(0, x > c1, z)",
         "z": "piecewise(-a*z, x > c1, -b*z, x > c2, 0)"}
    r = det(f, {"y": "x"}, trafo={"x": "2", "z": "1"})
    assert r.identifiable and r.info["coordinates"] == ["a", "b", "c1", "c2", "k"]


def test_parameter_in_another_regime():
    # (iii) a parameter of any regime that is not excluded counts, whether or not the
    # trajectory reaches that regime
    f = {"x": "piecewise(-k2*x, x < 1, -k1*x)"}
    r = det(f, {"y": "x"}, trafo={"x": "2"})
    assert r.identifiable and r.info["coordinates"] == ["k1", "k2"]


def test_surface_kept_or_moved():
    # (iv) a scaling of every regime that moves a switching surface is broken by it
    f = {"x": "piecewise(-k1*x, x > 1, -k2*x)"}
    r = det(f, {"y": "c*x"})
    assert r.identifiable and r.dim == 4
    assert not det(f, {"y": "c*x"}, scalings_only=True).symmetries
    g = {"x": "piecewise(-k1*x, x > thr, -k2*x)"}
    s = det(g, {"y": "c*x"}, scalings_only=True)
    assert len(s.symmetries) == 1 and s.symmetries[0]["weights"]["thr"] == "1"
    assert det(g, {"y": "c*x"}, fixed=["thr"]).identifiable


def test_time_event_and_input():
    # (v) events on an input: a symmetry keeps the event times and the input values
    f = {"x": "b*u - piecewise(k1, x > c, k2)*x", "u": "0"}
    ev = [{"var": "u", "time": 0, "value": 0, "method": "replace"},
          {"var": "u", "time": 1, "value": 1, "method": "replace"}]
    r = det(f, {"y": "s*x"}, inputs=["u"], events=ev, reconstruct=True)
    assert (r.rank, r.dim) == (5, 6)
    same_span(r, [{"x": "x", "b": "b", "c": "c", "s": "-s"}])
    # an event time in a parameter is kept fixed by every symmetry
    ev = [{"var": "u", "time": 0, "value": 0, "method": "replace"},
          {"var": "u", "time": "te", "value": 1, "method": "replace"}]
    r = det(f, {"y": "s*x"}, inputs=["u"], events=ev, reconstruct=True)
    assert (r.rank, r.dim) == (6, 7)
    same_span(r, [{"x": "x", "b": "b", "c": "c", "s": "-s"}])


def test_two_conditions():
    # (vi) a symmetry acts on every condition with its own initial values
    f = {"x": "piecewise(-k1*x, x > c, -k2*x)"}
    cond = {"rows": ["A", "B"], "cols": {}}
    r = det(f, {"y": "s*x"}, trafo=[{"x": "2"}, {"x": "xb"}], conditions=cond)
    assert r.identifiable and r.dim == 5
    r = det(f, {"y": "s*x"}, trafo=[{"x": "xa"}, {"x": "xb"}], conditions=cond, reconstruct=True)
    assert (r.rank, r.dim) == (5, 6)
    assert r.symmetries[0]["weights"] == {"c": "1", "s": "-1", "xa": "1", "xb": "1"}
    # a surface shared by the conditions must be kept in each of them
    r = det(f, [{"y": "s*x"}, {"y": "x"}], trafo=[{"x": "xa"}, {"x": "xb"}], conditions=cond)
    assert r.identifiable and r.dim == 6


def test_equilibrate():
    # (vii) resting starts: a symmetry maps the resting states of every regime to themselves
    f = {"x": "kb + b*u - piecewise(k1, x > c, k2)*x", "u": "0"}
    ev = [{"var": "u", "time": 0, "value": 1, "method": "replace"}]
    r = det(f, {"y": "s*x"}, inputs=["u"], events=ev, equilibrate=True, reconstruct=True)
    assert (r.rank, r.dim) == (5, 6)
    same_span(r, [{"kb": "kb", "b": "b", "c": "c", "s": "-s"}])


def test_infeasible_regime_is_excluded():
    # (viii) a sign pattern the declared signs contradict is no regime, so a parameter
    # that acts only there is no coordinate; a surface between regimes with the same
    # dynamics is invisible, and its threshold is free
    f = {"x": "piecewise(-k1*x, x > c1, -k2*x, x > c1 + d, -k3*x)"}
    r = det(f, {"y": "x"}, reconstruct=True)
    sw = r.info["switching"]
    assert len(sw["regimes"]) == 3 and len(sw["excluded"]) == 1
    assert r.info["coordinates"] == ["x", "c1", "d", "k1", "k3"]
    assert sw["conditions"][0]["visibility"] == {"c1 - x = 0": "visible",
                                                 "c1 + d - x = 0": "invisible"}
    assert (r.rank, r.dim) == (4, 5)
    same_span(r, [{"d": "1"}])


def test_switch_on_a_hidden_state():
    # a switch that acts only on states that never reach the observables is invisible: a
    # direction that moves its surface is not lost, and the hidden parameters are free
    f = {"x": "-k*x", "z": "piecewise(-a*z, z > c, -b*z)"}
    r = det(f, {"y": "s*x"}, reconstruct=True)
    assert r.info["switching"]["conditions"][0]["visibility"] == {"c - z = 0": "invisible"}
    assert (r.rank, r.dim) == (2, 7)
    same_span(r, [{"x": "x", "s": "-s"}, {"z": "1"}, {"a": "1"}, {"b": "1"}, {"c": "1"}])
    # the same switch made visible keeps only the scaling of the threshold with z
    r = det(f, {"y": "s*x", "w": "z"}, reconstruct=True)
    assert r.info["switching"]["conditions"][0]["visibility"] == {"c - z = 0": "visible"}
    assert (r.rank, r.dim) == (6, 7)
    same_span(r, [{"x": "x", "s": "-s"}])


def test_named_regimes():
    # the regimes an experiment visits restrict the rule to them: a parameter that acts
    # only in a regime outside the named set, and a surface no named regimes share, are
    # not identifiable
    f = {"x": "piecewise(-k1*x, x > c, -k2*x)"}
    r = det(f, {"y": "x"}, trafo={"x": "2"}, regimes=["x > c"], reconstruct=True)
    assert (r.rank, r.dim) == (1, 3)
    same_span(r, [{"k2": "1"}, {"c": "1"}])
    assert r.info["switching"]["conditions"][0]["declared"] == ["c - x < 0 at the start"]
    assert list(r.info["assumptions"]["starts"].values()) == [["c - x < 0 at the start"]]
    # a start that contradicts the known initial values is an error
    with pytest.raises(ValueError, match="initial values give"):
        det({"x": "piecewise(-k1*x, x > 1, -k2*x)"}, {"y": "x"}, trafo={"x": "2"},
            regimes=["x < 1"])
    # both regimes named: the result of the rule over all regimes
    r = det(f, {"y": "x"}, trafo={"x": "2"}, regimes=["x > c", "x < c"])
    assert r.identifiable


def test_unsupported():
    f = {"x": "piecewise(-k1*x, x > c, -k2*x)"}
    with pytest.raises(ValueError, match="gauge"):
        det(f, {"y": "x"}, gauge=None)
    with pytest.raises(ValueError, match="reduce_cq"):
        det(f, {"y": "x"}, reduce_cq=True)


_SW = {"x": "piecewise(-k1*x, x > c, -k2*x)"}
_ROOT2 = [{"var": "x", "root": "(x - c)*(x - d)", "value": "0", "method": "replace"}]


@pytest.mark.parametrize("f, kw, match", [
    (_SW, {"free_initial": ["x"]}, "does not support `free_initial`"),
    (_SW, {"engine": "symbolic"}, 'needs engine="modular"'),
    ({"time": "1", **_SW}, {}, "clashes with the clock"),
    (_SW, {"regimes": {"zz": ["x > c"]}}, "unknown conditions: zz"),
    (_SW, {"regimes": "x > c"}, "a list of regimes or a dict"),
    (_SW, {"regimes": ["x > d"]}, "names no switching surface"),
    (_SW, {"regimes": [["x > c", "x < c"]]}, "names both sides of c - x = 0"),
    ({"x": "-k*x"}, {"regimes": ["x > c"]}, "has no switch that the declared signs leave open"),
    ({"x": "piecewise(-k1*x, x > c1, -k2*x, x > c1 + d, -k3*x)"},
     {"regimes": [["x < c1", "x > c1 + d"]]}, "names no feasible regime"),
    ({"x": "ifelse(x > c, -k1*x)"}, {}, r"ifelse\(\) needs three arguments"),
    ({"x": "piecewise(-k1*x, k1 - k2, -k2*x)"}, {}, "is not a comparison"),
    ({"x": "piecewise(-k1*x, abs(x - c) > d, -k2*x)"}, {}, "condition on a switched expression"),
    ({"x": "a - k*x"}, {"trafo": {"x": "0"}, "events": _ROOT2}, "is not one switching surface"),
    ({"x": "-k*x + " + " + ".join(f"piecewise(a{i}, x > c{i}, 0)" for i in range(13))}, {},
     "8192 sign patterns, more than 4096"),
    ({"x": "b - piecewise(k1*x^5 + x, x > c, k2*x)"}, {"equilibrate": True}, "has no closed form"),
])
def test_documented_errors(f, kw, match):
    # each unsupported or contradictory input stops before any analysis
    with pytest.raises(ValueError, match=match):
        det(f, {"y": "x"}, **kw)


def test_surface_without_a_rational_point_is_undecided():
    # no rational point on the surface to compare the jets at: the verdict stays open
    r = det({"x": "piecewise(-k1*x, x^2 > 2, -k2*x)"}, {"y": "x"})
    assert (r.rank, r.dim) == (3, 3)
    assert r.info["switching"]["conditions"][0]["visibility"] == {"x^2 - 2 = 0": "undecided"}


@pytest.mark.parametrize("f, g, events, weights", [
    ({"x": "-k*x"}, {"y": "piecewise(s1*x, x > c, s2*x)"}, None,
     {"c": "1", "s1": "-1", "s2": "-1", "x": "1"}),
    ({"x": "piecewise(-k1*x, x > c, -k2*x)"}, {"y": "s*x"},
     [{"var": "x", "time": "1", "value": "2", "method": "multiply", "root": None}],
     {"c": "1", "s": "-1", "x": "1"}),
])
def test_scaling_through_a_switched_observable_or_event(f, g, events, weights):
    # a scaling of the state and its threshold survives a switched observable and a
    # multiplicative event alike
    r = det(f, g, events=events, reconstruct=True)
    assert (r.rank, r.dim) == (4, 5)
    assert [d["weights"] for d in r.symmetries] == [weights]
    assert r.info["switching"]["conditions"][0]["visibility"] == {"c - x = 0": "visible"}


def test_heaviside_jump_observed_directly():
    r = det({"x": "-k*x + a*Heaviside(x - c)"}, {"y": "x"}, reconstruct=True)
    assert r.identifiable and (r.rank, r.dim) == (4, 4)


def test_regimes_per_condition():
    # each condition keeps to its named regime, so neither sees a surface
    cg = {"rows": ["a", "b"], "cols": {"x0": [3, 1]}}
    r = det({"x": "piecewise(-k1*x, x > 2, -k2*x)"}, {"y": "x"}, trafo={"x": "x0"}, fixed=["x0"],
            conditions=cg, regimes={"a": ["x > 2"], "b": ["x < 2"]})
    assert r.identifiable and r.info["coordinates"] == ["k1", "k2"]
    assert [c["visibility"] for c in r.info["switching"]["conditions"]] == [{}, {}]


def test_time_switch():
    # a switch in time alone is a surface like any other: its time is kept
    f = {"x": "piecewise(-k1*x, time > tau, -k2*x)"}
    r = det(f, {"y": "x"}, trafo={"x": "x0"})
    assert r.identifiable and r.dim == 4
    assert r.info["switching"]["surfaces"] == ["tau - time = 0"]


def test_root_event():
    # a reset at a root surface: a symmetry commutes with the reset and keeps the surface
    f = {"x": "a - k*x"}
    ev = [{"var": "x", "root": "x - c", "value": "0", "method": "replace"}]
    r = det(f, {"y": "x"}, trafo={"x": "0"}, events=ev)
    assert r.identifiable and r.info["coordinates"] == ["a", "c", "k"]
    r = det(f, {"y": "s*x"}, trafo={"x": "0"}, events=ev, reconstruct=True)
    assert (r.rank, r.dim) == (3, 4)
    same_span(r, [{"a": "a", "c": "c", "s": "-s"}])


def test_visibility_of_a_step_input_is_shared_across_switching_times(monkeypatch):
    # the jets that show a switch are computed once for conditions that differ only in the switching time
    from symident.detection import hybrid
    calls = []
    series = hybrid.output_series
    monkeypatch.setattr(hybrid, "output_series", lambda *a, **k: calls.append(1) or series(*a, **k))
    f = {"x": "k*piecewise(0, time - T < 0, 1) - d*x", "z": "x - e*z"}
    cg = {"rows": [f"c{i}" for i in range(40)], "cols": {"T": list(range(1, 41))}}
    r = det(f, {"y": "s*z"}, trafo={"x": "0", "z": "0"}, conditions=cg)
    vis = [c["visibility"] for c in r.info["switching"]["conditions"]]
    assert vis == [{"T - time = 0": "visible"}] * 40 and len(calls) == 3


def test_output_series_are_the_taylor_coefficients_modulo_p():
    from symident.sym.taylor import NoSeriesError, output_series
    x, k, s = spy.symbols("x k s")
    p = 1000003
    (h,) = output_series({x: -k * x}, [s * x / (1 + x)], {x: 2, k: 3, s: 5}, 3, p)
    inv3 = pow(3, p - 2, p)
    assert h == [10 * inv3 % p, -10 * inv3 % p, -5 * inv3 % p]
    with pytest.raises(NoSeriesError):
        output_series({x: -spy.exp(x)}, [x], {x: 2}, 3, p)


def test_a_switched_input_model_at_rest_keeps_its_directions():
    f = {"x1": "kb + kin*u - (k1 + piecewise(kl, x1 > thr, kl2))*x1", "x2": "k1*x1 - k2*x2", "u": "0"}
    m = si.Model(f, {"y": "s*x2"}, inputs=["u"], regimes=["x1 > thr"],
                 events=[{"var": "u", "time": 0, "value": "dose", "method": "replace"}],
                 conditions={"rows": ["low", "high"], "cols": {"dose": [1, 10]}})
    r = si.detect(m, equilibrate=True, reconstruct=True)
    assert (r.rank, r.dim) == (4, 8)
    assert sorted(sorted(s.support) for s in r.symmetries) == \
        [["k1", "kb", "kin", "kl"], ["kb", "kin", "s"], ["kl2"], ["thr"]]


def test_a_field_with_integer_weights_is_a_scaling():
    r = si.detect(si.Model({"x": "-k*x", "z": "piecewise(-a*z, z > c, -b*z)"}, {"y": "x"}), reconstruct=True)
    kinds = {tuple(s.support): s.type for s in r.symmetries}
    assert kinds[("a",)] == kinds[("b",)] == "scaling" and kinds[("c",)] == "general"
    assert {tuple(s.support): s.weights for s in r.symmetries}[("a",)] == {"a": "1"}


def test_a_direction_the_rule_does_not_prove_is_a_candidate_not_a_symmetry():
    m = si.Model({"x1": "kb + kin*u - (k1 + kl)*x1", "x2": "k1*x1 - k2*x2", "u": "0"}, {"y": "s*x2"},
                 inputs=["u"], events=[{"var": "u", "time": 0, "value": "dose", "method": "replace"},
                                       {"var": "x2", "root": "x1 - c", "value": "0", "method": "replace"}],
                 conditions={"rows": ["low", "high"], "cols": {"dose": [1, 10]}})
    r = si.detect(m, equilibrate=True, reconstruct=True)
    assert (r.rank, r.dim) == (5, 7)
    assert {s.level for s in r.symmetries} == {"exact"}
    cands = r.info["switching"]["candidates"]
    assert len(r.symmetries) + len(cands) == r.dim - r.rank
    assert all(c["support"] and c["type"] in ("scaling", "general") for c in cands)
    assert r.info["rank_status"]["status"] == "lower bound" and r.info["switching"]["uncertified"]
    assert "up to" in repr(r) and "Candidates, not proven" in repr(r)


def test_a_direction_that_moves_a_root_surface_is_no_symmetry():
    m = si.Model({"x": "-k*x", "z": "a"}, {"y": "s*x"},
                 events=[{"var": "x", "root": "z - c", "value": "x0", "method": "replace"}])
    r = si.detect(m, reconstruct=True)
    assert all(s.level == "exact" for s in r.symmetries)
    assert {"a", "z"} not in [set(s.support) for s in r.symmetries]
    assert r.info["rank_status"]["status"] == "lower bound"
