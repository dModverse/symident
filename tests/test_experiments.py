"""The count of several experiments: shared parameters, free starts, and the premise of no rational first integral."""

from fractions import Fraction

import pytest
import sympy as spy

import symident as si
from symident.sym.msolve import MsolveMissingError, _msolve_binary
from symident.sym.numfield import NumberField, charpoly, cycle_type, rank_over_q


def _has_msolve():
    try:
        _msolve_binary()
    except MsolveMissingError:
        return False
    return True


needs_msolve = pytest.mark.skipif(not _has_msolve(), reason="needs msolve")


def _series(f, g, params):
    from symident.detection.discrete import Series

    ic = {x: x for x in f}
    f = dict(f, time="1")
    ic["time"] = "0"
    parts = [{"segments": [{"f": f, "g": g, "ic": ic, "t": "0", "label": ""}]}]
    return parts, [x for x in f if x != "time"] + params, Series(parts, [x for x in f if x != "time"] + params)


def test_a_number_field_reduces_modulo_its_polynomial():
    K = NumberField([-2, 0, 1])
    s = K.from_t([0, 1])
    assert K.mul(s, s) == K.rational(2)
    assert K.mul(s, K.inv(s)) == K.one
    # det(z - A) of the companion matrix of z^2 - 2
    assert charpoly(K, [[K.zero, K.one], [K.rational(2), K.zero]]) == [K.one, K.zero, K.rational(-2)]
    # s maps to a square root of 2 modulo p at a prime of degree one
    for p in (7, 17, 23):
        for tau in K.primes_of_degree_one(p):
            assert tau * tau % p == 2 and K.residue(s, p, tau) == tau
    # a polynomial with a leading coefficient becomes monic in s = lc * t
    K3 = NumberField([1, 0, 3])
    t = K3.from_t([0, 1])
    assert K3.add(K3.scale(K3.mul(t, t), Fraction(3)), K3.one) == K3.zero


def test_the_cycle_types_of_a_polynomial_modulo_primes():
    # z^3 - 2: irreducible modulo 7, a root and an irreducible quadratic modulo 5, not square-free modulo 3
    assert cycle_type([1, 0, 0, -2], 7) == [3]
    assert cycle_type([1, 0, 0, -2], 5) == [2, 1]
    assert cycle_type([1, 0, 0, -2], 3) is None
    assert rank_over_q([(Fraction(1), Fraction(2)), (Fraction(2), Fraction(4))]) == 1


@needs_msolve
def test_a_cycle_of_three_states_has_no_rational_first_integral():
    from symident.detection import firstint

    *_, s = _series(
        {"x1": "-k1*x1 + k3*x3 + u0", "x2": "k1*x1 - k2*x2", "x3": "k2*x2 - (k3 + k4)*x3"},
        {"y": "x1"},
        ["k1", "k2", "k3", "k4", "u0"],
    )
    c = firstint.certificate(s)
    assert c["proven"] and c["field"] == "Q"
    (b,) = c["blocks"]
    assert b["degree"] == 3 and b["group"] == "S_3" and set(b["primes"]) >= {"n-cycle", "(n-1)-cycle"}


@needs_msolve
def test_a_cascade_takes_the_parameters_into_a_number_field():
    from symident.detection import firstint

    # two blocks of one state: their rational traces are dependent over Q, those over Q(alpha) are not
    *_, s = _series({"x1": "c - a*x1", "x2": "x1 - b*x2"}, {"y": "x2"}, ["a", "b", "c"])
    c = firstint.certificate(s)
    assert c["proven"] and c["field"] != "Q" and c["alpha"] is not None
    assert [b["degree"] for b in c["blocks"]] == [1, 1]


@needs_msolve
@pytest.mark.parametrize(
    "f, g, params, why",
    [
        ({"x1": "-k1*x1 + k2*x2", "x2": "k1*x1 - k2*x2"}, {"y": "x1"}, ["k1", "k2"], "eigenvalue 0"),
        ({"x1": "w*x2", "x2": "-w*x1"}, {"y": "x1"}, ["w"], "trace of the Jacobian at the equilibrium is 0"),
        ({"x1": "(w + x3)*x2", "x2": "-w*x1", "x3": "0"}, {"y1": "x2", "y2": "x3"}, ["w"], "no isolated equilibrium"),
        ({"x": "-k*x + time"}, {"y": "x"}, ["k"], "clock"),
    ],
)
def test_a_first_integral_or_the_clock_leaves_the_premise_unproven(f, g, params, why):
    from symident.detection import firstint

    *_, s = _series(f, g, params)
    c = firstint.certificate(s)
    assert not c["proven"] and why in c["why"]


@needs_msolve
def test_the_count_of_two_experiments_is_that_of_the_parameters_and_the_first_starts():
    import random

    from symident.detection import discrete, multiexp
    from symident.gfp import PRIMES

    parts, coords, s = _series({"x": "-k*x"}, {"y": "x^2"}, ["k"])
    starts, why = multiexp.eligible(s)
    assert why is None and [str(v) for v in starts.values()] == ["x"]
    pk, ck = multiexp.stacked(parts, coords, s, starts, 2)
    assert ck == ["x", "k", "x_e2"] and [p["segments"][0]["label"] for p in pk] == ["e1:", "e2:"]
    sk = discrete.Series(pk, ck)
    rng = random.Random(3)
    pt = {spy.Symbol(c): spy.Integer(rng.randint(2, 1000)) for c in ck}
    fib = discrete.Fiber(sk, 2, True)
    # each start goes to the negative independently: four points, two values of (x, k)
    assert discrete.complex_count(fib, pt, PRIMES[0]) == 4
    fib.proj = len(coords)
    assert discrete.complex_count(fib, pt, PRIMES[0]) == 2
    # over Q, from the parametrization of all four points
    assert discrete.q_count(fib, pt) == 2


def test_a_fixed_start_or_an_event_leaves_the_route_out():
    from symident.detection import multiexp
    from symident.detection.discrete import Series

    seg = {"f": {"x": "-k*x"}, "g": {"y": "x"}, "ic": {"x": "1"}, "t": "0", "label": ""}
    assert "fixed" in multiexp.eligible(Series([{"segments": [seg]}], ["k"]))[1]
    seg = dict(seg, ic={"x": "x"})
    ev = {
        "f": seg["f"],
        "g": seg["g"],
        "events": [{"var": "x", "method": "add", "value": "1"}],
        "t": "1",
        "label": "@1",
    }
    assert "events" in multiexp.eligible(Series([{"segments": [seg, ev]}], ["x", "k"]))[1]
    assert "one condition" in multiexp.eligible(Series([{"segments": [seg]}, {"segments": [seg]}], ["x", "k"]))[1]
    # a start coordinate that is a parameter as well
    seg = dict(seg, f={"x": "-k*x + x0"}, ic={"x": "x0"})
    assert "also a parameter" in multiexp.eligible(Series([{"segments": [seg]}], ["x0", "k"]))[1]


@needs_msolve
def test_two_compartments_count_their_swap_on_several_experiments():
    m = si.Model({"x1": "-(k21 + k01)*x1 + k12*x2 + u", "x2": "k21*x1 - k12*x2"}, {"y": "x2"}, known_inputs=["u"])
    r = si.detect(m, cores=1)
    info = r.discrete(points=1, experiments=True)["info"]
    ex = info["experiments"]
    assert ex["why"] is None and ex["experiments"] >= 2 and ex["count"] == 2
    assert ex["first_integrals"]["proven"]
    cert = info["certificate"]
    assert info["count"] == 2 and info["count_final"] and cert["route"]["experiments"] == ex["experiments"]
    assert cert["complete"] in ("verified", "axiom")
    assert any("divides the size of the fiber" in p for p in cert["proven"])
    assert "counted on experiments with shared parameters" in str(r)


@needs_msolve
def test_the_twisted_oscillator_leaves_the_route_to_one_experiment():
    m = si.Model({"x1": "(w + x3)*x2", "x2": "-w*x1", "x3": "0"}, {"y1": "x2", "y2": "x3"})
    info = si.detect(m, cores=1).discrete(points=1, experiments=True)["info"]
    # x3 is a conservation law: on a level set the oscillator has the trace 0
    assert "first integral" in info["experiments"]["why"] and "trace" in info["experiments"]["why"]
    assert "route" not in info["certificate"]
    # one experiment leaves w and -(w + x3)
    assert info["count"] == 2


def test_a_conservation_law_is_found_over_the_field_of_the_parameters():
    from symident.detection import multiexp

    seg = {"f": {"x1": "-k1*x1*x3 + k2*x2", "x2": "(k1*x1*x3 - k2*x2)/v", "x3": "c - d*x3 + u", "u": "0"}}
    (law,) = multiexp.conservation_laws(seg, ["x1", "x2", "x3"])
    assert law == {"x1": 1, "x2": spy.Symbol("v")}
    assert multiexp.conservation_laws(dict(seg, f=dict(seg["f"], x3="c - d*x3 + u*x1")), ["x1", "x2", "x3"]) == [law]


@needs_msolve
def test_the_premise_holds_on_the_level_set_of_a_conservation_law():
    from symident.detection import firstint, multiexp

    f = {"x1": "-k1*x1*x3 + k2*x2", "x2": "k1*x1*x3 - k2*x2", "x3": "c - d*x3"}
    parts, coords, s = _series(f, {"y1": "x1", "y2": "x3"}, ["c", "d", "k1", "k2"])
    laws = multiexp.conservation_laws(parts[0]["segments"][0], ["x1", "x2", "x3"])
    assert not firstint.certificate(s)["proven"]
    c = firstint.certificate(s, laws)
    assert c["proven"] and c["laws"] and any(b.get("zero_eigenvalues") == 1 for b in c["blocks"])
    # the second experiment starts on the level set of the first: the pivot x1 of the law follows from it
    starts, _ = multiexp.eligible(s)
    pk, ck = multiexp.stacked(parts, coords, s, starts, 2, laws)
    assert ck == coords + ["x2_e2", "x3_e2"]
    assert spy.sympify(pk[1]["segments"][0]["ic"]["x1"]) == spy.sympify("x1 + x2 - x2_e2")


@needs_msolve
def test_two_experiments_on_one_level_set_count_a_globally_identifiable_exchange():
    m = si.Model({"x1": "-k1*x1 + k2*x2", "x2": "k1*x1 - k2*x2"}, {"y1": "x1", "y2": "x2"})
    info = si.detect(m, cores=1).discrete(points=1, experiments=True)["info"]
    ex = info["experiments"]
    assert ex["why"] is None and ex["first_integrals"]["laws"] and ex["count"] == 1
    assert info["count"] == 1 and info["count_final"] and "route" in info["certificate"]
