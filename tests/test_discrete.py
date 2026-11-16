"""Discrete symmetries: the other points of the domain with the output of a generic one."""

import pytest
import sympy as spy

import symident as si
from symident.sym.msolve import MsolveMissingError, _msolve_binary


def _has_msolve():
    try:
        _msolve_binary()
    except MsolveMissingError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _has_msolve(), reason="needs msolve")


def same(a, b):
    return spy.simplify(spy.sympify(a) - spy.sympify(b)) == 0


def two_compartment():
    return si.Model({"x1": "-(k21 + k01)*x1 + k12*x2 + u", "x2": "k21*x1 - k12*x2"}, {"y": "x2"},
                    known_inputs=["u"])


def test_two_compartment_exchanges_the_elimination_rates():
    r = si.detect(two_compartment(), cores=1)
    out = r.discrete(points=[{"x1": 1, "x2": 2, "k01": 3, "k12": 5, "k21": 7},
                             {"x1": 1, "x2": 2, "k01": 4, "k12": 4, "k21": 7}])
    (s,) = out["symmetries"]
    # the coordinates SIAN.jl reports locally but not globally identifiable
    assert s["level"] == "exact" and set(s["changes"]) == {"x1", "k01", "k12"}
    assert same(s["map"]["k01"], "k12") and same(s["map"]["k12"], "k01")
    assert same(s["map"]["x1"], "x1 + (k01 - k12)*x2/k21")
    # sigma(theta) is in the domain while its initial value of x1 stays positive
    assert s["preserves_domain"] is None and len(s["domain_conditions"]) == 1
    assert [spy.sympify(e) for e in s["fixed_set"]] == [spy.sympify("k01 - k12")]
    info = out["info"]
    assert info["count"] == 2 and info["count_final"] and info["order_stable"]
    assert info["fundamental_domain"] == ["k01 < k12"]
    # on the fixed set the two solutions meet
    assert [c["in_domain"] for c in info["counts"]] == [2, 1]
    assert r.discrete_symmetries == out["symmetries"]


def test_a_symmetry_outside_the_domain_is_not_relevant():
    r = si.detect(si.Model({"x": "-theta^2*x"}, {"y": "x"}), cores=1, discrete={"points": 2})
    (s,) = r.discrete_symmetries
    assert same(s["map"]["theta"], "-theta") and s["preserves_domain"] is False and s["level"] == "exact"
    info = r.info["discrete"]
    assert info["count"] == 2 and info["fundamental_domain"] == []
    assert all(c["real"] == 2 and c["in_domain"] == 1 for c in info["counts"])


def test_the_count_in_the_domain_differs_between_regions():
    m = si.Model({"x": "-a*x", "z": "-(b - a)^2*z"}, {"y1": "x", "y2": "z"}, trafo={"x": "1", "z": "1"})
    out = si.detect(m, cores=1).discrete(points=[{"a": 2, "b": 1}, {"a": 1, "b": 3}])
    (s,) = out["symmetries"]
    assert same(s["map"]["b"], "2*a - b") and s["level"] == "exact"
    assert s["preserves_domain"] is None and [spy.sympify(c.split(">")[0]) for c in s["domain_conditions"]] == \
        [spy.sympify("2*a - b")]
    assert [c["in_domain"] for c in out["info"]["counts"]] == [2, 1]


def test_rates_identified_only_as_a_set():
    # HIV model of Rey Barreiro and Villaverde (2023): c, k1 + w1 and w2 enter only through symmetric functions
    m = si.Model({"x1": "-b*x1*x4 - d*x1 + s", "x2": "b*q1*x1*x4 - w1*x2 - k1*x2",
                  "x3": "b*x1*q2*x4 - x3*w2 + k1*x2", "x4": "-c*x4 + k2*x3"}, {"y1": "x1", "y2": "x4"})
    r = si.detect(m, cores=1, reconstruct=True)
    assert not r.identifiable
    out = r.discrete(points=1)
    info = out["info"]
    assert info["count"] == 6 and info["count_final"] and "reduced" in info
    assert len(out["symmetries"]) == 5 and all(s["level"] == "exact" for s in out["symmetries"])
    # the three rates in the coordinates of the reduced model, ordered by the fundamental domain
    rates = {spy.sympify(s["map"]["c"]) for s in out["symmetries"] if "c" in s["map"]} | {spy.Symbol("c")}
    # SIAN.jl reports c and w2 locally but not globally identifiable; k1 and w1 alone are not identifiable
    assert {"c", "w2"} <= {x for s in out["symmetries"] for x in s["changes"]}
    assert len(rates) == 3
    assert len(info["fundamental_domain"]) == 3
    red = r.reduce()
    assert red.discrete is not None and red.discrete["fundamental_domain"] == info["fundamental_domain"]


def test_a_square_outside_the_domain_leaves_the_model_globally_identifiable():
    # beta-cell model of Finegood et al. (1995), as in SIAN.jl: SIAN reports alpha locally but not globally
    # identifiable over the complex numbers, from alpha -> -alpha
    m = si.Model({"G": "u1 - (c + si*II)*G",
                  "beta": "beta*((1/(48*24*60))/(1 + (8/G)^2) - (1/(40*24*60))/(1 + (G/5)^8))",
                  "II": "p1*beta*(G^2/(alpha^2 + G^2)) - gm*II"}, {"y1": "G"}, known_inputs=["u1"])
    r = si.detect(m, cores=1, reconstruct=True)
    out = r.discrete(points=2)
    (s,) = out["symmetries"]
    assert s["changes"] == ["alpha"] and same(s["map"]["alpha"], "-alpha")
    assert s["level"] == "exact" and s["preserves_domain"] is False
    info = out["info"]
    assert info["count"] == 2 and info["count_final"] and info["fundamental_domain"] == []
    assert all(c["real"] == 2 and c["in_domain"] == 1 for c in info["counts"])


def test_a_vanishing_denominator_is_no_point_of_the_fiber():
    # the x4 equation of the p53 model of Rey Barreiro and Villaverde (2023), the other states as known inputs:
    # where p22^4 = -1 a denominator and a numerator vanish together, which the saturation must exclude
    f = {"x4": "p20 - p21*(1-p24)*(1-p25)/((p22^4)+1) - (p20*x4) + (p21*(x3^4))"
               " + (1+p23*u1)*(1-p24*x1)*(1-p25*x2)/(p22^4+x3^4)"}
    m = si.Model(f, {"y4": "x4"}, known_inputs=["u1", "x1", "x2", "x3"])
    out = si.detect(m, cores=1).discrete(points=1)
    (s,) = out["symmetries"]
    assert same(s["map"]["p22"], "-p22") and s["preserves_domain"] is False
    info = out["info"]
    # p22 -> -p22 and the two complex roots of p22^4
    assert info["count"] == 4 and [(c["complex"], c["real"], c["in_domain"]) for c in info["counts"]] == [(4, 2, 1)]


def test_discrete_needs_one_condition():
    m = si.Model({"x": "-k*x"}, {"y": "x"}, conditions={"rows": ["a", "b"], "cols": {"k": ["k1", "k2"]}})
    with pytest.raises(ValueError, match="one condition"):
        si.detect(m, cores=1).discrete()
