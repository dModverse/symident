"""Points of the fiber by monodromy and their Krawczyk boxes: proposals that only proven maps turn into results."""

from fractions import Fraction

import numpy as np
import pytest
import sympy as spy

import symident as si
from symident.detection import monodromy as mono
from symident.detection.discrete import Fiber, Series
from symident.detection.fieldproof import generators
from symident.sym.msolve import MsolveMissingError, _msolve_binary


def _has_msolve():
    try:
        _msolve_binary()
    except MsolveMissingError:
        return False
    return True


pytestmark = pytest.mark.skipif(not _has_msolve(), reason="needs msolve")


def _series(model):
    r = si.detect(model, cores=1)
    coords = list(r.info["coordinates"])
    parts, _ = r._fiber_parts(2 * len(coords), "discrete()")
    return Series(parts, coords)


def _contains(box, z):
    return all(re[0] <= v.real <= re[1] and im[0] <= v.imag <= im[1] for (re, im), v in zip(box, z))


def test_the_boxes_of_a_swap_hold_the_point_and_its_image_and_are_disjoint():
    s = _series(si.Model({"x1": "-a*x1", "x2": "-b*x2"}, {"y": "x1 + x2"}, trafo={"x1": "1", "x2": "1"}))
    fib = Fiber(s, [3], True)
    info, maps = mono.propose(s, fib, target=2)
    a, b = (Fraction(info["base"][c]) for c in ("a", "b"))
    assert info["certified"] == 2 and mono.disjoint(*info["boxes"])
    assert sum(_contains(bx, [complex(a), complex(b)]) for bx in info["boxes"]) == 1
    assert sum(_contains(bx, [complex(b), complex(a)]) for bx in info["boxes"]) == 1
    assert [{str(k): str(v) for k, v in m.items()} for m in maps] == [{"a": "b", "b": "a"}]


def test_the_krawczyk_operator_proves_no_zero_in_a_box_without_one():
    s = _series(si.Model({"x": "-k^4*x"}, {"y": "x"}, trafo={"x": "x0"}))
    base = {c: spy.Rational(3, 2) for c in s.cs}
    x = np.array([1.5, 1.5], dtype=complex)
    rows = mono.square_rows(s, [2], 0, x)
    assert mono.krawczyk(s, [2], 0, rows, base, x, [1e-9, 1e-9]) is not None
    # the zeros of the square subsystem are i^m * 3/2 in k: a box of radius 0.1 around 1.8 holds none
    assert mono.krawczyk(s, [2], 0, rows, base, np.array([1.8, 1.5], dtype=complex), [0.1, 0.1]) is None


def test_a_fourth_power_gives_four_certified_points_and_the_maps_over_q_of_i():
    s = _series(si.Model({"x": "-k^4*x"}, {"y": "x"}, trafo={"x": "x0"}))
    info, maps = mono.propose(s, Fiber(s, [3], True), target=4)
    k = complex(Fraction(info["base"]["k"]))
    assert info["certified"] == 4
    for z in (k, 1j * k, -k, -1j * k):
        assert sum(_contains(bx, [z, complex(Fraction(info["base"]["x0"]))]) for bx in info["boxes"]) == 1
    K = spy.Symbol("k")
    assert sorted((m[K] / K for m in maps), key=str) == sorted([-1, spy.I, -spy.I], key=str)


def test_the_orbit_polynomial_of_a_group_over_q_of_i_has_rational_coefficients():
    k, x0 = spy.symbols("k x0")
    grp = [{k: k, x0: x0}, {k: spy.I * k, x0: x0}, {k: -k, x0: x0}, {k: -spy.I * k, x0: x0}]
    alpha, fs, _, others = generators(grp, [k, x0])
    assert alpha == k and others == []
    assert {spy.expand(f) for f in fs} == {x0, -k ** 4}


def test_the_paths_of_the_monodromy_stop_at_their_cap_with_the_cap_named():
    from symident.detection import work
    s = _series(si.Model({"x": "-k^4*x"}, {"y": "x"}, trafo={"x": "x0"}))
    meter = work.Meter("the monodromy", 0, "term products")
    info, _ = mono.propose(s, Fiber(s, [3], True), meter, target=4)
    assert meter.over and info["why"] == "the monodromy reached the cap of 0 term products"
