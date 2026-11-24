"""Input-output projections of the exact count: support modulo a prime, coefficients interpolated in the parameters,
the relation checked as a polynomial identity."""

import math
import random
from fractions import Fraction

import numpy as np
import pytest

import symident as si
from symident.detection import fieldproof, ioproj, work
from symident.gfp import all_primes
from symident.sym.msolve import MsolveMissingError, _msolve_binary


def _has_msolve():
    try:
        _msolve_binary()
    except MsolveMissingError:
        return False
    return True


def _captured(model, monkeypatch):
    """The arguments of the first exact count of discrete() on the model."""
    got = []

    def record(*args, **kw):
        got.append((args, kw))
        return {"exact": False, "why": "off", "generators": [], "seconds": 0.0}
    with monkeypatch.context() as m:
        m.setattr(fieldproof, "exact_fiber", record)
        si.detect(model, mode="global", cores=1)
    assert got
    return got[0]


def _search_by_projections(model, monkeypatch, cap=10 ** 10):
    """The exact count in this process with the dense relations off: only projections give relations."""
    args, kw = _captured(model, monkeypatch)
    series, orders, grp, syms, inputs = args[:5]
    monkeypatch.setattr(fieldproof, "io_functions", lambda *a, **k: iter(()))
    monkeypatch.setattr(fieldproof, "state_relation", lambda *a, **k: None)
    fieldproof._REL.clear()
    return fieldproof._exact_fiber(series, list(orders), grp, list(syms), cap, list(inputs), random.Random(1),
                                   list(kw.get("known", args[5] if len(args) > 5 else ())), [])


def test_rational_reconstruction_of_one_variable_finds_unbalanced_degrees():
    p = all_primes()[0]
    rng = random.Random(2)
    N = [rng.randrange(p) for _ in range(9)]
    D = [rng.randrange(p) for _ in range(3)] + [1]

    def ev(c, t):
        return sum(x * pow(t, k, p) for k, x in enumerate(c)) % p
    ts = list(range(1, 15))
    got = ioproj.ratrec1(ts, [ev(N, t) * pow(ev(D, t), -1, p) % p for t in ts], p)
    assert got is not None and got[1] == D and len(got[0]) == len(N)
    # two points short of the degrees: no answer rather than a wrong one
    assert ioproj.ratrec1(ts[:12], [ev(N, t) * pow(ev(D, t), -1, p) % p for t in ts[:12]], p) is None


def test_sparse_numerators_find_the_terms_of_a_polynomial():
    q = all_primes()[0]
    terms = {(1, 0, 2, 0): 3, (0, 1, 0, 1): 5, (2, 2, 0, 0): 7, (0, 0, 0, 0): 11}

    def f(x, p):
        return np.array([sum(c * math.prod(pow(xi, e, p) for xi, e in zip(x, m)) for m, c in terms.items()) % p])
    co = ioproj.Coefficients(None, 4, random.Random(3), math.inf)
    got = co._sparse(f, np.array([1]), [(0, 0, 0, 0)], [ioproj._box(4, 4, [2, 2, 2, 2])], q)
    assert got == [{m: c % q for m, c in terms.items()}]


@pytest.mark.skipif(not _has_msolve(), reason="needs msolve")
def test_a_projection_is_proven_and_a_wrong_coefficient_fails(monkeypatch):
    m = si.Model({"x": "-a*x^2 + b*z", "z": "-c*z"}, {"y": "x"}, trafo={"x": "x0", "z": "1"})
    args, _ = _captured(m, monkeypatch)
    series, syms = args[0], args[3]
    fieldproof._REL.clear()
    rel = fieldproof._relations(series, syms)
    rng = random.Random(1)
    p = all_primes()[0]
    idx = [k for k, s in enumerate(syms) if s in set(series.params)]
    pos = [rel.names.index(str(syms[k])) for k in idx]
    h = fieldproof._profiles(rel, rng)[0]
    assert h == [2]
    pr = ioproj.Projection.output(rel, h, 0, pos, rng)
    S = pr.search([rng.randrange(1, p) for _ in idx], p, work.ticks() + 10 ** 10)
    # x'' + 2 a x x' + c x' + a c x^2
    assert sorted(S) == sorted([(0, 0, 1), (1, 1, 0), (0, 1, 0), (2, 0, 0)])
    assert ioproj.series_rank(rel, pr, S, syms, rng, p) == len(S) - 1
    co = ioproj.Coefficients(pr, len(idx), rng, work.ticks() + 10 ** 10)
    assert co.run()
    assert ioproj.identity(rel, pr, co, pos)
    k = next(iter(co.nums[0]))
    co.nums[0][k] += Fraction(1, 7)
    assert not ioproj.identity(rel, pr, co, pos)


@pytest.mark.skipif(not _has_msolve(), reason="needs msolve")
def test_the_exact_count_holds_with_projections_alone(monkeypatch):
    m = si.Model({"x": "-a*x^2 + b*z", "z": "-c*z"}, {"y": "x"}, trafo={"x": "x0", "z": "1"})
    out = _search_by_projections(m, monkeypatch)
    assert out["exact"] is True
    assert any(r.startswith("projection of observable 0") and r.endswith("proven") for r in out["relations"])


@pytest.mark.skipif(not _has_msolve(), reason="needs msolve")
def test_a_rational_model_is_proven_through_its_projection(monkeypatch):
    m = si.Model({"x": "-a*x/(k + x) + z", "z": "-c*z"}, {"y": "x"}, trafo={"x": "x0", "z": "1"})
    out = _search_by_projections(m, monkeypatch)
    assert out["exact"] is True
    assert any(r.endswith("proven") for r in out["relations"])


@pytest.mark.skipif(not _has_msolve(), reason="needs msolve")
def test_the_start_of_an_unobserved_state_has_a_state_projection(monkeypatch):
    m = si.Model({"x": "-a*x + z", "z": "-c*z^2"}, {"y": "x"}, trafo={"x": "x0", "z": "z0"})
    args, _ = _captured(m, monkeypatch)
    series, syms = args[0], args[3]
    fieldproof._REL.clear()
    z0 = next(s for s in syms if str(s) == "z0")
    got = fieldproof.state_projection(series, syms, z0, random.Random(1), work.ticks() + 10 ** 10)
    # z = x' + a x: linear in z, with a coefficient that reads a
    assert got is not None and {str(s) for s in got["reads"]} == {"a"} and got["terms"] == 3


@pytest.mark.skipif(not _has_msolve(), reason="needs msolve")
def test_a_start_quadratic_below_the_profile_has_a_state_projection_with_the_next_derivative(monkeypatch):
    m = si.Model({"x": "z^2 - a*x", "z": "b"}, {"y": "x"}, trafo={"x": "x0", "z": "z0"})
    args, _ = _captured(m, monkeypatch)
    series, syms = args[0], args[3]
    fieldproof._REL.clear()
    z0 = next(s for s in syms if str(s) == "z0")
    got = fieldproof.state_projection(series, syms, z0, random.Random(1), work.ticks() + 10 ** 10)
    assert got is not None and {str(s) for s in got["reads"]} == {"a", "b"} and got["terms"] == 3


@pytest.mark.skipif(not _has_msolve(), reason="needs msolve")
def test_the_exact_output_coefficients_are_the_taylor_coefficients_of_a_rational_model(monkeypatch):
    import sympy as spy
    m = si.Model({"x": "-a*x/(k + x) + z", "z": "-c*z"}, {"y": "x/(1 + z)"}, trafo={"x": "x0", "z": "1"})
    args, _ = _captured(m, monkeypatch)
    series, orders = args[0], args[1]
    outs = fieldproof._Outputs(series, orders)
    R, F, cs = outs.exact(4)
    t = {spy.Symbol(f"t{i}"): s for i, s in enumerate(series.cs)}

    def expr(q):
        den = spy.prod([F.fac[k].as_expr() ** e for k, e in q[1].items()])
        return (q[0].as_expr() / den).xreplace(t)
    a, k, c, x, z, x0 = (spy.Symbol(n) for n in ("a", "k", "c", "x", "z", "x0"))
    f = {x: -a * x / (k + x) + z, z: -c * z}
    d = x / (1 + z)
    for j, (_i, order) in enumerate(outs.index):
        if order > 4:
            continue
        want = (d / math.factorial(order)).xreplace({x: x0, z: 1})
        assert spy.cancel(expr(cs[j]) - want) == 0
        d = sum(spy.diff(d, v) * fv for v, fv in f.items())
    p = all_primes()[0]
    rng = random.Random(4)
    pts = [[rng.randrange(1, p) for _ in series.cs] for _ in range(5)]
    vals, ok = outs.mod_many(pts, p)
    for x_, v, good in zip(pts, vals, ok):
        old = series.numeric(dict(zip(series.cs, x_)), outs.order, 0, p)
        assert good and [int(w) for w in v] == [int(old[i][kk][0]) % p for i, kk in outs.index]


@pytest.mark.skipif(not _has_msolve(), reason="needs msolve")
def test_counts_of_points_bound_the_newton_polytope_of_a_projection(monkeypatch):
    m = si.Model({"x": "-a*x^2 + b*z", "z": "-c*z"}, {"y": "x"}, trafo={"x": "x0", "z": "1"})
    args, _ = _captured(m, monkeypatch)
    series, syms = args[0], args[3]
    fieldproof._REL.clear()
    rel = fieldproof._relations(series, syms)
    rng = random.Random(5)
    p = all_primes()[0]
    idx = [k for k, s in enumerate(syms) if s in set(series.params)]
    pos = [rel.names.index(str(syms[k])) for k in idx]
    pr = ioproj.Projection.output(rel, [2], 0, pos, rng)
    theta = [rng.randrange(1, p) for _ in idx]
    mons = pr.polytope(theta, p, work.ticks() + 10 ** 10)
    # x'' + 2 a x x' + c x' + a c x^2: degrees 2, 1, 1 in x, x', x'', total 2, weighted by order 3
    assert pr.newton[(1, 1, 1)] == (1, 2)
    assert [pr.newton[ioproj._unit(3, v)][1] for v in range(3)] == [2, 1, 1]
    assert pr.newton[(1, 2, 3)][1] == 3
    assert {(0, 0, 1), (1, 1, 0), (0, 1, 0), (2, 0, 0)} <= set(mons)
    # with boxes too small for the projection the support comes from the polytope
    S = pr.search(theta, p, work.ticks() + 10 ** 10, cap=4)
    assert sorted(S) == sorted([(0, 0, 1), (1, 1, 0), (0, 1, 0), (2, 0, 0)])


@pytest.mark.skipif(not _has_msolve(), reason="needs msolve")
def test_a_projection_of_higher_degree_in_the_start_is_not_searched_for_a_linear_one(monkeypatch):
    m = si.Model({"x": "z^2 - a*x", "z": "b"}, {"y": "x"}, trafo={"x": "x0", "z": "z0"})
    args, _ = _captured(m, monkeypatch)
    series, syms = args[0], args[3]
    fieldproof._REL.clear()
    rel = fieldproof._relations(series, syms)
    rng = random.Random(6)
    p = all_primes()[0]
    idx = [k for k, s in enumerate(syms) if s in set(series.params)]
    pos = [rel.names.index(str(syms[k])) for k in idx]
    k = next(j for j, x in enumerate(rel.X) if str(x) == "z")
    pr = ioproj.Projection.state(rel, [2], k, pos, rng)
    # z^2 = x' + a x: degree two in z, so no relation linear in z below the profile
    assert pr.search([rng.randrange(1, p) for _ in idx], p, work.ticks() + 10 ** 10, cap=4, top_degree=1) is None
    assert pr.newton[ioproj._unit(pr.nb, pr.top)][1] == 2


def test_relations_skip_derivatives_of_derivatives_above_the_term_limit(monkeypatch):
    m = si.Model({"x": "-a*x^3 + b*x*z + z^2", "z": "-c*z^2 + x"}, {"y": "x"}, trafo={"x": "x0", "z": "z0"})
    args, _ = _captured(m, monkeypatch)
    series, syms = args[0], args[3]
    fieldproof._REL.clear()
    rel = fieldproof._relations(series, syms)
    J = ioproj.jets_of(rel)
    monkeypatch.setattr(ioproj, "_JET", 4)
    z0 = next(s for s in syms if str(s) == "z0")
    fieldproof.state_relation(series, syms, z0, random.Random(1), work.ticks() + 10 ** 9, cap=40)
    # no derivative is taken of one with more than four terms
    lst = J._obs[(0, str(rel.y[0]))][0]
    assert all(len(N) <= 4 for N, _, _ in lst[:-1])


def test_no_projection_reads_the_derivative_of_one_above_the_term_limit(monkeypatch):
    m = si.Model({"x": "-a*x^3 + b*x*z + z^2", "z": "-c*z^2 + x"}, {"y": "x"}, trafo={"x": "x0", "z": "z0"})
    args, _ = _captured(m, monkeypatch)
    series, syms = args[0], args[3]
    fieldproof._REL.clear()
    rel = fieldproof._relations(series, syms)
    monkeypatch.setattr(ioproj, "_JET", 4)
    assert ioproj.Projection.output(rel, [3], 0, [], random.Random(1)) is None
    lst = ioproj.jets_of(rel)._obs[(0, str(rel.y[0]))][0]
    assert all(len(N) <= 4 for N, _, _ in lst[:-1])


@pytest.mark.skipif(not _has_msolve(), reason="needs msolve")
def test_coefficients_above_the_height_of_the_engine_primes_are_rebuilt(monkeypatch):
    # a constant of 200 bits in the coefficients: beyond the rational reconstruction of the engine primes
    K = 2 ** 200 + 235
    m = si.Model({"x": "-a*x^2 + z", "z": f"-{K}*c*z"}, {"y": "x"}, trafo={"x": "x0", "z": "1"})
    args, _ = _captured(m, monkeypatch)
    series, syms = args[0], args[3]
    fieldproof._REL.clear()
    rel = fieldproof._relations(series, syms)
    rng = random.Random(1)
    p = all_primes()[0]
    idx = [k for k, s in enumerate(syms) if s in set(series.params)]
    pos = [rel.names.index(str(syms[k])) for k in idx]
    pr = ioproj.Projection.output(rel, [2], 0, pos, rng)
    assert len(pr.search([rng.randrange(1, p) for _ in idx], p, work.ticks() + 10 ** 10)) == 4
    co = ioproj.Coefficients(pr, len(idx), rng, work.ticks() + 10 ** 10)
    assert co.run()
    assert ioproj.identity(rel, pr, co, pos)
    assert any(abs(c.numerator) >= K for d in co.nums for c in d.values())


def _search_by_elimination(model, monkeypatch, cap=10 ** 10):
    """The exact count in this process with the dense relations, the projections and the relations of the starts off:
    only the elimination gives relations and starts."""
    args, kw = _captured(model, monkeypatch)
    series, orders, grp, syms, inputs = args[:5]
    monkeypatch.setattr(fieldproof, "io_functions", lambda *a, **k: iter(()))
    monkeypatch.setattr(fieldproof, "projection_functions", lambda *a, **k: iter(()))
    monkeypatch.setattr(fieldproof, "state_relation", lambda *a, **k: None)
    monkeypatch.setattr(fieldproof, "state_projection", lambda *a, **k: None)
    fieldproof._REL.clear()
    return fieldproof._exact_fiber(series, list(orders), grp, list(syms), cap, list(inputs), random.Random(1),
                                   list(kw.get("known", args[5] if len(args) > 5 else ())), [])


def _saturable():
    return si.Model({"x1": "a1*(x2 - x1) - ka*x1/(kc + x1)", "x2": "a2*(x1 - x2)"}, {"y": "x1"})


def test_the_elimination_gives_the_relation_of_the_observable(monkeypatch):
    import sympy as spy

    from symident.detection import elim
    args, _ = _captured(_saturable(), monkeypatch)
    series, syms = args[0], args[3]
    fieldproof._REL.clear()
    rel = fieldproof._relations(series, syms)
    got = elim.eliminate_relations(rel, random.Random(1), math.inf)
    (P,) = got["relations"].values()
    y0, y1, y2, a1, a2, ka, kc = spy.symbols("si_y0_0 si_y0_1 si_y0_2 a1 a2 ka kc")
    names = {n: spy.Symbol(n) for n in got["ring"].names}
    expr = spy.sympify(str(P).replace("^", "**"), locals=names)
    # (kc + y)^2 y'' + (a1 + a2) (kc + y)^2 y' + ka kc y' + a2 ka y (kc + y), up to a constant
    want = (kc + y0) ** 2 * y2 + (a1 + a2) * (kc + y0) ** 2 * y1 + ka * kc * y1 + a2 * ka * y0 * (kc + y0)
    assert spy.cancel(expr / want).is_number


def test_the_elimination_drops_a_factor_nonzero_along_a_trajectory(monkeypatch):
    from symident.detection import elim
    args, _ = _captured(_saturable(), monkeypatch)
    series, syms = args[0], args[3]
    fieldproof._REL.clear()
    rel = fieldproof._relations(series, syms)
    got = elim.eliminate_relations(rel, random.Random(1), math.inf)
    (P,) = got["relations"].values()
    R = got["ring"]
    el = elim.Eliminator(rel, random.Random(2), math.inf)
    a1, y0 = (R.gens[R.at[n]] for n in ("a1", "si_y0_0"))
    kept = el.keep(P * (a1 + y0) * (y0 + 1), extra=[a1 - 1])
    assert kept == P or kept == -P


@pytest.mark.skipif(not _has_msolve(), reason="needs msolve")
def test_the_exact_count_holds_with_a_relation_by_elimination(monkeypatch):
    out = _search_by_elimination(_saturable(), monkeypatch)
    assert out["exact"] is True
    assert any(r.startswith("elimination of observable 0") and r.endswith("coefficients proven")
               for r in out["relations"])
    # the start of the unobserved state from the equation of degree one in it where it is eliminated
    assert "x2" in {g["function"] for g in out["generators"] if g["numerator"] == "elimination"}


def test_the_first_subresultant_gives_the_common_root():
    import flint

    from symident.detection import elim
    ctx = flint.fmpz_mpoly_ctx.get(("v", "a", "b", "c"), "lex")
    v, a, b, c = ctx.gens()
    f, g = (v - a) * (v - b) * (v + 1), (v - a) * (v - c)
    A, B = elim._subresultant1(f, g, 0)
    # A v + B is a combination of f and g, so it vanishes at their common root v = a
    assert (A * a + B).is_zero() and not A.is_zero()


def test_the_elimination_stops_at_its_cap_on_terms(monkeypatch):
    from symident.detection import elim
    args, _ = _captured(_saturable(), monkeypatch)
    series, syms = args[0], args[3]
    fieldproof._REL.clear()
    rel = fieldproof._relations(series, syms)
    with pytest.raises(ValueError, match="past the cap elimination_terms of 2"):
        elim.eliminate_relations(rel, random.Random(1), 2)


@pytest.mark.skipif(not _has_msolve(), reason="needs msolve")
def test_the_relation_by_elimination_names_its_cap_in_the_log(monkeypatch):
    with work.caps(elimination_terms=2):
        out = _search_by_elimination(_saturable(), monkeypatch)
    assert out["exact"] is False
    assert any(r.startswith("elimination: a polynomial of") and r.endswith("past the cap elimination_terms of 2")
               for r in out["relations"])


@pytest.mark.skipif(not _has_msolve(), reason="needs msolve")
def test_the_exact_search_ends_at_its_cap_on_entries_with_the_cap_named(monkeypatch):
    out = _search_by_elimination(_saturable(), monkeypatch, cap=0)
    assert out["exact"] is False
    assert out["why"].endswith("before the end of the search: the search reached the cap search_entries of 0 "
                               "entries of the matrices reduced")
