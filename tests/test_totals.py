"""Conserved totals under equilibrate: the result does not depend on the pivots, the names or the order of states
and totals, and reduce_cq agrees with the pivots."""

import re
import warnings

import pytest
import sympy as spy

import symident as si
from symident.detection.cq import canonical_laws
from symident.errors import InputError

R = spy.Rational
U = [{"var": "u", "time": 0, "value": 1, "method": "replace"}]


def _sub(text, ren):
    return re.sub(r"[A-Za-z_][A-Za-z0-9_]*", lambda m: ren.get(m.group(0), m.group(0)), str(text))


class Case:
    """A model with conserved totals and a resting point of it, renamed, reordered and detected."""

    def __init__(self, odes, obs, totals, point, events=U, inputs=("u",), fixed=()):
        self.odes, self.obs, self.totals, self.point = odes, obs, totals, point
        self.events, self.inputs, self.fixed = events, list(inputs), list(fixed)

    def renamed(self, ren):
        return Case({ren.get(k, k): _sub(v, ren) for k, v in self.odes.items()},
                    {k: _sub(v, ren) for k, v in self.obs.items()},
                    {k: _sub(v, ren) for k, v in self.totals.items()},
                    {ren.get(k, k): v for k, v in self.point.items()},
                    [dict(e, var=ren.get(e["var"], e["var"]), value=_sub(e["value"], ren)) for e in self.events],
                    [ren.get(x, x) for x in self.inputs], [ren.get(x, x) for x in self.fixed])

    def detect(self, order=None, totals_order=None, **kw):
        odes = self.odes if order is None else {k: self.odes[k] for k in order}
        totals = self.totals if totals_order is None else {k: self.totals[k] for k in totals_order}
        m = si.Model(odes=odes, observables=self.obs, totals=totals, events=self.events, inputs=self.inputs,
                     fixed=self.fixed)
        kw.setdefault("equilibrate", True)
        kw.setdefault("reconstruct", True)
        kw.setdefault("cores", 1)
        return si.detect(m, **kw)

    def span(self, r, back=None):
        """The directions of `r` mapped to the parameters and every resting state at the resting point."""
        back = back or {}
        vals = dict(self.point)
        states = [s for s in self.odes if s not in self.inputs]
        coords = list(r.info["coordinates"])
        sym = {k: spy.Symbol(k) for k in set(vals) | set(coords) | set(states) | set(self.totals)}
        at = {sym[k]: v for k, v in vals.items()}
        for t, e in self.totals.items():
            at[sym[t]] = spy.sympify(e.replace("^", "**"), locals=sym).subs(at)
        free = [c for c in coords if c in states]
        Y = [s for s in states if s not in free]
        eqs = [spy.sympify(self.odes[s].replace("^", "**"), locals=sym).subs({sym.get(u, spy.Symbol(u)): 0
                                                                           for u in self.inputs})
               for s in states]
        eqs += [spy.sympify(self.totals[c].replace("^", "**"), locals=sym) - sym[c] for c in coords if c in self.totals]
        GY = spy.Matrix([[spy.diff(e, sym[y]) for y in Y] for e in eqs]).subs(at)
        Gc = spy.Matrix([[spy.diff(e, sym[c]) for c in coords] for e in eqs]).subs(at)
        dY, free_par = GY.gauss_jordan_solve(-Gc)
        assert free_par.shape[0] == 0
        names = sorted([c for c in coords if c not in self.totals] + Y, key=lambda k: back.get(k, k))
        rows = []
        for s in r.symmetries:
            g = s.sympy()
            assert g is not None, "a direction without a closed form"
            xi = spy.Matrix([g.get(c, 0) for c in coords]).subs(at)
            full = {c: xi[i] for i, c in enumerate(coords)}
            full.update({y: (dY * xi)[i] for i, y in enumerate(Y)})
            rows.append([spy.nsimplify(full.get(k, 0)) for k in names])
        return rows


def summary(r):
    return (r.rank, r.dim, r.info["rank_status"]["status"], sorted((s.level, s.kind) for s in r.symmetries))


def same_span(a, b):
    ra = spy.Matrix(a).rank() if a else 0
    rb = spy.Matrix(b).rank() if b else 0
    return ra == rb == (spy.Matrix(a + b).rank() if a or b else 0)


def _rest_ab(k1, k2, a):
    return {"A": a, "B": k1 * a / k2}


EXCHANGE = Case({"A": "-k1*(1+u)*A + k2*B", "B": "k1*(1+u)*A - k2*B", "u": "0"}, {"y": "s*B"}, {"T": "A + B"},
                dict(k1=R(3, 7), k2=R(5, 11), s=R(13, 3), **_rest_ab(R(3, 7), R(5, 11), R(17, 5))))
VOLUMES = Case({"A": "(-k1*(1+u)*A + k2*B)/V1", "B": "(k1*(1+u)*A - k2*B)/V2", "u": "0"}, {"y": "s*B"},
               {"T": "V1*A + V2*B"},
               dict(k1=R(3, 7), k2=R(5, 11), s=R(13, 3), V1=R(2, 3), V2=R(7, 5), **_rest_ab(R(3, 7), R(5, 11), R(4))))
TWO = Case({"A": "(-k1*(1+u)*A + k2*B)/V1", "B": "(k1*(1+u)*A - k2*B)/V2", "C": "(-k3*B*C + k4*D)/V1",
            "D": "(k3*B*C - k4*D)/V2", "u": "0"}, {"y1": "s1*B", "y2": "D"}, {"T1": "V1*A + V2*B", "T2": "V1*C + V2*D"},
           dict(k1=R(3, 7), k2=R(5, 11), k3=R(2, 9), k4=R(7, 4), s1=R(13, 3), V1=R(2, 3), V2=R(7, 5),
                A=R(4), B=R(3, 7) * 4 / R(5, 11), C=R(6), D=R(2, 9) * (R(3, 7) * 4 / R(5, 11)) * 6 / R(7, 4)))


@pytest.mark.parametrize("case", [EXCHANGE, VOLUMES, TWO], ids=["exchange", "volumes", "two totals"])
def test_the_result_does_not_depend_on_pivots_names_or_order(case):
    ref = case.detect()
    base = case.span(ref)
    states = [s for s in case.odes if s not in case.inputs]
    per_total = [[x for x in states if re.search(rf"\b{x}\b", t)] for t in case.totals.values()]
    sets = [[a] for a in per_total[0]] if len(per_total) == 1 else [[a, b] for a in per_total[0] for b in per_total[1]]
    runs = [case.detect(free_initial=p) for p in sets]
    runs += [case.detect(order=list(reversed(case.odes))), case.detect(reduce_cq=True)]
    if len(case.totals) > 1:
        runs.append(case.detect(totals_order=list(reversed(case.totals))))
    for r in runs:
        assert summary(r) == summary(ref)
        assert same_span(case.span(r), base)
    # renamed species and parameters, the states in reverse order
    ren = {s: "Z" + t for s, t in zip(sorted(states), reversed(sorted(states)))}
    ren.update({p: "q" + p for p in case.point if p not in states})
    other = case.renamed(ren)
    r = other.detect(order=list(reversed(other.odes)))
    assert summary(r) == summary(ref)
    back = {v: k for k, v in ren.items()}
    assert same_span(other.span(r, back), base)
    assert sorted(ref.info["settings"]["free_initial"]) == sorted(back[x] for x in r.info["settings"]["free_initial"])


LINEAR = {"A": "-k1*A + k2*B", "B": "k1*A - k2*B"}


@pytest.mark.parametrize("var, time, method", [("A", 0, "add"), ("B", 0, "replace"), ("A", 1, "add"),
                                                ("B", 0, "multiply")])
def test_reduce_cq_applies_an_event_on_a_species_of_a_total(var, time, method):
    # the total changes by E(x) - x at the event; with y = s*B, A is eliminated
    case = Case(LINEAR, {"y": "s*B"}, {"T": "A + B"},
                dict(k1=R(3, 7), k2=R(5, 11), s=R(13, 3), dose=R(2), **_rest_ab(R(3, 7), R(5, 11), R(17, 5))),
                events=[{"var": var, "time": time, "value": "dose", "method": method}], inputs=(), fixed=("dose",))
    pivots, reduced = case.detect(), case.detect(reduce_cq=True)
    assert reduced.info["settings"]["eliminated"] == ["A"] and "T" in reduced.info["coordinates"]
    assert summary(pivots) == summary(reduced) and pivots.rank == 3
    assert same_span(case.span(pivots), case.span(reduced))


def test_a_dose_on_a_species_changes_the_total_in_later_segments():
    # the dose at t = 1 raises the total: without it the total and the rates would share a scaling
    m = si.Model(odes=LINEAR, observables={"y": "B"}, totals={"T": "A + B"},
                 events=[{"var": "A", "time": 1, "value": "dose", "method": "add"}], fixed=["dose"])
    r = si.detect(m, equilibrate=True, reduce_cq=True, cores=1)
    assert (r.rank, r.dim) == (3, 3) and r.info["rank_status"]["status"] == "proven"


BINDING = {"A": "-kon*A*B + koff*C", "B": "-kon*A*B + koff*C", "C": "kon*A*B - koff*C"}


def _binding(**kw):
    m = si.Model(odes=BINDING, observables={"y": "s*C"}, totals=kw.pop("totals", {"TA": "A + C", "TB": "B + C"}),
                 events=[{"var": "A", "time": 0, "value": "dose", "method": "add"}], fixed=["dose"])
    return si.detect(m, equilibrate=True, cores=1, **kw)


@pytest.mark.parametrize("pivots", [["A", "B"], ["A", "C"], ["C", "B"], ["B", "C"]])
def test_every_valid_pivot_set_of_totals_sharing_a_species_solves(pivots):
    ref = _binding()
    for totals in ({"TA": "A + C", "TB": "B + C"}, {"TB": "B + C", "TA": "A + C"}):
        r = _binding(free_initial=pivots, totals=totals)
        assert (r.rank, r.dim) == (ref.rank, ref.dim) == (4, 5)
        assert sorted(r.info["settings"]["free_initial"]) == sorted(pivots)


@pytest.mark.parametrize("pivots, match", [
    (["C", "C"], "names C twice"),
    (["A", "Z"], "Z, which is not a state"),
    (["A", "C", "B"], "do not leave independent"),
])
def test_an_invalid_free_initial_is_refused(pivots, match):
    with pytest.raises(InputError, match=match):
        _binding(free_initial=pivots)


def test_free_initial_takes_one_state_per_total():
    m = si.Model(odes=LINEAR, observables={"y": "s*B"}, totals={"T": "A + B"})
    with pytest.raises(InputError, match="single total T"):
        si.detect(m, equilibrate=True, free_initial=["A", "B"])
    m = si.Model(odes={**LINEAR, "C": "k3*A - c*C"}, observables={"y": "s*B + C"}, totals={"T": "A + B"})
    with pytest.raises(InputError, match="C, which is in no conserved total"):
        si.detect(m, equilibrate=True, free_initial=["C"])


ACR = {"A": "-k1*A*B + k2*B", "B": "k1*A*B - k2*B"}


def test_a_state_at_zero_at_rest_takes_the_free_value_where_free_initial_names_it():
    # A + B -> 2B, B -> A: B rests at zero, or A = k2/k1 with B free
    m = si.Model(odes=ACR, observables={"y": "s*B"}, totals={"T": "A + B"},
                 events=[{"var": "B", "time": 0, "value": "dose", "method": "add"}], fixed=["dose"])
    zero = si.detect(m, equilibrate=True, reconstruct=True, cores=1)
    assert any("B rests at zero" in b and "free_initial=['B']" in b for b in zero.info["branches"])
    assert zero.info["settings"]["free_initial"] == ["A"]
    reduced = si.detect(m, equilibrate=True, reduce_cq=True, reconstruct=True, cores=1)
    assert summary(reduced) == summary(zero)
    away = si.detect(m, equilibrate=True, free_initial=["B"], cores=1)
    assert away.info["settings"]["free_initial"] == ["B"] and away.rank != zero.rank
    assert any("free_initial: B rests away from zero" in b for b in away.info["branches"])
    lost = si.Model(odes={"x": "-k*x", "y": "k*x"}, observables={"o": "s*y"}, totals={"T": "x + y"},
                    events=[{"var": "x", "time": 0, "value": "d", "method": "add"}], fixed=["d"])
    with pytest.raises(InputError, match="x, which rests at zero at every resting state"):
        si.detect(lost, equilibrate=True, free_initial=["x"])


def test_the_totals_of_odes_are_nonnegative_in_every_order():
    import itertools
    seen = set()
    for order in itertools.permutations(BINDING):
        m = si.Model(odes={k: BINDING[k] for k in order}, observables={"y": "s*C"})
        laws = set()
        for t in m.totals.values():
            assert "-" not in t
            laws.add(frozenset(re.findall(r"[A-Z]", t)))
        seen.add(frozenset(laws))
        rx = si.Model(reactions={"species": list(order), "stoichiometry": [
            [{"A": -1, "B": -1, "C": 1}[s] for s in order], [{"A": 1, "B": 1, "C": -1}[s] for s in order]],
            "rates": ["kon*A*B", "koff*C"]}, observables={"y": "s*C"})
        assert {frozenset(re.findall(r"[A-Z]", t)) for t in rx.totals.values()} == laws
    assert seen == {frozenset({frozenset("AC"), frozenset("BC")})}
    # one relation between four nonnegative laws: three of them, all nonnegative
    basis = canonical_laws([[1, 1, 0, 0], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 0, 1]], 4)
    assert len(basis) == 3 and all(min(v) >= 0 for v in basis)


def test_a_total_of_both_signs_is_noted_under_reduce_cq():
    m = si.Model(odes={"A": "k*(1+u)", "B": "k*(1+u)", "u": "0"}, observables={"y": "A + s*B"}, inputs=["u"],
                 events=U)
    assert m.totals["total1"] == "A - B"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = si.detect(m, reduce_cq=True, cores=1)
    assert any("total1 = A - B has coefficients of both signs" in x for x in r.info["limits"])


def test_the_exact_scalings_certify_the_order_under_reduce_cq_as_with_pivots():
    for kw in ({}, {"reduce_cq": True}):
        r = EXCHANGE.detect(reconstruct=False, **kw)
        assert r.info["lie_certified"] and r.info["order_bound"] == 0.0
