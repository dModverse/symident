"""The resting chart of discrete() under equilibrate: the resting value of each state inside a transcendental term is
a new coordinate, and one coordinate per such state is solved from the balances in exchange (docs/method/global.rst,
"Rational form of the fiber")."""

import random

import sympy as spy

from ..sym.parse import model_sympify
from .relax import _fresh, _non_rational, _str


def _atoms(e):
    return [a for a in spy.preorder_traversal(e) if _non_rational(a)]


def _nonzero_at(e, syms, rng):
    """Whether e is nonzero at a random rational point: then e is not the zero function."""
    pt = {s: spy.Rational(rng.randint(2, 10 ** 4), rng.randint(2, 10 ** 4)) for s in syms}
    try:
        v = spy.cancel(e.xreplace(pt))
    except ZeroDivisionError:
        return False
    return v.is_number and v != 0 and v.is_finite


def _solve_linear(eqs, unknowns, syms, rng):
    """The unique solution of equations linear in `unknowns` with a coefficient matrix that is invertible at a
    random point, else None."""
    try:
        A, b = spy.linear_eq_to_matrix(eqs, unknowns)
    except (spy.polys.polyerrors.PolynomialError, ValueError):
        return None
    if any(c.has(*unknowns) for c in list(A) + list(b)):
        return None
    if not _nonzero_at(A.det(method="berkowitz"), syms, rng):
        return None
    return dict(zip(unknowns, A.LUsolve(b)))


def _triangular(bal, unknowns):
    """{state: value} solving balances one at a time where a balance is linear in the only unknown left, or None."""
    sol, todo = {}, dict(bal)
    while todo:
        hit = None
        for x, e in todo.items():
            e = spy.together(e.xreplace(sol))
            left = [u for u in unknowns if u not in sol and e.has(u)]
            if len(left) == 1 and spy.Poly(spy.fraction(e)[0], left[0]).degree() == 1:
                hit = (x, left[0], spy.fraction(e)[0])
                break
        if hit is None:
            return None if set(unknowns) - set(sol) else sol
        x, u, num = hit
        P = spy.Poly(num, u)
        sol[u] = -P.coeff_monomial(1) / P.coeff_monomial(u)
        del todo[x]
    return sol if not set(unknowns) - set(sol) else None


def rest_chart(odes, observables, inputs, coords, positive, fixed=(), events=(), trafo=None):
    """{"odes", "observables", "trafo", "events", "exchanged", "rest", "positive", "coordinates"}: the model on the
    resting set in the chart where the resting value R_x of each state x inside a transcendental term is a
    coordinate and a coordinate p, entering a balance linearly, is the solution of the balances. A ValueError
    says why no chart is found."""
    rng = random.Random("rest chart")
    # the trafo of the parameters, a section of a reduction, is applied first
    tp = {spy.Symbol(k): model_sympify(str(v)) for k, v in (trafo or {}).items()}
    f = {spy.Symbol(x): model_sympify(str(v)).xreplace(tp) for x, v in odes.items()}
    g = {y: model_sympify(str(v)).xreplace(tp) for y, v in observables.items()}
    events = [{**ev, "value": _str(model_sympify(str(ev["value"])).xreplace(tp))} for ev in events or []]
    ins = {spy.Symbol(u) for u in inputs}
    cs = {spy.Symbol(c) for c in coords} - set(f)
    rhs = {x: e.xreplace({u: spy.Integer(0) for u in ins}) for x, e in f.items()}
    start = {x: spy.Integer(0) for x in f if x in ins}
    for x, e in rhs.items():
        if x not in start and (str(x) in coords or e == 0):
            start[x] = x
    U = [x for x in f if x not in start]
    H = sorted({s for x in U for a in _atoms(rhs[x]) for s in a.free_symbols & set(U)}, key=str)
    if not H:
        raise ValueError("no resting state lies inside a transcendental term")
    taken = {str(s) for e in list(f.values()) + list(g.values()) for s in e.free_symbols} | {str(x) for x in f}
    R = {x: _fresh(f"si_{x}_rest", taken) for x in H}
    held = {**start, **R}
    dummies = {}

    def flat(e):
        # each transcendental term at rest is one symbol, with the term at the resting values
        e = e.xreplace(held)
        for a in sorted(_atoms(e), key=lambda a: -spy.count_ops(a)):
            if a.free_symbols & set(U):
                raise ValueError(f"{_str(a)} reads a resting state that no transcendental term determines")
            d = dummies.setdefault(a, spy.Dummy("w"))
            e = e.xreplace({a: d})
        return e
    bal = {x: flat(spy.together(rhs[x])) for x in U}
    if any(_atoms(e) for e in bal.values()):
        raise ValueError("a transcendental term at rest is nested")
    rest = [x for x in U if x not in H]
    syms = set(cs) | set(R.values()) | set(dummies.values())
    sol = _triangular({x: bal[x] for x in rest}, rest)
    if sol is None:
        raise ValueError("the balances of " + ", ".join(map(str, rest)) + " are not solved one by one")
    hb = [spy.fraction(spy.together(bal[x].xreplace(sol)))[0] for x in H]
    pos = None if positive is True else {str(p) for p in (positive or ())}
    inside = {s for a in dummies for s in a.free_symbols}

    def candidates(e):
        return [p for p in sorted(e.free_symbols & cs, key=str)
                if p not in inside and str(p) not in fixed and spy.Poly(e, p).degree() == 1]
    nb = {p: sum(p in e.free_symbols for e in hb) for p in cs}
    exch = None
    for key in (lambda p: (pos is None or str(p) in pos, nb[p], str(p)),
                lambda p: (nb[p], pos is None or str(p) in pos, str(p))):
        pick = []
        for e in sorted(hb, key=lambda e: len(candidates(e))):
            c = [p for p in sorted(candidates(e), key=key) if p not in pick]
            if not c:
                break
            pick.append(c[0])
        if len(pick) == len(hb):
            exch = _solve_linear(hb, pick, syms, rng)
            if exch is not None:
                break
    if exch is None:
        raise ValueError("no coordinates enter the balances of " + ", ".join(map(str, H)) + " linearly with an "
                         "invertible coefficient matrix")
    back = {d: a.xreplace(held) for a, d in dummies.items()}
    exch = {p: v.xreplace(back) for p, v in exch.items()}
    sol = {x: v.xreplace(exch).xreplace(back) for x, v in sol.items()}

    def sub(e):
        return model_sympify(str(e)).xreplace(exch)
    trafo = {str(x): _str(v) for x, v in {**start, **R, **sol}.items()}
    trafo = {k: _str(sub(v)) for k, v in trafo.items()}
    new_pos = positive
    if pos is not None:
        new_pos = sorted((pos - {str(p) for p in exch}) | {str(R[x]) for x in H if str(x) in pos})
    evs = []
    for ev in events or []:
        if spy.Symbol(ev["var"]) in inside or spy.Symbol(ev["var"]) in H:
            raise ValueError(f"an event changes {ev['var']}, read by a transcendental term")
        evs.append({**ev, "value": _str(sub(ev["value"]))})
    return {"odes": {str(x): _str(sub(v)) for x, v in f.items()},
            "observables": {y: _str(sub(v)) for y, v in g.items()}, "trafo": trafo, "events": evs,
            "exchanged": {str(p): _str(v) for p, v in exch.items()}, "rest": {str(x): str(r) for x, r in R.items()},
            "positive": new_pos,
            "coordinates": [c for c in coords if spy.Symbol(c) not in exch] + [str(R[x]) for x in H]}
