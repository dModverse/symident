"""Schwartz-Zippel bound on the chance that the kernel check at random points passes a wrong closed form."""

import functools
import math

import sympy as spy

from ..errors import SYMPY_FAILS, sympy_fail
from ..sym.parse import model_sympify


def _degs(e):
    """Total degrees of the numerator and the denominator of the rational expression e."""
    num, den = spy.fraction(spy.together(e))
    syms = sorted(e.free_symbols, key=str)
    if not syms:
        return 0, 0
    return int(spy.Poly(num, *syms).total_degree()), int(spy.Poly(den, *syms).total_degree())


def _rational(e):
    return all(not a.is_Function for a in spy.preorder_traversal(e)) and \
        all(a.exp.is_Integer for a in e.atoms(spy.Pow))


def _over_common_denominator(exprs, X):
    """The numerator and denominator degrees in the symbols X of each expression, and the degree of a common
    denominator, the product of the distinct denominators, which bounds that of their least common multiple."""
    parts, distinct = [], {}
    for e in exprs:
        num, den = spy.fraction(spy.together(e))
        parts.append((int(spy.Poly(num, *X).total_degree()), int(spy.Poly(den, *X).total_degree())))
        if not den.is_number:
            distinct.setdefault(spy.expand(den), den)
    return parts, sum(int(spy.Poly(d, *X).total_degree()) for d in distinct.values())


@functools.lru_cache(maxsize=256)
def _model_degrees(f_items, g_items, x0_items):
    f = {k: model_sympify(v) for k, v in f_items}
    g = [model_sympify(v) for _, v in g_items]
    x0 = [model_sympify(v) for _, v in x0_items]
    if not all(_rational(e) for e in list(f.values()) + g + x0):
        return None
    syms = sorted(set().union(*[e.free_symbols for e in list(f.values()) + g + x0]) or {spy.Symbol("_")}, key=str)
    parts, dB = _over_common_denominator(list(f.values()), syms)
    dA = max([dn + dB - dd for dn, dd in parts] + [0])
    gd = [_degs(h) for h in g]
    # the initial values enter the coordinates over their common denominator
    d0 = max(1, _vector_degree(x0)) if x0 else 1
    return dA, dB, gd, d0


def field_degrees(F, G, X):
    """(dA, dB, gd) as in _model_degrees of the field F and the functions G in the symbols X, None if not rational."""
    try:
        F, G = [spy.sympify(e) for e in F], [spy.sympify(e) for e in G]
        if not all(_rational(e) for e in F + G) or sum(spy.count_ops(e) for e in F + G) > _ROW_OPS:
            return None
        X = list(X) or [spy.Symbol("_")]
        parts, dB = _over_common_denominator(F, X)
        dA = max([dn + dB - dd for dn, dd in parts] + [0])
        return dA, dB, _over_common_denominator(G, X)[0]
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None


def no_degree(exprs):
    """Why `exprs` have no degree bound: the first term that is not rational, or their size; None otherwise."""
    from ..sym.parse import to_r
    try:
        es = [model_sympify(str(e)) for e in exprs]
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return "an expression SymPy does not read"
    for e in es:
        bad = next((a for a in spy.preorder_traversal(e)
                    if a.is_Function or (a.is_Pow and not a.exp.is_Integer)), None)
        if bad is not None:
            return f"{to_r(bad)} is not rational"
    if sum(spy.count_ops(e) for e in es) > _ROW_OPS:
        return f"the terms exceed {_ROW_OPS} operations"
    return None


def state_degrees(model_lines, obs_lines, states):
    """field_degrees() of the right-hand sides and observables in the states, every other symbol a constant."""
    try:
        f = [model_sympify(str(l).split("=", 1)[1]) for l in model_lines]
        g = [model_sympify(str(l).split("=", 1)[1]) for l in obs_lines]
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    return field_degrees(f, g, [spy.Symbol(str(x)) for x in states])


def total_degrees(model_lines, obs_lines):
    """field_degrees() of the right-hand sides and observables in all their symbols, None where not rational."""
    try:
        f = [model_sympify(str(l).split("=", 1)[1]) for l in model_lines]
        g = [model_sympify(str(l).split("=", 1)[1]) for l in obs_lines]
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    syms = sorted(set().union(*[e.free_symbols for e in f + g]), key=str)
    return field_degrees(f, g, syms)


def lie_row_degree(md, k):
    """Degree bound of the cleared rows of the Lie order k from (dA, dB, gd), as in identity_degree."""
    dA, dB, gd = md
    return max([a_h + k * max(dQ + dB + dA - 1, 0) + dQ + dB for a_h, dQ in gd] + [0])


def rows_degree(rows):
    """The largest degree of a row over the common denominator of the row, None where not rational or too large."""
    out = 0
    try:
        for r in rows:
            r = [spy.sympify(e) for e in r if spy.sympify(e) != 0]
            if not r:
                continue
            if sum(spy.count_ops(e) for e in r) > _ROW_OPS or not all(_rational(e) for e in r):
                return None
            out = max(out, _vector_degree(r))
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    return out


# operations of one row past which no degree is taken
_ROW_OPS = 20000


def identity_degree(f, g, x0, order, vector):
    """Degree bound of the cleared identity sum_j O_kj v_j up to Lie order `order`, None where not rational."""
    try:
        md = _model_degrees(tuple(sorted(f.items())), tuple(sorted(g.items())), tuple(sorted(x0.items())))
        if md is None:
            return None
        dA, dB, gd, d0 = md
        comps = [model_sympify(v) for v in vector.values()]
        if not all(_rational(e) for e in comps):
            return None
        W = spy.Integer(1)
        for e in comps:
            W = spy.lcm(W, spy.fraction(spy.together(e))[1])
        syms = sorted(set().union(*[e.free_symbols for e in comps]) or {spy.Symbol("_")}, key=str)
        dW = int(spy.Poly(W, *syms).total_degree())
        dV = max(_degs(e)[0] + dW - _degs(e)[1] for e in comps)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    worst = 0
    for a_h, dQ in gd:
        # each Lie derivative raises the numerator of P / (Q^e B^m) by dQ + dB + dA - 1
        deg_p = a_h + order * max(dQ + dB + dA - 1, 0)
        row = deg_p + dQ + dB
        worst = max(worst, row + dV)
    # initial values substituted into the coordinates
    return int(worst * d0)


def _vector_degree(exprs):
    """Max of the numerator degrees and the denominator degree of `exprs` over a common denominator."""
    exprs = list(exprs)
    syms = sorted(set().union(*[e.free_symbols for e in exprs]) or {spy.Symbol("_")}, key=str)
    W = spy.Integer(1)
    for e in exprs:
        W = spy.lcm(W, spy.fraction(spy.together(e))[1])
    dW = int(spy.Poly(W, *syms).total_degree())
    return max([dW] + [_degs(e)[0] + dW - _degs(e)[1] for e in exprs])


def balance_degrees(polys):
    """(prod d_i, sum (d_i - 1)) of the total degrees d_i of the polynomials `polys`, None where one is not a
    polynomial or too large."""
    bez, jac = 1, 0
    try:
        for P in polys:
            P = spy.sympify(P)
            if P == 0:
                continue
            if spy.count_ops(P) > _ROW_OPS or not _rational(P):
                return None
            syms = sorted(P.free_symbols, key=str)
            if not syms:
                return None
            d = int(spy.Poly(P, *syms).total_degree())
            bez, jac = bez * max(d, 1), jac + max(d - 1, 0)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    return bez, jac


def rest_identity_degree(identity, jac, bezout, other):
    """Degree of the bad parameter set of an identity of degree `identity` on resting varieties: the cleared identity
    times the Bezout numbers `bezout` of the resting systems it reads, plus the parameters `other` over which a solve
    leaves the resting component."""
    return int((identity + jac) * math.prod(bezout) + sum(other))


def chain_degree(f, g, x0, events, order, gap_order, vector, times=None):
    """Degree bound of the cleared identity of the chain rows with events after the start, None where not rational."""
    fs = list(f) if isinstance(f, (list, tuple)) else [f] * (len(events) + 1)
    try:
        mds = [_model_degrees(tuple(sorted(fj.items())), tuple(sorted(g.items())), tuple(sorted(x0.items())))
               for fj in fs]
        if any(md is None for md in mds):
            return None
        comps = [model_sympify(v) for v in vector.values()]
        x0e = [model_sympify(v) for v in x0.values()]
        evs = [[(str(e["var"]), model_sympify(str(e["value"])), str(e.get("method", "replace"))) for e in grp]
               for grp in events]
        tau = [model_sympify(str(t)) if t not in (None, "") else spy.Integer(0)
               for t in (list(times) if times else [None] * (len(evs) + 1))]
        if not all(_rational(e) for e in comps + x0e + tau + [v for grp in evs for _, v, _ in grp]):
            return None
        states = {spy.Symbol(x) for fj in fs for x in fj}
        if any(v.free_symbols & states for grp in evs for _, v, _ in grp):
            return None
        moving = [bool(t.free_symbols) for t in tau]
        # the derivative of each moving start time along v joins the vector
        dirs = [sum((spy.diff(t, spy.Symbol(k)) * c for k, c in zip(vector, comps)), spy.Integer(0))
                for t, mv in zip(tau, moving) if mv]
        dV = _vector_degree(comps + dirs)
        M = int(gap_order)
        Gs = [max(_vector_degree(x0e) if x0e else 1, 1)]
        for j, grp in enumerate(evs):
            dA, dB = mds[j][0], mds[j][1]
            if moving[j] or moving[j + 1]:
                # gap^m = a^m b^(M'-m) over b^M', with c_(M'+1) for the derivative in the gap
                Mg, dG = M + 1, _vector_degree([tau[j + 1] - tau[j]])
                T = max(max(1 + m * (dB + dA - 1), 0) + 2 * (Mg - m) * dB + Mg * dG for m in range(Mg + 1))
                DX = max(T, 2 * Mg * dB + Mg * dG, 1)
            else:
                # the state series: c_m over B^(2m), every term over B^(2M), gap^m adds m
                T = max(max(1 + m * (dB + dA - 1), 0) + 2 * (M - m) * dB + m for m in range(M + 1))
                DX = max(T, 2 * M * dB, 1)
            dv = _vector_degree([v for _, v, _ in grp]) if grp else 0
            Gs.append(DX * Gs[-1] + dv)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    Ms = M + 1 if any(moving) else M
    top = order + 1 if any(moving) else order
    worst = 0
    for j, Gj in enumerate(Gs):
        dA, dB, gd, _ = mds[min(j, len(mds) - 1)]
        for a_h, dQ in gd:
            for k in range(top + 1):
                Dh = a_h + (k + 1) * dQ + k * (dQ + 2 * dB + dA)
                worst = max(worst, 2 * (Ms + 1) * Dh * Gj + dV)
    return int(worst)


def drawn(points):
    """The most points a kernel check of the running session drew for the `points` that passed."""
    from .. import state
    ses = state.get("session")
    return max([int(points)] + list((ses.notes["draws"] if ses is not None else None) or []))


def kernel_bound(f, g, x0, order, vectors, q, points, events=None, gap_order=0, draws=None):
    """{"degree", "points", "draws", "prime", "probability"} for the closed forms `vectors`, None if inapplicable."""
    if events:
        degs = [chain_degree(f, g, x0, events, order, gap_order, v) for v in vectors]
    else:
        degs = [identity_degree(f, g, x0, order, v) for v in vectors]
    if not degs or any(d is None for d in degs):
        return None
    d = max(degs)
    draws = max(int(draws or points), points)
    # the passing points are `points` of the draws: a union over the subsets, each a uniform sample
    return {"degree": d, "points": points, "draws": draws, "prime": int(q),
            "probability": float(min(1.0, len(vectors) * math.comb(draws, points) * (min(1.0, d / q)) ** points))}
