"""Schwartz-Zippel bound on the chance that the kernel check at random points passes a wrong closed form."""

import functools

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


@functools.lru_cache(maxsize=256)
def _model_degrees(f_items, g_items, x0_items):
    f = {k: model_sympify(v) for k, v in f_items}
    g = [model_sympify(v) for _, v in g_items]
    x0 = [model_sympify(v) for _, v in x0_items]
    if not all(_rational(e) for e in list(f.values()) + g + x0):
        return None
    dens = [spy.fraction(spy.together(e))[1] for e in f.values()]
    syms = sorted(set().union(*[e.free_symbols for e in list(f.values()) + g + x0]) or {spy.Symbol("_")}, key=str)
    B = spy.Integer(1)
    for d in dens:
        B = spy.lcm(B, d)
    dB = int(spy.Poly(B, *syms).total_degree())
    dA = max([_degs(e)[0] + dB - _degs(e)[1] for e in f.values()] + [0])
    gd = [_degs(h) for h in g]
    d0 = max([1] + [max(_degs(e)) for e in x0])
    return dA, dB, gd, d0


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


def kernel_bound(f, g, x0, order, vectors, q, points, events=None, gap_order=0):
    """{"degree", "points", "prime", "probability"} for the closed forms `vectors`, or None where it does not apply."""
    if events:
        degs = [chain_degree(f, g, x0, events, order, gap_order, v) for v in vectors]
    else:
        degs = [identity_degree(f, g, x0, order, v) for v in vectors]
    if not degs or any(d is None for d in degs):
        return None
    d = max(degs)
    return {"degree": d, "points": points, "prime": int(q),
            "probability": float(len(vectors) * (min(1.0, d / q)) ** points)}
