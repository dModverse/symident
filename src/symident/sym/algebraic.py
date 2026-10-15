"""Polynomial relations of values given with radicals."""

import sympy as spy

from ..errors import SYMPY_FAILS, sympy_fail


def radical_relation(name, expr):
    """The irreducible factors in `name` of a polynomial vanishing at name = expr, radicals eliminated by resultants."""
    X = spy.Symbol(str(name))
    e = spy.sympify(expr)
    rads = sorted((a for a in e.atoms(spy.Pow) if a.exp.is_Rational and not a.exp.is_Integer),
                  key=spy.count_ops)
    if not rads:
        return None
    try:
        F = spy.numer(spy.together(X - e))
        rels = []
        for k in range(len(rads)):
            a = rads[k]
            r = spy.Symbol(f"_rad{k}")
            sub = {a: r ** int(a.exp.p)}
            F = F.xreplace(sub)
            rels.append((r, spy.numer(spy.together(r ** int(a.exp.q) - a.base))))
            rads = rads[:k + 1] + [w.xreplace(sub) for w in rads[k + 1:]]
        F = spy.numer(spy.together(F))
        for r, rel in reversed(rels):
            F = spy.resultant(F, rel, r)
        _, factors = spy.factor_list(spy.expand(F))
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    out = [f for f, _ in factors if X in f.free_symbols]
    return out or None
