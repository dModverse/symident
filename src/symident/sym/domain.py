"""The declared domain: coordinates and expressions declared positive."""

import re

import sympy as spy

_NAME = re.compile(r"[A-Za-z_.][A-Za-z0-9_.]*$")


def declared(positive):
    """(names, expressions): the names declared positive (None for all) and the declared expressions h > 0."""
    if positive is True:
        return None, []
    if positive in (False, None):
        return set(), []
    names, exprs = set(), []
    for p in positive:
        p = str(p).strip()
        if _NAME.match(p):
            names.add(p)
        else:
            from .parse import model_sympify
            exprs.append(model_sympify(p))
    return names, exprs


def is_positive(e, positive, real=True):
    """Whether e > 0 follows from the declared names positive, others real if `real`, declared expressions positive."""
    names, exprs = declared(positive)
    e = spy.sympify(e)
    for h in exprs:
        s = spy.Dummy("h", positive=True)
        for v in sorted(h.free_symbols & e.free_symbols, key=str):
            c = spy.diff(h, v)
            if c.free_symbols or c == 0:
                continue
            e = e.xreplace({v: (s - (h - c * v)) / c})
            break
    sub = {}
    for x in e.free_symbols:
        if isinstance(x, spy.Dummy):
            continue
        if names is None or str(x) in names:
            sub[x] = spy.Symbol(str(x), positive=True)
        elif real:
            sub[x] = spy.Symbol(str(x), real=True)
    return bool(e.xreplace(sub).is_positive)
