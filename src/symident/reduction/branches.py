"""Branch charts of discrete symmetries: one trafo per image of the fundamental domain, chained with the section."""

import sympy as spy

from ..errors import SYMPY_FAILS, InputError, sympy_fail
from ..sym.domain import is_positive
from ..sym.parse import model_sympify


def _side_text(e):
    """`e > 0` as an inequality between its positive and its negative terms, e.g. "k01 > k12"."""
    pos, neg = [], []
    for t in spy.Add.make_args(spy.expand(e)):
        (neg if t.could_extract_minus_sign() else pos).append(t)
    if not pos or not neg:
        return f"{e} > 0"
    return f"{spy.Add(*pos)} > {-spy.Add(*neg)}".replace("**", "^")


def _fresh(name, taken):
    while name in taken:
        name += "_"
    taken.add(name)
    return name


def branch_charts(discrete, trafo, coords, positive):
    """{"branches": [...]} or {"why": ...}: for a discrete symmetry sigma whose fixed set is one hypersurface q = 0
    with q(sigma(theta)) = -c q(theta), c > 0, the two sides s q > 0, each the image of the other under sigma. On a
    side, delta = s q(theta) > 0 replaces a coordinate x in which q is linear, chosen where x(delta, theta') is
    positive, so that free values in the positive orthant map one to one onto the side; the chart is chained with
    the trafo of the section. `discrete` is Reduction.discrete, `trafo` the section (None: the identity)."""
    maps = [s for s in (discrete or {}).get("symmetries") or [] if s.get("level") != "open"]
    if not maps:
        return {"branches": []}
    if len(maps) > 1:
        return {
            "why": "branch charts cover one discrete symmetry; the fiber has "
            f"{len(maps) + 1} points with the output of a generic one"
        }
    sig = maps[0]
    dcoords = list(discrete.get("coordinates") or coords)
    syms = {c: spy.Symbol(c) for c in dcoords}
    try:
        smap = {syms.get(k, spy.Symbol(k)): model_sympify(str(v)) for k, v in sig["map"].items()}
        qs = [model_sympify(str(q)) for q in sig.get("fixed_set") or []]
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return {"why": "the map does not parse"}
    if len(qs) != 1:
        return {"why": "the fixed set of the map is not one hypersurface"}
    q = qs[0]
    flip = spy.cancel(-q.xreplace(smap) / q)
    if not is_positive(flip, positive):
        return {"why": f"the map does not exchange the sides of {q} = 0"}
    taken = set(dcoords) | set(trafo or {})
    for v in (trafo or {}).values():
        taken |= {str(s) for s in model_sympify(str(v)).free_symbols}
    delta = spy.Symbol(_fresh("delta", taken))
    pos_d = (True if positive is True else set(positive or ()) | {str(delta)}) if positive is not False else False
    out = []
    for s in (1, -1):
        cands = []
        for x in sorted(q.free_symbols & set(syms.values()), key=str):
            a = spy.diff(q, x)
            if a.has(x) or a == 0:
                continue
            sol = spy.solve(s * q - delta, x)
            if len(sol) != 1:
                continue
            xe = spy.cancel(sol[0])
            reach = is_positive(xe, pos_d)
            inj = is_positive(a, positive) or is_positive(-a, positive)
            # by structure: a certified positive value first, then the smaller expression
            cands.append(((not reach, not inj, spy.count_ops(xe)), x, xe, reach, inj))
        if not cands:
            return {"why": f"{q} is linear in no coordinate"}
        best = min(c[0] for c in cands)
        _, x, xe, reach, inj = next(c for c in cands if c[0] == best)
        base = dict(trafo) if trafo else {c: c for c in coords}
        sub = {x: xe}
        chart = {}
        for k, v in base.items():
            e = model_sympify(str(v)).xreplace(sub)
            chart[k] = str(spy.cancel(e)).replace("**", "^")
        if str(x) not in chart:
            chart[str(x)] = str(xe).replace("**", "^")
        unproven = []
        if not inj:
            unproven.append(f"injective: the coefficient {a} of {x} in {q} has one sign")
        if not reach:
            unproven.append(f"reach: {x} = {xe} is positive for positive {delta}")
        out.append(
            {
                "branch": _side_text(s * q),
                "trafo": chart,
                "new_parameters": {str(delta): str(s * q)},
                "solved": {str(x): str(xe).replace("**", "^")},
                "boundary": [f"{q} = 0"],
                "injective": bool(inj),
                "reach": bool(reach),
                "unproven": unproven,
                "map": sig["map"],
            }
        )
    return {"branches": out}


def pick_branch(branches, text):
    """The branch whose side is the inequality `text` ("a > b" or "a < b"), an InputError otherwise."""
    op = ">" if ">" in text else "<" if "<" in text else None
    if op is None:
        raise InputError(f"`branch` must be an inequality such as 'k1 > k2', not {text!r}")
    sgn = 1 if op == ">" else -1
    lhs, rhs = text.split(op, 1)
    try:
        e = sgn * (model_sympify(lhs.strip().lstrip("=")) - model_sympify(rhs.strip().lstrip("=")))
    except SYMPY_FAILS:
        raise InputError(f"`branch` {text!r} does not parse") from None
    for b in branches:
        side = model_sympify(b["new_parameters"][next(iter(b["new_parameters"]))])
        r = spy.cancel(e / side)
        if r.is_number and r > 0:
            return b
    names = ", ".join(repr(b["branch"]) for b in branches) or "none"
    raise InputError(f"`branch` {text!r} is no side of a discrete symmetry; the branches are {names}")
