"""Sections of directions that translate linear and logarithmic coordinates at rates free of the moved coordinates."""

import itertools

import sympy as spy

from .. import state
from ..errors import SYMPY_FAILS, sympy_fail
from ..rexpr import sym_sort
from .expr import red_locals, red_sympify
from .sign import DOMAIN, positive_set, sgn

# pin sets of a block of several directions whose minor is tried
_PIN_SETS = 2000


def _model(e):
    return str(e).replace("**", "^")


def linear_flow(preps, fixed=(), prefer=None):
    """A global section of directions that move every coordinate at a rate free of the moved coordinates.

    A positive moved coordinate is taken in the logarithm, another one as is (docs, "Linear flows"). None
    where a component depends on a moved coordinate or no pin set has a minor of one sign. `prefer` holds
    ``observation_only`` (coordinates pinned first) and ``states`` (initial values, pinned last).
    """
    prefer = prefer or {}
    if not preps:
        return None
    moved = sym_sort(list(dict.fromkeys(c for pr in preps for c in pr["support"])))
    if set(moved) & set(fixed or ()) or len(moved) < len(preps):
        return None
    # a declared bound in a moved coordinate makes the domain no product of lines
    if any(set(red_locals([b])) & set(moved) for b in DOMAIN.get("bounds") or []):
        return None
    pos = positive_set()
    is_log = {c: pos is None or c in set(pos) for c in moved}
    msyms = {spy.Symbol(c) for c in moved}
    try:
        cols = []
        for pr in preps:
            den = pr.get("den") or "1"
            d = red_sympify(den, red_locals([den]))
            if not (d.is_number and d != 0) and sgn(d) == 0:
                return None
            loc = red_locals(list(pr["comps"].values()) + moved)
            col = {}
            for c in moved:
                e = red_sympify(pr["comps"].get(c, "0"), loc)
                if is_log[c]:
                    e = spy.cancel(e / spy.Symbol(c))
                if e.free_symbols & msyms or e.has(spy.exp, spy.log):
                    return None
                col[c] = e
            cols.append(col)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    r = len(preps)
    obs = set(prefer.get("observation_only") or ())
    states = set(prefer.get("states") or ())

    def size(c):
        return sum(int(spy.count_ops(col[c])) + len(col[c].free_symbols) for col in cols)
    # by structure: observation-only coordinates first, initial values last, then linear before logarithmic,
    # then the smaller components; the name breaks remaining ties only
    cand = [c for c in moved if any(col[c] != 0 for col in cols)]
    order = sorted(cand, key=lambda c: (c not in obs, c in states, is_log[c], size(c), c.encode()))
    skip = int(state.get("transversal_skip") or 0)
    found, tried = [], 0
    for P in itertools.combinations(order, r):
        tried += 1
        if tried > _PIN_SETS:
            break
        M = spy.Matrix([[cols[k][c] for k in range(r)] for c in P])
        try:
            det = spy.factor(spy.cancel(M.det(method="berkowitz")))
            ok = det != 0 and sgn(det) != 0
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            ok = False
        if ok:
            found.append((list(P), M, det))
            if len(found) > skip:
                break
    if not found:
        return None
    P, M, det = found[min(skip, len(found) - 1)]
    th = {c: spy.log(spy.Symbol(c, positive=True)) if is_log[c] else spy.Symbol(c) for c in moved}
    try:
        Minv = M.inv(method="LU")
        xP = spy.Matrix([th[c] for c in P])
        meaning, solved_domain = {}, {}
        for c in moved:
            if c in P:
                continue
            row = spy.Matrix([[cols[k][c] for k in range(r)]])
            a = (row * Minv).applyfunc(spy.cancel)
            inv = th[c] - (a * xP)[0, 0]
            if is_log[c]:
                inv = spy.powsimp(spy.Symbol(c, positive=True) * spy.exp(spy.expand(-(a * xP)[0, 0])))
            else:
                inv = spy.expand(inv)
            inv = inv.xreplace({s: spy.Symbol(str(s)) for s in inv.free_symbols})
            meaning[c] = _model(inv)
            solved_domain[c] = "positive" if is_log[c] else "real"
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    pins = {c: "1" if is_log[c] else "0" for c in P}
    coords = ", ".join([f"log({c})" if is_log[c] else c for c in moved])
    pin_txt = ", ".join(f"{c} = {v}" for c, v in pins.items())
    certs = [
        f"linear flow: in the coordinates {coords} every generator component depends only on coordinates "
        f"the block does not move, so the generators commute and every orbit is the translate x + Xi(u) s, "
        f"s in R^{r}, with every moved coordinate on the whole line",
        f"section {pin_txt}: the minor of the components on {', '.join(P)} is {_model(det)}, of one sign on "
        f"the domain, so x_P + Xi_P(u) s = 0 has exactly one solution s on every orbit",
        "section met at most once: two points of one orbit on the section have equal pinned coordinates, so "
        "equal s",
        "every level set of the invariants is one orbit: the invariants and u fix the point of the orbit on "
        "the section, and every orbit meets it",
    ]
    note = (f"{pin_txt} meets every orbit once; the other moved coordinates become the invariants listed "
            f"under the same names")
    return {"type": "curved", "support": moved, "stage": "linear flow", "status": "reduced",
            "target": len(moved) - r, "count": "exact", "invariants": list(meaning.values()),
            "certificates": certs, "reason": None, "module_combos": None, "transversal": P, "section": None,
            "pins": pins, "new_parameters": meaning, "solved_domain": solved_domain, "face": False,
            "injective": True, "reach": True, "section_monotone": False, "coverage": "total",
            "unproven_entries": [], "sign_conditions": [], "gauge_note": note, "preps": preps}
