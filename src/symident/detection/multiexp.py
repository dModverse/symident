"""The count of several experiments: K copies of a model with free starts share the parameters, each with other
starts, and the count of their fiber on the parameters and the starts of the first bounds the fiber of the model where
the model has no nonconstant rational first integral (docs/method/global.rst, "Several experiments")."""

import math
import re
import time

import sympy as spy

from ..gfp import PRIMES

# the most experiments the choice tries
_MAX_EXPERIMENTS = 8


def eligible(series):
    """(starts, why): per state with an unknown series the coordinate it starts from, or None and the reason where
    the route does not apply: one condition without events and switches, every state starting from a coordinate
    that nothing else reads, and no clock read."""
    if len(series.parts) != 1:
        return None, "the route takes a model with one condition"
    part = series.parts[0]
    if len(part["segs"]) != 1 or series.has_events:
        return None, "the route takes a model without events"
    seg = part["segs"][0]
    if seg.get("free") or seg.get("on") is not None:
        return None, "the route takes a model without switches"
    st = part["st"]
    ns, known = len(st), set(part["known"])
    params = list(series.params)
    starts = {}
    for i, x in enumerate(st):
        if i in known:
            continue
        v = seg["x0"][i][0]
        if v is None or not v.is_Symbol or v not in series.cs:
            return None, f"the start of {x} is fixed ({v}): the flow from a later start leaves the family of starts"
        if v in starts.values():
            return None, f"two states start from the coordinate {v}"
        starts[i] = v
    # a coordinate a start takes must enter the model only there
    used = set()
    for P in [P for N, D in list(seg["F"]) + list(seg["G"]) for P in (N, D)]:
        for m in P:
            used |= {params[j - ns] for j, e in enumerate(m) if e and j >= ns}
    for v in starts.values():
        if v in used:
            return None, f"the start coordinate {v} is also a parameter of the model"
    clocks = {i for i in known if all(not any(m) for P in seg["F"][i] for m in P) and any(seg["F"][i][0].values())}
    reads = {j for N, D in [seg["F"][i] for i in range(ns) if i not in known] + list(seg["G"])
             for P in (N, D) for m in P for j, e in enumerate(m[:ns]) if e}
    if reads & clocks:
        return None, "the model reads the clock"
    return starts, None


def conservation_laws(seg, states):
    """A basis of the linear forms sum_i c_i x_i in `states` that the right-hand sides of the segment keep constant,
    sum_i c_i f_i = 0 identically in the states and inputs, the coefficients rational in the other symbols; each
    as {state: coefficient}, in reduced echelon form (a pivot state with coefficient one per law)."""
    from sympy.polys.matrices import DomainMatrix

    from ..sym.parse import model_sympify
    xs = [spy.Symbol(x) for x in states]
    rhs = [model_sympify(str(seg["f"][x])) for x in states]
    variables = [spy.Symbol(x) for x in seg["f"]]
    others = sorted(set().union(*[e.free_symbols for e in rhs]) - set(variables), key=str)
    nums, dens = zip(*[spy.fraction(spy.together(e)) for e in rhs])
    common = spy.lcm_list(list(dens)) if dens else spy.Integer(1)
    cols = [spy.Poly(spy.expand(n * spy.cancel(common / d)), *variables) for n, d in zip(nums, dens)]
    monos = sorted({m for P in cols for m in P.monoms()})
    if not monos:
        return [{str(x): spy.Integer(1)} for x in xs]
    rows = [[P.coeff_monomial(mm) for P in cols] for mm in monos]
    dom = spy.QQ.frac_field(*others) if others else spy.QQ
    M = DomainMatrix.from_Matrix(spy.Matrix(rows)).convert_to(dom)
    null = M.nullspace()
    if null.shape[0] == 0:
        return []
    R = null.rref()[0].to_Matrix()
    out = []
    for r in range(R.rows):
        law = {str(x): spy.factor(R[r, j]) for j, x in enumerate(xs) if R[r, j] != 0}
        if law:
            out.append(law)
    return out


def _fresh(name, k, taken):
    """The name of `name` in experiment k, clear of `taken`, which it joins."""
    out = f"{name}_e{k}"
    while out in taken:
        out += "_"
    taken.add(out)
    return out


def stacked(parts, coords, series, starts, K, laws=()):
    """(parts, coords): K experiments of the single condition of `parts`, the first with the starts of the model, the
    others with new start coordinates, on the level set of the conservation `laws` of the first: the pivot state of
    each law starts where the law takes the value of the first experiment. The coordinates of the model come
    first."""
    from ..sym.parse import model_sympify
    seg = parts[0]["segments"][0]
    st = [str(x) for x in series.parts[0]["st"]]
    taken = set(coords) | set(seg["f"]) | {w for e in list(seg["f"].values()) + list(seg["g"].values())
                                          for w in re.findall(r"[A-Za-z_]\w*", str(e))}
    first = {st[i]: spy.Symbol(str(v)) for i, v in starts.items()}
    pivots = [next(iter(law)) for law in laws]
    out, extra = [], []
    for k in range(K):
        ic = dict(seg["ic"])
        if k:
            new = {}
            for i, v in starts.items():
                if st[i] not in pivots:
                    new[st[i]] = spy.Symbol(_fresh(str(v), k + 1, taken))
                    extra.append(str(new[st[i]]))
            for x, law in zip(pivots, laws):
                c = {y: model_sympify(str(e)) for y, e in law.items()}
                g = sum((c[y] * first[y] for y in c), spy.Integer(0))
                new[x] = g - sum((c[y] * new[y] for y in c if y != x), spy.Integer(0))
            ic.update({x: str(e).replace("**", "^") for x, e in new.items()})
        out.append({"segments": [dict(seg, ic=ic, label=f"e{k + 1}:")]})
    return out, list(coords) + extra


def plan(parts, coords, series, starts, ks=None, laws=()):
    """[(K, series, crit)] per number of experiments K in `ks` (by default 2 up to the parameters over the
    observables plus one, at most 8) whose stacked model has full rank: the orders of the rank criterion per
    observable of each experiment, the lowest highest order first and among equal ones the fewest experiments."""
    from .discrete import Series
    if ks is None:
        n_y = len(series.parts[0]["segs"][0]["G"])
        n_t = len(coords) - len(starts)
        ks = range(2, min(_MAX_EXPERIMENTS, max(2, math.ceil(n_t / max(1, n_y)) + 1)) + 1)
    out = []
    for K in ks:
        pk, ck = stacked(parts, coords, series, starts, K, laws)
        sk = Series(pk, ck)
        crit = sk.output_orders(2 * len(ck), PRIMES[0])
        if crit is not None:
            out.append((K, sk, crit))
    out.sort(key=lambda e: (max(e[2]), e[0]))
    return out


def count(parts, coords, series, positive, floor, experiments=None):
    """The multi-experiment count: {"why"} where the route does not apply, else {"experiments", "orders", "count",
    "first_integrals", "fiber", "crit", "seconds"}, with "failed" where msolve gives no count. The experiments share
    the level set of the conservation laws of the first; the count modulo a prime takes the orders from the rank
    criterion of the stacked model up, until it is stable or reaches `floor`, the orbit of the proven maps."""
    from . import discrete, firstint
    from .discrete import choose_order
    t0 = time.monotonic()
    starts, why = eligible(series)
    if why is not None:
        return {"why": why}
    st = [str(x) for x in series.parts[0]["st"]]
    # linear first integrals are declared by the laws: the experiments share their level set
    laws = conservation_laws(parts[0]["segments"][0], [st[i] for i in starts])
    fi = firstint.certificate(series, laws)
    if not fi["proven"]:
        return {"why": "the absence of rational first integrals is not proven: " + str(fi["why"]),
                "first_integrals": fi}
    given = isinstance(experiments, int) and not isinstance(experiments, bool) and experiments > 1
    plans = plan(parts, coords, series, starts, [experiments] if given else None, laws)
    if not plans:
        return {"why": "the stacked experiments do not reach full rank", "first_integrals": fi}
    K, sk, crit = plans[0]
    # a function proven in the field of the outputs of one experiment holds on the first of them
    pad = (0,) * (len(sk.cs) - len(series.cs))
    sk.pins = [tuple({mm + pad: c for mm, c in P.items()} for P in pin) for pin in series.pins]
    discrete._FAILURES.clear()
    fib, stable, cc = choose_order(sk, positive, 1, 2 * len(sk.coords), floor=floor, proj=len(coords))
    out = {"why": None, "experiments": K, "orders": list(fib.orders), "count": cc, "stable": stable,
           "first_integrals": fi, "fiber": fib, "crit": crit, "series": sk,
           "seconds": round(time.monotonic() - t0, 2)}
    if cc is None:
        out["failed"] = discrete.no_count_reason(fib.order)
    return out
