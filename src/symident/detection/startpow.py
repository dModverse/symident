"""Free powers x^alpha of a state that starts at 0 in discrete(): the series in t and the layers S^m of
docs/method/global.rst, "Rational form of the fiber", with the normalized power N = (x / c_x)^alpha as a state
and the leading coefficient c_x of x a jet."""

import random
from fractions import Fraction

import sympy as spy

from ..sym.domain import is_positive

# the highest order at which the leading coefficient of the base is searched
_ORDER_CAP = 12


def power_exponent(e, cs):
    """(c0, c, alpha) of an exponent c0 + c*alpha with integers c0 >= 0, c > 0 and a coordinate alpha, else None."""
    syms = e.free_symbols
    if len(syms) != 1 or not syms <= set(cs):
        return None
    a = next(iter(syms))
    try:
        P = spy.Poly(e, a)
    except spy.PolynomialError:
        return None
    if P.degree() != 1:
        return None
    c, c0 = P.coeff_monomial(a), P.coeff_monomial(1)
    if not (c.is_Integer and c > 0 and c0.is_Integer and c0 >= 0):
        return None
    return int(c0), int(c), a


def start_powers(exprs, zero_states, cs):
    """{(x, alpha)} of the powers x^(c0 + c*alpha) in `exprs` with x among `zero_states`."""
    out = set()
    for e in exprs:
        for pw in spy.sympify(e).atoms(spy.Pow):
            if pw.base in zero_states:
                pe = power_exponent(pw.exp, cs)
                if pe is not None:
                    out.add((pw.base, pe[2]))
    return out


def substitute(e, x, alpha, N, Z=1):
    """`e` with each x^(c0 + c*alpha) as x^c0 * (Z*N)^c."""
    def repl(pw):
        if pw.base != x:
            return pw
        pe = power_exponent(pw.exp, {alpha})
        if pe is None or pe[2] != alpha:
            return pw
        return x ** pe[0] * (Z * N) ** pe[1]
    return spy.sympify(e).replace(lambda a: a.is_Pow and a.base == x, repl)


def leading(f, ic, x, alpha):
    """(k, c): the order k of the first Taylor coefficient of the state x that is not zero, and that coefficient
    c, a rational function of the other symbols, in the model where every power of x with the exponent alpha is 0;
    a ValueError where k passes the cap. `f` and `ic` map state names to expressions; the start of x is 0."""
    from .discrete import Series
    N = spy.Dummy("N")
    f0 = {s: substitute(spy.sympify(v), x, alpha, N).xreplace({N: 0}) for s, v in f.items()}
    syms = sorted({s for v in list(f0.values()) + [spy.sympify(v) for v in ic.values()] for s in v.free_symbols}
                  - {spy.Symbol(s) for s in f}, key=str)
    part = {"segments": [{"f": {s: str(v) for s, v in f0.items()}, "g": {}, "t": "0",
                          "ic": {s: str(v) for s, v in ic.items()}}]}
    ser = Series([part], [str(s) for s in syms])
    ib = [str(s) for s in ser.parts[0]["st"]].index(str(x))
    rng = random.Random("leading coefficient")
    pt = [Fraction(rng.randint(2, 10 ** 6), rng.randint(2, 10 ** 6)) for _ in syms]
    if ser._outputs(pt, _ORDER_CAP, 0, Fraction(1), Fraction(0)) is None:
        raise ValueError(f"the model has a pole at the start of {x}")
    k = next((n for n, v in enumerate(ser.states0[ib]) if v != 0), None)
    if k is None:
        raise ValueError(f"{x} has no Taylor coefficient other than 0 up to order {_ORDER_CAP}")
    # nonzero at a point is nonzero as a function; the coefficients below k vanish identically
    F, *gens = spy.field(",".join(str(s) for s in syms), spy.QQ) if syms else (None,)
    one = F.one if F is not None else Fraction(1)
    ser._outputs(list(gens), k, 0, one, one * 0)
    if any(v != 0 for v in ser.states0[ib][:k]):
        raise ValueError(f"the Taylor coefficients of {x} below order {k} do not vanish identically")
    c = ser.states0[ib][k]
    return k, spy.sympify(c.as_expr() if hasattr(c, "as_expr") else spy.Rational(c.numerator, c.denominator))


def definite(e, positive):
    """Whether e, a rational function, has one sign on the domain: numerator and denominator positive or negative
    by the declared signs."""
    num, den = spy.fraction(spy.cancel(spy.together(e)))
    return all(is_positive(spy.expand(t), positive) or is_positive(spy.expand(-t), positive) for t in (num, den))


def normalize(parts, cs, states):
    """`parts` (sympy entries, as in rationalize) with each free power of a state that starts at 0 written through
    the normalized power N = (x/c_x)^alpha, a state of the part named in part["spow"]; a power that does not enter
    as c_x^(-alpha) x^alpha with the leading coefficient c_x of x is left in place."""
    taken = {str(s) for part in parts for seg in part["segments"] for d in ("f", "g") for v in seg[d].values()
             for s in spy.sympify(v).free_symbols} | {str(s) for s in states}
    for part in parts:
        segs = part["segments"]
        seg = segs[0]
        ic = seg.get("ic")
        if len(segs) > 1 or ic is None or any(v is None for v in ic.values()):
            continue
        zero = {spy.Symbol(x) for x, v in ic.items() if spy.sympify(v) == 0}
        found = start_powers(list(seg["f"].values()) + list(seg["g"].values()), zero, cs)
        if len(found) != 1:
            continue
        ((x, alpha),) = found
        try:
            k, c = leading(seg["f"], ic, x, alpha)
        except ValueError:
            continue
        N, Z = spy.Symbol(_fresh("si_N", taken)), spy.Dummy("Z")
        cZ = {}

        def unit(e):
            # c_x^(w*alpha) is Z^w
            def repl(pw):
                q = spy.cancel(pw.exp / alpha)
                if not q.is_Integer:
                    return pw
                if spy.cancel(pw.base - c) == 0:
                    return Z ** q
                if spy.cancel(pw.base * c - 1) == 0:
                    return Z ** (-q)
                return pw
            return e.replace(lambda a: a.is_Pow and a.exp.has(alpha) and not a.base.free_symbols & {x}, repl)
        new = {}
        for d in ("f", "g"):
            for key, v in seg[d].items():
                e = unit(substitute(v, x, alpha, N, Z))
                if c == 1:
                    e = e.xreplace({Z: spy.Integer(1)})
                if e.has(Z):
                    e = spy.cancel(e)
                cZ[(d, key)] = e.has(Z)
                new[(d, key)] = e
        if any(cZ.values()):
            continue
        for (d, key), e in new.items():
            seg[d][key] = e
        seg["f"][str(N)] = spy.Integer(0)
        ic[str(N)] = spy.Integer(0)
        part["spow"] = {"N": str(N), "state": str(x), "alpha": str(alpha), "k": k, "leading": str(c)}
    return parts


def _fresh(stem, taken):
    k = 1
    while f"{stem}{k}" in taken:
        k += 1
    taken.add(f"{stem}{k}")
    return f"{stem}{k}"


def _reads(polys, ns):
    """The states that the polynomials `polys` read."""
    return {j for P in polys for m in P for j in range(ns) if m[j]}


def lam(n, m, k, alpha, one):
    """lambda_{n,m} = n - k*m + k*m*alpha: the derivative of the term of index n and layer m."""
    return (n - k * m) * one + (k * m) * alpha


def valuations(F, G, ns, iN, known):
    """(v, vy): per state and per observable a lower bound on n - k*m over the terms of layer m >= 1, None where
    no such term exists: N has 0, a state one more than the least it reads, an observable the least it reads."""
    rd = [_reads(F[i], ns) for i in range(ns)]
    v = [None] * ns
    v[iN] = 0
    changed = True
    while changed:
        changed = False
        for i in range(ns):
            if i == iN or i in known:
                continue
            lo = [v[j] for j in rd[i] if v[j] is not None]
            if lo and (v[i] is None or min(lo) + 1 < v[i]):
                v[i] = min(lo) + 1
                changed = True
    vy = []
    for M, E in G:
        lo = [v[j] for j in _reads((M, E), ns) if v[j] is not None]
        vy.append(min(lo) if lo else None)
    return v, vy


def needs(F, G, ns, sp, orders, known):
    """The highest index of each state that the outputs up to `orders` read: an observable reads the states at
    the order of the observable, the equation of index n of a state reads the right-hand side at index n - 1, and
    N reads the base at the same index."""
    need = [0] * ns
    for (M, E), n in zip(G, orders):
        for j in _reads((M, E), ns):
            need[j] = max(need[j], n)
    changed = True
    while changed:
        changed = False
        for i in range(ns):
            src = {sp["base"]} if i == sp["N"] else set() if i in known else _reads(F[i], ns)
            lo = need[i] if i == sp["N"] else need[i] - 1
            for j in src:
                if need[j] < lo:
                    need[j], changed = lo, True
    return need


def layered_flow(F, xs, params, order, o, z, skip, sp, alpha):
    """The series of the states of a part with a normalized power, xs[i][n] the layers of index n, from
    D(x) x' = N(x) with x' of index n and layer m lambda_{n+1,m} times the term of index n + 1, and N = S Q with
    n c Q_n = sum_j (alpha j + j - n) x_{k+j} Q_{n-j}; False at a pole."""
    from .discrete import _Eps, _lead, _ser_poly
    one, zero = o.c[0], z.c[0]
    L, k, iN, ib = len(o.c), sp["k"], sp["N"], sp["base"]
    dx = {i: [] for i in range(len(F)) if i not in skip and i != iN}
    Q = [o]
    c = None
    for n in range(order + 1):
        if n == k:
            c = xs[ib][k].c[0]
            if c == 0:
                return False
        if n >= k:
            j = n - k
            while len(Q) <= j:
                q = len(Q)
                acc = sum((xs[ib][k + i] * (alpha * i + (i - q) * one) * Q[q - i] for i in range(1, q + 1)), z)
                Q.append(acc / (q * c))
            xs[iN][n] = _Eps([zero] + Q[j].c[:-1])
        else:
            xs[iN][n] = z
        if n == order:
            break
        for i in dx:
            N, D = F[i]
            nser = _ser_poly(N, xs, params, n, o, z)
            dser = _ser_poly(D, xs, params, n, o, z)
            if _lead(dser[0]) == 0:
                return False
            rest = sum((dser[l_] * dx[i][n - l_] for l_ in range(1, n + 1)), z)
            d = (nser[n] - rest) / dser[0]
            dx[i].append(d)
            # a term of layer m has index at least k*m
            xs[i][n + 1] = _Eps([d.c[m] / lam(n + 1, m, k, alpha, one) if n + 1 >= k * m else zero
                                 for m in range(L)])
    return True


def layered_equations(F, xs, params, need, o, z, skip, sp, alpha, Q, add_eq, add_sat, limit):
    """The equations of the prolonged model of a part with a normalized power up to the index `need` of each state:
    D(x) x' = N(x) per index and layer, and n c Q_n = sum_j (alpha j + j - n) x_{k+j} Q_{n-j}, with c the leading
    coefficient of the base saturated."""
    from .discrete import _Eps, _ser_poly
    one = o.c[0]
    L, k, iN, ib = len(o.c), sp["k"], sp["N"], sp["base"]
    c = xs[ib][k].c[0]
    add_sat(c)
    for q in range(1, len(Q)):
        acc = sum((xs[ib][k + i] * (alpha * i + (i - q) * one) * Q[q - i] for i in range(1, q + 1)), z)
        add_eq(Q[q] * (q * c) - acc)
    for n in range(len(xs[iN])):
        xs[iN][n] = _Eps([z.c[0]] + Q[n - k].c[:-1]) if k <= n < k + len(Q) else z
    for i, (N, D) in enumerate(F):
        if i in skip or i == iN:
            continue
        top = need[i]
        nser = _ser_poly(N, xs, params, top - 1, o, z, limit) if top else []
        dser = _ser_poly(D, xs, params, top - 1, o, z, limit) if top else []
        dx = [_Eps([lam(n + 1, m, k, alpha, one) * xs[i][n + 1].c[m] for m in range(L)]) for n in range(top)]
        for n in range(top):
            add_eq(sum((dser[l_] * dx[n - l_] for l_ in range(0, n + 1)), z) - nser[n])
        add_sat(_ser_poly(D, xs, params, 0, o, z, limit)[0])


def proof(series, pi, positive):
    """{"premises", "k", "row", ...}: the conditions under which every point of the fiber in the domain has the
    exponent alpha of the point (docs/method/global.rst, "Rational form of the fiber"), checked exactly:
    the leading coefficient c_x and the lowest coefficient of layer 1 of one observable have one sign on the
    domain; else they are named as premises."""
    from .discrete import _Eps, _eval, _ser_poly
    part = series.parts[pi]
    sp = part["spow"]
    seg = part["segs"][0]
    k, ib = sp["k"], sp["base"]
    premises = []
    out = {"premises": premises}
    names = [str(s) for s in series.cs]
    F, *gens = spy.field(",".join(names), spy.QQ)
    val = dict(zip(series.cs, gens))
    alpha = val[series.cs[sp["alpha_index"]]]
    params = [val[s] for s in series.params]
    rows = [r for r, v in enumerate(sp["vy"]) if v is not None]
    if not rows:
        raise ValueError(f"no observable reads the power of {sp['state']}")
    r = min(rows, key=lambda r: sp["vy"][r])
    n1 = sp["vy"][r] + k
    order = max(n1, k)
    one, zero = F.one, F.zero
    o, z = _Eps([one, zero]), _Eps([zero, zero])
    kser = series._known_series(pi, order)
    xs = []
    for i, (_, num, den) in enumerate(seg["x0"]):
        if i in kser:
            xs.append([o * c for c in kser[i]])
            continue
        x0 = _eval(num, gens, zero) / _eval(den, gens, zero)
        xs.append([o * x0] + [z] * order)
    if not layered_flow(seg["F"], xs, params, order, o, z, set(kser), sp, alpha):
        raise ValueError("the series of the model has a pole at the start")
    c = xs[ib][k].c[0]
    M, E = seg["G"][r]
    mser = _ser_poly(M, xs, params, n1, o, z)
    eser = _ser_poly(E, xs, params, n1, o, z)
    y = []
    for n in range(n1 + 1):
        y.append((mser[n] - sum((eser[l_] * y[n - l_] for l_ in range(1, n + 1)), z)) / eser[0])
    d = y[n1].c[1]
    cx, dx = spy.sympify(c.as_expr()), spy.sympify(d.as_expr())
    out.update({"leading": str(cx), "observable": series.labels[_row(series, pi, r)], "index": n1,
                "coefficient": str(dx)})
    if d == 0:
        raise ValueError(f"the lowest coefficient of layer 1 of {out['observable']} vanishes")
    if not definite(cx, positive):
        premises.append(f"the leading coefficient {cx} of {sp['state']} is not zero on the domain")
    if not definite(dx, positive):
        premises.append(f"the coefficient {dx} of t^({n1 - k} + {k}*{sp['alpha']}) in {out['observable']} is not "
                        "zero on the domain")
    return out


def _row(series, pi, r):
    """The index among the labels of the series of observable r of the first segment of part pi."""
    i = 0
    for pj, part in enumerate(series.parts):
        for seg in part["segs"]:
            if pj == pi and seg is part["segs"][0]:
                return i + r
            i += len(seg["labels"])
    return r
