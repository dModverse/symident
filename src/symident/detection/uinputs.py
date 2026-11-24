"""Unknown inputs in the count of discrete symmetries: jet chains with free starts, unknowns of the fiber that are no
coordinates, and the verified count with them (docs/method/conditions.rst, "Unknown and known inputs")."""

import math

import sympy as spy

from .. import session
from ..gfp import PRIMES
from ..rexpr import get_symbols
from ..sym.parse import model_sympify


def unknown_chain(f, g, ic, unknown, length, unsupported, what):
    """(f, ic, jets): the right-hand sides with each unknown input a chain of `length` jet states, their starts free
    (None), and per jet name the input and the order; ValueError for a setting the count does not handle."""
    from .core import _input_jets

    if unsupported:
        raise ValueError(f"{what} handles unknown inputs in one condition without events, switches and equilibrate")
    taken = set(get_symbols(list(f.values()) + list(g.values()))) | set(f)
    f, jets = _input_jets(f, unknown, length, taken)
    of = {x: [w, k] for w, ns in jets.items() for k, x in enumerate(ns)}
    return f, dict(ic, **{x: None for x in of}), of


def _chains(seg):
    """{input: {jet symbol: order}} of the chains of the unknown inputs of a segment."""
    out = {}
    for n, (w, k) in (seg.get("input_of") or {}).items():
        out.setdefault(w, {})[spy.Symbol(n)] = int(k)
    return out


def relative_degrees(series, cap):
    """Per observable row of the series {input: r}, r the least order whose Lie derivative of the observable reads a
    jet of the input, by symbolic derivatives over Q up to `cap`; an input not read up to `cap` is left out. The
    derivative of order r + m reads jets up to order m and the jet of order m with the coefficient of the input in
    the derivative of order r, which is not zero."""
    have = getattr(series, "_reldeg", None)
    if have is not None and have[0] >= cap:
        return have[1]
    out = []
    for part in series.raw_parts:
        seg = part["segments"][0]
        chains = _chains(seg)
        F = {spy.Symbol(k): model_sympify(str(v)) for k, v in seg["f"].items()}
        for h in seg["g"].values():
            e, r = spy.cancel(model_sympify(str(h))), {}
            for k in range(cap + 1):
                fs = e.free_symbols
                for w, c in chains.items():
                    if w not in r and fs & set(c):
                        r[w] = k
                if len(r) == len(chains) or k == cap:
                    break
                e = spy.cancel(sum((spy.diff(e, x) * F[x] for x in fs if x in F), spy.Integer(0)))
            out.append(r)
    series._reldeg = (cap, out)
    return out


def unread_jets(series, orders):
    """The (part, state) of the jets of unknown inputs above the orders that the output coefficients up to `orders`
    read: the coefficient of order n of an observable of relative degree r reads the jets up to n - r."""
    rd = relative_degrees(series, max(orders))
    top = {}
    for r, n in zip(rd, orders):
        for w, k in r.items():
            top[w] = max(top.get(w, -1), int(n) - k)
    out = set()
    for pi, part in enumerate(series.raw_parts):
        seg = part["segments"][0]
        idx = {x: i for i, x in enumerate(seg["f"])}
        for name, (w, k) in (seg.get("input_of") or {}).items():
            if int(k) > top.get(w, -1):
                out.add((pi, idx[name]))
    return out


def _rank_mod(rows, p):
    """The rank of the integer matrix `rows` modulo p."""
    a = [list(r) for r in rows]
    rank, ncol = 0, len(a[0]) if a else 0
    for c in range(ncol):
        piv = next((i for i in range(rank, len(a)) if a[i][c] % p), None)
        if piv is None:
            continue
        a[rank], a[piv] = a[piv], a[rank]
        inv = pow(a[rank][c], -1, p)
        a[rank] = [x * inv % p for x in a[rank]]
        for i in range(len(a)):
            if i != rank and a[i][c] % p:
                f = a[i][c]
                a[i] = [(x - f * y) % p for x, y in zip(a[i], a[rank])]
        rank += 1
    return rank


def isolated(fib, p):
    """(ok, jets): whether the output coefficients up to the orders of the fiber have a Jacobian in the coordinates and
    the input jets they read of full column rank at a random point modulo p, and the number of those jets. Then the
    points of the fiber are isolated in the coordinates and these jets (Step 1 of the verified count)."""
    from .discrete import _Dual

    series = fib.series
    zero = unread_jets(series, fib.orders)
    free = [key for key in series.inputs if key not in zero]
    n, q = len(series.cs), len(free)
    rng = session.current().rng("input jets", tuple(series.coords))
    m = n + q
    cv = [_Dual(rng.randint(2, p - 1), [int(i == j) for j in range(m)], p) for i in range(n)]
    old = series.input_values
    series.input_values = {key: _Dual(0, [0] * m, p) for key in zero}
    series.input_values.update(
        {key: _Dual(rng.randint(2, p - 1), [int(n + k == j) for j in range(m)], p) for k, key in enumerate(free)}
    )
    try:
        ys = series._outputs(cv, fib.order, 0, _Dual(1, [0] * m, p), _Dual(0, [0] * m, p))
    except ZeroDivisionError:
        ys = None
    finally:
        series.input_values = old
    if ys is None:
        return False, None
    rows = [list(y[k][0].d) for y, top in zip(ys, fib.orders) for k in range(top + 1)]
    return _rank_mod(rows, p) == m, q


def _degrees(fib):
    """The total degrees in (theta, input jets, u) of the fiber equations, as discrete._fiber_degrees, with the drawn
    jets of the inputs as coordinates of the point."""
    from ..sym.degrees import Atoms, Deg, common
    from .discrete import _sparse, _sparse_deg

    A = Atoms()
    series = fib.series
    old = series.input_values
    zero = unread_jets(series, fib.orders)
    series.input_values = {key: Deg.const(A, 0) if key in zero else Deg(A, 1) for key in series.inputs}
    try:
        outs = series._outputs([Deg(A, 1) for _ in series.cs], fib.order, 0, Deg.const(A, 1), Deg.const(A, 0))
    except ZeroDivisionError:
        outs = None
    finally:
        series.input_values = old
    if outs is None:
        return None
    vals = [[y[k][0] for k in range(n + 1)] for y, n in zip(outs, fib.orders)]
    dq, nums = common([v for row in vals for v in row])
    it = iter(nums)
    bnum = [[next(it) for _ in row] for row in vals]
    degs = [_sparse_deg(P) for P in fib.static] + [1 + sum(_sparse_deg(f) for f in fib.sat)]
    degs += [2 * max(_sparse_deg(_sparse(n)), _sparse_deg(_sparse(d))) for n, d in series.pins]
    for (E, M), b, n in zip(fib.outs, bnum, fib.orders):
        for k in range(n + 1):
            terms = [_sparse_deg(E[l_][0]) + b[k - l_] for l_ in range(k + 1) if E[l_][0] and b[k - l_] is not None]
            if M[k][0]:
                terms.append(_sparse_deg(M[k][0]) + dq)
            if terms:
                degs.append(max(terms))
    return degs


def count_certificate(fib, k, out):
    """discrete.count_certificate() for a model with unknown inputs: the jets of the inputs at the point drawn with
    the coordinates, the points isolated in the coordinates and the jets that the output reads, and the Bezout number
    of the whole system."""
    from .discrete import _MIN_BITS, _VERIFIED_AIM, _VERIFIED_BITS, q_count

    series = fib.series
    ok, nread = isolated(fib, PRIMES[0])
    if not ok:
        out["why"] = (
            "the output coefficients do not determine the coordinates and the jets of the unknown inputs "
            "they read at a random point modulo a prime"
        )
        return out
    degs = _degrees(fib)
    if degs is None:
        out["why"] = "the Taylor coefficients have a pole"
        return out
    D = math.prod(max(1, int(d)) for d in degs)
    nu = len(fib.allnames) + 1
    out["bezout"] = D
    deg = (nu + 2 * D) * D
    bits = max(_MIN_BITS, math.ceil(math.log2(deg) - math.log2(_VERIFIED_AIM)))
    if bits > _VERIFIED_BITS:
        out["why"] = f"the point would need coordinates of {bits} bits for a bound below {_VERIFIED_AIM:g}"
        return out
    out["bits"] = bits
    rng = session.current().rng("verified count", tuple(fib.coords))
    pt = {s: spy.Integer(rng.randint(1, 2**bits)) for s in fib.syms}
    old = series.input_values
    zero = unread_jets(series, fib.orders)
    series.input_values = {key: spy.Integer(0 if key in zero else rng.randint(1, 2**bits)) for key in series.inputs}
    try:
        k2 = q_count(fib, pt)
    finally:
        series.input_values = old
    out["count"] = k2
    out["input_jets"] = nread
    if not k2:
        out["why"] = f"no finite count over Q at the random integer point of {bits} bits"
        return out
    if k2 != k:
        out["why"] = f"the count over Q at a random integer point is {k2}, not {k}"
        return out
    out["bound"] = 2.0 ** (math.log2(deg) - bits)
    return out


def preserved(fib, sig, points=2, deeper=0):
    """discrete.output_preserved() for a model with unknown inputs: at random rational points theta, the equations of
    the fiber up to `deeper` orders higher, modulo a prime, with the coordinates at sigma(theta), have a solution in
    the jets of the states and the inputs."""
    from ..sym.modp import _modp_rational
    from .discrete import Fiber, _gb_count

    p = PRIMES[0]
    rng = session.current().rng("input check", tuple(fib.coords))
    deep = Fiber(fib.series, [n + deeper for n in fib.orders], fib.positive)
    for _ in range(points):
        pt = {s: spy.Rational(rng.randint(1, 97), rng.randint(1, 13)) for s in fib.syms}
        try:
            img = [_modp_rational(sig.get(s, s).xreplace(pt), p) for s in fib.syms]
            sysd = deep.system(pt, p, extra=[{((i, 1),): 1, (): (-c) % p} for i, c in enumerate(img)])
        except (ZeroDivisionError, TypeError):
            return False
        if sysd is None:
            return False
        for block in sysd["blocks"]:
            if not _gb_count(block["names"], p, block["eqs"], 60.0):
                return False
    return True


def input_state_map(parts, sig, states):
    """[map] of the states and the unknown inputs that proves sigma, or None: phi takes a state that starts at a
    coordinate named like the state to the image of that coordinate and keeps the others, and psi, the value of each
    input, is solved from F(phi(x), sigma(theta), psi) = Dphi(x) F(x, theta, w) and h(phi(x), sigma(theta), psi) =
    h(x, theta, w) where these are linear in psi; every identity holds over Q(x, theta, w), and phi maps the given
    initial values. (phi(x(t)), psi(t)) is then a solution at sigma(theta) with the output of the solution at theta."""
    from ..errors import SYMPY_FAILS, sympy_fail

    if len(parts) != 1 or len(parts[0]["segments"]) != 1:
        return None
    seg = parts[0]["segments"][0]
    of = seg.get("input_of") or {}
    W = [spy.Symbol(n) for n, (w, k) in of.items() if int(k) == 0]
    X = [spy.Symbol(x) for x in states if x in seg["f"]]
    try:
        F = {spy.Symbol(k): model_sympify(str(v)) for k, v in seg["f"].items()}
        H = [model_sympify(str(h)) for h in seg["g"].values()]
        ic = {spy.Symbol(k): model_sympify(str(v)) for k, v in seg["ic"].items() if v is not None}
        high = {spy.Symbol(n) for n, (w, k) in of.items() if int(k) > 0}
        if any(F[x].free_symbols & high for x in X) or any(h.free_symbols & high for h in H):
            # an input that enters through a derivative
            return None
        th = {s: e for s, e in sig.items() if s not in set(X)}
        phi = {
            x: sig.get(ic[x], ic[x]).xreplace({ic[y]: y for y in X if ic.get(y) == y}) if ic.get(x) == x else x
            for x in X
        }
        psi = [spy.Dummy(str(w)) for w in W]
        img = {**th, **phi, **dict(zip(W, psi))}
        eqs = [F[x].xreplace(img) - sum((spy.diff(phi[x], y) * F[y] for y in X), spy.Integer(0)) for x in X]
        eqs += [h.xreplace(img) - h for h in H]
        eqs = [spy.numer(spy.together(e)) for e in eqs]
        if any(e != 0 and spy.Poly(e, *psi).total_degree() > 1 for e in eqs):
            return None
        sol = spy.linsolve(eqs, psi)
        if not sol or len(sol.args) != 1 or sol.args[0].free_symbols & set(psi):
            return None
        sub = dict(zip(psi, sol.args[0]))
        if any(spy.cancel(e.xreplace(sub)) != 0 for e in eqs):
            return None
        for x in X:
            if (
                x in ic
                and ic[x] != x
                and spy.cancel(phi[x].xreplace({y: ic[y] for y in X if y in ic}) - ic[x].xreplace(th)) != 0
            ):
                return None
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    out = {str(x): str(e) for x, e in phi.items() if e != x}
    out.update({str(w): str(spy.cancel(sub[d])) for w, d in zip(W, psi) if spy.cancel(sub[d] - w) != 0})
    return [out]
