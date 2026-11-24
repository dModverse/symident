"""The exact fiber of discrete(): L = Q(c)(alpha) with prod_g (Z - g(alpha)) over Q(c), Q(c) the field of the output
coefficients, proven by identities of rational functions (docs/method/global.rst, "The exact count")."""

import itertools
import math
import pickle
import random
import re
import sys
import time
from fractions import Fraction

import numpy as np
import sympy as spy

from .. import _core, session
from ..gfp import all_primes, rref_modp
from . import ioproj, work

# the largest number of unknown coefficients of a rational expression in the output coefficients
_UNKNOWNS = 1000
# the terms of a function evaluated in Python, more in C++
_SMALL_TERMS = 24
# the largest number of monomials of an input-output relation, and of unknowns of a rational function interpolated
_RELATION = 300
_INTERP = 3600
# the monomials of the state relations tried before the input-output relations
_SMALL = 40
# the share of the cap of the search left where the model has no polynomial input-output relations
_WITHOUT_RELATIONS = 1 / 15
# the monomials of the dense relations tried before the projections, and the share of the cap of all relations
_FIRST = 120
_SHARE = 0.75
# the largest number of proven functions in a small system
_SUBSYSTEM = 8
# the share of the cap of the small dense relations of a state or of the observables
_DENSE = 0.1
# the steps along a chain of known states that the relations keep
_CHAIN = 8
# the largest group whose generators are expanded
_GROUP = 8

# the _Var of each target function, and the _Relations of each series, built once
_TARGETS: dict = {}
_REL: dict = {}


class _Factored:
    """Quotients (N, {k: e}) = N / prod_k F_k^e of polynomials in the coordinates over a shared list F of monic
    factors, without gcds: a sum takes the largest power of each factor, so that the denominators grow linearly
    along a recursion."""

    def __init__(self, R, cs, params):
        self.R, self.fac, self.at, self._pw = R, [], {}, {}
        self.gens = R.gens
        self.pidx = [cs.index(s_) for s_ in params]

    def const(self, c):
        c = Fraction(c)
        return (self.R.ground_new(self.R.domain.convert(c.numerator) / self.R.domain.convert(c.denominator)), {})

    def zero(self):
        return (self.R.zero, {})

    def _poly(self, terms, idx):
        out = self.R.zero
        for m, c in terms:
            t = self.const(c)[0]
            for j, e in zip(idx, m):
                if e:
                    t = t * self.gens[j] ** e
            out = out + t
        return out

    def coef(self, terms):
        """A polynomial in the parameters, [(exponents, Fraction)]."""
        return (self._poly(terms, self.pidx), {})

    def start(self, num, den):
        """num / den for polynomials {exponents: Fraction} in the coordinates; None where den is zero."""
        idx = range(len(self.gens))
        d = self._poly(den.items(), idx)
        return None if not d else self.quo(self._poly(num.items(), idx), d)

    def factor(self, P):
        P = P.monic()
        if P not in self.at:
            self.at[P] = len(self.fac)
            self.fac.append(P)
        return self.at[P]

    def power(self, k, e):
        if (k, e) not in self._pw:
            self._pw[(k, e)] = self.fac[k] ** e
        return self._pw[(k, e)]

    def quo(self, n, d):
        """n / d for polynomials, d nonzero."""
        c = self.R.domain.quo(self.R.domain.one, d.LC)
        return (n * c, {}) if d.is_ground else (n * c, {self.factor(d): 1})

    def mul(self, a, b):
        if not a[0] or not b[0]:
            return self.zero()
        e = dict(a[1])
        for k, x in b[1].items():
            e[k] = e.get(k, 0) + x
        return (a[0] * b[0], e)

    def scale(self, a, c):
        return (a[0] * self.const(c)[0], a[1])

    def total(self, terms):
        """The sum of the quotients over the largest power of each factor."""
        terms = [t for t in terms if t[0]]
        T = {}
        for _, e in terms:
            for k, x in e.items():
                T[k] = max(T.get(k, 0), x)
        out = self.R.zero
        for n, e in terms:
            for k, x in T.items():
                if x - e.get(k, 0):
                    n = n * self.power(k, x - e.get(k, 0))
            out = out + n
        return (out, T) if out else self.zero()

    def pole(self, b):
        return not b[0]

    def div(self, a, b):
        """a / b for b with a nonzero numerator: the numerator of b joins the factors."""
        num, ea = self.quo(a[0], b[0])
        for k, x in a[1].items():
            ea[k] = ea.get(k, 0) + x
        for k, x in b[1].items():
            if ea.get(k, 0) >= x:
                ea[k] -= x
            else:
                num = num * self.power(k, x - ea.get(k, 0))
                ea[k] = 0
        return (num, {k: x for k, x in ea.items() if x})


class _Modular:
    """The arithmetic of _Taylor modulo a prime p below 2^31 at many points of the coordinates at once, as int64
    arrays; `bad` marks the points at a pole."""

    def __init__(self, pts, p, cs, params):
        self.X, self.p = np.asarray(pts, dtype=np.int64) % p, p
        self.n = len(self.X)
        self.pidx = [cs.index(s_) for s_ in params]
        self.bad = np.zeros(self.n, dtype=bool)
        self._pw = {}

    def const(self, c):
        c = Fraction(c)
        return np.full(self.n, c.numerator % self.p * pow(c.denominator, -1, self.p) % self.p, dtype=np.int64)

    def zero(self):
        return np.zeros(self.n, dtype=np.int64)

    def _xpow(self, j, e):
        if (j, e) not in self._pw:
            self._pw[(j, e)] = ioproj._pow(self.X[:, j], e, self.p)
        return self._pw[(j, e)]

    def _poly(self, terms, idx):
        out = self.zero()
        for m, c in terms:
            t = self.const(c)
            for j, e in zip(idx, m):
                if e:
                    t = t * self._xpow(j, e) % self.p
            out = (out + t) % self.p
        return out

    def coef(self, terms):
        return self._poly(terms, self.pidx)

    def start(self, num, den):
        idx = range(self.X.shape[1])
        d = self._poly(den.items(), idx)
        self.bad |= d == 0
        d[d == 0] = 1
        return self._poly(num.items(), idx) * ioproj._inv(d, self.p) % self.p

    def mul(self, a, b):
        return a * b % self.p

    def scale(self, a, c):
        return a * self.const(c) % self.p

    def total(self, terms):
        out = self.zero()
        for t in terms:
            out = (out + t) % self.p
        return out

    def pole(self, b):
        self.bad |= b == 0
        return bool(self.bad.all())

    def div(self, a, b):
        b = b.copy()
        b[b == 0] = 1
        return a * ioproj._inv(b, self.p) % self.p


class _Taylor:
    """The Taylor coefficients of the observables at the start of the first segment of a condition, in the
    arithmetic `ar` (_Factored or _Modular), by the recursion D(x) x' = N(x) order by order, each monomial in the
    states from one of lower degree: no symbolic derivative and no gcd."""

    def __init__(self, series, ar, pi, order):
        self.ok, self.ar = False, ar
        part = series.parts[pi]
        seg = part["segs"][0]
        ns = len(part["st"])
        kser = series._known_series(pi, order)

        def split(P):
            # {state exponents: coefficient, a polynomial in the parameters}
            out = {}
            for m, c in P.items():
                out.setdefault(tuple(m[:ns]), []).append((m[ns:], c))
            return {k: ar.coef(v) for k, v in out.items()}

        polys = [split(P) for N, D in seg["F"] for P in (N, D)] + [split(P) for M, E in seg["G"] for P in (M, E)]
        mons, at, parent = [(0,) * ns], {(0,) * ns: 0}, [None]

        def add(m):
            if m in at:
                return
            v = next(i for i, e in enumerate(m) if e)
            low = m[:v] + (m[v] - 1,) + m[v + 1 :]
            add(low)
            at[m] = len(mons)
            mons.append(m)
            parent.append((at[low], v))

        for P in polys:
            for m in P:
                add(m)
        self.polys, self.at = polys, at
        X = []
        for i in range(ns):
            if i in kser:
                X.append([ar.const(c) for c in kser[i]])
                continue
            x0 = ar.start(seg["x0"][i][1], seg["x0"][i][2])
            if x0 is None:
                return
            X.append([x0] + [ar.zero() for _ in range(order)])
        ser = [[ar.const(1)] + [ar.zero() for _ in range(order)]] + [[] for _ in mons[1:]]
        Dser = {i: [] for i in range(ns) if i not in kser}
        for k in range(order + 1):
            for j in range(1, len(mons)):
                pj, v = parent[j]
                ser[j].append(ar.total([ar.mul(ser[pj][l], X[v][k - l]) for l in range(k + 1)]))
            if k == order:
                break
            for i in Dser:
                Nk = self.value(2 * i, ser, k)
                Dser[i].append(self.value(2 * i + 1, ser, k))
                if k == 0 and ar.pole(Dser[i][0]):
                    return
                rest = ar.total([ar.scale(ar.mul(Dser[i][l], X[i][k + 1 - l]), -(k + 1 - l)) for l in range(1, k + 1)])
                X[i][k + 1] = ar.div(ar.total([Nk, rest]), ar.scale(Dser[i][0], k + 1))
        self.y = []
        for r in range(len(seg["G"])):
            M = [self.value(2 * ns + 2 * r, ser, k) for k in range(order + 1)]
            E = [self.value(2 * ns + 2 * r + 1, ser, k) for k in range(order + 1)]
            if ar.pole(E[0]):
                return
            y = []
            for k in range(order + 1):
                rest = ar.total([ar.scale(ar.mul(E[l], y[k - l]), -1) for l in range(1, k + 1)])
                y.append(ar.div(ar.total([M[k], rest]), E[0]))
            self.y.append(y)
        self.ok = True

    def value(self, q, ser, k):
        return self.ar.total([self.ar.mul(c, ser[self.at[m]][k]) for m, c in self.polys[q].items()])


def output_rows(series):
    """The indices of the rows of Series._outputs that are Taylor coefficients of an observable at the start of an
    experiment: the first segment of each condition, without the copies that start free."""
    rows, i = [], 0
    for part in series.parts:
        for j, seg in enumerate(part["segs"]):
            for _ in seg["labels"]:
                if j == 0 and not seg.get("free") and seg.get("on") is None:
                    rows.append(i)
                i += 1
    return rows


def _rat_recon(a, m):
    """The rational number n/d with |n|, d <= sqrt(m/2) and n = a d mod m, None where there is none."""
    a %= m
    bound = math.isqrt(m // 2)
    r0, r1, s0, s1 = m, a, 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        s0, s1 = s1, s0 - q * s1
    if s1 == 0 or abs(s1) > bound or math.gcd(r1, abs(s1)) != 1:
        return None
    return Fraction(r1, s1)


def _poly_terms(e, syms):
    """The terms (exponents, numerator, denominator) of a polynomial in `syms` with rational coefficients."""
    P = spy.Poly(e, *syms, domain="QQ")
    return [(m, int(c.numerator), int(c.denominator)) for m, c in P.terms()]


def _eval_terms(terms, x, p):
    out = 0
    for m, a, b in terms:
        v = a * pow(b, -1, p) % p
        for xi, e in zip(x, m):
            if e:
                v = v * pow(xi, e, p) % p
        out += v
    return out % p


class _Outputs:
    """The output coefficients used: values modulo p at points, and exact values as quotients in the coordinates."""

    def __init__(self, series, orders):
        self.series = series
        self.rows = output_rows(series)
        self.orders = orders
        self.order = max(orders[i] for i in self.rows) if self.rows else 0
        self.index = [(i, k) for i in self.rows for k in range(orders[i] + 1)]
        self._exact = {}
        self.where, i = {}, 0
        for pi, part in enumerate(series.parts):
            for seg in part["segs"]:
                for r in range(len(seg["labels"])):
                    self.where[i] = (pi, r)
                    i += 1

    def _taylor(self, ar, order):
        """{index: value} of the coefficients up to `order` in the arithmetic `ar`; None at a pole."""
        taylor, out = {}, {}
        for j, (i, k) in enumerate(self.index):
            if k > order:
                continue
            pi, r = self.where[i]
            if pi not in taylor:
                taylor[pi] = _Taylor(self.series, ar, pi, order)
            if not taylor[pi].ok:
                return None
            out[j] = taylor[pi].y[r][k]
        return out

    def mod_many(self, pts, p):
        """The coefficients at the points modulo p, one row each, and the rows at no pole."""
        ar = _Modular(pts, p, self.series.cs, self.series.params)
        got = self._taylor(ar, self.order)
        if got is None:
            return None, np.zeros(len(pts), dtype=bool)
        return np.array([got[j] for j in range(len(self.index))], dtype=np.int64).T.reshape(len(pts), -1), ~ar.bad

    def mod(self, x, p):
        vals, ok = self.mod_many([x], p)
        return None if not ok[0] else [int(v) for v in vals[0]]

    def exact(self, order):
        """(ring, factors, {index: quotient}) of the coefficients up to `order` as _Factored quotients in the
        coordinates; None at a pole."""
        if order not in self._exact:
            R = spy.ring(",".join(f"t{i}" for i in range(len(self.series.cs))), spy.QQ)[0]
            F = _Factored(R, self.series.cs, self.series.params)
            out = self._taylor(F, order)
            self._exact[order] = None if out is None else (R, F, out)
        return self._exact[order]


def generators(grp, syms):
    """(alpha, F, traces, others): a function alpha of the coordinates with |G| distinct images under the group
    `grp` (maps {symbol: expr}, the identity first), the functions F to prove in the field of the outputs, the
    coordinates the group fixes and the coefficients of prod_g (Z - g(alpha)), the traces sum_g g(theta_i) of the
    moved coordinates, which may help, and the other coordinates, to prove in F(alpha); None without alpha."""
    k = len(grp)
    imgs = {s: [spy.cancel(g.get(s, s)) for g in grp] for s in syms}
    fixed = [s for s in syms if all(spy.cancel(v - s) == 0 for v in imgs[s])]
    if k == 1:
        return None, list(syms), [], []
    moved = [s for s in syms if s not in fixed]

    def images(a):
        return [spy.cancel(a.xreplace({s: g.get(s, s) for s in syms})) for g in grp]

    def distinct(ims):
        return all(spy.cancel(a - b) != 0 for a, b in itertools.combinations(ims, 2))

    # the simplest alpha: one coordinate, else a combination of two or three with the simplest images
    order = sorted(moved, key=lambda s: sum(spy.count_ops(v) for v in imgs[s]))
    cands = (
        list(order)
        + [a + t * b for a, b in itertools.combinations(order, 2) for t in (2, 3)]
        + [a + 2 * b + 5 * c for a, b, c in itertools.combinations(order, 3)]
    )
    alpha = next((a for a in cands if distinct(imgs[a] if a in imgs else images(a))), None)
    if alpha is None:
        return None
    aims = images(alpha) if not isinstance(alpha, spy.Symbol) else imgs[alpha]
    # prod_g (Z - g(alpha)) over the rational function field of the coordinates, constant term first; maps over
    # Q(i) give coefficients over Q(i), which must lie in Q(theta)
    K = spy.polys.fields.field(",".join(str(s) for s in syms), spy.QQ)[0]
    gauss = any(v.has(spy.I) for vs in imgs.values() for v in vs)
    Kg = spy.polys.fields.field(",".join(str(s) for s in syms), spy.QQ_I)[0] if gauss else K

    def over_q(f):
        e = spy.cancel(spy.expand(f.as_expr())) if gauss else None
        return f if not gauss else None if e.has(spy.I) else K.from_expr(e)

    P = [Kg.one]
    for a in (Kg.from_expr(a) for a in aims):
        P = [(P[j - 1] if j else Kg.zero) - (a * P[j] if j < len(P) else Kg.zero) for j in range(len(P) + 1)]
    P = [over_q(c) for c in P]
    if any(c is None for c in P):
        return None
    # the traces sum_g g(theta_i) of the moved coordinates, which F(alpha) needs besides the coefficients
    traces = [over_q(sum((Kg.from_expr(v) for v in imgs[s]), Kg.zero)) for s in moved]
    traces = [t for t in traces if t is not None]
    F = [K.from_expr(x) for x in fixed] + P[:-1]
    F = [f for n_, f in enumerate(F) if not (f.numer.is_ground and f.denom.is_ground) and f not in F[:n_]]
    traces = [
        t
        for n_, t in enumerate(traces)
        if not (t.numer.is_ground and t.denom.is_ground) and t not in F and t not in traces[:n_]
    ]
    return alpha, [f.as_expr() for f in F], [t.as_expr() for t in traces], [s for s in moved if s != alpha]


def _monomials(R, d):
    return [m for t in range(d + 1) for m in _exps(R, t)]


def _parents(mons):
    """For each monomial but the first (the constant), the index of a monomial of one degree less and the variable
    that multiplies it."""
    at = {m: i for i, m in enumerate(mons)}
    out = [None]
    for m in mons[1:]:
        v = next(i for i, e in enumerate(m) if e)
        out.append((at[m[:v] + (m[v] - 1,) + m[v + 1 :]], v))
    return out


def _mono_values(vals, parents, p):
    out = [1]
    for par in parents[1:]:
        out.append(out[par[0]] * vals[par[1]] % p)
    return out


def _exps(R, t):
    if R == 0:
        return [()] if t == 0 else []
    return [(a,) + rest for a in range(t, -1, -1) for rest in _exps(R - 1, t - a)]


class _Pool:
    """Random points modulo each prime with the output coefficients there, drawn once and shared."""

    def __init__(self, outs, n, rng):
        self.outs, self.n, self.rng, self.pts = outs, n, rng, {}

    def get(self, p, m):
        pts = self.pts.setdefault(p, [])
        while len(pts) < m:
            xs = [[self.rng.randrange(1, p) for _ in range(self.n)] for _ in range(m - len(pts) + 4)]
            vals, ok = self.outs.mod_many(xs, p)
            if not ok.any():
                continue
            pts += [(x, [int(v) for v in vals[k]]) for k, x in enumerate(xs) if ok[k]]
        return pts[:m]


class _Var:
    """A function known to lie in the field of the output coefficients: an output coefficient (index `c`) or a
    function of the coordinates proven to lie there (`expr`)."""

    def __init__(self, name, c=None, expr=None, syms=()):
        self.name, self.c, self.expr = name, c, expr
        if expr is not None:
            num, den = spy.fraction(spy.cancel(spy.together(expr)))
            self.tn, self.td = _poly_terms(num, syms), _poly_terms(den, syms)
            self.num, self.den = num, den
            self.free = expr.free_symbols

    def value(self, x, c, p):
        if self.c is not None:
            return c[self.c]
        key = (p, id(x))
        memo = self.__dict__.setdefault("memo", {})
        if key not in memo:
            if len(self.tn) + len(self.td) > _SMALL_TERMS:
                nv, dv = self._large_value(x, p)
            else:
                dv = _eval_terms(self.td, x, p)
                nv = _eval_terms(self.tn, x, p) if dv else 0
            memo[key] = None if not dv else nv * pow(dv, -1, p) % p
        return memo[key]

    def _large_value(self, x, p):
        """(numerator, denominator) at x modulo p in C++, both at once."""
        if not hasattr(self, "_arr"):
            terms = self.tn + self.td
            self._arr = (
                np.array([m for m, _, _ in terms], dtype=np.int32).reshape(len(terms), -1),
                [(a, b) for _, a, b in terms],
                np.array([0, len(self.tn), len(terms)], dtype=np.int64),
            )
        E, co, off = self._arr
        res = self.__dict__.setdefault("_res", {})
        if p not in res:
            res[p] = np.array([a % p * pow(b, -1, p) % p for a, b in co], dtype=np.int64)
        nv, dv = _core.io_poly_eval(E, res[p], off, np.array([x], dtype=np.int64) % p, p, 1)[0]
        return int(nv), int(dv)


def _find(f, W, syms, outs, pool, d, stop, mons=None):
    """(A, B) as text with f = A(W)/B(W) of total degree d, or on the monomials `mons` (lowest degree first),
    proven by _check; None where none is found."""
    target = _TARGETS.setdefault((f, tuple(syms)), _Var("f", expr=f, syms=syms))
    mons = mons or _monomials(len(W), d)
    U = 2 * len(mons)
    E = np.array(mons, dtype=np.int32).reshape(len(mons), len(W))
    primes = all_primes()[:3]
    pts, basis, test = {}, None, {}
    for p in primes:
        fvs, vals = [], []
        for x, c in pool.get(p, U + 12):
            fv = target.value(x, c, p)
            v = [w.value(x, c, p) for w in W]
            if fv is None or any(a is None for a in v):
                continue
            fvs.append(fv)
            vals.append(v)
        if work.ticks() > stop or not vals:
            return None
        # the monomials at every point in C++, then times -f for the columns of B
        M = _core.io_mono_matrix(np.array(vals, dtype=np.int64).reshape(len(vals), len(W)), E, p, ioproj._threads())
        test.setdefault(p, [int(t) for t in M[0]])
        rows = np.hstack([M, M * ((-np.array(fvs, dtype=np.int64)) % p)[:, None] % p])
        work.tick(rows.shape)
        ker = _core.io_kernel(np.ascontiguousarray(rows), p, ioproj._threads())
        piv, free = list(ker["pivots"]), list(ker["free"])
        if not free:
            return None
        key = (tuple(piv), tuple(free))
        basis = basis or key
        if key != basis:
            continue
        pts[p] = [[int(x) for x in v] for v in ker["basis"]]
    if len(pts) < len(primes):
        return None
    M = math.prod(pts)
    p0, n = primes[0], len(mons)
    for vi in range(len(basis[1])):
        # a vector with B zero at a point is a relation among the variables, not an expression of f
        if not sum(b_ * t for b_, t in zip(pts[p0][vi][n:], test[p0])) % p0:
            continue
        vec = []
        for j in range(U):
            r = sum(pts[p][vi][j] * (M // p) * pow(M // p, -1, p) for p in pts)
            q = _rat_recon(r % M, M)
            if q is None:
                break
            vec.append(q)
        if len(vec) == U:
            got = _check(vec, mons, W, target, syms, outs)
            if got is not None:
                return got
    return None


def _check(vec, mons, W, target, syms, outs):
    """(A, B) of `vec` where A(W) den_f = num_f B(W) and B(W) != 0 hold as polynomials in the coordinates, over
    shared factors of the denominators and without gcds; None otherwise."""
    n = len(mons)
    terms = [m for m, q in zip(mons + mons, vec) if q]
    used = sorted({i for m in terms for i, e in enumerate(m) if e})
    order = max([outs.index[W[i].c][1] for i in used if W[i].c is not None], default=0)
    got = outs.exact(order)
    if got is None:
        return None
    R, F, cs = got
    tsym = dict(zip(syms, [spy.Symbol(f"t{i}") for i in range(len(syms))]))

    def ring(e):
        return R.from_expr(spy.sympify(e).xreplace(tsym))

    fr = {i: cs[W[i].c] if W[i].c is not None else F.quo(ring(W[i].num), ring(W[i].den)) for i in used}
    pw = {}

    def power(i, e):
        if (i, e) not in pw:
            pw[(i, e)] = fr[i] if e == 1 else F.mul(power(i, e - 1), fr[i])
        return pw[(i, e)]

    def value(coefs):
        out = []
        for m, q in zip(mons, coefs):
            if not q:
                continue
            t = F.const(q)
            for i in used:
                if m[i]:
                    t = F.mul(t, power(i, m[i]))
            out.append(t)
        return F.total(out)

    B = value(vec[n:])
    if not B[0]:
        return None
    A = value(vec[:n])
    diff = F.total([F.mul(A, (ring(target.den), {})), F.scale(F.mul(B, (ring(target.num), {})), Fraction(-1))])
    if diff[0]:
        return None
    names = [W[i].name for i in range(len(W))]

    def text(coefs):
        return (
            " + ".join(
                f"({q})" + "".join(f"*{nm}" + (f"^{e}" if e > 1 else "") for nm, e in zip(names, m) if e)
                for m, q in zip(mons, coefs)
                if q
            )
            or "0"
        )

    return text(vec[:n]), text(vec[n:])


def _reynolds(grp, syms, degree):
    """The sums over the group of the monomials of `degree` in the coordinates the group moves, one at a time."""
    moved = [s for s in syms if any(spy.cancel(g.get(s, s) - s) != 0 for g in grp)]
    for m in itertools.combinations_with_replacement(moved, degree):
        e = spy.prod(m)
        h = spy.cancel(sum((e.xreplace({s: g.get(s, s) for s in syms}) for g in grp), spy.Integer(0)))
        if not h.has(spy.I):
            yield h


class _Relations:
    """The linear relations over Q(theta) among the monomials phi_i in the derivatives of the observables and the
    known inputs along the trajectory of the first condition (docs/method/global.rst, "The exact count"): the kernel of
    the matrix of the Taylor coefficients of the phi_i, whose entries are polynomials in output coefficients."""

    def __init__(self, series, syms):
        self.ok = False
        part = series.parts[0]
        seg = part["segs"][0]
        if seg.get("free") or seg.get("on") is not None:
            return
        st = list(part["st"])
        ns = len(st)
        params = list(series.params)
        known = set(part["known"])

        def reads(j):
            N, D = seg["F"][j]
            return {i for P in (N, D) for m in P for i in range(ns) if m[i]}

        # the known states (inputs and their jets, the clock) within _CHAIN steps of the others: a relation of
        # order m reads the jets of an input up to m, and the long chains would make the field too large
        keep = {j for j in range(ns) if j not in known}
        obs = {i for M, E in seg["G"] for P in (M, E) for m in P for i in range(ns) if m[i]}
        front = (set().union(*[reads(j) for j in keep]) | obs) & known if keep else obs & known
        last = set()
        for _ in range(_CHAIN):
            new = front - keep
            keep |= new
            front = set().union(*[reads(j) for j in new]) & known if new else set()
            last = new
        cut = {j for j in last if reads(j) - keep}
        kept = sorted(keep)
        names = [str(st[j]) for j in kept] + [str(p) for p in params]
        extra = [str(c) for c in syms if str(c) not in names]
        # rational right-hand sides and observables: the field of the states and coordinates; the polynomial ring
        # where every denominator is a number, whose arithmetic needs no gcd
        polynomial = all(all(not any(m) for m in D) for _, D in seg["F"]) and all(
            all(not any(m) for m in E) for _, E in seg["G"]
        )
        self.R, *gens = (spy.ring if polynomial else spy.field)(",".join(names + extra), spy.QQ)
        self.names = names + extra
        X = [None] * ns
        for j, g in zip(kept, gens):
            X[j] = g
        P = gens[len(kept) : len(kept) + len(params)]

        def poly(Pd):
            out = self.R.zero
            for m, c in Pd.items():
                t = self.R(spy.Rational(c.numerator, c.denominator))
                for g, e in zip(X + list(P), m):
                    if e:
                        t = t * g**e
                out = out + t
            return out

        def quo(N, D):
            if polynomial:
                (c,) = D.values()
                return poly(N) * self.R(spy.Rational(1) / spy.Rational(c.numerator, c.denominator))
            return poly(N) / poly(D)

        # the last jets kept stay constant: the relations tried read fewer derivatives than the chain holds
        self.f = [None if j not in keep else self.R.zero if j in cut else quo(*seg["F"][j]) for j in range(ns)]
        self.cut = cut
        self.y = [quo(M, E) for M, E in seg["G"]]
        self.X = X
        self.read = sorted({i for j in range(ns) if j not in known and j in keep for i in known & reads(j)})
        self.known = sorted(known & keep)
        self.series, self.part, self.ns = series, part, ns
        self.syms = syms
        self.ny = len(self.y)
        self.observed = []
        self.ok = True

    def observe(self, i):
        """State i joins the observables: the series of the state lies in the field of the outputs."""
        if i not in self.observed:
            self.observed.append(i)
            self.y.append(self.X[i])
            if hasattr(self, "_lie"):
                self._lie.append([self.X[i]])

    def lie(self, q):
        return sum((q.diff(x) * fx for x, fx in zip(self.X, self.f) if x is not None), self.R.zero)

    def jets(self, ms):
        """The symbolic derivatives of each observable r up to order ms[r], none for ms[r] < 0."""
        if not hasattr(self, "_lie"):
            self._lie = [[y] for y in self.y]
        ys = []
        for r, m in enumerate(ms):
            d = self._lie[r]
            while len(d) <= m:
                d.append(self.lie(d[-1]))
            ys.append(d[: m + 1])
        return ys

    def inputs(self, m):
        """The known states read by the others and their derivatives along the known chain, up to order m."""
        out = list(self.read)
        for i in self.read:
            j = i
            for _ in range(m):
                nxt = [k for k in self.known if self.f[j] == self.X[k]]
                if not nxt:
                    break
                j = nxt[0]
                if j not in out:
                    out.append(j)
        return out


def _relations(series, syms):
    """The _Relations of a series, built once: the derivatives and their compiled functions are shared."""
    key = (id(series), tuple(str(s_) for s_ in syms))
    if key not in _REL:
        _REL[key] = (series, _Relations(series, syms))
    return _REL[key][1]


def _relation_matrix(rel, x, p, ms, degree, K):
    S = ioproj.taylor_rows(rel, ms, dict(zip(rel.series.cs, x)), p, K)
    if S is None:
        return None
    mons = _monomials(S.shape[0], degree)
    ser = _core.io_series_monomials(
        S, np.array(mons, dtype=np.int32).reshape(len(mons), S.shape[0]), p, ioproj._threads()
    )
    return ser.T.tolist(), mons


def _kernel_canonical(M, p):
    """(pivots, free, basis) of the kernel of M over GF(p), each basis vector the identity on the free columns."""
    M = np.array(M, dtype=np.int64)
    work.tick(M.shape)
    ker = _core.io_kernel(M, p, ioproj._threads())
    return tuple(ker["pivots"]), tuple(ker["free"]), [[int(x) for x in v] for v in ker["basis"]]


def _crt_rational(vals):
    """{prime: residue} as a rational number by the Chinese remainder theorem, None without a bounded one."""
    M = math.prod(vals)
    r = sum(v * (M // p) * pow(M // p, -1, p) for p, v in vals.items())
    return _rat_recon(r % M, M)


class _Samples:
    """Points modulo each prime with the values of a vector function there, kept for every degree."""

    def __init__(self, values, n, rng, primes):
        self.values, self.n, self.rng, self.pts = values, n, rng, {p: [] for p in primes}

    def get(self, p, m, stop):
        pts = self.pts[p]
        while len(pts) < m:
            if work.ticks() > stop:
                return None
            x = [self.rng.randrange(1, p) for _ in range(self.n)]
            v = self.values(x, p)
            if v is not None:
                pts.append((x, v))
        return pts[:m]


def _polys_at_once(S, entries, scale, primes, stop, degrees=range(0, 12), box=None):
    """{entry: (monomials, coefficients)}: polynomials of the least degree in `degrees` with the values of
    `entries` (times scale(x, p) where given), one linear system per prime and degree for all of them; with `box`,
    only the monomials whose exponent of variable i is at most box[i]."""
    out = {}
    for d in degrees:
        todo = [e for e in entries if e not in out]
        if not todo:
            break
        mons = _monomials(S.n, d)
        if box is not None:
            mons = [m for m in mons if all(e <= b for e, b in zip(m, box))]
        U = len(mons)
        if U > _INTERP:
            break
        par = _parents(mons) if box is None else None
        sols = {}
        for p in primes:
            pts = S.get(p, U + 8, stop)
            if pts is None:
                return out
            rows = []
            for x, v in pts:
                sc = scale(x, p) if scale else 1
                mv = (
                    _mono_values(x, par, p)
                    if par is not None
                    else [math.prod((pow(xi, k, p) for xi, k in zip(x, m) if k), start=1) % p for m in mons]
                )
                rows.append(mv + [v[e] * sc % p for e in todo])
            rows = np.array(rows, dtype=np.int64)
            work.tick(rows.shape)
            rr = rref_modp(rows, p)
            piv = [int(c) for c in rr["piv"][: rr["rank"]]]
            if piv[:U] != list(range(U)):
                break
            for k, e in enumerate(todo):
                if U + k in piv:
                    continue
                sols.setdefault(e, {})[p] = [int(rr["R"][i, U + k]) for i in range(U)]
        for e, by in sols.items():
            if len(by) < len(primes):
                continue
            coef = [_crt_rational({p: by[p][i] for p in primes}) for i in range(U)]
            if all(c_ is not None for c_ in coef):
                out[e] = (mons, coef)
    return out


def _line_degrees(S, width, p, stop, top=24):
    """Per entry the degrees (numerator, denominator) of the restriction to a random line, the least sum first and
    a polynomial first; the total degrees of the entries at a generic line. None where not found."""
    rng = S.rng
    b = [rng.randrange(1, p) for _ in range(S.n)]
    d = [rng.randrange(1, p) for _ in range(S.n)]
    pts = []
    t = 0
    while len(pts) < 2 * top + 6:
        t += 1
        if work.ticks() > stop or t > 6 * top:
            return None
        v = S.values([(bi + t * di) % p for bi, di in zip(b, d)], p)
        if v is not None:
            pts.append((t, v))
    out = []
    for e in range(width):
        got = None
        for tot in range(0, top + 1):
            for dd in range(0, tot + 1):
                dn = tot - dd
                rows = [
                    [pow(t, i, p) for i in range(dn + 1)] + [(-v[e] * pow(t, i, p)) % p for i in range(dd + 1)]
                    for t, v in pts
                ]
                if _kernel_canonical(rows, p)[2]:
                    got = (dn, dd)
                    break
            if got:
                break
        out.append(got)
    return out


def _variable_degrees(S, entries, p, stop, top):
    """Per variable the highest degree of the entries in it alone, the others at random values: a box of
    monomials that holds every term, at a generic point. None where not found."""
    rng = S.rng
    out = []
    for i in range(S.n):
        base = [rng.randrange(1, p) for _ in range(S.n)]
        pts, t = [], 0
        while len(pts) < top + 6:
            t += 1
            if work.ticks() > stop or t > 4 * top + 20:
                return None
            x = list(base)
            x[i] = t
            v = S.values(x, p)
            if v is not None:
                pts.append((t, v))
        deg = 0
        for e in entries:
            for dn in range(0, top + 1):
                rows = [[pow(t, k, p) for k in range(dn + 1)] + [(-v[e]) % p] for t, v in pts]
                if _kernel_canonical(rows, p)[2]:
                    deg = max(deg, dn)
                    break
            else:
                return None
        out.append(deg)
    return out


def _rational_one(S, e, dn, dd, primes, stop):
    """(numerator monomials and coefficients, denominator monomials and coefficients) of entry e with total degrees
    at most dn and dd, None where the kernel is not one vector."""
    mn, md = _monomials(S.n, dn), _monomials(S.n, dd)
    pn, pd = _parents(mn), _parents(md)
    U = len(mn) + len(md)
    if U > _INTERP:
        return None
    vecs = {}
    for p in primes:
        pts = S.get(p, U + 8, stop)
        if pts is None:
            return None
        rows = [
            [v[e] * t % p for t in _mono_values(x, pd, p)] + [(-t) % p for t in _mono_values(x, pn, p)] for x, v in pts
        ]
        piv, free, basis = _kernel_canonical(rows, p)
        if len(basis) != 1:
            return None
        vecs[p] = (piv, free, basis[0])
    if len({(a, b) for a, b, _ in vecs.values()}) != 1:
        return None
    coef = [_crt_rational({p: vecs[p][2][i] for p in primes}) for i in range(U)]
    if any(c_ is None for c_ in coef):
        return None
    return (mn, coef[len(md) :]), (md, coef[: len(md)])


def _interpolate(values, n, rng, stop):
    """Rational functions of the n variables, one per entry of the vector function `values(x, p)` (None at a bad
    point), rebuilt over Q: the degrees from a random line, the polynomials of all entries at once, the rational
    entries over the denominator of one of them; candidates only."""
    primes = all_primes()[:2]
    S = _Samples(values, n, rng, primes)
    first = S.get(primes[0], 1, stop)
    if not first:
        return None
    width = len(first[0][1])
    degs = _line_degrees(S, width, primes[0], stop)
    if degs is None or any(g is None for g in degs):
        return None
    zero = [Fraction(0)]
    out = {e: ([(0,) * n], zero, [Fraction(1)]) for e, g in enumerate(degs) if g == (0, 0) and not first[0][1][e]}
    polys = [e for e, g in enumerate(degs) if e not in out and g[1] == 0]
    if polys:
        top = max(degs[e][0] for e in polys)
        # most coefficients of relations are sparse: the monomials within the degrees in each variable
        box = _variable_degrees(S, polys, primes[0], stop, top) if top > 2 else None
        got = _polys_at_once(S, polys, None, primes, stop, [top], box)
        if len(got) < len(polys):
            return None
        for e, (m_, c_) in got.items():
            out[e] = (m_, c_, [Fraction(1)] + [Fraction(0)] * (len(m_) - 1))
    rest = [e for e in range(width) if e not in out]
    if not rest:
        return [out[e] for e in range(width)]
    # the denominator of the entry with the highest one, then the numerators over it at once
    e0 = max(rest, key=lambda e: degs[e][1])
    one = _rational_one(S, e0, degs[e0][0], degs[e0][1], primes, stop)
    if one is None:
        return None
    den = one[1]
    dterms = [(m_, int(c_.numerator), int(c_.denominator)) for m_, c_ in zip(*den) if c_]

    def scale(x, p):
        return _eval_terms(dterms, x, p)

    top = max(degs[e][0] - degs[e][1] for e in rest) + degs[e0][1]
    nums = _polys_at_once(S, rest, scale, primes, stop, range(max(0, top), top + 3))
    if len(nums) < len(rest):
        return None
    for e, (m_, c_) in nums.items():
        # numerator and denominator over one list of monomials
        allm = sorted(set(m_) | set(den[0]), key=lambda t: (sum(t), t))
        nn = dict(zip(m_, c_))
        dd = dict(zip(*den))
        out[e] = (allm, [nn.get(t, Fraction(0)) for t in allm], [dd.get(t, Fraction(0)) for t in allm])
    return [out[e] for e in range(width)]


def _base(rel, ms):
    """The derivatives of the observables, up to ms[r] for observable r, and the input states, as ioproj.Quo."""
    J = ioproj.jets_of(rel)
    return [q for r, m in enumerate(ms) if m >= 0 for q in J.observable(r, m)] + [
        J.state(i) for i in rel.inputs(max(ms))
    ]


def _configs(ny, ni, cap, floor=0):
    """(orders per observable, degree) of the relations tried with more than `floor` and at most `cap` monomials,
    fewest monomials first; an order -1 leaves the observable out."""
    out = []
    # every combination of orders for a few observables; for many, each one alone and all at one order
    if ny <= 3:
        vectors = itertools.product(range(-1, 6), repeat=ny)
    else:
        vectors = [tuple(m if r == q else -1 for r in range(ny)) for q in range(ny) for m in range(6)] + [
            (m,) * ny for m in range(6)
        ]
    for ms in vectors:
        if max(ms) < 0:
            continue
        nvar = sum(m + 1 for m in ms) + ni(max(ms))
        for degree in range(1, 7):
            nmon = math.comb(nvar + degree, degree)
            if nmon > cap:
                break
            if nmon > floor:
                out.append((nmon, ms, degree))
    return [(ms, degree) for _, ms, degree in sorted(out)]


def _covers(a, b):
    """Whether config a = (orders, degree) holds every monomial of config b."""
    return a[1] >= b[1] and all(x >= y for x, y in zip(a[0], b[0]))


def _compiled(rel, ms, degree):
    """((base, parents), monomials): the derivatives of the observables and the input states, evaluated together, and
    the exponent vectors of the monomials phi in them."""
    base = ioproj.Base(rel, _base(rel, ms))
    mons = _monomials(len(base.funcs), degree)
    return (base, _parents(mons)), mons


def _eval_monomials(compiled, pt, p):
    """The monomials phi at the point modulo p; None where a denominator vanishes there."""
    (base, parents), _ = compiled
    ok, vals = base.values(np.array([pt], dtype=np.int64) % p, p)
    if not ok[0]:
        return None
    return _mono_values([int(v) for v in vals[0]], parents, p)


def _rebuild(inter, tsy):
    """The interpolated entries as expressions in the symbols `tsy`, None where a denominator is zero."""
    out = []
    for mons_, nn, dd in inter:
        N = sum(
            (
                spy.Rational(q.numerator, q.denominator) * spy.prod([s_**e for s_, e in zip(tsy, mm)])
                for mm, q in zip(mons_, nn)
                if q
            ),
            spy.Integer(0),
        )
        Dn = sum(
            (
                spy.Rational(q.numerator, q.denominator) * spy.prod([s_**e for s_, e in zip(tsy, mm)])
                for mm, q in zip(mons_, dd)
                if q
            ),
            spy.Integer(0),
        )
        if Dn == 0:
            return None
        out.append(spy.cancel(N / Dn))
    return out


def _ring_coeffs(rel, syms, lam):
    """The coefficients lam, rational functions of the coordinates, times their common denominator, as polynomials
    of the ring of the jets."""
    J = ioproj.jets_of(rel)
    den = spy.Integer(1)
    for v in lam.values():
        den = spy.lcm(den, spy.fraction(spy.cancel(v))[1])
    names = {s_: spy.Symbol(str(s_)) for s_ in syms}
    return {j: J.R.from_expr(spy.cancel(v * den).xreplace(names)) for j, v in lam.items() if v != 0}


def _identity(rel, syms, lam, phis):
    """Whether sum_j lam[j] phis[j] is zero, over the common denominator of the phis and of the coefficients."""
    J = ioproj.jets_of(rel)
    co = _ring_coeffs(rel, syms, lam)
    return ioproj.combine(J, list(co.values()), [phis[j] for j in co])[0] == 0


def _phis(rel, ms, degree, extra=()):
    """The monomials phi in the base functions as ioproj.Quo, then each state of `extra` times those of one degree
    less."""
    J = ioproj.jets_of(rel)
    base = _base(rel, ms)
    phis = ioproj.products(J, base, _monomials(len(base), degree))
    low = phis[: math.comb(len(base) + degree - 1, degree - 1)]
    return phis + [ioproj.Quo(q.numer * J.X[e], q.exps) for e in extra for q in low]


def _kernel_at(rel, jets, xp, p, pos, rng, rows_needed, extra=None):
    rows = []
    while len(rows) < rows_needed:
        pt = [rng.randrange(1, p) for _ in rel.names]
        for i_, v in zip(pos, xp):
            pt[i_] = v
        r = _eval_monomials(jets, pt, p)
        if r is None:
            continue
        if extra is not None:
            # the columns x_i psi_k: the state times the monomials of one degree less
            r = r + [pt[extra[0]] * r[k] % p for k in range(extra[1])]
        rows.append(r)
    return _kernel_canonical(rows, p)


def state_relation(series, syms, coord, rng, stop, cap=None):
    """For a coordinate that is the start of a state x_i: a relation sum_j lambda_j phi_j(y, u) + x_i sum_k mu_k
    psi_k(y, u) = 0 along every trajectory, checked as an identity of rational functions, with mu_k psi_k nonzero,
    so that x_i(0) = -sum_j lambda_j phi_j / sum_k mu_k psi_k at the start; {"reads": the coordinates the
    coefficients read} or None."""
    rel = _relations(series, syms)
    if not rel.ok:
        return None
    seg = rel.part["segs"][0]
    i = next((k for k, v in enumerate(seg["x0"]) if v[0] is not None and v[0] == coord), None)
    if i is None or i in rel.known:
        return None
    idx = [k for k, s_ in enumerate(syms) if s_ in set(series.params)]
    pos = [rel.names.index(str(syms[k])) for k in idx]
    xi = rel.names.index(str(rel.X[i]))
    p0 = all_primes()[0]
    for ms, degree in _configs(len(rel.y), lambda m: len(rel.inputs(m)), cap or _RELATION):
        if work.ticks() > stop:
            return None
        if not ioproj.jets_of(rel).within(ms):
            continue
        jets = _compiled(rel, ms, degree)
        nmon = len(jets[1])
        nlow = math.comb(len(_base(rel, ms)) + degree - 1, degree - 1)
        ext = (xi, nlow)
        piv, free, basis = _kernel_at(
            rel, jets, [rng.randrange(1, p0) for _ in idx], p0, pos, rng, nmon + nlow + 7, ext
        )
        # a free column x_i psi_k: the basis vector of that column has the state at a nonzero coefficient
        cols = [c for c in free if c >= nmon]
        if not cols:
            continue
        col = free.index(cols[0])

        def values(xp, p, piv=piv, free=free, jets=jets, col=col, ext=ext, nmon=nmon, nlow=nlow):
            pv, fr, b = _kernel_at(rel, jets, xp, p, pos, rng, nmon + nlow + 7, ext)
            if (pv, fr) != (piv, free):
                return None
            return [b[col][j] for j in piv]

        inter = _interpolate(values, len(idx), rng, stop)
        if inter is None:
            continue
        exprs = _rebuild(inter, [syms[k] for k in idx])
        if exprs is None:
            continue
        lam = {free[col]: spy.Integer(1)}
        lam.update(zip(piv, exprs))
        J = ioproj.jets_of(rel)
        phis = _phis(rel, ms, degree, [i])
        # the denominator sum_k mu_k psi_k of x_i is not zero, over the common denominators
        mu = _ring_coeffs(rel, syms, {j: v for j, v in lam.items() if j >= nmon})
        den, T = ioproj.combine(
            J, list(mu.values()), [ioproj.Quo(phis[j].numer.exquo(J.X[i]), phis[j].exps) for j in mu]
        )
        if den == 0 or not _identity(rel, syms, lam, phis):
            continue
        # and not zero at the start of the trajectory, with the factors of the denominator
        start = {
            spy.Symbol(str(x)): (v[0] if v[0] is not None else spy.Symbol(str(x)))
            for x, v in zip(rel.X, seg["x0"])
            if x is not None
        }
        if any(spy.cancel(P.as_expr().xreplace(start)) == 0 for P in [den] + [J.fac[k] for k in T]):
            continue
        return {
            "reads": set().union(*[v.free_symbols for v in lam.values()]),
            "orders": ms,
            "degree": degree,
            "state": i,
        }
    return None


def io_functions(series, syms, rng, stop, log, cap=_RELATION, floor=0, found=None):
    """Functions of the coordinates proven to lie in the field of the output coefficients: the entries of the
    canonical kernel basis of the relation matrix, each relation checked as an identity of rational functions in
    the states and coordinates, the kernel dimension by the rank at one point modulo a prime. A generator: each
    configuration of orders and degree with more than `floor` and at most `cap` monomials that gives relations yields
    their new entries."""
    rel = _relations(series, syms)
    if not rel.ok:
        log.append("no relations for this model")
        return
    p0 = all_primes()[0]
    idx = [k for k, s_ in enumerate(syms) if s_ in set(series.params)]
    pos = [rel.names.index(str(syms[k])) for k in idx]
    found = set() if found is None else found
    done = []
    for ms, degree in _configs(len(rel.y), lambda m: len(rel.inputs(m)), cap, floor):
        if work.ticks() > stop:
            return
        # a configuration holding one that gave relations holds their multiples and derivatives: the relations
        # found later come with new observables
        if any(_covers((ms, degree), d) for d in done) or not ioproj.jets_of(rel).within(ms):
            continue
        jets = _compiled(rel, ms, degree)
        nmon = len(jets[1])
        piv, free, basis = _kernel_at(rel, jets, [rng.randrange(1, p0) for _ in idx], p0, pos, rng, nmon + 6)
        if not basis or len(basis) > 6:
            continue
        # the theorem needs the kernel of M along the trajectory to be the space of the relations
        K = nmon + 6
        got = _relation_matrix(rel, [rng.randrange(1, p0) for _ in syms], p0, ms, degree, K)
        if got is None or len(_kernel_canonical(got[0], p0)[0]) != nmon - len(basis):
            continue

        def values(xp, p, piv=piv, free=free, jets=jets, nmon=nmon):
            pv, fr, b = _kernel_at(rel, jets, xp, p, pos, rng, nmon + 6)
            if (pv, fr) != (piv, free):
                return None
            return [v[j] for v in b for j in piv]

        inter = _interpolate(values, len(idx), rng, stop)
        if inter is None:
            log.append(f"relations at orders {ms}, degree {degree}: not interpolated")
            continue
        exprs = _rebuild(inter, [syms[k] for k in idx])
        if exprs is None:
            continue
        phis = _phis(rel, ms, degree)
        # one relation per free column: 1 there, the interpolated entries on the pivots
        it = iter(exprs)
        lams = ({fc: spy.Integer(1), **{pc: next(it) for pc in piv}} for fc in free)
        if not all(_identity(rel, syms, lam, phis) for lam in lams):
            log.append(f"relations at orders {ms}, degree {degree}: identity fails")
            continue
        done.append((ms, degree))
        new = _new(exprs, found)
        log.append(f"relations at orders {ms}, degree {degree}: {len(free)} proven")
        yield new


def _new(exprs, found):
    """The expressions that are no numbers and not in `found`, which takes them."""
    out = []
    for e in exprs:
        key = str(e)
        if not e.is_number and key not in found:
            found.add(key)
            out.append(e)
    return out


def _profiles(rel, rng):
    """The parametric profiles of the relations, computed once."""
    if not hasattr(rel, "_profiles"):
        unknown = [j for j in range(rel.ns) if rel.X[j] is not None and j not in rel.known]
        rel._profiles = ioproj.profiles(rel, unknown, all_primes()[0], rng)
    return rel._profiles


def projection_functions(series, syms, rng, stop, log, found=None):
    """Functions of the coordinates proven to lie in the field of the output coefficients: the coefficients of the
    projections, per profile and observable the relation of the derivative of the observable on top of the profile
    with the derivatives below (docs/method/global.rst, "The exact count"). Each relation is checked as an identity of
    polynomials, and the rank of the Taylor coefficients of the relation along the trajectory of the first condition,
    one less than the number of terms at a point modulo a prime, puts the coefficients in the field of the outputs.
    A generator: each projection proven yields the coefficients not found before."""
    rel = _relations(series, syms)
    if not rel.ok:
        return
    found = set() if found is None else found
    p0 = all_primes()[0]
    idx = [k for k, s_ in enumerate(syms) if s_ in set(series.params)]
    pos = [rel.names.index(str(syms[k])) for k in idx]
    params = [syms[k] for k in idx]
    # the projections tried before, also by an earlier pass with fewer observables
    seen = rel.__dict__.setdefault("_projections", set())
    for h in _profiles(rel, rng):
        # the supports of the projections of a profile first, then the relations with the fewest terms first
        found_here = []
        for i in range(len(h)):
            if work.ticks() > stop:
                return
            pr = ioproj.Projection.output(rel, h, i, pos, rng)
            where = f"projection of observable {i} at profile {tuple(h)}"
            if pr is None:
                if (tuple(h), i) not in seen:
                    seen.add((tuple(h), i))
                    log.append(f"{where}: a derivative of one with more than {ioproj._JET} terms")
                continue
            key = (tuple(pr.ms), pr.top, tuple(str(rel.y[r]) for r, m in enumerate(pr.ms) if m >= 0))
            if key in seen:
                continue
            seen.add(key)
            S = pr.search([rng.randrange(1, p0) for _ in idx], p0, stop)
            if S is not None:
                _supports(rel)[(tuple(h), i)] = S
            if S is None or len(S) < 2:
                log.append(f"{where}: no relation found")
            elif len(S) > ioproj.TERMS:
                log.append(f"{where}: {len(S)} terms")
            else:
                found_here.append((len(S), i, where, pr))
        for _, _i, where, pr in sorted(found_here, key=lambda x: x[:2]):
            if work.ticks() > stop:
                return
            S = pr.mons
            rk = ioproj.series_rank(rel, pr, S, syms, rng, p0)
            if rk != len(S) - 1:
                log.append(f"{where}: rank {rk} of {len(S)} terms along the trajectory")
                continue
            co = ioproj.Coefficients(pr, len(idx), rng, stop)
            if not co.run():
                log.append(f"{where}: {len(S)} terms, not interpolated ({co.why})")
                continue
            if not ioproj.identity(rel, pr, co, pos):
                log.append(f"{where}: {len(S)} terms, identity fails")
                continue
            new = _new(co.expressions(params), found)
            log.append(f"{where}: {len(S)} terms proven")
            yield new


def _elim_child():
    """elim.eliminate_relations() on the pickled input of stdin, the relations pickled to stdout as dictionaries."""
    from . import elim

    series, syms, seed, terms = pickle.loads(sys.stdin.buffer.read())
    rel = _relations(series, syms)
    try:
        got = elim.eliminate_relations(rel, random.Random(seed), terms)
    except ValueError as exc:
        sys.stdout.buffer.write(pickle.dumps({"why": str(exc)}))
        return
    sys.stdout.buffer.write(
        pickle.dumps(
            {
                "order": got["ring"].order,
                "orders": got["orders"],
                "steps": got["steps"],
                "relations": {r: P.to_dict() for r, P in got["relations"].items()},
                "states": {
                    j: None if ab is None else tuple(P.to_dict() for P in ab) for j, ab in got["states"].items()
                },
            }
        )
    )


def elimination_functions(series, syms, rng, stop, log, found=None):
    """Functions of the coordinates proven in the field of the output coefficients: per observable the coefficients of
    the relation by elimination (elim.py), over one of fewest terms, those whose gradients raise the rank and as many
    more (docs/method/global.rst, "The exact count"). The relation vanishes along every trajectory by construction, and
    the rank of the Taylor coefficients of the monomials of the relation along the trajectory, one less than their
    number at a point modulo a prime, makes it the only one on the support. The elimination runs in a separate process,
    which starts where the search has not reached `stop` (work.ticks()) and ends at the cap elimination_terms of
    work.CAPS; a generator as projection_functions()."""
    rel = _relations(series, syms)
    if not rel.ok or getattr(rel, "_elim_tried", False):
        return
    rel._elim_tried = True
    try:
        from . import elim
    except ImportError:
        log.append("elimination: python-flint is not installed")
        return
    if work.ticks() > stop:
        return
    payload = pickle.dumps((series, list(syms), rng.getrandbits(64), work.CAPS["elimination_terms"]))
    try:
        r = work.run(
            [sys.executable, "-c", "from symident.detection.fieldproof import _elim_child; _elim_child()"],
            input=payload,
        )
        got = pickle.loads(r.stdout) if r.returncode == 0 and r.stdout else None
    except (pickle.UnpicklingError, EOFError):
        got = None
    if got is None or "why" in got:
        log.append(f"elimination: {got['why']}" if got else "elimination: the process failed")
        return
    R = elim._Ring(rel, got["order"])
    _elimination_states(rel, R, got.get("states") or {})
    p0 = all_primes()[0]
    gens = [spy.Symbol(n) for n in R.params]

    def expr(co):
        return sum((spy.Integer(c) * spy.prod([g**k for g, k in zip(gens, m)]) for m, c in co.items()), spy.Integer(0))

    for obs, d in got["relations"].items():
        P = R.ctx.from_dict(d)
        built = elim.relation_projection(rel, R, P)
        where = f"elimination of observable {obs}"
        if built is None:
            log.append(f"{where}: the relation reads a known state outside the base functions")
            continue
        pr, support, coefs = built
        rk = ioproj.series_rank(rel, pr, support, syms, rng, p0)
        if rk != len(support) - 1:
            log.append(f"{where}: rank {rk} of {len(support)} terms along the trajectory")
            continue
        ref, order = elim.normalized(coefs)
        _, pins = elim.gradient_select(coefs, ref, order, len(gens), p0, rng)
        den = expr(coefs[ref])
        exprs = [spy.cancel(expr(coefs[j]) / den) for j in pins]
        rel.__dict__.setdefault("_eliminated", set()).add(obs)
        log.append(
            f"{where}: relation of {len(support)} terms ({len(P)} with the coordinates), rank {rk} along "
            f"the trajectory, {len(pins)} coefficients proven"
        )
        yield _new(exprs, found)


def _elimination_states(rel, R, states):
    """rel._elim_states: per state eliminated with A x + B = 0 along every trajectory, A nonzero at a point of one,
    the coordinates that A and B read, the parameters and the starts of the states they read."""
    seg = rel.part["segs"][0]
    out = {}
    for j, ab in states.items():
        if ab is None:
            continue
        used = {i for d in ab for m in d for i, k in enumerate(m) if k}
        reads = set()
        for i in used:
            n = R.names[i]
            if n in R.params:
                reads.add(spy.Symbol(n))
            elif n.startswith("si_x"):
                v = seg["x0"][int(n[4:])][0]
                reads |= set() if v is None else spy.sympify(v).free_symbols
        out[j] = {"reads": reads, "state": j}
    rel._elim_states = out


def elimination_state(series, syms, coord):
    """For a coordinate that is the start of a state the elimination removed with A x + B = 0, A nonzero at a point
    of a trajectory: {"reads", "state"}, the coordinates that A and B read; None otherwise."""
    rel = _relations(series, syms)
    if not rel.ok:
        return None
    seg = rel.part["segs"][0]
    i = next((k for k, v in enumerate(seg["x0"]) if v[0] is not None and v[0] == coord), None)
    return None if i is None else getattr(rel, "_elim_states", {}).get(i)


def state_projection(series, syms, coord, rng, stop):
    """For a coordinate that is the start of a state x_i: the projection of x_i, a relation linear in x_i with the
    derivatives of the observables below a profile, checked as an identity of polynomials, with the coefficient of
    x_i nonzero at the start, as in state_relation(); {"reads", "state", "terms"} or None."""
    rel = _relations(series, syms)
    if not rel.ok:
        return None
    seg = rel.part["segs"][0]
    i = next((k for k, v in enumerate(seg["x0"]) if v[0] is not None and v[0] == coord), None)
    if i is None or i in rel.known or rel.X[i] is None:
        return None
    p0 = all_primes()[0]
    idx = [k for k, s_ in enumerate(syms) if s_ in set(series.params)]
    pos = [rel.names.index(str(syms[k])) for k in idx]
    for h in _profiles(rel, rng)[:2]:
        # the derivatives below the profile, then with the next derivative of one observable below the degree of
        # the projection, where the start is algebraic of higher degree over those below; small supports first,
        # since a search without a relation runs up to the cap
        lifts = ((r, d) for r in range(len(h)) for d in [_top_degree(rel, h, r, pos, rng, stop)] if d and d > 1)
        tries = itertools.chain([(None, ioproj.CAP // 4)], ((lift, ioproj.CAP) for lift in lifts), [(None, ioproj.CAP)])
        for lift, cap in tries:
            if work.ticks() > stop:
                return None
            pr = ioproj.Projection.state(rel, h, i, pos, rng, lift)
            if pr is None:
                continue
            S = pr.search([rng.randrange(1, p0) for _ in idx], p0, stop, cap=cap, top_degree=1)
            if S is None or len(S) > ioproj.TERMS or not any(m[pr.top] for m in S) or all(m[pr.top] for m in S):
                continue
            co = ioproj.Coefficients(pr, len(idx), rng, stop)
            if not co.run() or not ioproj.identity(rel, pr, co, pos):
                continue
            if not _nonzero_at_start(rel, pr, co, seg, syms, idx, pos, rng):
                continue
            exprs = co.expressions([syms[k] for k in idx])
            return {"reads": set().union(*[e.free_symbols for e in exprs]), "state": i, "terms": len(S)}
    return None


def _supports(rel):
    """{(profile, observable): support} of the projections found, None where none was found."""
    return rel.__dict__.setdefault("_supports", {})


def _top_degree(rel, h, r, pos, rng, stop):
    """The degree of the projection of observable r at profile h in the derivative on top, None where it is not
    found."""
    key = (tuple(h), r)
    pr = ioproj.Projection.output(rel, h, r, pos, rng)
    if pr is None:
        return None
    if key not in _supports(rel):
        p0 = all_primes()[0]
        _supports(rel)[key] = pr.search([rng.randrange(1, p0) for _ in pos], p0, stop)
    S = _supports(rel)[key]
    return None if not S else max(m[pr.top] for m in S)


def _nonzero_at_start(rel, pr, co, seg, syms, idx, pos, rng):
    """Whether the coefficient of the top state of a state projection is nonzero at the start of the first
    condition: a nonzero value at one point modulo a prime."""
    p = all_primes()[0]
    names = rel.names
    for _ in range(8):
        vals = {s_: rng.randrange(1, p) for s_ in syms}
        theta = [vals[syms[k]] for k in idx]
        pt = [rng.randrange(1, p) for _ in names]
        for k, v in zip(pos, theta):
            pt[k] = v
        ok = True
        for x, v in zip(rel.X, seg["x0"]):
            if x is None or str(x) not in names or v[0] is None:
                continue
            e = spy.sympify(v[0])
            num, den = spy.fraction(spy.together(e.xreplace({s_: vals[s_] for s_ in syms})))
            if int(den) % p == 0:
                ok = False
                break
            pt[names.index(str(x))] = int(num) % p * pow(int(den) % p, -1, p) % p
        a = ioproj.coefficient_values(co, theta, p) if ok else None
        if a is None:
            continue
        got = pr.base.values(np.array([pt], dtype=np.int64), p)
        if not got[0][0]:
            continue
        B = got[1][0]
        total = 0
        for c, m in zip(a, pr.mons):
            if not m[pr.top]:
                continue
            t = c
            for v, e in enumerate(m):
                if e and v != pr.top:
                    t = t * pow(int(B[v]), e, p) % p
            total += t
        return total % p != 0
    return False


def _normalized(w, P, syms):
    """(the proven function w over m, m) with m the largest monomial in the proven coordinates P that divides the
    numerator of w, over the denominator of w where that reads P only; None where m is one."""
    key = frozenset(P)
    if getattr(w, "_norm", (None,))[0] == key:
        return w._norm[1]
    Ps = sorted(P & w.num.free_symbols, key=str)
    m = spy.Integer(1)
    if Ps:
        mons = spy.Poly(w.num, *Ps).monoms()
        m = spy.prod([x ** min(mm[i] for mm in mons) for i, x in enumerate(Ps)])
    if w.den.free_symbols <= P:
        m = m / w.den
    res = None if m == 1 else (_Var(f"{w.name}n", expr=spy.cancel(w.expr / m), syms=syms), m)
    w._norm = (key, res)
    return res


def exact_fiber(series, orders, grp, syms, inputs=(), known=(), pre=()):
    """{"exact", "why", "generators", "seconds"}: whether the field fixed by the group `grp` of exact maps lies in
    the field of the output coefficients, so that the fiber at a generic point is the orbit (docs/method/global.rst,
    "The exact count"). The search runs in a separate process and ends at the caps of work.CAPS: search_entries,
    the entries of the matrices over GF(p) it reduces, and elimination_terms."""
    t0 = time.monotonic()
    out = {"exact": False, "why": None, "generators": [], "seconds": 0.0}
    if len(grp) > _GROUP:
        out["why"] = f"the group of the exact maps has more than {_GROUP} elements"
        return out
    jet = re.compile("|".join(re.escape(u) + r"(_d\d+)?" for u in inputs)) if inputs else None
    drawn = [str(x) for x in getattr(series, "drawn_starts", ())]
    if series.known or any(jet is None or not jet.fullmatch(x) for x in drawn):
        out["why"] = "the model holds constants that are no coordinates, taken at drawn values"
        return out
    seed = session.current().rng("exact fiber", tuple(series.coords)).getrandbits(64)
    from ..sym import msolve

    payload = pickle.dumps(
        (
            series,
            list(orders),
            grp,
            list(syms),
            list(inputs),
            seed,
            list(known),
            list(pre),
            msolve._MSOLVE_THREADS,
            work.MSOLVE_MEMORY,
            dict(work.CAPS),
        )
    )
    try:
        r = work.run(
            [sys.executable, "-c", "from symident.detection.fieldproof import _child; _child()"], input=payload
        )
        got = pickle.loads(r.stdout) if r.returncode == 0 and r.stdout else None
    except (pickle.UnpicklingError, EOFError):
        got = None
    if got is None:
        out["why"] = "the search failed"
        out["seconds"] = round(time.monotonic() - t0, 2)
        return out
    got["seconds"] = round(time.monotonic() - t0, 2)
    return got


def _child():
    """The search of exact_fiber() on the pickled input of stdin, the result pickled to stdout."""
    series, orders, grp, syms, inputs, seed, known, pre, threads, memory, caps = pickle.loads(sys.stdin.buffer.read())
    ioproj.THREADS[0] = threads
    work.MSOLVE_MEMORY = memory
    work.CAPS.update(caps)
    res = _exact_fiber(series, orders, grp, syms, caps["search_entries"], inputs, random.Random(seed), known, pre)
    sys.stdout.buffer.write(pickle.dumps(res))


def _exact_fiber(series, orders, grp, syms, cap, inputs, rng, known=(), pre=()):
    out = _search(series, orders, grp, syms, cap, inputs, rng, known, pre)
    out.setdefault("proven", [])
    # the supports of the projections past ioproj.TERMS, whose coefficients discrete.pinned_count() may count
    out["large"] = large_projections(series, syms)
    return out


def large_projections(series, syms):
    """[{"profile", "observable", "support"}] of the projections found with more than ioproj.TERMS terms."""
    key = (id(series), tuple(str(s_) for s_ in syms))
    if key not in _REL:
        return []
    rel = _REL[key][1]
    done = rel.__dict__.get("_eliminated", set())
    return [
        {"profile": list(h), "observable": i, "support": [tuple(int(x) for x in m) for m in S]}
        for (h, i), S in _supports(rel).items()
        if S and len(S) > ioproj.TERMS and i not in done
    ]


def _search(series, orders, grp, syms, cap, inputs, rng, known=(), pre=()):
    t0 = work.ticks()
    # the clock of the search: the entries of the matrices over GF(p) reduced, at most `cap`
    stop = t0 + cap
    ended = [f"the search reached the cap search_entries of {int(cap):,} entries of the matrices reduced"]
    out = {"exact": False, "why": None, "generators": [], "entries": 0}
    gens = generators(grp, syms)
    if gens is None:
        out["why"] = "no linear form of the coordinates separates the orbit"
        return out
    alpha, fs, traces, others = gens
    outs = _Outputs(series, orders)
    if not outs.index:
        out["why"] = "no observable at the start of an experiment"
        return out
    pool = _Pool(outs, len(syms), rng)
    a, b = pool.get(all_primes()[0], 2)
    # the coefficients that are numbers add nothing to the field
    W = [_Var(f"c{j}", c=j) for j in sorted(range(len(a[1])), key=lambda j: outs.index[j][1]) if a[1][j] != b[1][j]]
    # the known constants of the model: in the base field Q(kappa), with the output coefficients
    kappa = {spy.Symbol(k) for k in known}
    W += [_Var(str(k), expr=k, syms=syms) for k in sorted(kappa, key=str)]
    # functions an earlier search of this detection proved in the field of the outputs
    W += [_Var(f"p{n_}", expr=spy.sympify(e), syms=syms) for n_, e in enumerate(pre)]
    fs = [f for f in fs if f not in kappa]
    pending = sorted(fs, key=spy.count_ops)
    extra = (h for deg in (1, 2) for h in _reynolds(grp, syms, deg))
    log = []
    out["relations"], out["io_functions"] = log, []
    found = set()

    def relations(end):
        # the small dense configurations, the projections, the relations by elimination, the large dense
        # configurations
        return itertools.chain(
            io_functions(series, syms, rng, min(end, work.ticks() + _DENSE * cap), log, cap=_FIRST, found=found),
            projection_functions(series, syms, rng, end, log, found),
            elimination_functions(series, syms, rng, end, log, found),
            io_functions(series, syms, rng, end, log, floor=_FIRST, found=found),
        )

    io = relations(t0 + _SHARE * cap)
    tried = set()
    scaled = {}

    def subsystems(f):
        # the proven functions whose symbols outside the proven coordinates lie in those of f and at most two more,
        # with the proven coordinates they read: few variables, so that higher degrees fit
        P = {w.expr for w in W if w.c is None and w.expr.is_Symbol}
        sym_of = {w.expr: w for w in W if w.c is None and w.expr.is_Symbol}
        Uf = spy.sympify(f).free_symbols - P
        cand = [w for w in W if w.c is None and not w.expr.is_Symbol and w.free - P]
        others = sorted(set().union(*[w.free - P for w in cand]) - Uf, key=str)
        out = []
        p = all_primes()[0]
        pts = pool.get(p, 2)
        for k in range(3):
            for E in itertools.combinations(others, k):
                allowed = Uf | set(E)
                V, seen = [], set()
                for w in sorted((w for w in cand if (w.free - P) <= allowed), key=lambda w: len(w.tn) + len(w.td)):
                    # over the monomial in proven coordinates that divides it
                    nw = _normalized(w, P, syms)
                    if nw is not None:
                        scaled[id(nw[0])] = (w, nw[1])
                        w = nw[0]
                    # one of the functions that differ by a constant factor
                    vals = [w.value(x, c, p) for x, c in pts]
                    if None in vals or not vals[0]:
                        continue
                    key = vals[1] * pow(vals[0], -1, p) % p
                    if key not in seen:
                        seen.add(key)
                        V.append(w)
                V = V[:_SUBSYSTEM]
                if len(V) < len(allowed):
                    continue
                read = set().union(*[w.free for w in V]) & P
                Ps = [sym_of[x] for x in sorted(read, key=str)]
                if (V, Ps) not in out:
                    out.append((V, Ps))
        return out

    def products(nv, nP, cap):
        # monomials of degree dv in the functions times degree dp in the proven coordinates, fewest first
        pairs = sorted(
            (2 * math.comb(nv + dv, dv) * math.comb(nP + dp, dp), dv, dp) for dv in range(1, 4) for dp in range(0, 9)
        )
        for U, dv, dp in pairs:
            if U > cap:
                break
            mons = [a + b for a in _monomials(nv, dv) for b in _monomials(nP, dp)]
            yield sorted(mons, key=sum)

    def prove(f, coefficients, cap=_UNKNOWNS):
        # the functions proven on the coordinates of f first, then small systems of proven functions, then all
        # proven, then with the output coefficients
        support = spy.sympify(f).free_symbols
        near = [w for w in W if w.c is None and w.free <= support]
        cs = [w for w in W if w.c is not None]
        top = max((outs.index[w.c][1] for w in cs), default=0)
        # the output coefficients join lowest order first
        low = [near + [w for w in cs if outs.index[w.c][1] <= j] for j in range(top + 1)] if coefficients else []
        for V, Ps in subsystems(f):
            key = (f, tuple(id(w) for w in V + Ps), cap, "products")
            if key in tried:
                continue
            tried.add(key)
            for mons in products(len(V), len(Ps), cap):
                if work.ticks() > stop:
                    break
                got = _find(f, V + Ps, syms, outs, pool, 0, stop, mons)
                if got is not None:
                    # a proven function over a monomial in proven coordinates joins the tower before f
                    for v in V:
                        if id(v) in scaled and re.search(rf"\b{v.name}\b", got[0] + got[1]) and v not in W:
                            w, m = scaled[id(v)]
                            W.append(v)
                            out["proven"].append(str(v.expr))
                            out["generators"].append(
                                {"function": str(v.expr), "numerator": f"(1)*{w.name}", "denominator": str(m)}
                            )
                    return got
        sets = [near, [w for w in W if w.c is None]] + low + ([W] if coefficients and cap == _UNKNOWNS else [])
        for V in sets:
            key = (f, tuple(sorted(id(w) for w in V)), cap)
            if key in tried:
                continue
            tried.add(key)
            for d in range(1, 9):
                if not V or 2 * math.comb(len(V) + d, d) > cap or work.ticks() > stop:
                    break
                got = _find(f, V, syms, outs, pool, d, stop)
                if got is not None:
                    return got
        return None

    out["proven"] = [str(w_.expr) for w_ in W if w_.c is None and w_.expr not in kappa]

    def take(new):
        for h in new:
            W.append(_Var(f"r{len(W)}", expr=h, syms=syms))
            out["io_functions"].append(str(h))
            out["proven"].append(str(h))

    def add(f, got):
        out["proven"].append(str(f))
        out["generators"].append({"function": str(f), "numerator": got[0], "denominator": got[1]})
        W.append(_Var(f"f{len(out['generators'])}", expr=f, syms=syms))

    # each function proven joins the variables of the next: a tower of fields inside the field of the outputs
    def step(coefficients, cap=_UNKNOWNS):
        done = next(((f, got) for f in pending for got in [prove(f, coefficients, cap)] if got is not None), None)
        if done is not None:
            pending.remove(done[0])
            add(*done)
        return done is not None

    states = {}
    io_restart = [False]
    starts = {v[0] for part in series.parts for v in part["segs"][0].get("x0", []) if v[0] is not None}

    def from_state(size):
        # the start of a state: x_i(0) from a relation whose coefficients are rational in coordinates in F; small
        # relations first, the large ones once the input-output relations are done
        known = {w.expr for w in W if w.c is None and w.expr.is_Symbol}
        for f in list(pending):
            if not f.is_Symbol:
                continue
            # the start of a state the elimination removed with an equation of degree one in it
            es = elimination_state(series, syms, f)
            if es is not None and es["reads"] <= known:
                pending.remove(f)
                add(f, ("elimination", f"coefficients in {sorted(map(str, es['reads']))}"))
                return True
            if states.get(f, (None, 0))[1] < size:
                # the projection before the large dense relations
                sr = state_projection(series, syms, f, rng, stop) if size == _RELATION else None
                end = min(stop, work.ticks() + (_DENSE if size == _RELATION else _DENSE / 3) * cap)
                sr = sr or state_relation(series, syms, f, rng, end, size)
                states[f] = (sr, size)
            sr = states[f][0]
            if sr is not None and sr["reads"] <= known:
                pending.remove(f)
                add(f, ("state relation", f"coefficients in {sorted(map(str, sr['reads']))}"))
                # the state joins the observables, and the relations are searched again with it
                rel = _relations(series, syms)
                if sr["state"] not in rel.observed and rel.X[sr["state"]] not in rel.y and len(rel.y) < 6:
                    rel.observe(sr["state"])
                    io_restart[0] = True
                return True
        return False

    while pending and work.ticks() < stop:
        # small searches in the output coefficients before the next relations, the large ones after them
        if step(False) or from_state(_SMALL) or step(True, 300):
            continue
        # where only starts of states are left, their relations before the next input-output relations
        if all(f in starts for f in pending) and from_state(_RELATION):
            continue
        t1 = work.ticks()
        if io_restart[0]:
            io_restart[0] = False
            io = relations(stop)
        new = next(io, None)
        out["relation_entries"] = out.get("relation_entries", 0) + work.ticks() - t1
        if new is not None:
            take(new)
            continue
        if from_state(_RELATION):
            continue
        if not out["io_functions"]:
            # without input-output relations only the search in the output coefficients remains
            if work.ticks() + _WITHOUT_RELATIONS * cap < stop:
                stop = work.ticks() + _WITHOUT_RELATIONS * cap
                ended[0] = (
                    f"without input-output relations the search ends {int(_WITHOUT_RELATIONS * cap):,} "
                    "entries of the matrices reduced later"
                )
        if step(True):
            continue
        for h in extra:
            if h.is_number or any(w.expr is not None and spy.cancel(w.expr - h) == 0 for w in W):
                continue
            got = prove(h, True)
            if got is not None:
                add(h, got)
                break
        else:
            break
    if not pending:
        # the traces join F where they are found; they are not needed for the theorem
        for t in sorted(traces, key=spy.count_ops):
            got = prove(t, True, 300)
            if got is not None:
                add(t, got)
    if pending:
        out["entries"] = work.ticks() - t0
        out["why"] = f"{pending[0]} was not found as a rational function of the output coefficients" + (
            f" before the end of the search: {ended[0]}" if work.ticks() >= stop else ""
        )
        return out
    # the other coordinates in F(alpha): alpha joins the proven functions as a variable of these searches only
    if others:
        A = _Var("alpha", expr=alpha, syms=syms)
        # a tower in F(alpha): each coordinate proven there joins the variables of the next
        Fa = [A]

        def in_f_alpha(th):
            # the proven functions of the coordinates that the images of th and alpha read first
            S = set().union(spy.sympify(alpha).free_symbols, *[spy.sympify(g.get(th, th)).free_symbols for g in grp])
            near = [w for w in W if w.c is None and w.free <= S]
            cs = [w for w in W if w.c is not None]
            top = max((outs.index[w.c][1] for w in cs), default=0)
            low = [near + Fa + [w for w in cs if outs.index[w.c][1] <= j] for j in range(top + 1)]
            for V in [near + Fa] + low + [[w for w in W if w.c is None] + Fa, W + Fa]:
                key = (th, tuple(id(w) for w in V), "alpha")
                if key in tried:
                    continue
                tried.add(key)
                for d in range(1, 9):
                    if 2 * math.comb(len(V) + d, d) > _UNKNOWNS or work.ticks() > stop:
                        break
                    got = _find(th, V, syms, outs, pool, d, stop)
                    if got is not None:
                        return got
            return None

        proven_syms = set()
        for th in sorted(others, key=lambda e: sum(spy.count_ops(g.get(e, e)) for g in grp)):
            got = in_f_alpha(th)
            if got is None:
                # the start of a state: x_i(0) from a relation whose coefficients read coordinates in F(alpha)
                sr = state_relation(series, syms, th, rng, stop) or state_projection(series, syms, th, rng, stop)
                inF = {w.expr for w in W if w.c is None and w.expr.is_Symbol} | proven_syms
                if sr is not None and sr["reads"] <= inF:
                    got = ("state relation", f"coefficients in {sorted(map(str, sr['reads']))}")
            # more of F: the trace of th, then the next relations
            while got is None and work.ticks() < stop:
                tr = sum((spy.sympify(g.get(th, th)) for g in grp), spy.Integer(0))
                gt = (
                    prove(tr, True) if not any(w.expr is not None and spy.cancel(w.expr - tr) == 0 for w in W) else None
                )
                if gt is not None:
                    add(tr, gt)
                else:
                    new = next(io, None)
                    if not new:
                        break
                    take(new)
                got = in_f_alpha(th)
            if got is None:
                out["entries"] = work.ticks() - t0
                out["why"] = f"{th} was not found as a rational function of the proven functions and {alpha}"
                return out
            out["generators"].append(
                {"function": str(th), "numerator": got[0], "denominator": got[1], "over": f"F({alpha})"}
            )
            Fa.append(_Var(f"a{len(Fa)}", expr=th, syms=syms))
            proven_syms.add(th)
    out["alpha"] = str(alpha)
    out["entries"] = work.ticks() - t0
    out["exact"] = True
    return out
