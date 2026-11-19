"""Discrete symmetries: other coordinate values in the domain with the output of a generic one."""

import ast
import itertools
import math
import os
import re
import subprocess
import tempfile
from fractions import Fraction

import numpy as np
import sympy as spy

from .. import session
from ..errors import SYMPY_FAILS, sympy_fail
from ..gfp import PRIMES, rat_recon, rref_modp
from ..sym.domain import declared, is_positive
from ..sym.gfpoly import _gf_roots
from ..sym.modp import _modp_rational
from ..sym.parse import model_sympify

# total degree past which a coefficient of a characteristic polynomial is not interpolated
_DEG_CAP = 6
# sample points of the real count in the domain
_POINTS = 5
# orders past the fiber at which a reconstructed map is checked on the output, and raises of the order on a failure
_DEEPER = 4
_RAISES = 2
_NO_FINITE_COUNT = ("msolve found no finite count: the equations of the fiber have a component of positive dimension "
                    "up to order {order}, which the denominators and coordinates do not exclude, or msolve ran out "
                    "of time")


class BudgetError(Exception):
    """The reconstruction of the maps ran out of its time budget."""


# ---- the equations of the fiber ------------------------------------------------------------

class _Dual:
    """A value with its gradient over the coordinates, modulo p."""
    __slots__ = ("v", "d", "p")

    def __init__(self, v, d, p):
        self.v, self.d, self.p = v % p, d, p

    def _of(self, x):
        if isinstance(x, _Dual):
            return x
        x = Fraction(x)
        return _Dual(x.numerator * pow(x.denominator, -1, self.p), [0] * len(self.d), self.p)

    def __add__(self, x):
        x = self._of(x)
        return _Dual(self.v + x.v, [(a + b) % self.p for a, b in zip(self.d, x.d)], self.p)

    __radd__ = __add__

    def __neg__(self):
        return _Dual(-self.v, [(-a) % self.p for a in self.d], self.p)

    def __sub__(self, x):
        return self + (-self._of(x))

    def __rsub__(self, x):
        return self._of(x) - self

    def __mul__(self, x):
        x = self._of(x)
        p = self.p
        return _Dual(self.v * x.v, [(self.v * b + x.v * a) % p for a, b in zip(self.d, x.d)], p)

    __rmul__ = __mul__

    def __truediv__(self, x):
        x = self._of(x)
        p = self.p
        if x.v == 0:
            raise ZeroDivisionError("division by a value zero modulo p")
        iv = pow(x.v, -1, p)
        return _Dual(self.v * iv, [((a * x.v - self.v * b) * iv * iv) % p for a, b in zip(self.d, x.d)], p)

    def __rtruediv__(self, x):
        return self._of(x) / self

    def __pow__(self, e):
        out = self._of(1)
        for _ in range(e):
            out = out * self
        return out

    def __eq__(self, x):
        return self.v == self._of(x).v

    __hash__ = None


def _ser_mul(a, b, n, zero):
    """The product of two power series truncated after degree n."""
    return [sum((a[i] * b[k - i] for i in range(k + 1)), zero) for k in range(n + 1)]


def _ser_poly(P, xs, params, n, one, zero):
    """A polynomial {exponents: coefficient} in states and parameters with the states replaced by the series xs."""
    out = [zero] * (n + 1)
    cache = {}

    def power(i, e):
        if (i, e) not in cache:
            cache[(i, e)] = xs[i] if e == 1 else _ser_mul(power(i, e - 1), xs[i], n, zero)
        return cache[(i, e)]
    ns = len(xs)
    for m, c in P.items():
        coef = one * c
        for j, e in enumerate(m[ns:]):
            if e:
                coef = coef * params[j] ** e
        ser = [coef] + [zero] * n
        for i, e in enumerate(m[:ns]):
            if e:
                ser = _ser_mul(ser, power(i, e), n, zero)
        out = [u + v for u, v in zip(out, ser)]
    return out


class _Eps:
    """A power series in the formal gap variable eps, truncated after a fixed degree, over the coefficients of any
    ring: the gaps between events are eps times the given gaps, as in the local analysis."""
    __slots__ = ("c",)

    def __init__(self, c):
        self.c = list(c)

    def _lift(self, x):
        return x if isinstance(x, _Eps) else _Eps([x] + [x * 0] * (len(self.c) - 1))

    def __add__(self, x):
        x = self._lift(x)
        return _Eps([a + b for a, b in zip(self.c, x.c)])

    __radd__ = __add__

    def __neg__(self):
        return _Eps([-a for a in self.c])

    def __sub__(self, x):
        return self + (-self._lift(x))

    def __rsub__(self, x):
        return self._lift(x) - self

    def __mul__(self, x):
        if not isinstance(x, _Eps):
            return _Eps([a * x for a in self.c])
        return _Eps([sum((self.c[i] * x.c[k - i] for i in range(1, k + 1)), self.c[0] * x.c[k])
                     for k in range(len(self.c))])

    __rmul__ = __mul__

    def __truediv__(self, x):
        if not isinstance(x, _Eps):
            return _Eps([a / x for a in self.c])
        q = []
        for k in range(len(self.c)):
            q.append((self.c[k] - sum((x.c[i] * q[k - i] for i in range(1, k + 1)), self.c[0] * 0)) / x.c[0])
        return _Eps(q)

    def __rtruediv__(self, x):
        return self._lift(x) / self


def _lead(x):
    """The constant term in eps, which decides whether a value can be divided by."""
    return x.c[0] if isinstance(x, _Eps) else x


def _at_gap(xs, dt, gap, one, zero):
    """The state at the end of a gap dt * eps from its Taylor coefficients xs: sum_k xs[k] (dt eps)^k in eps."""
    c = [zero] * (gap + 1)
    w = one
    for k in range(min(gap, len(xs) - 1) + 1):
        xk = xs[k].c if isinstance(xs[k], _Eps) else [xs[k]]
        for m in range(k, gap + 1):
            if m - k < len(xk):
                c[m] = c[m] + xk[m - k] * w
        w = w * dt
    return _Eps(c)


def _shift(xs, d, order, gap, one, zero):
    """The Taylor coefficients up to `order` at the point d * eps away from the point of xs: coefficient k is
    sum_n C(n, k) xs[n] (d eps)^(n - k), exact in eps up to `gap` when xs reaches order + gap."""
    out = []
    for k in range(order + 1):
        c = [zero] * (gap + 1)
        w = one
        for n in range(k, min(len(xs), k + gap + 1)):
            xn = xs[n].c if isinstance(xs[n], _Eps) else [xs[n]]
            f = math.comb(n, k)
            for m in range(n - k, gap + 1):
                if m - (n - k) < len(xn):
                    c[m] = c[m] + xn[m - (n - k)] * w * f
            w = w * d
        out.append(_Eps(c))
    return out


def _coeffs(x):
    """The coefficients in eps of a value, one for a value of the first segment."""
    return x.c if isinstance(x, _Eps) else [x]


def _flow(F, xs, params, order, one, zero, skip=()):
    """The Taylor coefficients xs[i][k + 1] from coefficient k of D(x) x' = N(x), with D at the start nonzero; False
    at a pole. `skip` holds the states whose series is known."""
    for k in range(order):
        for i, (N, D) in enumerate(F):
            if i in skip:
                continue
            nser = _ser_poly(N, xs, params, k, one, zero)
            dser = _ser_poly(D, xs, params, k, one, zero)
            if _lead(dser[0]) == 0:
                return False
            rest = sum((dser[l] * (k + 1 - l) * xs[i][k + 1 - l] for l in range(1, k + 1)), zero)
            xs[i][k + 1] = (nser[k] - rest) / ((k + 1) * dser[0])
    return True


def _value(P, xs, params, one, zero):
    """A polynomial {exponents: coefficient} in states and parameters at the values xs of the states."""
    return _ser_poly(P, [[x] for x in xs], params, 0, one, zero)[0]


class Series:
    """The model in prolonged form, as in SIAN: the derivatives of the states at the start of each segment are
    unknowns, tied by the coefficients of D(x) x' = N(x) and E(x) y = M(x) as power series in t.

    `parts` holds one entry per condition, the coordinates shared: {"segments": [...]}, the first segment
    {"f", "g", "ic", "t", "label"}, each later one {"f", "g", "events", "t", "label"} with the events at its start
    time t ({"var", "method", "value"}, values read the states before the events). Past an event every
    coefficient is a power series in the formal variable eps, each gap being eps times the given gap
    (docs/method.rst, "Conditions and events"). A segment whose start holds coordinates is expanded at the start
    of the condition, so that its series depends on the time of the event."""

    def __init__(self, parts, coords):
        self.coords = list(coords)
        cs = [spy.Symbol(c) for c in self.coords]
        self.cs = cs
        segs = [seg for part in parts for seg in part["segments"]]
        texts = []
        for seg in segs:
            texts += list(seg["f"].values()) + list(seg["g"].values()) + \
                [v for v in (seg.get("ic") or {}).values() if v is not None] + [seg.get("on") or "0"]
            texts += [e["value"] for e in seg.get("events") or []] + [seg.get("t") or "0"]
        allsyms = set().union(*[model_sympify(str(e)).free_symbols for e in texts])
        states = {spy.Symbol(x) for seg in segs for x in seg["f"]}
        self.params = [c for c in cs if c not in states]
        rng = session.current().rng("discrete known", tuple(self.coords))
        # a symbol that is neither a state nor a coordinate is known: one value in a point and in the solutions
        self.known = {x: spy.Integer(rng.randint(2, 9)) for x in sorted(allsyms - states - set(cs), key=str)}
        self.parts = [self._part(part, rng) for part in parts]
        self.labels = [lab for part in self.parts for seg in part["segs"] for lab in seg["labels"]]
        # the rows of the copies with a free start: their values at sigma(theta) need the image of the start
        self.free_rows = {k for k, (part, seg, _) in enumerate((p, s_, y) for p in self.parts for s_ in p["segs"]
                                                               for y in s_["labels"]) if part["segs"][0].get("free")}
        self.has_events = any(len(part["segs"]) > 1 for part in self.parts)
        self._kser, self._jac = {}, {}

    def _part(self, part, rng):
        st = [spy.Symbol(x) for x in part["segments"][0]["f"]]
        gens = st + self.params
        cs = self.cs

        def frac(e):
            num, den = spy.fraction(spy.cancel(spy.together(model_sympify(str(e)).xreplace(self.known))))
            return _dict(num, gens), _dict(den, gens)
        def poly(text):
            e = spy.cancel(model_sympify(str(text)).xreplace(self.known))
            num, den = spy.fraction(e)
            if not den.is_number or (num.free_symbols - set(cs)):
                raise ValueError(f"discrete() needs event times polynomial in the coordinates, not {e}")
            return e
        t0 = poly(part["segments"][0].get("t", "0"))
        # each segment is expanded at its start, or, where the start holds coordinates, at the start of the
        # condition: there the series of a later start tells the time of the event
        anchor = t0
        out = {"st": st, "segs": []}
        for j, seg in enumerate(part["segments"]):
            if list(seg["f"]) != [str(x) for x in st]:
                raise ValueError("discrete(): the segments of a condition have other states")
            s = {"F": [frac(seg["f"][str(x)]) for x in st], "G": [frac(v) for v in seg["g"].values()],
                 "labels": [f"{seg.get('label', '')}{y}" for y in seg["g"]]}
            if j == 0:
                # the start of each state: a coordinate, a number, or a rational function of the coordinates; without
                # initial values (a copy switched into a regime) every state starts free, a number at the point
                # and a variable of its own in the fiber
                free = seg["ic"] is None or None in seg["ic"].values()
                X0 = {k: model_sympify(str(v)) for k, v in (seg["ic"] or {}).items() if v is not None}
                s["x0"], s["free"] = [], {}
                for i, x in enumerate(st):
                    if seg["ic"] is None or (str(x) in seg["ic"] and seg["ic"][str(x)] is None):
                        s["free"][i] = spy.Integer(rng.randint(2, 9))
                        s["x0"].append((None, None, None))
                        continue
                    v = spy.sympify(X0.get(str(x), x)).xreplace(self.known)
                    if x not in cs and v == x:
                        v = spy.Integer(rng.randint(2, 9))
                    num, den = spy.fraction(spy.cancel(v))
                    s["x0"].append((v, _dict(num, cs), _dict(den, cs)))
                # a copy switched into a regime starts on the surface it crosses: q(z, theta) = 0, solved at the point
                # for a state in which q is linear
                s["on"] = None
                if free and seg.get("on") is not None:
                    N, D = frac(seg["on"])
                    ns = len(st)
                    lin = [i for i in range(ns) if max(m[i] for m in N) == 1 and
                           all(not any(m[:i] + m[i + 1:ns]) or m[i] == 0 for m in N if m[i] == 1)]
                    lin = lin or [i for i in range(ns) if max(m[i] for m in N) == 1]
                    if not lin:
                        raise ValueError(f"discrete() needs the switching surface {seg['on']} = 0 linear in a state")
                    i = lin[0]

                    def drop(m, i=i):
                        return m[:i] + (0,) + m[i + 1:]
                    s["on"] = (i, {drop(m): c for m, c in N.items() if m[i] == 1},
                               {m: c for m, c in N.items() if m[i] == 0}, N, D)
                # a state with a numeric start whose right-hand side reads only such states (the jets of a known
                # input, the clock) has a known series: it enters as numbers, not as unknowns
                ns, known = len(st), set()
                while True:
                    more = {i for i in range(ns) if i not in known and s["x0"][i][0] is not None and
                            s["x0"][i][0].is_number and
                            all(not e or j_ in known for P in s["F"][i] for m in P for j_, e in enumerate(m[:ns])) and
                            all(not any(m[ns:]) for P in s["F"][i] for m in P)}
                    if not more:
                        break
                    known |= more
                out["known"] = known
            else:
                after = {}
                for e in seg["events"]:
                    x = spy.Symbol(e["var"])
                    v = model_sympify(str(e["value"]))
                    cur = after.get(x, x)
                    after[x] = v if e["method"] == "replace" else cur + v if e["method"] == "add" else cur * v
                s["event"] = {i: frac(after[x]) for i, x in enumerate(st) if x in after and after[x] != x}
                tj = poly(seg["t"])
                at = t0 if tj.free_symbols else tj
                # the gap from the expansion point before, and the start from the own expansion point
                s["dt"] = _dict(spy.expand(tj - anchor), cs)
                s["off"] = _dict(spy.expand(tj - at), cs) if tj.free_symbols else None
                anchor = at
            out["segs"].append(s)
        return out

    def _known_series(self, pi, order):
        """{state index: Taylor coefficients up to `order`} of the states of part `pi` with a known series."""
        key = (pi, order)
        if key not in self._kser:
            part = self.parts[pi]
            seg, known = part["segs"][0], part["known"]
            one, zero = Fraction(1), Fraction(0)

            def start(v):
                v = spy.Rational(v)
                return Fraction(int(v.p), int(v.q))
            xs = [[start(seg["x0"][i][0])] + [zero] * order if i in known else [zero] * (order + 1)
                  for i in range(len(part["st"]))]
            nop = [zero] * len(self.params)
            for k in range(order):
                for i in sorted(known):
                    N, D = seg["F"][i]
                    nser = _ser_poly(N, xs, nop, k, one, zero)
                    dser = _ser_poly(D, xs, nop, k, one, zero)
                    rest = sum((dser[l] * (k + 1 - l) * xs[i][k + 1 - l] for l in range(1, k + 1)), zero)
                    xs[i][k + 1] = (nser[k] - rest) / ((k + 1) * dser[0])
            self._kser[key] = {i: xs[i] for i in known}
        return self._kser[key]

    def _outputs(self, cvals, order, gap, one, zero):
        """Per observable of each segment of each condition, the Taylor coefficients up to `order`, each as its
        coefficients in eps up to `gap` (one for the first segment), the coordinates at `cvals` in the arithmetic
        of one and zero; None at a pole."""
        val = dict(zip(self.cs, cvals))
        params = [val[s] for s in self.params]
        out = []
        for pi, part in enumerate(self.parts):
            xs = None
            for j, seg in enumerate(part["segs"]):
                if j == 0:
                    kser = self._known_series(pi, order)
                    xs = []
                    for i, (_, num, den) in enumerate(seg["x0"]):
                        if i in kser:
                            xs.append([one * c for c in kser[i]])
                            continue
                        if i in seg["free"]:
                            v = seg["free"][i]
                            xs.append([one * Fraction(int(v.p), int(v.q))] + [zero] * order)
                            continue
                        d = _eval(den, cvals, zero)
                        if d == 0:
                            return None
                        xs.append([_eval(num, cvals, zero) / d] + [zero] * order)
                    if seg["on"] is not None:
                        i, A, B = seg["on"][:3]
                        start = [x[0] for x in xs]
                        a = _value(A, start, params, one, zero)
                        if a == 0:
                            return None
                        xs[i][0] = -_value(B, start, params, one, zero) / a
                    o, z = one, zero
                else:
                    o, z = _Eps([one] + [zero] * gap), _Eps([zero] * (gap + 1))
                    dt = _eval(seg["dt"], cvals, zero)
                    pre = [_at_gap(x, dt, gap, one, zero) for x in xs]
                    w0 = list(pre)
                    for i, (N, D) in seg["event"].items():
                        d = _value(D, pre, params, o, z)
                        if _lead(d) == 0:
                            return None
                        w0[i] = _value(N, pre, params, o, z) / d
                    kser = {}
                    if seg["off"] is not None:
                        # the series at the event, one coefficient in eps longer per order, then at the anchor
                        n = order + gap
                        xs = [[w] + [z] * n for w in w0]
                        if not _flow(seg["F"], xs, params, n, o, z):
                            return None
                        off = _eval(seg["off"], cvals, zero)
                        xs = [_shift(x, -off, order, gap, one, zero) for x in xs]
                    else:
                        xs = [[w] + [z] * order for w in w0]
                if (j == 0 or seg["off"] is None) and not _flow(seg["F"], xs, params, order, o, z, kser):
                    return None
                for M, E in seg["G"]:
                    mser = _ser_poly(M, xs, params, order, o, z)
                    eser = _ser_poly(E, xs, params, order, o, z)
                    if _lead(eser[0]) == 0:
                        return None
                    y = []
                    for k in range(order + 1):
                        y.append((mser[k] - sum((eser[l] * y[k - l] for l in range(1, k + 1)), z)) / eser[0])
                    out.append([_coeffs(v) for v in y])
        return out

    def numeric(self, point, order, gap=0, p=0):
        """The Taylor coefficients of the observables at `point` ({symbol: Rational}) up to `order` and `gap`, one
        list per observable and segment, as Fractions or, with p, as integers modulo p; None at a pole."""
        cv = [Fraction(int(spy.Rational(point[s]).p), int(spy.Rational(point[s]).q)) for s in self.cs]
        try:
            if not p:
                return self._outputs(cv, order, gap, Fraction(1), Fraction(0))
            out = self._outputs([_Dual(0, [], p) + c for c in cv], order, gap, _Dual(1, [], p), _Dual(0, [], p))
        except ZeroDivisionError:
            return None
        return None if out is None else [[[x.v for x in ym] for ym in y] for y in out]

    def output_orders(self, cap, p, gap=0):
        """Per observable the least order whose next Taylor coefficient adds no rank to the Jacobian in the
        coordinates, as in SIAN, at a random point modulo p; None where the rank stays below the number of
        coordinates up to `cap`."""
        # the coefficients up to an order depend on none above it: the series grows by doubling up to `cap`
        n = min(cap, 8)
        while True:
            out = self._output_orders(n, p, gap)
            if out != "short" or n >= cap:
                return None if out == "short" else out
            n = min(cap, 2 * n)

    def jacobian(self, order, p, gap=0):
        """The output coefficients up to `order` at the random point of output_orders modulo p, as duals with the
        gradient over the coordinates; None at a pole."""
        key = (p, gap)
        have = self._jac.get(key)
        if have is not None and have[0] >= order:
            return None if have[1] is None else [y[:order + 1] for y in have[1]]
        rng = session.current().rng("discrete orders", tuple(self.coords))
        m = len(self.cs)
        cv = [_Dual(rng.randint(2, p - 1), [int(i == j) for j in range(m)], p) for i in range(m)]
        one, zero = _Dual(1, [0] * m, p), _Dual(0, [0] * m, p)
        try:
            ys = self._outputs(cv, order, gap, one, zero)
        except ZeroDivisionError:
            ys = None
        self._jac[key] = (order, ys)
        return ys

    def _output_orders(self, cap, p, gap):
        m = len(self.cs)
        ys = self.jacobian(cap, p, gap)
        if ys is None:
            return None
        basis, piv = [], []

        def adds(row):
            row = list(row)
            for b, c in zip(basis, piv):
                if row[c]:
                    f = row[c]
                    row = [(x - f * y) % p for x, y in zip(row, b)]
            c = next((j for j, x in enumerate(row) if x), None)
            if c is None:
                return False
            inv = pow(row[c], -1, p)
            basis.append([x * inv % p for x in row])
            piv.append(c)
            return True
        orders = [0] * len(ys)
        active = list(range(len(ys)))
        for k in range(cap + 1):
            for j in list(active):
                if any([adds(v.d) for v in ys[j][k]]):
                    orders[j] = k
                elif k:
                    active.remove(j)
            if len(basis) == m or not active:
                break
        if len(basis) == m:
            return orders
        # an observable may add rank after an order that adds none (a state at rest moves later): each keeps the
        # last order that adds rank, until the rank is full
        basis.clear()
        piv.clear()
        orders = [0] * len(ys)
        for k in range(cap + 1):
            for j in range(len(ys)):
                if any([adds(v.d) for v in ys[j][k]]):
                    orders[j] = k
            if len(basis) == m:
                return orders
        return "short"

    def prolonged(self, orders, gap=0):
        """(names, equations, outputs, sat) up to the order of each observable in `orders` and `gap`: the jet
        variables after the coordinates, the equations of the model, the starts and the events as {exponents:
        Fraction} over coordinates and jets, the series (E, M) of each observable as E[l][a] and M[k][m] (Taylor
        coefficient, then coefficient in eps), and the factors that must not vanish."""
        order = max(orders)
        from sympy.polys.domains import QQ
        from sympy.polys.rings import ring
        names = [str(c) for c in self.cs]
        jets = {}
        for pi, part in enumerate(self.parts):
            kser = self._known_series(pi, order)
            for j, seg in enumerate(part["segs"]):
                for i in range(len(part["st"])):
                    if j == 0:
                        if i in kser:
                            continue
                        v = seg["x0"][i][0]
                        lo = 0 if v is None or (not (v.is_Symbol and v in self.cs) and not v.is_number) else 1
                        for k in range(lo, order + 1):
                            jets[(pi, j, i, k, 0)] = f"si_j{pi}_{j}_{i}_{k}"
                    else:
                        # the start after an event is a jet of its own where the event divides by the states, and
                        # at an anchor before the event
                        ev = seg["event"].get(i)
                        lo = 0 if seg["off"] is not None or (ev is not None and any(any(m) for m in ev[1])) else 1
                        for k in range(lo, order + 1):
                            for m in range(gap + 1):
                                jets[(pi, j, i, k, m)] = f"si_j{pi}_{j}_{i}_{k}_{m}"
        R, *gens = ring(",".join(names + list(jets.values())), QQ)
        g = dict(zip(names + list(jets.values()), gens))
        cof = {str(c): g[str(c)] for c in self.cs}
        params = [g[str(p)] for p in self.params]
        pad = (0,) * len(jets)

        def in_ring(P):
            return R.from_dict({mm + pad: c for mm, c in P.items()}) if P else R.zero
        eqs, outs = [], []
        sat = [cof[str(c)] for c in self.cs]

        def add_eq(e):
            eqs.extend(_coeffs(e))

        def add_sat(e):
            e = _lead(e)
            if not e.is_ground:
                sat.append(e)
        row = 0
        for pi, part in enumerate(self.parts):
            kser = self._known_series(pi, order)
            xs = None
            for j, seg in enumerate(part["segs"]):
                if j == 0:
                    xs = []
                    for i in range(len(part["st"])):
                        v, num, den = seg["x0"][i]
                        if i in kser:
                            xs.append([R(spy.Rational(c.numerator, c.denominator)) for c in kser[i]])
                            continue
                        if v is None:
                            # a free start: a variable without an equation
                            a0 = g[jets[(pi, 0, i, 0, 0)]]
                        elif (pi, 0, i, 0, 0) in jets:
                            a0 = g[jets[(pi, 0, i, 0, 0)]]
                            # the start is a polynomial in the coordinates; the jet variables follow them in R
                            add_eq(in_ring(den) * a0 - in_ring(num))
                        elif v.is_number:
                            a0 = R(spy.Rational(v))
                        else:
                            a0 = cof[str(v)]
                        xs.append([a0] + [g[jets[(pi, 0, i, k, 0)]] for k in range(1, order + 1)])
                    if seg["on"] is not None:
                        N, D = seg["on"][3:]
                        start = [x[0] for x in xs]
                        add_eq(_value(N, start, params, R.one, R.zero))
                        add_sat(_value(D, start, params, R.one, R.zero))
                    o, z = R.one, R.zero
                else:
                    o, z = _Eps([R.one] + [R.zero] * gap), _Eps([R.zero] * (gap + 1))
                    pre = [_at_gap(x, in_ring(seg["dt"]), gap, R.one, R.zero) for x in xs]
                    xs = []
                    for i in range(len(part["st"])):
                        def jet(k, i=i):
                            return _Eps([g[jets[(pi, j, i, k, m)]] for m in range(gap + 1)])
                        ws = [jet(k) for k in range(1, order + 1)]
                        w0 = pre[i]
                        if (pi, j, i, 0, 0) in jets:
                            w0 = jet(0)
                        # the state at the event: the series at the anchor where the start holds coordinates
                        at = _at_gap([w0] + ws, in_ring(seg["off"]), gap, R.one, R.zero) \
                            if seg["off"] is not None else w0
                        if i in seg["event"]:
                            N, D = seg["event"][i]
                            nv, dv = _value(N, pre, params, o, z), _value(D, pre, params, o, z)
                            if (pi, j, i, 0, 0) in jets:
                                add_eq(dv * at - nv)
                                add_sat(dv)
                            else:
                                c = next(iter(D.values()))
                                w0 = nv * R(spy.Rational(c.denominator, c.numerator))
                        elif seg["off"] is not None:
                            add_eq(at - pre[i])
                        xs.append([w0] + ws)
                for i, (N, D) in enumerate(seg["F"]):
                    if j == 0 and i in kser:
                        continue
                    nser = _ser_poly(N, xs, params, order - 1, o, z) if order else []
                    dser = _ser_poly(D, xs, params, order - 1, o, z) if order else []
                    for k in range(order):
                        add_eq(sum((dser[l] * (k + 1 - l) * xs[i][k + 1 - l] for l in range(0, k + 1)), z) - nser[k])
                    add_sat(_ser_poly(D, xs, params, 0, o, z)[0])
                for M, E in seg["G"]:
                    n = orders[row]
                    row += 1
                    mser = _ser_poly(M, xs, params, n, o, z)
                    eser = _ser_poly(E, xs, params, n, o, z)
                    add_sat(eser[0])
                    outs.append(([[_rdict(c) for c in _coeffs(e)] for e in eser],
                                 [[_rdict(c) for c in _coeffs(m)] for m in mser]))
        return list(jets.values()), [_rdict(e) for e in eqs if e], outs, [_rdict(f) for f in sat]


def _dict(e, syms):
    """A polynomial in `syms` as {exponents: Fraction}."""
    P = spy.Poly(spy.expand(e), *syms)
    return {m: Fraction(int(spy.Rational(c).p), int(spy.Rational(c).q)) for m, c in P.terms()}


def _rdict(P):
    return {m: Fraction(int(c.numerator), int(c.denominator)) for m, c in P.terms()}


def _clear(P, p=0):
    """{exponents: integer} of a polynomial with Fraction coefficients, times the lcm of the denominators, mod p."""
    L = math.lcm(*[c.denominator for c in P.values()]) if P else 1
    out = {m: c.numerator * (L // c.denominator) for m, c in P.items()}
    if p:
        out = {m: c % p for m, c in out.items()}
    return {m: c for m, c in out.items() if c}


class Fiber:
    """The points of the prolonged model whose observables have the Taylor coefficients of a given point."""

    def __init__(self, series, orders, positive=True, gap=0):
        self.series = series
        self.coords = list(series.coords)
        self.syms = [spy.Symbol(c) for c in self.coords]
        # one order per observable of each segment; a number is the same order for all
        self.orders = [int(orders)] * len(series.labels) if isinstance(orders, int) else [int(n) for n in orders]
        self.order = max(self.orders)
        # the coefficients in eps after an event; the state at the end of a gap needs its jets up to `gap`
        self.gap = min(int(gap), self.order) if series.has_events else 0
        self.positive = positive
        self.known = series.known
        self.names = [f"x{i}" for i in range(len(self.syms))]
        jets, static, outs, sat = series.prolonged(self.orders, self.gap)
        self.allnames = self.names + jets
        # the monomials as (variable, exponent) pairs: the system has hundreds of variables and few terms
        self.static = [_sparse(P) for P in static]
        self.outs = [([[_sparse(c) for c in row] for row in E], [[_sparse(c) for c in row] for row in M])
                     for E, M in outs]
        self.sat = []
        for f in (_sparse(_clear(f)) for f in sat):
            if f not in self.sat:
                self.sat.append(f)
        # per characteristic, whether msolve gets the separating form (decided at the first call)
        self.sep_mode = {}

    def values(self, point, p=0):
        """The Taylor coefficients of the observables at `point` ({symbol: Rational}), modulo p with p, None at a
        pole."""
        return self.series.numeric(point, self.order, self.gap, p)

    def system(self, point, p=0):
        """{"blocks", "fixed"}: the msolve input of the points with the coefficients of `point`, over Q (p = 0) or
        GF(p). The variables that a linear equation in one of them fixes to a number are substituted, `fixed`
        holds the coordinates among those. The rest splits into blocks without a shared variable in an equation
        or a factor of the saturation, each {"names", "sep", "eqs"} with `sep` its coordinates; the fiber is the
        product of the fibers of the blocks. None where the point has a pole or the system has no solution."""
        vals = self.values(point, p)
        if vals is None:
            return None
        polys = [_clear(P, p) for P in self.static]
        # coefficient k in t and m in eps of E(x) y = M(x), with y the values at the point
        for (E, M), b, n in zip(self.outs, vals, self.orders):
            for k in range(n + 1):
                for m in range(len(M[k])):
                    P = {mm: -c for mm, c in M[k][m].items()}
                    for l in range(k + 1):
                        for a in range(m + 1):
                            for mm, c in E[l][a].items():
                                P[mm] = P.get(mm, 0) + c * b[k - l][m - a]
                    polys.append(_clear({mm: c for mm, c in P.items() if c}, p))
        polys = [P for P in polys if P]
        sat = [{m: c % p if p else c for m, c in f.items()} for f in self.sat]
        red = _reduce(polys, sat, len(self.allnames), len(self.names), p)
        if red is None:
            return None
        polys, sat, fixed, gone = red
        keep = [i for i in range(len(self.allnames)) if i not in fixed and i not in gone]
        # blocks: the components of the variables joined by an equation or a factor of the saturation
        parent = {i: i for i in keep}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for P in polys + sat:
            vs = sorted({v for mm in P for v, _ in mm})
            for a, b_ in zip(vs, vs[1:]):
                parent[find(a)] = find(b_)
        comp = {}
        for i in keep:
            comp.setdefault(find(i), []).append(i)
        m = len(self.names)
        blocks = []
        for idx in comp.values():
            loc = set(idx)
            names = [self.allnames[i] for i in idx]
            eqs = [_ms_sparse(_clear(P, p), self.allnames) for P in polys if {v for mm in P for v, _ in mm} & loc]
            fac = {(): 1}
            for f in sat:
                if {v for mm in f for v, _ in mm} & loc:
                    fac = _smul(fac, _clear(f, p), p)
            # points where a coordinate or a denominator at the start vanishes do not count; msolve reads no
            # brackets, so si_w multiplies each term
            eqs.append(_ms_sparse(fac, self.allnames, "si_w") + "-1")
            blocks.append({"names": names, "sep": [self.names[i] for i in idx if i < m], "eqs": eqs})
        return {"blocks": blocks, "fixed": {self.names[i]: v for i, v in fixed.items() if i < m}}


def _sparse(P):
    """A polynomial {exponents: coefficient} with each monomial as (variable, exponent) pairs."""
    return {tuple((i, e) for i, e in enumerate(mm) if e): c for mm, c in P.items()}


def _smul(a, b, p):
    """The product of two sparse polynomials, modulo p unless p is 0."""
    out = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            d = dict(ma)
            for v, e in mb:
                d[v] = d.get(v, 0) + e
            mm = tuple(sorted(d.items()))
            out[mm] = (out.get(mm, 0) + ca * cb) % p if p else out.get(mm, 0) + ca * cb
    return {mm: c for mm, c in out.items() if c}


def _ms_sparse(P, names, extra=None):
    """A sparse polynomial as msolve input, expanded, each term times `extra` if given."""
    out = []
    for mm, c in P.items():
        mono = "*".join(names[v] + (f"^{e}" if e > 1 else "") for v, e in mm)
        if extra:
            mono = f"{mono}*{extra}" if mono else extra
        out.append(f"{c}*{mono}" if mono else f"{c}")
    return "+".join(out).replace("+-", "-") or "0"


def _reduce(polys, sat, nv, m0, p):
    """(polys, sat, fixed, gone) of the sparse system after the fixings of symFiberReduce (cpp/kernel.cpp): the
    variables that an equation a*v + b fixes, substituted in turn, and the jets from m0 on in no factor that one
    equation holds only as the monomial v, which leave with it; None without a solution. Modulo p in C++, over Q
    here."""
    if p:
        return _reduce_mod(polys, sat, nv, m0, p)
    return _reduce_q(polys, sat, nv, m0)


def _encode(polys):
    start, coef, tstart, var, ex = [0], [], [0], [], []
    for P in polys:
        for mm, c in P.items():
            coef.append(int(c))
            for v, e in mm:
                var.append(v)
                ex.append(e)
            tstart.append(len(var))
        start.append(len(coef))
    return tuple(np.asarray(x, dtype=np.int32) for x in (start, coef, tstart, var, ex))


def _decode(d):
    start, coef, tstart, var, ex = (list(map(int, d[k])) for k in ("start", "coef", "tstart", "var", "exp"))
    out = []
    for j in range(len(start) - 1):
        P = {}
        for t in range(start[j], start[j + 1]):
            P[tuple(zip(var[tstart[t]:tstart[t + 1]], ex[tstart[t]:tstart[t + 1]]))] = coef[t]
        out.append(P)
    return out


def _reduce_mod(polys, sat, nv, m0, p):
    from .. import _core
    out = _core.sym_fiber_reduce(_encode(polys), _encode(sat), nv, m0, float(p))
    if not out["ok"]:
        return None
    fixed = dict(zip(map(int, out["fixed_var"]), map(int, out["fixed_val"])))
    return _decode(out["polys"]), _decode(out["sat"]), fixed, set(map(int, out["gone"]))


def _reduce_q(polys, sat, nv, m0, p=0):
    """_reduce() in Python, over Q (p = 0) or GF(p)."""
    def div(a, b):
        return a * pow(b, -1, p) % p if p else Fraction(a) / b

    def subst(P, v, x):
        out, hit = {}, False
        for mm, c in P.items():
            e = dict(mm).get(v)
            if e:
                hit = True
                c = c * pow(x, e, p) % p if p else c * Fraction(x) ** e
                mm = tuple(t for t in mm if t[0] != v)
            out[mm] = (out.get(mm, 0) + c) % p if p else out.get(mm, 0) + c
        return {mm: c for mm, c in out.items() if c} if hit else P

    def linear_var(P):
        vs = {mm[0][0] for mm in P if mm}
        if len(vs) != 1 or any(mm and (len(mm) != 1 or mm[0][1] != 1) for mm in P):
            return None
        return vs.pop()
    polys, sat = [dict(P) for P in polys], [dict(f) for f in sat]
    alive = [bool(P) for P in polys]
    salive = [True] * len(sat)
    occ, socc = {}, {}
    for j, P in enumerate(polys):
        for v in {v for mm in P for v, _ in mm}:
            occ.setdefault(v, []).append(j)
    for j, f in enumerate(sat):
        for v in {v for mm in f for v, _ in mm}:
            socc.setdefault(v, []).append(j)
    fixed, queue = {}, list(range(len(polys)))
    while queue:
        j = queue.pop(0)
        if not alive[j]:
            continue
        v = linear_var(polys[j])
        if v is None or v in fixed:
            continue
        x = div(-polys[j].get((), 0), polys[j][((v, 1),)])
        fixed[v] = x
        for k in occ.get(v, ()):
            if not alive[k]:
                continue
            new = subst(polys[k], v, x)
            if new is polys[k]:
                continue
            polys[k] = new
            if not new:
                alive[k] = False
            elif list(new) == [()]:
                return None
            else:
                queue.append(k)
        for k in socc.get(v, ()):
            if not salive[k]:
                continue
            new = subst(sat[k], v, x)
            if new is sat[k]:
                continue
            sat[k] = new
            if not new:
                return None
            if list(new) == [()]:
                salive[k] = False
    if fixed:
        if any(a and list(P) == [()] for a, P in zip(alive, polys)):
            return None
        if any(a and not f for a, f in zip(salive, sat)):
            return None
        salive = [a and list(f) != [()] for a, f in zip(salive, sat)]
    polys = [P for a, P in zip(alive, polys) if a]
    sat = [f for a, f in zip(salive, sat) if a]
    insat = {v for f in sat for mm in f for v, _ in mm}
    count = {}
    for P in polys:
        for v in {v for mm in P for v, _ in mm}:
            count[v] = count.get(v, 0) + 1
    gone = set()
    while True:
        hit = None
        for j, P in enumerate(polys):
            for mm in P:
                if len(mm) == 1 and mm[0][1] == 1:
                    v = mm[0][0]
                    if v >= m0 and v not in insat and count.get(v) == 1 and \
                            sum(1 for nn in P if any(w == v for w, _ in nn)) == 1:
                        hit = (j, v)
                        break
            if hit:
                break
        if hit is None:
            break
        j, v = hit
        gone.add(v)
        for w in {w for mm in polys[j] for w, _ in mm}:
            count[w] -= 1
        polys.pop(j)
    return polys, sat, fixed, gone


def _eval(P, x, zero=Fraction(0)):
    return sum((c * math.prod((xi ** e for xi, e in zip(x, m) if e), start=zero + 1) for m, c in P.items()), zero)


def _msolve(names, p, eqs, flags, timeout, sep=()):
    """The parsed output of msolve on `eqs` in `names` + si_w over GF(p) (p = 0: Q), or None on failure. With
    `sep`, a last variable si_t is a linear form in these names with fixed coefficients: it separates distinct
    points, where si_w need not, so msolve parametrizes by it without trying other forms."""
    from ..sym import msolve
    flags = list(flags) + ["-t", str(msolve._MSOLVE_THREADS)]
    if p:
        # the probabilistic sparse linear algebra of msolve over GF(p): the counts are at random points anyway
        flags += ["-l", "44"]
    names = list(names) + ["si_w"]
    eqs = list(eqs)
    if sep:
        eqs.append("si_t" + "".join(f"-{3 + 2 * i}*{n}" for i, n in enumerate(sep)))
        names.append("si_t")
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.ms"), os.path.join(td, "out.ms")
        with open(fi, "w") as fh:
            fh.write(",".join(names) + "\n" + str(p) + "\n" + ",\n".join(eqs) + "\n")
        try:
            r = subprocess.run([msolve._msolve_binary(), "-f", fi, "-o", fo] + flags, capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        txt = ""
        if os.path.exists(fo):
            with open(fo) as fh:
                txt = fh.read()
    if r.returncode != 0 or not txt.strip():
        return None
    txt = txt.strip().rstrip(":")
    if p == 0:
        # dyadic interval ends a / 2^k as exact fractions
        txt = re.sub(r"(-?\d+)\s*/\s*2\^(\d+)", r"Fraction(\1, 2**\2)", txt)
        return eval(txt, {"Fraction": Fraction, "__builtins__": {}})
    return ast.literal_eval(txt)


# ---- counts --------------------------------------------------------------------------------

def real_points(fib, point, timeout, precision=64):
    """The real points of the fiber of `point` as boxes [(lo, hi)] per coordinate, None on failure."""
    sysd = fib.system(point)
    if sysd is None:
        return None
    per = []
    for block in sysd["blocks"]:
        at = {n: i for i, n in enumerate(block["names"])}
        for prec in (precision, 4 * precision, 16 * precision):
            data = _solve(fib, 0, block, ["-P", "0", "-p", str(prec)], timeout)
            if data is None or data[0] != 0:
                return None
            sols = [{n: tuple(sol[at[n]]) for n in block["sep"]} for sol in data[1][1]]
            # a box that straddles zero is refined
            if all(lo > 0 or hi < 0 for sol in sols for lo, hi in sol.values()):
                break
        per.append(sols)
    boxes = []
    for combo in itertools.product(*per):
        merged = {k: v for sol in combo for k, v in sol.items()}
        # a fixed coordinate takes its value on every point
        boxes.append([(sysd["fixed"][n], sysd["fixed"][n]) if n in sysd["fixed"] else merged[n] for n in fib.names])
    return boxes


def in_domain(fib, box):
    """True, False or None (undecided) for a box of a real point and the declared domain."""
    names, exprs = declared(fib.positive)
    for c, (lo, hi) in zip(fib.coords, box):
        if names is not None and c not in names:
            continue
        if hi < 0 or (lo <= 0 <= hi and hi - lo == 0):
            return False
        if lo <= 0:
            return None
    mid = {s: spy.Rational((lo + hi) / 2) for s, (lo, hi) in zip(fib.syms, box)}
    for h in exprs:
        v = spy.sympify(h).xreplace(mid).xreplace(fib.known)
        if not v.is_number:
            return None
        if v <= 0:
            return False
    return True


def _solve(fib, p, block, flags, timeout):
    """_msolve() on a block of Fiber.system(), with the separating form si_t only where the first call on the block
    fails without it: msolve may then try several forms when si_w does not separate the points, and the dense form
    costs more where it does."""
    names, eqs, sep = block["names"], block["eqs"], block["sep"]
    key = (p == 0, tuple(names))
    mode = fib.sep_mode.get(key)
    if mode is not None or not sep:
        return _msolve(names, p, eqs, flags, timeout, sep if mode else ())
    plain = _msolve(names, p, eqs, flags, timeout)
    if plain is not None:
        fib.sep_mode[key] = False
        return plain
    with_sep = _msolve(names, p, eqs, flags, timeout, sep)
    fib.sep_mode[key] = with_sep is not None
    return with_sep


def complex_count(fib, point, p, timeout):
    """The number of points of the fiber over the algebraic closure of GF(p), None on failure: per block the number
    of standard monomials of the leading ideal of a Gröbner basis, the dimension of the quotient, without the
    parametrization of the points."""
    try:
        pt = {s: spy.Integer(_modp_rational(spy.Rational(v), p)) for s, v in point.items()}
        sysd = fib.system(pt, p)
    except ZeroDivisionError:
        return None
    if sysd is None:
        return None
    n = 1
    for block in sysd["blocks"]:
        k = _gb_count(block["names"], p, block["eqs"], timeout, block["sep"])
        if not k:
            return None
        n *= k
    return n


def _gb_count(names, p, eqs, timeout, sep=(), cap=5000):
    """The number of points of the ideal of `eqs` in `names` + si_w over the algebraic closure of GF(p) with
    distinct values of the coordinates `sep`; None where it is not finite, the quotient above `cap`, or on failure.
    From the reduced Gröbner basis of msolve: the standard monomials span the quotient, the multiplication by a
    random linear form in `sep` acts on it, and the distinct eigenvalues of that matrix are the distinct values of
    the form on the points. A block without coordinates counts one point where it has any."""
    from ..sym import msolve
    names = list(names) + ["si_w"]
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.ms"), os.path.join(td, "out.ms")
        with open(fi, "w") as fh:
            fh.write(",".join(names) + "\n" + str(p) + "\n" + ",\n".join(eqs) + "\n")
        try:
            r = subprocess.run([msolve._msolve_binary(), "-f", fi, "-o", fo, "-g", "2", "-l", "44",
                                "-t", str(msolve._MSOLVE_THREADS)], capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        txt = ""
        if os.path.exists(fo):
            with open(fo) as fh:
                txt = fh.read()
    if r.returncode != 0:
        return None
    at = {n: i for i, n in enumerate(names)}
    body = "".join(line for line in txt.splitlines() if not line.startswith("#")).strip().rstrip(":").strip("[]")
    gb = [_gb_poly(t, at, p) for t in body.split(",") if t.strip()]
    if any(list(P) == [()] for P in gb):
        return 0
    lead = [next(iter(P)) for P in gb]
    basis = _standard_monomials([dict(m) for m in lead], len(names), cap)
    if basis is None:
        return None
    if not any(n in at for n in sep):
        return 1 if basis else 0
    rng = session.current().rng("discrete form", tuple(names))
    form = {((at[n], 1),): rng.randint(1, p - 1) for n in sep if n in at}
    index = {m: k for k, m in enumerate(basis)}
    M = []
    for b in basis:
        row = [0] * len(basis)
        for m, c in _gb_normal(_smul(form, {b: 1}, p), gb, lead, p).items():
            row[index[m]] = (row[index[m]] + c) % p
        M.append(row)
    return _distinct_eigenvalues(M, p, rng)


def _gb_poly(text, at, p):
    """A polynomial of the msolve output, leading term first, as {sparse monomial: coefficient}."""
    out = {}
    for term in text.strip().split("+"):
        c, mono = 1, {}
        for f in term.split("*"):
            f = f.strip()
            v, _, k = f.partition("^")
            if v in at:
                mono[at[v]] = mono.get(at[v], 0) + int(k or 1)
            else:
                c = c * int(f) % p
        key = tuple(sorted(mono.items()))
        out[key] = (out.get(key, 0) + c) % p
    return {m: c for m, c in out.items() if c}


def _divides(a, b):
    """Whether the sparse monomial a divides b."""
    d = dict(b)
    return all(d.get(v, 0) >= e for v, e in a)


def _gb_normal(P, gb, lead, p):
    """The normal form of P by the reduced Gröbner basis `gb` with leading monomials `lead`, all monic."""
    out = {}
    P = dict(P)
    while P:
        m = max(P, key=lambda mm: (sum(e for _, e in mm), _revlex(mm)))
        c = P.pop(m)
        j = next((j for j, l in enumerate(lead) if _divides(l, m)), None)
        if j is None:
            out[m] = (out.get(m, 0) + c) % p
            continue
        q = dict(m)
        for v, e in lead[j]:
            q[v] -= e
        q = tuple((v, e) for v, e in sorted(q.items()) if e)
        for mm, cc in _smul({q: c}, gb[j], p).items():
            if mm == m:
                continue
            P[mm] = (P.get(mm, 0) - cc) % p
            if not P[mm]:
                del P[mm]
    return {m: c for m, c in out.items() if c}


def _revlex(mm):
    """A sort key that orders monomials of one degree by grevlex, the first variable largest."""
    return tuple(-e for _, e in sorted(mm, key=lambda t: -t[0])) if mm else ()


def _distinct_eigenvalues(M, p, rng):
    """The number of distinct eigenvalues of the matrix M over the algebraic closure of GF(p): the degree of the
    square-free part of the minimal polynomial of M on a random vector."""
    n = len(M)
    v = [rng.randint(0, p - 1) for _ in range(n)]
    krylov, rows, piv = [], [], []
    while True:
        w = list(v)
        coef = [0] * (len(rows) + 1)
        coef[len(rows)] = 1
        for r, c in zip(rows, piv):
            if w[c]:
                f = w[c]
                w = [(x - f * y) % p for x, y in zip(w, r[0])]
                coef = [(x - f * y) % p for x, y in zip(coef, r[1] + [0] * (len(coef) - len(r[1])))]
        c = next((k for k, x in enumerate(w) if x), None)
        if c is None:
            minpoly = coef
            break
        inv = pow(w[c], -1, p)
        rows.append(([x * inv % p for x in w], [x * inv % p for x in coef]))
        piv.append(c)
        krylov.append(v)
        v = [sum(v[i] * M[i][j] for i in range(n)) % p for j in range(n)]
    f = minpoly
    df = [(k * f[k]) % p for k in range(1, len(f))]
    g = _gcd_mod(f, df, p)
    return len(f) - len(g)


def _gcd_mod(a, b, p):
    """The monic gcd of two polynomials mod p, coefficients from the constant term up."""
    def trim(x):
        while x and not x[-1]:
            x = x[:-1]
        return x
    a, b = trim(list(a)), trim(list(b))
    while b:
        inv = pow(b[-1], -1, p)
        while len(a) >= len(b) and a:
            f = a[-1] * inv % p
            off = len(a) - len(b)
            a = trim([(x - f * (b[k - off] if k >= off else 0)) % p for k, x in enumerate(a)])
        a, b = b, a
    inv = pow(a[-1], -1, p)
    return [x * inv % p for x in a]


def _standard_monomials(lead, nv, cap):
    """The monomials in nv variables that no monomial of `lead` ({variable: exponent}) divides, sparse; None where
    there are infinitely many or more than `cap`."""
    pure = {}
    for e in lead:
        if len(e) == 1:
            (v, k), = e.items()
            pure[v] = min(pure.get(v, k), k)
    if len(pure) < nv:
        return None
    free = [v for v in range(nv) if pure[v] > 1]
    lead = [e for e in lead if all(v in free for v in e)]
    seen, todo = {()}, [()]
    while todo:
        m = todo.pop()
        d = dict(m)
        for v in free:
            nxt = dict(d)
            nxt[v] = nxt.get(v, 0) + 1
            if any(all(nxt.get(u, 0) >= k for u, k in e.items()) for e in lead):
                continue
            key = tuple(sorted(nxt.items()))
            if key not in seen:
                seen.add(key)
                todo.append(key)
                if len(seen) > cap:
                    return None
    return sorted(seen)


def _boxes_of_maps(fib, pt, maps, check=True):
    """The points of the fiber at `pt` as exact boxes, the point and its images under `maps`, or None where an image
    is not distinct or, with `check`, the coefficients there differ from those at the point."""
    try:
        imgs = [dict(pt)] + [{s: spy.Rational(sig.get(s, s).xreplace(pt)) for s in fib.syms} for sig in maps]
    except (ZeroDivisionError, TypeError):
        return None
    vals = [tuple(im[s] for s in fib.syms) for im in imgs]
    if len(set(vals)) != len(vals):
        return None
    if not check:
        return [[(v, v) for v in t] for t in vals]
    here = fib.values(pt)
    for im in imgs[1:]:
        there = fib.values(im)
        if here is None or there is None or [r for k, r in enumerate(here) if k not in fib.series.free_rows] != \
                [r for k, r in enumerate(there) if k not in fib.series.free_rows]:
            return None
    return [[(v, v) for v in t] for t in vals]


def domain_point(fib, rng):
    """A random rational point with every declared coordinate positive."""
    return {s: spy.Rational(rng.randint(1, 60), rng.randint(1, 12)) for s in fib.syms}


# ---- the rational univariate representation over GF(p) -------------------------------------

def _rur(fib, point, p, timeout):
    """{"deg", "elim", "den", "par", "lin", "vnames"} of the fiber of `point` (integers mod p), None on failure."""
    try:
        pt = {s: spy.Integer(_modp_rational(spy.Rational(v), p)) for s, v in point.items()}
    except ZeroDivisionError:
        return None
    try:
        sysd = fib.system(pt, p)
    except ZeroDivisionError:
        return None
    if sysd is None:
        return None
    blocks = []
    for block in sysd["blocks"]:
        data = _solve(fib, p, block, ["-P", "1"], timeout)
        if data is None or data[0] != 0:
            return None
        _, _, deg, vnames, lin, par = data[1]
        if deg <= 0 or par[0] != 1:
            return None
        blocks.append({"deg": int(deg), "elim": [int(c) % p for c in par[1][0][1]],
                       "den": [int(c) % p for c in par[1][1][1]],
                       "par": [[int(c) % p for c in q[0][1]] for q in par[1][2]], "lin": [int(c) % p for c in lin],
                       "vnames": list(vnames)})
    return {"deg": math.prod(b["deg"] for b in blocks), "blocks": blocks,
            "fixed": {n: int(v) % p for n, v in sysd["fixed"].items()}}


def _pmod(a, f, p):
    """a mod f over GF(p), lowest degree first."""
    a = list(a)
    n = len(f) - 1
    inv = pow(f[-1], p - 2, p)
    while len(a) - 1 >= n and any(a):
        if a[-1] == 0:
            a.pop()
            continue
        c = a[-1] * inv % p
        s = len(a) - 1 - n
        for i in range(n + 1):
            a[s + i] = (a[s + i] - c * f[i]) % p
        a.pop()
    return a + [0] * (n - len(a))


def _pmul(a, b, f, p):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] = (out[i + j] + x * y) % p
    return _pmod(out, f, p)


def _pinv(a, f, p):
    """The inverse of a modulo f over GF(p), None if it has none."""
    n = len(f) - 1
    # solve a * x = 1 in the n-dimensional algebra as a linear system
    M = np.zeros((n, n + 1), dtype=np.int64)
    e = [1] + [0] * (n - 1)
    for j in range(n):
        col = _pmul(a, e, f, p)
        M[:, j] = col
        e = [0] + e[:-1]
    M[0, n] = 1
    R = rref_modp(M, p)
    if R["rank"] < n or n in list(R["piv"]):
        return None
    x = [0] * n
    for r, c in enumerate(R["piv"]):
        x[int(c)] = int(R["R"][r, n])
    return x


def charpolys(rur, p, names=None):
    """{variable: coefficients c_0..c_{K-1} of the monic characteristic polynomial} over the fiber, the product of
    the fibers of the blocks: a variable of a block of k points has its polynomial over the block to the power
    K/k; a fixed coordinate c has (Z - c)^K. None on failure."""
    K = rur["deg"]
    out = {}
    for b in rur["blocks"]:
        cp = _charpolys_block(b, p, names)
        if cp is None:
            return None
        e = K // b["deg"]
        for nm, cs in cp.items():
            poly = [1]
            for _ in range(e):
                poly = _plain_mul(poly, cs + [1], p)
            out[nm] = poly[:K]
    for nm, c in rur.get("fixed", {}).items():
        out[nm] = [math.comb(K, j) * pow(-c, K - j, p) % p for j in range(K)]
    return out


def _plain_mul(a, b, p):
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            out[i + j] = (out[i + j] + x * y) % p
    return out


def _charpolys_block(rur, p, names=None):
    """{variable: coefficients c_0..c_{k-1} of the monic characteristic polynomial} over the points of a block."""
    f, k = rur["elim"], rur["deg"]
    dinv = _pinv(rur["den"], f, p)
    if dinv is None:
        return None
    vn = rur["vnames"]
    elems = {}
    for j, q in enumerate(rur["par"]):
        elems[vn[j]] = [(-c) % p for c in _pmul(q, dinv, f, p)]
    # the last variable from the linear form t = sum lin_j x_j
    last = len(vn) - 1
    if rur["lin"][last] == 0:
        return None
    t = _pmod([0, 1], f, p)
    acc = list(t)
    for j in range(last):
        acc = [(a - rur["lin"][j] * b) % p for a, b in zip(acc, elems[vn[j]])]
    il = pow(rur["lin"][last], p - 2, p)
    elems[vn[last]] = [a * il % p for a in acc]
    out = {}
    for nm, el in elems.items():
        if names is not None and nm not in names:
            # the jets: the maps need the coordinates only
            continue
        # power sums tr(x^j) of multiplication by x, then Newton's identities
        basis = []
        e = [1] + [0] * (k - 1)
        for _ in range(k):
            basis.append(e)
            e = _pmul(e, [0, 1], f, p)
        M = np.array([_pmul(el, b, f, p) for b in basis], dtype=object).T
        P = np.identity(k, dtype=object)
        s = []
        for _ in range(k):
            P = np.array([[sum(int(P[i, l]) * int(M[l, j]) for l in range(k)) % p for j in range(k)]
                          for i in range(k)], dtype=object)
            s.append(sum(int(P[i, i]) for i in range(k)) % p)
        e_ = [1]
        for m in range(1, k + 1):
            v = sum(((-1) ** (i - 1)) * e_[m - i] * s[i - 1] for i in range(1, m + 1)) % p
            e_.append(v * pow(m, p - 2, p) % p)
        # Z^k - e1 Z^(k-1) + e2 Z^(k-2) - ...
        out[nm] = [((-1) ** (k - j)) * e_[k - j] % p for j in range(k)]
    return out


def fiber_points(rur, p):
    """The points of the fiber with coordinates in GF(p), as {variable: value}: the products of the points of
    the blocks, with the fixed coordinates."""
    out = []
    for combo in itertools.product(*[_fiber_points_block(b, p) for b in rur["blocks"]]):
        pt = {k: v for d in combo for k, v in d.items()}
        pt.update(rur.get("fixed", {}))
        out.append(pt)
    return out


def _fiber_points_block(rur, p):
    """The points of a block with coordinates in GF(p), as {variable: value}."""
    vn, last = rur["vnames"], len(rur["vnames"]) - 1

    def ev(coeffs, x):
        acc = 0
        for c in reversed(coeffs):
            acc = (acc * x + c) % p
        return acc
    out = []
    for t in _gf_roots(list(reversed(rur["elim"])), p):
        dv = ev(rur["den"], t)
        if dv == 0:
            continue
        dinv = pow(dv, p - 2, p)
        vals = {vn[j]: (-ev(q, t)) * dinv % p for j, q in enumerate(rur["par"])}
        rest = (t - sum(rur["lin"][j] * vals[vn[j]] for j in range(last))) % p
        vals[vn[last]] = rest * pow(rur["lin"][last], p - 2, p) % p
        out.append(vals)
    return out


# ---- interpolation of the coefficients -------------------------------------------------------

class _Sampler:
    """Characteristic polynomials of the fiber at integer points, memoized per point and prime."""

    def __init__(self, fib, timeout, budget=math.inf):
        import time
        self.fib, self.timeout, self.memo = fib, timeout, {}
        self.deadline = time.monotonic() + budget
        self.rng = session.current().rng("discrete points", tuple(fib.coords))
        self.k = None
        self.pool, self.moved = [], {}

    def point(self):
        return tuple(self.rng.randint(2, 10 ** 6) for _ in self.fib.syms)

    def points(self, n):
        """The first `n` points of one pool, shared by every coefficient."""
        while len(self.pool) < n:
            self.pool.append(self.point())
        return self.pool[:n]

    def moved_point(self, base, i):
        """`base` with coordinate i changed, the same for every coefficient."""
        if i not in self.moved:
            pt = list(base)
            pt[i] = self.rng.randint(2, 10 ** 6)
            self.moved[i] = tuple(pt)
        return self.moved[i]

    def rur(self, pt, p):
        """The representation of the fiber at `pt` modulo p, None at a point of another size."""
        import time
        key = (pt, p)
        if key not in self.memo:
            if time.monotonic() > self.deadline:
                raise BudgetError()
            point = dict(zip(self.fib.syms, [spy.Integer(v) for v in pt]))
            rur = _rur(self.fib, point, p, self.timeout)
            if rur is not None and self.k is not None and rur["deg"] != self.k:
                rur = None
            self.memo[key] = [rur, None, None]
        return self.memo[key][0]

    def at(self, pt, p):
        """The characteristic polynomials of the coordinates over the fiber at `pt` modulo p."""
        rur = self.rur(pt, p)
        m = self.memo[(pt, p)]
        if rur is not None and m[1] is None:
            m[1] = charpolys(rur, p, set(self.fib.names)) or False
        return m[1] or None

    def points_of(self, pt, p):
        """The points of the fiber at `pt` with coordinates in GF(p)."""
        rur = self.rur(pt, p)
        m = self.memo[(pt, p)]
        if rur is not None and m[2] is None:
            m[2] = fiber_points(rur, p)
        return m[2] or []


def _monomials(nv, d):
    return [m for t in range(d + 1) for m in _exps(nv, t)]


def _exps(nv, t):
    if nv == 0:
        return [()] if t == 0 else []
    return [(a,) + r for a in range(t, -1, -1) for r in _exps(nv - 1, t - a)]


def _support(oracle, sam, base, p):
    """The coordinates the value of `oracle` depends on, from changes of one coordinate at a time."""
    v0 = oracle(base, p)
    if v0 is None:
        return None
    out = []
    for i in range(len(base)):
        v1 = oracle(sam.moved_point(base, i), p)
        if v1 is None:
            return None
        if v1 != v0:
            out.append(i)
    return out


def interpolate(oracle, sam, base, primes, cap=_DEG_CAP):
    """The rational function of the coordinates with the values `oracle(point, p)`, None past degree `cap`."""
    sup = _support(oracle, sam, base, primes[0])
    if sup is None:
        return None
    syms = [sam.fib.syms[i] for i in sup]
    for d in range(cap + 1):
        mons = _monomials(len(sup), d)
        nu = 2 * len(mons)
        pts = sam.points(nu + 4)
        sols = []
        # one prime first, more only where the check at an independent prime fails
        for p in primes[:3]:
            rows = []
            for pt in pts:
                c = oracle(pt, p)
                if c is None:
                    continue
                vals = [math.prod(pow(pt[i] % p, e, p) for i, e in zip(sup, m)) % p for m in mons]
                rows.append(vals + [(-c * v) % p for v in vals])
            if len(rows) < nu + 2:
                break
            R = rref_modp(np.array(rows, dtype=np.int64), p)
            if nu - R["rank"] != 1:
                break
            free = [j for j in range(nu) if j not in set(int(x) for x in R["piv"])]
            vec = [0] * nu
            vec[free[0]] = 1
            for r, c in enumerate(R["piv"]):
                vec[int(c)] = (-int(R["R"][r, free[0]])) % p
            # the first nonzero coefficient of the denominator is one
            j0 = next((j for j in range(len(mons), nu) if vec[j]), None)
            if j0 is None:
                break
            inv = pow(vec[j0], p - 2, p)
            sols.append([x * inv % p for x in vec])
            r = _recon(sols, primes, mons, syms)
            if r is not None and _holds(oracle, sam, r, primes[-1]):
                return r
    return None


def _recon(sols, primes, mons, syms):
    """The rational function from its coefficient vectors modulo the first primes, None where none is bounded."""
    rec = rat_recon(np.array(sols, dtype=np.int64).T, primes[:len(sols)])
    if any(dd == "0" for dd in rec["den"]):
        return None
    coef = [spy.Rational(int(n), int(dd)) for n, dd in zip(rec["num"], rec["den"])]
    num = sum((c * spy.Mul(*[s ** e for s, e in zip(syms, m)]) for c, m in zip(coef[:len(mons)], mons)),
              spy.Integer(0))
    den = sum((c * spy.Mul(*[s ** e for s, e in zip(syms, m)]) for c, m in zip(coef[len(mons):], mons)),
              spy.Integer(0))
    return None if den == 0 else spy.cancel(num / den)


def _holds(oracle, sam, r, p):
    """Whether `r` matches `oracle` at two check points modulo p, apart from the pool."""
    if not hasattr(sam, "checks"):
        sam.checks = [sam.point() for _ in range(2)]
    for pt in sam.checks:
        c = oracle(pt, p)
        if c is None:
            return False
        try:
            v = _modp_rational(r.xreplace(dict(zip(sam.fib.syms, [spy.Integer(x) for x in pt]))), p)
        except ZeroDivisionError:
            return False
        if v != c % p:
            return False
    return True


# ---- the maps ------------------------------------------------------------------------------

def _rational_roots(coeffs, Z):
    """The roots in Q(theta) of Z^k + sum c_j Z^j, without repeats."""
    k = len(coeffs)
    poly = Z ** k + sum((c * Z ** j for j, c in enumerate(coeffs)), spy.Integer(0))
    num = spy.fraction(spy.together(poly))[0]
    out = []
    for fac, _ in spy.factor_list(num, Z)[1]:
        if spy.degree(fac, Z) == 1:
            a, b = spy.Poly(fac, Z).all_coeffs()
            out.append(spy.cancel(-b / a))
    return out


def _labels(sam, roots, pt, p):
    """{label: fiber point} at `pt`: the label holds, per separating coordinate, the index of the root its value
    matches; a point off every root lies on an irrational branch and has none. None where two roots meet."""
    point = dict(zip(sam.fib.syms, [spy.Integer(v) for v in pt]))
    try:
        vals = {s: [_modp_rational(r.xreplace(point), p) for r in rs] for s, rs in roots.items()}
    except ZeroDivisionError:
        return None
    if any(len(set(v)) < len(v) for v in vals.values()):
        return None
    out = {}
    for fp in sam.points_of(pt, p):
        lab = []
        for s, v in vals.items():
            x = fp[sam.fib.names[sam.fib.syms.index(s)]]
            lab.append(v.index(x) if x in v else None)
        if None not in lab:
            out[tuple(lab)] = fp
    return out


def rational_maps(fib, timeout, budget=math.inf):
    """(count, maps) of _rational_maps(), maps None where the budget of seconds runs out."""
    try:
        return _rational_maps(fib, timeout, budget)
    except BudgetError:
        return None, None


def _rational_maps(fib, timeout, budget):
    """(count, maps): the size of the fiber and the rational maps sigma onto other points of it, None if unknown.

    Coordinates with the fewest arguments come first: their characteristic polynomials over the fiber are
    interpolated and factored until the rational roots separate the points; the other coordinates are then
    interpolated on each branch."""
    sam = _Sampler(fib, timeout, budget)
    primes = list(PRIMES)
    base = sam.point()
    cp = sam.at(base, primes[0])
    if cp is None:
        return None, None
    k = len(next(iter(cp.values())))
    sam.k = k
    if k == 1:
        return 1, []
    Z = spy.Symbol("si_Z")

    def coeff(nm, j):
        def oracle(pt, p):
            c = sam.at(pt, p)
            return None if c is None else c[nm][j]
        return oracle
    size = {}
    for nm in fib.names:
        sup = set()
        for j in range(k):
            s_ = _support(coeff(nm, j), sam, base, primes[0])
            if s_ is None:
                return k, None
            sup |= set(s_)
        size[nm] = len(sup)
    roots, labels = {}, None
    for nm in sorted(fib.names, key=lambda n: (size[n], fib.names.index(n))):
        coeffs = [interpolate(coeff(nm, j), sam, base, primes) for j in range(k)]
        if any(c is None for c in coeffs):
            continue
        rs = _rational_roots(coeffs, Z)
        if len(rs) < 2:
            continue
        roots[fib.syms[fib.names.index(nm)]] = rs
        lab = _labels(sam, roots, sam.points(1)[0], primes[1])
        # separated: no point of a rational branch shares its label
        if lab is not None and len(lab) == sum(1 for fp in sam.points_of(sam.points(1)[0], primes[1])
                                                if _on_roots(sam, roots, fp, sam.points(1)[0], primes[1])):
            labels = lab
            break
    if labels is None:
        return k, None
    seps = list(roots)
    maps = []
    for lab in labels:
        sig = {s: roots[s][i] for s, i in zip(seps, lab)}
        if all(sig[s] == s for s in seps):
            continue

        def branch(nm, lab=lab):
            def oracle(pt, p):
                ls = _labels(sam, roots, pt, p)
                if not ls or lab not in ls:
                    return None
                return ls[lab][nm]
            return oracle
        for s, nm in zip(fib.syms, fib.names):
            if s in sig:
                continue
            r = interpolate(branch(nm), sam, base, primes)
            if r is None:
                return k, None
            sig[s] = r
        maps.append(sig)
    return k, maps


def _on_roots(sam, roots, fp, pt, p):
    """Whether every separating coordinate of the fiber point `fp` takes the value of one of its roots."""
    point = dict(zip(sam.fib.syms, [spy.Integer(v) for v in pt]))
    for s, rs in roots.items():
        x = fp[sam.fib.names[sam.fib.syms.index(s)]]
        if x not in {_modp_rational(r.xreplace(point), p) for r in rs}:
            return False
    return True


def output_preserved(fib, sig, points=3, deeper=0):
    """Whether the output coefficients agree at sigma(theta) and theta at random rational points, up to the order of
    the fiber plus `deeper`, in t and after an event in eps."""
    rng = session.current().rng("discrete check", tuple(fib.coords))
    order, gap = fib.order + deeper, (fib.gap + deeper if fib.series.has_events else 0)
    for _ in range(points):
        pt = {s: spy.Rational(rng.randint(1, 97), rng.randint(1, 13)) for s in fib.syms}
        img = {s: e.xreplace(pt) for s, e in sig.items()}
        a, b = fib.series.numeric(pt, order, gap), fib.series.numeric(img, order, gap)
        if a is not None and b is not None:
            # a copy with a free start keeps the output only from the image of its start, which is no coordinate
            a = [r for k, r in enumerate(a) if k not in fib.series.free_rows]
            b = [r for k, r in enumerate(b) if k not in fib.series.free_rows]
        if a is None or b is None or any(x != y for x, y in zip(a, b)):
            return False
    return True


def patterns(series, order, gap, p):
    """Per coordinate the output coefficients (row, order, order in eps) with a derivative in it that is not zero at
    the random point of the Jacobian modulo p; None at a pole. A permutation of the coordinates that keeps the output
    maps each coordinate onto one with the same set."""
    ys = series.jacobian(order, p, gap)
    if ys is None:
        return None
    out = [set() for _ in series.cs]
    for r, y in enumerate(ys):
        if r in series.free_rows:
            continue
        for k, v in enumerate(y):
            for m, c in enumerate(v):
                for i, d in enumerate(c.d):
                    if d:
                        out[i].add((r, k, m))
    return [frozenset(x) for x in out]


def candidates(series, order, gap, p, deadline):
    """The permutations of the coordinates that keep the output coefficients up to `order` (and `gap` in eps) at a
    random point modulo p: the transpositions within a class of equal patterns, then the products of two
    transpositions that fail alone. Each as {symbol: image} over all coordinates; the search stops at `deadline`."""
    import time
    pats = patterns(series, order, gap, p)
    if pats is None:
        return []
    groups = {}
    for i, x in enumerate(pats):
        if x:
            groups.setdefault(x, []).append(i)
    rng = session.current().rng("discrete candidates", tuple(series.coords))
    pt = [rng.randint(2, p - 1) for _ in series.cs]

    def vals(v):
        out = series.numeric(dict(zip(series.cs, v)), order, gap, p)
        return None if out is None else [r for k, r in enumerate(out) if k not in series.free_rows]

    def swapped(pairs):
        v = list(pt)
        for i, j in pairs:
            v[i], v[j] = v[j], v[i]
        return v
    base = vals(pt)
    if base is None:
        return []
    found, failed = [], []
    for idx in groups.values():
        for i, j in itertools.combinations(idx, 2):
            if time.monotonic() > deadline:
                return found
            (found if vals(swapped([(i, j)])) == base else failed).append([(i, j)])
    for a, b in itertools.combinations(failed, 2):
        if time.monotonic() > deadline:
            break
        if {*a[0]} & {*b[0]} or any(pats[a[0][0]] == pats[x] for x in b[0]):
            continue
        if vals(swapped(a + b)) == base:
            found.append(a + b)
    out = []
    for pairs in found:
        sig = {s: s for s in series.cs}
        for i, j in pairs:
            sig[series.cs[i]], sig[series.cs[j]] = series.cs[j], series.cs[i]
        out.append(sig)
    return out


def _block_map(F, H, X, x0, th):
    """phi affine in the states of the block that sigma reaches, the identity elsewhere: the states whose right-hand
    side or start holds a coordinate that sigma moves, closed under the states that read them. The coefficients
    solve the invariance of the right-hand sides, the outputs and the starts, where these are linear in them; None
    without a solution."""
    moved = {s for s, e in th.items() if e != s}
    reads = {x: F[x].free_symbols & set(X) for x in X}
    block = {x for x in X if F[x].free_symbols & moved or (x0 and x0[x].free_symbols & moved)}
    while True:
        more = {y for y in X if y not in block and reads[y] & block}
        if not more:
            break
        block |= more
    if not block or len(block) > 12:
        return None
    basis = sorted(block | {y for x in block for y in reads[x]}, key=str)
    a = {}
    phi = {x: x for x in X}
    for x in sorted(block, key=str):
        cs = [spy.Dummy(f"a_{x}_{y}") for y in basis] + [spy.Dummy(f"a_{x}")]
        a[x] = cs
        phi[x] = sum((c * y for c, y in zip(cs, basis)), cs[-1])
    unknowns = [c for cs in a.values() for c in cs]
    both = {**th, **phi}
    eqs = []

    def coeffs(e):
        num = spy.fraction(spy.together(e))[0]
        return spy.Poly(spy.expand(num), *X).coeffs() if num != 0 else []
    for x in block:
        rhs = sum((spy.diff(phi[x], y) * F[y] for y in X), spy.Integer(0))
        eqs += coeffs(F[x].xreplace(both) - rhs)
    eqs += [c for h in H for c in coeffs(h.xreplace(both) - h)]
    if x0:
        at0 = {y: x0[y] for y in X}
        eqs += [phi[x].xreplace(at0) - x0[x].xreplace(th) for x in block]
    eqs = [spy.expand(e) for e in eqs if e != 0]
    if any(spy.Poly(e, *unknowns).total_degree() > 1 for e in eqs):
        return None
    sol = spy.linsolve(eqs, unknowns)
    if not sol:
        return None
    vals = dict(zip(unknowns, next(iter(sol))))
    free = {u: 0 for v in vals.values() for u in v.free_symbols & set(unknowns)}
    return {x: spy.cancel(e.xreplace(vals).xreplace(free)) for x, e in phi.items()}


def _state_map(f, g, ic, sig, states):
    """A map phi of the states that takes the model at theta onto the model at sigma(theta), over Q(x, theta), or
    None. With the initial values `ic`, a state whose initial value is a coordinate of its own, the state itself
    or a parameter found nowhere else, takes the image of that coordinate with every such coordinate read as its
    state. Another state is mapped to itself, or, where that fails, to the solution of h(phi, sigma) = h(x, theta)
    for the output h and its Lie derivatives when these equations are linear in phi; with `ic`, phi must map the
    initial value at theta to the initial value at sigma(theta)."""
    F = {spy.Symbol(k): model_sympify(v) for k, v in f.items()}
    X = [spy.Symbol(x) for x in states]
    H = [model_sympify(h) for h in g.values()]
    used = set().union(*[model_sympify(v).free_symbols for v in list(f.values()) + list(g.values())])
    own = {}
    for x in X if ic is not None else ():
        q = model_sympify(str(ic.get(str(x), x)))
        if q == x or (q.is_Symbol and q in sig and q not in used and
                      sum(model_sympify(str(ic.get(str(y), y))) == q for y in X) == 1):
            own[q] = x
    phi = {x: x for x in X}
    for q, x in own.items():
        phi[x] = sig.get(q, q).xreplace(own)
    th = {s: e for s, e in sig.items() if s not in set(X) and s not in own}
    x0 = {x: model_sympify(str(ic.get(str(x), x))) for x in X} if ic is not None else {}
    given = [x for x in X if x not in own.values()]

    def holds(phi):
        both = {**th, **phi}
        for x in X:
            rhs = sum((spy.diff(phi[x], y) * F[y] for y in X), spy.Integer(0))
            if spy.cancel(F[x].xreplace(both) - rhs) != 0:
                return False
        if any(spy.cancel(F[x].xreplace(both) - F[x]) != 0 for x in F if x not in set(X)):
            return False
        if any(spy.cancel(h.xreplace(both) - h) != 0 for h in H):
            return False
        if ic is None:
            return True
        # phi at theta maps x0(theta) to x0(sigma(theta)) for the given initial values
        at0 = {y: x0[y] for y in given}
        return all(spy.cancel(phi[x].xreplace(at0) - x0[x].xreplace(th)) == 0 for x in given)
    try:
        if holds(phi):
            return phi
        blk = _block_map(F, H, X, x0, th)
        if blk is not None and holds(blk):
            return blk
        # the solve from the outputs grows with the number of states
        if not given or len(given) > 12:
            return None
        P = {x: spy.Dummy(str(x)) for x in given}
        img = {**th, **{x: phi[x] for x in X if x not in P}, **P}
        eqs, rows = [], list(H)
        for _ in range(len(given) + 1):
            eqs += [spy.fraction(spy.together(h.xreplace(img) - h))[0] for h in rows]
            rows = [sum((spy.diff(h, y) * F[y] for y in F), spy.Integer(0)) for h in rows]
        eqs = [spy.expand(e) for e in eqs]
        if any(e != 0 and spy.Poly(e, *P.values()).total_degree() > 1 for e in eqs):
            return None
        sol = spy.linsolve(eqs, list(P.values()))
        if not sol or len(sol.args) != 1 or sol.args[0].free_symbols & set(P.values()):
            return None
        phi = {**phi, **{x: spy.cancel(e) for x, e in zip(P, sol.args[0])}}
        return phi if holds(phi) else None
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None


def state_map_proof(parts, sig, states, positive=True):
    """The maps of the states, one per segment of each condition, that prove sigma a symmetry over Q(x, theta), or
    None: in each segment a map phi of the states
    (_state_map, with the initial values in the first segment), and at each event the image of the event at theta
    equal to the event at sigma(theta) of the image before it, phi_after(E(x)) = E_sigma(phi_before(x)). With
    switches, each regime has such a map, which takes every switching surface q = 0 to itself: q(phi(x),
    sigma(theta)) = c q(x, theta) with c positive on the domain and free of the states. A model with root events is
    not proven."""
    X = [spy.Symbol(x) for x in states]
    th = {s: e for s, e in sig.items() if s not in set(X)}
    if any(part.get("roots") for part in parts):
        return None
    maps = []
    try:
        for part in parts:
            qs = [model_sympify(str(q)) for q in part.get("surfaces") or ()]
            prev = None
            for j, seg in enumerate(part["segments"]):
                ic = seg["ic"] if j == 0 and None not in (seg["ic"] or {None: None}).values() else None
                phi = _state_map(seg["f"], seg["g"], ic, sig, states)
                if phi is None:
                    return None
                for q in qs:
                    c = spy.cancel(q.xreplace({**th, **phi}) / q)
                    if c.free_symbols & set(X) or not is_positive(c, positive):
                        return None
                if j:
                    after = {}
                    for e in seg["events"]:
                        x = spy.Symbol(e["var"])
                        v = model_sympify(str(e["value"]))
                        cur = after.get(x, x)
                        after[x] = v if e["method"] == "replace" else cur + v if e["method"] == "add" else cur * v
                    E = {x: after.get(x, x) for x in X}
                    for x in X:
                        lhs = phi[x].xreplace(E)
                        rhs = E[x].xreplace({**th, **{y: prev[y] for y in X}})
                        if spy.cancel(lhs - rhs) != 0:
                            return None
                prev = phi
                maps.append({str(x): str(e) for x, e in phi.items() if e != x})
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    return maps


def domain_conditions(fib, sig):
    """(preserved, conditions): True if sigma maps the domain into itself, False if never, else None with the
    conditions on theta."""
    names, exprs = declared(fib.positive)
    pos = {s: spy.Symbol(str(s), positive=True) for s in fib.syms if names is None or str(s) in names}
    conds = []
    for s, e in sig.items():
        if s not in pos:
            continue
        if is_positive(e.xreplace(fib.known), fib.positive):
            continue
        if (e.xreplace(fib.known).xreplace(pos)).is_nonpositive:
            return False, [f"{e} > 0"]
        conds.append(f"{spy.factor(e)} > 0")
    for h in exprs:
        img = spy.sympify(h).xreplace(sig)
        if not is_positive(img, fib.positive):
            conds.append(f"{img} > 0")
    return (True if not conds else None), conds


def fixed_set(sig):
    """The factors of the numerators of sigma(theta) - theta: where sigma fixes theta."""
    out = []
    for s, e in sig.items():
        if e == s:
            continue
        n = spy.fraction(spy.cancel(e - s))[0]
        out.append(n)
    if not out:
        return []
    g = out[0]
    for n in out[1:]:
        g = spy.gcd(g, n)
    return [str(fac) for fac, _ in spy.factor_list(g)[1]] if g.free_symbols else []


def fundamental_domain(coords, states, maps):
    """Per map the inequality theta_c < sigma_c(theta) on the first coordinate c it moves in one order of the
    coordinates, parameters with simple images first: theta is the least of its orbit in that order."""
    syms = [spy.Symbol(c) for c in coords]
    order = sorted(syms, key=lambda x: (str(x) in set(states), sum(spy.count_ops(sig.get(x, x)) for sig in maps),
                                        coords.index(str(x))))
    out = []
    for sig in maps:
        c = next(x for x in order if sig.get(x, x) != x)
        out.append(f"{c} < {sig[c]}")
    return list(dict.fromkeys(out))


def choose_order(series, positive, start, cap, timeout, budget=math.inf, floor=1):
    """(fiber, stable, count): the fiber at the orders of the observables that SIAN's rank criterion gives (else
    `start` for each), raised together by one until the count over the algebraic closure equals the count one step
    higher, at most to `cap`. While raising, one count may take a quarter of `budget`, since a fiber with a component
    of positive dimension costs msolve much more than the finite fiber one order higher. The search stops after
    `budget` seconds once a finite count is found, else after three times the budget; without any finite count the
    last fiber gets one count with the full `timeout`. A count equal to `floor`, the size of an orbit of proven
    maps, ends the search: a higher order only removes points, and the orbit stays; the first count then gets the
    full `timeout`."""
    import time
    t0 = time.monotonic()
    step = min(timeout, max(30.0, budget / 4))
    crit = series.output_orders(cap, PRIMES[0])
    # at the orders of the rank criterion the system is square and has extra solutions; one order higher it is
    # smaller in count and faster to solve. Every observable starts at the highest of these orders: an observable
    # whose next coefficient adds no rank in the coordinates can still remove points, where the start is given
    base = [max(crit) + 1] * len(crit) if crit else [max(1, start)] * len(series.labels)
    rng = session.current().rng("discrete order", tuple(series.coords))
    point = {spy.Symbol(c): spy.Integer(rng.randint(2, 10 ** 6)) for c in series.coords}
    prev = found = None
    for t in range(cap + 1):
        orders = [n + t for n in base]
        spent = time.monotonic() - t0
        if max(orders) > cap or (prev is not None and spent > (budget if found else 3 * budget)):
            break
        # the gap order rises with the Lie order, from zero gaps on
        fib = Fiber(series, orders, positive, gap=t)
        cc = complex_count(fib, point, PRIMES[0], timeout if floor > 1 and prev is None else step)
        if prev is not None and cc is not None and cc == prev[1]:
            return prev[0], True, cc
        if cc == floor:
            # a higher order adds equations, so the fiber there lies in this one and holds the orbit of the point
            return fib, True, cc
        prev = (fib, cc)
        if cc is not None:
            found = prev
    if found is not None:
        return found[0], False, found[1]
    if prev is None:
        return Fiber(series, base, positive), False, None
    return prev[0], False, complex_count(prev[0], point, PRIMES[0], timeout)


def _same_map(a, b):
    return all(spy.cancel(a[s] - b.get(s, s)) == 0 for s in a)


def group(maps, syms, cap=64):
    """The elements of the group the maps generate under composition, the identity first; None past `cap`."""
    ident = {s: s for s in syms}
    group, todo = [ident], [ident]
    while todo:
        g = todo.pop()
        for m in maps:
            h = {s: spy.cancel(m[s].xreplace(g)) for s in syms}
            if not any(_same_map(h, x) for x in group):
                group.append(h)
                todo.append(h)
                if len(group) > cap:
                    return None
    return group


def _candidate_stage(series, parts, states, positive, deadline):
    """(maps, info) of the candidates: permutations of coordinates with equal patterns that keep the output modulo a
    prime, each with the state maps that prove it or None."""
    import time
    t0 = time.monotonic()
    crit = series.output_orders(2 * len(series.coords), PRIMES[0])
    if crit is None:
        return [], {"seconds": round(time.monotonic() - t0, 2), "tested": False}
    order = max(crit) + 1 + _DEEPER
    gap = _DEEPER if series.has_events else 0
    found = candidates(series, order, gap, PRIMES[0], deadline)
    out = [(sig, state_map_proof(parts, sig, states, positive)) for sig in found]
    return out, {"order": order, "found": len(found), "proven": sum(pf is not None for _, pf in out),
                 "seconds": round(time.monotonic() - t0, 2), "complete_search": time.monotonic() <= deadline}


def analyze(parts, coords, positive, states, order=None, start=None, points=_POINTS, timeout=600, budget=300):
    """The discrete symmetries of a locally identifiable model, as a dict; `parts` as in Series, one per condition.
    First the candidates: permutations of coordinates whose output coefficients have the same pattern, kept modulo a
    prime and proven by a map of the states. Then the count: without `order`, the order rises from `start` (the Lie
    order of the rank) until the count over the algebraic closure is stable; after an event the gap order rises with
    it from zero. The maps of the fiber are reconstructed within `budget` seconds."""
    import time
    t0 = time.monotonic()
    series = Series(parts, coords)
    syms = [spy.Symbol(c) for c in coords]
    timing = {}
    limits = []
    cand, cinfo = _candidate_stage(series, parts, states, positive, t0 + budget / 3)
    timing["candidates"] = cinfo["seconds"]
    if cinfo.get("complete_search") is False:
        limits.append(f"the candidates were tested for {budget / 3:g} s, not all of them")
    t1 = time.monotonic()
    left = max(30.0, budget - (t1 - t0))
    cc = None
    proven = [sig for sig, pf in cand if pf is not None]
    cgrp = group(proven, syms) if proven else None
    if order:
        fib, stable = Fiber(series, int(order), positive, gap=int(order)), None
    else:
        fib, stable, cc = choose_order(series, positive, int(start or 1), 2 * len(coords), timeout, left,
                                       floor=len(cgrp) if cgrp else 1)
    timing["count"] = round(time.monotonic() - t1, 2)
    k, maps, false = None, None, []
    if cc is not None and cgrp is not None and len(cgrp) == cc and len(proven) == len(cand):
        # the proven candidates account for every point counted: the fiber has no other map to reconstruct
        k, maps = cc, []
        timing["maps"] = 0.0
    elif order or cc is not None:
        t2 = time.monotonic()
        # a map of the fiber that does not keep the coefficients further up shows a fiber too large at this order:
        # the orders rise and the count is taken again
        for _ in range(_RAISES + 1):
            # a count of one has no maps to find
            k, maps = (1, []) if cc == 1 else rational_maps(fib, timeout, left)
            k = cc if k is None else k
            false = [sig for sig in maps or [] if not output_preserved(fib, sig, deeper=_DEEPER)]
            if not false or order or time.monotonic() - t1 > left:
                break
            fib, stable, cc = Fiber(series, [n + 2 for n in fib.orders], positive, gap=fib.gap + 2), False, None
        timing["maps"] = round(time.monotonic() - t2, 2)
        if false:
            limits.append(f"{len(false)} of the maps of the fiber at order {fib.order} change the coefficients up "
                          f"to order {fib.order + _DEEPER}: the count is that of the equations up to order "
                          f"{fib.order}, more than the values with the same output")
            maps = [sig for sig in maps if all(sig is not f for f in false)]
        if maps is None and time.monotonic() - t1 > left:
            limits.append(f"the closed forms of the maps were not found within the budget of {budget:g} s")
    else:
        limits.append(_NO_FINITE_COUNT.format(order=fib.order))
    # the candidates join the maps of the fiber
    proofs = {}
    every = list(maps or [])
    for sig, pf in cand:
        same = next((m for m in every if _same_map(sig, m)), None)
        if same is None:
            every.append(sig)
            same = sig
        proofs[id(same)] = pf
    out = []
    for sig in every:
        entry = {"map": {str(s): str(e) for s, e in sig.items() if e != s},
                 "changes": [str(s) for s in syms if sig[s] != s]}
        pf = proofs[id(sig)] if id(sig) in proofs else state_map_proof(parts, sig, states, positive)
        entry["level"] = "exact" if pf is not None else "checked"
        entry["found_by"] = [w for w, hit in (("candidates", id(sig) in proofs),
                                               ("fiber", any(sig is m for m in maps or []))) if hit]
        entry["state_maps"] = pf
        entry["preserves_domain"], entry["domain_conditions"] = domain_conditions(fib, sig)
        entry["fixed_set"] = fixed_set(sig)
        out.append(entry)
    exact = [sig for sig, e in zip(every, out) if e["level"] == "exact"]
    grp = group(exact, syms)
    orbit = None if grp is None else len(grp)
    # the proven maps account for every point counted: the reconstruction of the fiber adds nothing
    complete = k is not None and not false and orbit == k and all(e["level"] == "exact" for e in out)
    counts = []
    if k is not None:
        # the counts in the domain end with the time of the count: budget and one call of msolve
        counts, skipped = _domain_counts(fib, maps, k, points, timeout, coords, grp[1:] if complete else None,
                                         deadline=t0 + budget + timeout)
        if skipped:
            limits.append(f"the counts in the domain stopped after {budget + timeout:g} s: {skipped} of "
                          f"{len(counts)} points without a count")
    if k is None:
        k = next((c["complex"] for c in counts if c.get("complex")), None)
    relevant = [e for e in out if e.get("preserves_domain") is not False and e["level"] != "open"]
    info = {"order": fib.order, "orders": dict(zip(series.labels, fib.orders)), "gap_order": fib.gap,
            "order_stable": stable, "count": k,
            "counts": counts, "limits": limits,
            "closed_forms": maps is not None,
            "count_final": complete,
            "fundamental_domain": fundamental_domain(coords, states, [
                {spy.Symbol(s): model_sympify(v) for s, v in e["map"].items()} for e in relevant]),
            "candidates": cinfo, "timing": timing,
            "certificate": certificate(out, orbit, k, counts, complete)}
    return {"symmetries": out, "info": info}


def _domain_counts(fib, maps, k, points, timeout, coords, proven=None, deadline=math.inf):
    """(counts, skipped): per point of the domain the count over the algebraic closure and the real points in the
    domain, and the number of points left without a count at `deadline`. With the elements `proven` of a group of
    proven maps whose orbit has `k` points, the fiber at a point is the orbit of the point."""
    import time
    rng = session.current().rng("discrete domain", tuple(coords))
    pts = [domain_point(fib, rng) for _ in range(points)] if isinstance(points, int) else \
        [{spy.Symbol(str(k_)): spy.Rational(str(v)) for k_, v in dict(pt).items()} for pt in points]

    def count_at(pt):
        if proven is not None:
            boxes = _boxes_of_maps(fib, pt, proven, check=False)
            if boxes is not None:
                return None, boxes
        left = deadline - time.monotonic()
        if left <= 0:
            return "skipped", None
        cc = complex_count(fib, {s: v for s, v in pt.items()}, PRIMES[0], min(timeout, left)) \
            if all(v.q % PRIMES[0] for v in pt.values()) else None
        # where the maps account for every point counted, the point and its images are the fiber, exactly
        boxes = _boxes_of_maps(fib, pt, maps) if maps is not None and cc == k == len(maps) + 1 else None
        left = deadline - time.monotonic()
        if boxes is None and left <= 0:
            return cc, None
        return cc, boxes if boxes is not None else real_points(fib, pt, min(timeout, left))
    from ..sym import msolve
    # msolve gains little from threads: the points run as processes of one thread each
    workers = max(1, min(int(msolve._MSOLVE_THREADS), len(pts)))
    with msolve.msolve_threads(1):
        if workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(workers) as ex:
                done = list(ex.map(count_at, pts))
        else:
            done = [count_at(pt) for pt in pts]
    counts, skipped = [], sum(cc == "skipped" for cc, _ in done)
    done = [(None if cc == "skipped" else cc, boxes) for cc, boxes in done]
    for pt, (cc, boxes) in zip(pts, done):
        if boxes is None:
            counts.append({"point": {str(s): str(v) for s, v in pt.items()}, "complex": cc, "real": None,
                           "in_domain": None})
            continue
        dom = [in_domain(fib, b) for b in boxes]
        counts.append({"point": {str(s): str(v) for s, v in pt.items()}, "complex": cc, "real": len(boxes),
                       "in_domain": sum(d is True for d in dom),
                       "undecided": sum(d is None for d in dom)})
    return counts, skipped


def certificate(entries, orbit, k, counts, complete):
    """What the result rests on: the invariance proven per map, the size of the orbit, and the count."""
    # a point of the domain whose fiber is read from the orbit has no count over the algebraic closure
    agree = [c["complex"] for c in counts if c.get("complex") is not None]
    return {
        "invariance": [{"map": e["map"], "proven": e["level"] == "exact", "state_maps": e.get("state_maps")}
                       for e in entries],
        "identities": "f(phi(x), sigma(theta)) = Dphi(x) f(x, theta), h(phi(x), sigma(theta)) = h(x, theta) and "
                      "phi(x0(theta)) = x0(sigma(theta)), in every segment and through every event, as identities "
                      "of rational functions over Q, checked by cancellation",
        "orbit": orbit, "count": k, "prime": PRIMES[0], "counts_at_points": agree,
        "proven": ["the invariance of every map at level exact",
                   f"the orbit of a generic point under the group of the exact maps has {orbit} points"],
        "axiom": "the number of solutions modulo the prime at a random point equals the number over the complex "
                 "numbers at a generic point; no error bound is proven",
        "complete": "axiom" if complete else ("no" if k is not None and orbit is not None and orbit < k else "open")}


def observed_states(f, g, events=()):
    """The states the observables depend on through the right-hand sides and the event values, in the order of
    `f`: a state outside never changes the output."""
    from ..rexpr import get_symbols
    reads = {x: set(get_symbols([str(e)])) for x, e in f.items()}
    for e in events:
        reads.setdefault(e["var"], set()).update(get_symbols([str(e["value"])]))
    seen, todo = set(), set(get_symbols([str(v) for v in g.values()])) & set(f)
    while todo:
        x = todo.pop()
        seen.add(x)
        todo |= (reads.get(x, set()) & set(f)) - seen
    return [x for x in f if x in seen]


def resting_start(f, ic, inputs, coords):
    """{state: resting value} of a condition at rest, the inputs at zero: a state that is a coordinate holds its free
    resting value, a state without dynamics rests at its initial value, and the others solve the balances over
    Q(theta), where they have one rational solution with the most states away from zero. A ValueError where they
    have none."""
    from ..sym.parse import model_sympify
    zero = {spy.Symbol(u): spy.Integer(0) for u in inputs}
    rhs = {x: model_sympify(str(e)).xreplace(zero) for x, e in f.items()}
    out = {x: "0" for x in f if x in inputs}
    for x, e in rhs.items():
        if x in out:
            continue
        if x in coords:
            out[x] = x
        elif e == 0:
            out[x] = str(ic.get(x, x))
    held = {spy.Symbol(x): model_sympify(v) for x, v in out.items()}
    U = [spy.Symbol(x) for x in f if x not in out]
    if not U:
        return out
    bal = [spy.fraction(spy.together(rhs[str(x)].xreplace(held)))[0] for x in U]
    try:
        sols = spy.solve(bal, U, dict=True)
    except (NotImplementedError,) + SYMPY_FAILS as exc:
        sympy_fail(exc)
        sols = []
    sols = [s for s in sols if set(s) >= set(U) and not any(v.has(spy.I) or v.free_symbols & set(U)
                                                               for v in s.values())]
    if sols:
        best = max(sum(s[x] != 0 for x in U) for s in sols)
        sols = [s for s in sols if sum(s[x] != 0 for x in U) == best]
    if len(sols) != 1 or not all(_rational(sols[0][x]) for x in U):
        raise ValueError("discrete() under equilibrate needs a resting state with one rational closed form in the "
                         f"coordinates; the balances of {', '.join(map(str, U))} have {len(sols)} such solutions")
    out.update({str(x): str(spy.cancel(sols[0][x])) for x in U})
    return out


def _rational(e):
    """Whether `e` is a rational function, no root or other function."""
    return all(p.exp.is_Integer for p in e.atoms(spy.Pow)) and not e.atoms(spy.Function)
