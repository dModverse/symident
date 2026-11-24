"""Discrete symmetries: other coordinate values in the domain with the output of a generic one."""

import ast
import itertools
import math
import os
import re
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction

import numpy as np
import sympy as spy
from sympy.polys.domains import QQ
from sympy.polys.rings import ring
from sympy.utilities.iterables import strongly_connected_components

from .. import _core, session, state
from ..errors import SYMPY_FAILS, sympy_fail
from ..gfp import PRIMES, rat_recon, rref_modp
from ..rexpr import get_symbols
from ..sym import msolve
from ..sym.degrees import Atoms, Deg, common
from ..sym.domain import declared, is_positive
from ..sym.gfpoly import _gf_roots
from ..sym.modp import _modp_rational
from ..sym.parse import model_sympify
from . import startpow, work

# total degree past which a coefficient of a characteristic polynomial is not interpolated
_DEG_CAP = 6
# sample points of the real count in the domain
_POINTS = 5
# orders past the fiber at which a reconstructed map is checked on the output, and raises of the order on a failure
_DEEPER = 4
_RAISES = 2
# the raises of the orders of the count by one past the first, where the count is not yet stable
_ORDER_RAISES = 4
# the most terms of the equations of the fiber that a count reduces and hands to msolve
_FIBER_TERMS = 400000
# the most unknowns of the prolonged model, coordinates and jets, that a count takes: each monomial of the sparse ring
# stores one exponent per unknown, so n unknowns take n^2 words before the first equation
_FIBER_UNKNOWNS = 10000
# the most distinct points of a stage over Q whose rational roots are sought
_STAGE_ROOTS = 16
# the most points over Q whose projection is counted exactly
_PROJ_POINTS = 64
# the error bound the verified count aims at, and the least and most bits of a coordinate of the point drawn
_VERIFIED_AIM = 1e-6
_MIN_BITS = 64
_VERIFIED_BITS = 4096
# the most states whose map is solved from the outputs: the solve grows with the number of states
_SOLVE_STATES = 12


def _fiber_terms():
    """The most terms of the equations of the fiber: the switch FIBER_TERMS, else _FIBER_TERMS."""
    n = state.env_value("FIBER_TERMS")
    return int(n) if n not in (None, "") else _FIBER_TERMS


# why a call of msolve gave no count, by kind, since the last reset
_FAILURES: set = set()
# the work caps that stopped a proof of a map since the last reset, as reasons
_CAPPED: set = set()
_FAILED = {
    "memory": "msolve ran out of memory ({gib:g} GiB of address space per call)",
    "size": "a system past the caps of msolve ({unknowns} unknowns, {terms:,} terms)",
    "work": "msolve reached the cap msolve_entries of {entries:,} nonzero entries of the matrices of F4",
    "dimension": "the equations of the fiber have a component of positive dimension up to order {order}, which "
    "the denominators and coordinates do not exclude",
    "quotient": "the quotient of a block has more than {cap} standard monomials",
    "error": "msolve stopped with an error",
}
_NO_FINITE_COUNT = "msolve found no finite count: " + _FAILED["dimension"]


def _failed(r=None, kind=None):
    """Records why a call of msolve gave no count: `kind`, or the kind of the failed process `r`. A process ended by
    a signal ran out of the address space: msolve does not check the allocations."""
    if kind is None and getattr(r, "capped", None):
        kind = "work"
    if kind is None:
        err = (r.stderr or "") if isinstance(r.stderr, str) else ""
        kind = "memory" if r.returncode < 0 or "alloc" in err.lower() or "memory" in err.lower() else "error"
    _FAILURES.add(kind)


def no_count_reason(order):
    """The limit of a count without result, from the failures recorded since the last reset."""
    if not _FAILURES:
        return _NO_FINITE_COUNT.format(order=order)
    gib = work.MSOLVE_MEMORY / (1 << 30)
    cap = work.cap
    why = [
        _FAILED[k].format(
            order=order,
            gib=gib,
            cap=5000,
            unknowns=cap("msolve_unknowns"),
            terms=cap("msolve_terms"),
            entries=int(cap("msolve_entries")),
        )
        for k in ("memory", "work", "size", "dimension", "quotient", "error")
        if k in _FAILURES
    ]
    return f"msolve found no finite count up to order {order}: " + "; ".join(why)


class CapError(Exception):
    """A stage reached the work cap: the points of the reconstruction of the maps, or the work of the Taylor series of
    the outputs."""


# ---- the equations of the fiber ------------------------------------------------------------


class FiberTooLargeError(Exception):
    """The equations of the fiber have more than _FIBER_TERMS terms, or more than _FIBER_UNKNOWNS unknowns (the
    arguments n, "unknowns"): no count is taken."""


class _Dual:
    """A value and the gradient over the coordinates, modulo p."""

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

    __hash__ = None  # type: ignore[assignment]


def _ser_mul(a, b, n, zero, limit=None):
    """The product of two power series truncated after degree n. With `limit`, FiberTooLargeError where the
    coefficients of the product have more terms; no partial product is formed past the limit."""
    if limit is None:
        return [sum((a[i] * b[k - i] for i in range(k + 1)), zero) for k in range(n + 1)]
    out, done = [], 0
    for k in range(n + 1):
        acc = zero
        for i in range(k + 1):
            acc = _mul_add(acc, a[i], b[k - i], limit - done)
        done += _terms(acc)
        out.append(acc)
    return out


def _mul_add(acc, x, y, room):
    """acc + x*y with at most `room` terms, else FiberTooLargeError. A product of sparse polynomials whose term
    bound exceeds the room is formed in slices of x."""
    if isinstance(acc, _Eps) or isinstance(x, _Eps) or isinstance(y, _Eps):
        L = len(next(v for v in (acc, x, y) if isinstance(v, _Eps)).c)
        acc, x, y = (v.c if isinstance(v, _Eps) else [v] + [v * 0] * (L - 1) for v in (acc, x, y))
        c = list(acc)
        for k in range(L):
            for i in range(k + 1):
                c[k] = _mul_add(c[k], x[i], y[k - i], room - sum(_terms(u) for j, u in enumerate(c) if j != k))
        return _Eps(c)
    if not isinstance(x, dict) or not isinstance(y, dict) or len(x) * len(y) <= room:
        acc = acc + x * y
    else:
        items, step = list(x.items()), max(1, room // max(1, len(y)))
        for s in range(0, len(items), step):
            acc = acc + x.ring.dtype(items[s : s + step]) * y
            if _terms(acc) > room:
                raise FiberTooLargeError(room)
    if _terms(acc) > room:
        raise FiberTooLargeError(room)
    return acc


def _terms(x):
    """The terms of a sparse polynomial, summed over the coefficients of a series in eps."""
    return sum(_terms(c) for c in x.c) if isinstance(x, _Eps) else len(x) if isinstance(x, dict) else 1


def _ser_poly(P, xs, params, n, one, zero, limit=None):
    """A polynomial {exponents: coefficient} in states and parameters with the states replaced by the series xs.
    With `limit`, for series of sparse polynomials, FiberTooLargeError where a power, a product or the sum of the
    series has more terms."""
    out = [zero] * (n + 1)
    cache = {}

    def check(ser):
        if limit is not None and sum(_terms(t) for t in ser) > limit:
            raise FiberTooLargeError(limit)
        return ser

    def power(i, e):
        if (i, e) not in cache:
            cache[(i, e)] = check(xs[i] if e == 1 else _ser_mul(power(i, e - 1), xs[i], n, zero, limit))
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
                ser = _ser_mul(ser, power(i, e), n, zero, limit)
        out = check([u + v for u, v in zip(out, ser)])
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
        return _Eps(
            [sum((self.c[i] * x.c[k - i] for i in range(1, k + 1)), self.c[0] * x.c[k]) for k in range(len(self.c))]
        )

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
    """The state at the end of a gap dt * eps from the Taylor coefficients xs: sum_k xs[k] (dt eps)^k in eps."""
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


def _flow(F, xs, params, order, one, zero, skip=(), meter=None, weight=1):
    """The Taylor coefficients xs[i][k + 1] from coefficient k of D(x) x' = N(x), with D at the start nonzero; False
    at a pole. `skip` holds the states whose series is known. Coefficient k spends (k + 1)^2 times `weight` times the
    terms of F on `meter`; CapError past the cap."""
    size = sum(len(N) + len(D) for N, D in F) * weight
    for k in range(order):
        if meter is not None and meter.spend(size * (k + 1) ** 2):
            raise CapError()
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
    """The model in prolonged form, as in SIAN: the derivatives of the states at the start of each segment are unknowns,
    tied by the coefficients of D(x) x' = N(x) and E(x) y = M(x) as power series in t.

    `parts` holds one entry per condition, the coordinates shared: {"segments": [...]}, the first segment {"f", "g",
    "ic", "t", "label"}, each later one {"f", "g", "events", "t", "label"} with the events at the start time t ({"var",
    "method", "value"}, values read the states before the events). Past an event every coefficient is a power series in
    the formal variable eps, each gap being eps times the given gap (docs/method/conditions.rst, "Conditions and
    events"). A segment whose start holds coordinates is expanded at the start of the condition, so that the series
    depends on the time of the event."""

    # the Taylor series of jacobian() spend on `meter`; `capped` holds the order not formed within the cap
    meter = capped = None

    def __init__(self, parts, coords, positive=True):
        self.coords = list(coords)
        self.positive = positive
        cs = [spy.Symbol(c) for c in self.coords]
        self.cs = cs
        segs = [seg for part in parts for seg in part["segments"]]
        texts = []
        for seg in segs:
            texts += (
                list(seg["f"].values())
                + list(seg["g"].values())
                + [v for v in (seg.get("ic") or {}).values() if v is not None]
                + [seg.get("on") or "0"]
            )
            texts += [e["value"] for e in seg.get("events") or []] + [seg.get("t") or "0"]
        allsyms = set().union(*[model_sympify(str(e)).free_symbols for e in texts])
        states = {spy.Symbol(x) for seg in segs for x in seg["f"]}
        self.params = [c for c in cs if c not in states]
        rng = session.current().rng("discrete known", tuple(self.coords))
        # a symbol that is neither a state nor a coordinate is known: one value in a point and in the solutions
        self.known = {x: spy.Integer(rng.randint(2, 9)) for x in sorted(allsyms - states - set(cs), key=str)}
        # the states whose start is neither given nor a coordinate, taken at a drawn number
        self.drawn_starts = []
        self.parts = [self._part(part, rng) for part in parts]
        self.labels = [lab for part in self.parts for seg in part["segs"] for lab in seg["labels"]]
        # the rows of the copies with a free start: their values at sigma(theta) need the image of the start
        self.free_rows = {
            k
            for k, (part, seg, _) in enumerate((p, s_, y) for p in self.parts for s_ in p["segs"] for y in s_["labels"])
            if part["segs"][0].get("free")
        }
        self.has_events = any(len(part["segs"]) > 1 for part in self.parts)
        # the jets of unknown inputs: free starts that are no coordinates, with values drawn or set (uinputs.py)
        self.inputs = [(pi, i) for pi, part in enumerate(self.parts) for i in part["segs"][0]["inputs"]]
        self.input_values = None
        self.raw_parts = parts
        self._kser, self._jac = {}, {}
        # functions of the coordinates proven in the field of the outputs: equal at every point of the fiber; the
        # exponent of a free power of a state at 0 is equal on the domain
        self._pins, self._base_pins = [], []
        for pi, part in enumerate(self.parts):
            if part.get("spow") is not None:
                self._start_power(pi)

    @property
    def pins(self):
        return self._base_pins + [P for P in self._pins if P not in self._base_pins]

    @pins.setter
    def pins(self, value):
        self._pins = list(value)

    @property
    def start_powers(self):
        """The description of each free power of a state at 0, one per part that has one (docs/method/global.rst)."""
        return [part["spow"] for part in self.parts if part.get("spow") is not None]

    def _start_power(self, pi):
        part = self.parts[pi]
        sp, seg = part["spow"], part["segs"][0]
        v, vy = startpow.valuations(seg["F"], seg["G"], len(part["st"]), sp["N"], part["known"])
        if v[sp["base"]] is not None and v[sp["base"]] < sp["k"]:
            raise ValueError(
                f"discrete(): {sp['state']}^{sp['alpha']} enters the dynamics of {sp['state']} below "
                f"the order {sp['k']} of {sp['state']}"
            )
        sp["v"], sp["vy"] = v, vy
        a = self.cs[sp["alpha_index"]]
        if not is_positive(a, self.positive):
            raise ValueError(f"discrete() needs the exponent {a} of a power of a state at 0 declared positive")
        sp["proof"] = startpow.proof(self, pi, self.positive)
        self._base_pins.append((_dict(a, self.cs), _dict(spy.Integer(1), self.cs)))

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
        # each segment is expanded at the segment start, or, where the start holds coordinates, at the start of the
        # condition: there the series of a later start tells the time of the event
        anchor = t0
        out = {"st": st, "segs": []}
        for j, seg in enumerate(part["segments"]):
            if list(seg["f"]) != [str(x) for x in st]:
                raise ValueError("discrete(): the segments of a condition have other states")
            s = {
                "F": [frac(seg["f"][str(x)]) for x in st],
                "G": [frac(v) for v in seg["g"].values()],
                "labels": [f"{seg.get('label', '')}{y}" for y in seg["g"]],
            }
            if j == 0:
                # the start of each state: a coordinate, a number, or a rational function of the coordinates; without
                # initial values (a copy switched into a regime) every state starts free, a number at the point
                # and a separate variable in the fiber
                jets = set(seg.get("inputs") or ())
                free = seg["ic"] is None or None in [v for k, v in seg["ic"].items() if k not in jets]
                X0 = {k: model_sympify(str(v)) for k, v in (seg["ic"] or {}).items() if v is not None}
                s["x0"], s["free"], s["inputs"] = [], {}, {}
                for i, x in enumerate(st):
                    if str(x) in jets:
                        s["inputs"][i] = spy.Integer(rng.randint(2, 9))
                        s["x0"].append((None, None, None))
                        continue
                    if seg["ic"] is None or (str(x) in seg["ic"] and seg["ic"][str(x)] is None):
                        s["free"][i] = spy.Integer(rng.randint(2, 9))
                        s["x0"].append((None, None, None))
                        continue
                    v = spy.sympify(X0.get(str(x), x)).xreplace(self.known)
                    if x not in cs and v == x:
                        v = spy.Integer(rng.randint(2, 9))
                        self.drawn_starts.append(x)
                    num, den = spy.fraction(spy.cancel(v))
                    s["x0"].append((v, _dict(num, cs), _dict(den, cs)))
                # a copy switched into a regime starts on the surface it crosses: q(z, theta) = 0, solved at the point
                # for a state in which q is linear
                s["on"] = None
                if free and seg.get("on") is not None:
                    N, D = frac(seg["on"])
                    ns = len(st)
                    lin = [
                        i
                        for i in range(ns)
                        if max(m[i] for m in N) == 1
                        and all(not any(m[:i] + m[i + 1 : ns]) or m[i] == 0 for m in N if m[i] == 1)
                    ]
                    lin = lin or [i for i in range(ns) if max(m[i] for m in N) == 1]
                    if not lin:
                        raise ValueError(f"discrete() needs the switching surface {seg['on']} = 0 linear in a state")
                    i = lin[0]

                    def drop(m, i=i):
                        return m[:i] + (0,) + m[i + 1 :]

                    s["on"] = (
                        i,
                        {drop(m): c for m, c in N.items() if m[i] == 1},
                        {m: c for m, c in N.items() if m[i] == 0},
                        N,
                        D,
                    )
                # a state with a numeric start whose right-hand side reads only such states (the jets of a known
                # input, the clock) has a known series: it enters as numbers, not as unknowns
                ns, known = len(st), set()
                while True:
                    more = {
                        i
                        for i in range(ns)
                        if i not in known
                        and s["x0"][i][0] is not None
                        and s["x0"][i][0].is_number
                        and all(not e or j_ in known for P in s["F"][i] for m in P for j_, e in enumerate(m[:ns]))
                        and all(not any(m[ns:]) for P in s["F"][i] for m in P)
                    }
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
        if part.get("spow") is not None:
            out["spow"] = self._spow_part(part, out)
        return out

    def _spow_part(self, part, out):
        """The free power of a state at 0 of a part as indices: the state N of the normalized power, the base and
        the coordinate of the exponent."""
        sp = dict(part["spow"])
        names = [str(x) for x in out["st"]]
        if len(out["segs"]) > 1:
            raise ValueError("discrete() handles a free power of a state at 0 without events")
        sp["N"], sp["base"] = names.index(sp["N"]), names.index(sp["state"])
        a = spy.Symbol(sp["alpha"])
        if a not in self.params:
            raise ValueError(f"discrete() needs the exponent {a} of a power of a state at 0 as a coordinate")
        if out["segs"][0]["x0"][sp["base"]][0] != 0:
            raise ValueError(f"discrete(): {sp['state']} does not start at 0")
        sp["alpha_index"] = self.cs.index(a)
        out["known"] = out["known"] - {sp["N"]}
        return sp

    def _spow_jets(self, pi, order, kser):
        """{key: name} of the jets of a part with a free power of a state at 0: per state, index n and layer m >= 1
        those with n - k*m at least the valuation of the state, and the coefficients of Q likewise."""
        part = self.parts[pi]
        sp, seg = part["spow"], part["segs"][0]
        k, L, need = sp["k"], order // sp["k"] + 1, sp["need"]
        out = {}
        for i in range(len(part["st"])):
            if i in kser or i == sp["N"]:
                continue
            v = seg["x0"][i][0]
            lo = 0 if v is None or (not (v.is_Symbol and v in self.cs) and not v.is_number) else 1
            for n in range(lo, need[i] + 1):
                for m in range(L):
                    if m == 0 or (sp["v"][i] is not None and n >= 1 and n - k * m >= sp["v"][i]):
                        out[(pi, 0, i, n, m)] = f"si_j{pi}_0_{i}_{n}_{m}"
        # a term of Q of layer m >= 1 holds a term of the base of layer m or less over t^k
        w = None if sp["v"][sp["base"]] is None else sp["v"][sp["base"]] - k
        for q in range(1, need[sp["N"]] - k + 1):
            for m in range(L):
                if m == 0 or (w is not None and q - k * m >= w):
                    out[("q", pi, q, m)] = f"si_q{pi}_{q}_{m}"
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

            xs = [
                [start(seg["x0"][i][0])] + [zero] * order if i in known else [zero] * (order + 1)
                for i in range(len(part["st"]))
            ]
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

    def _outputs(self, cvals, order, gap, one, zero, meter=None, weight=1):
        """Per observable of each segment of each condition, the Taylor coefficients up to `order`, each as the
        coefficients in eps up to `gap` (one for the first segment), the coordinates at `cvals` in the arithmetic
        of one and zero; None at a pole, CapError past the cap of `meter` (_flow())."""
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
                        if i in seg["inputs"]:
                            v = (self.input_values or {}).get((pi, i), seg["inputs"][i])
                            xs.append(
                                [one * (Fraction(int(v.p), int(v.q)) if isinstance(v, spy.Rational) else v)]
                                + [zero] * order
                            )
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
                    sp = part.get("spow")
                    if sp is not None:
                        # the layers S^m of the free power of a state at 0, one per k orders
                        o, z = _Eps([one] + [zero] * (order // sp["k"])), _Eps([zero] * (order // sp["k"] + 1))
                        xs = [[o * c for c in x] for x in xs]
                        if not startpow.layered_flow(
                            seg["F"], xs, params, order, o, z, kser, sp, val[self.cs[sp["alpha_index"]]]
                        ):
                            return None
                else:
                    sp = None
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
                        if not _flow(seg["F"], xs, params, n, o, z, meter=meter, weight=weight):
                            return None
                        off = _eval(seg["off"], cvals, zero)
                        xs = [_shift(x, -off, order, gap, one, zero) for x in xs]
                    else:
                        xs = [[w] + [z] * order for w in w0]
                if (
                    sp is None
                    and (j == 0 or seg["off"] is None)
                    and not _flow(seg["F"], xs, params, order, o, z, kser, meter, weight)
                ):
                    return None
                if pi == 0 and j == 0:
                    # the series of the states at the start of the first condition, for the exact count
                    self.states0 = xs
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

    def output_orders(self, cap, p, gap=0, seed=()):
        """Per observable the least order whose next Taylor coefficient adds no rank to the Jacobian in the
        coordinates, as in SIAN, at a random point modulo p; None where the rank stays below the number of
        coordinates up to `cap`. The gradients `seed` enter the basis first."""
        # the coefficients up to an order depend on none above it: the series grows by doubling up to `cap`
        n = min(cap, 8)
        while True:
            out = self._output_orders(n, p, gap, seed)
            if out != "short" or n >= cap:
                return None if out == "short" else out
            n = min(cap, 2 * n)

    def jacobian(self, order, p, gap=0):
        """The output coefficients up to `order` at the random point of output_orders modulo p, as duals with the
        gradient over the coordinates; None at a pole, or where the series reaches the cap of `meter`, which
        `capped` then records."""
        key = (p, gap)
        have = self._jac.get(key)
        if have is not None and have[0] >= order:
            return None if have[1] is None else [y[: order + 1] for y in have[1]]
        m = len(self.cs)
        cv = self._jac_point(p)
        one, zero = _Dual(1, [0] * m, p), _Dual(0, [0] * m, p)
        try:
            ys = self._outputs(cv, order, gap, one, zero, self.meter, m + 1)
        except ZeroDivisionError:
            ys = None
        except CapError:
            # the series past the cap: no rows at this order, and none cached
            self.capped = order
            return None
        self._jac[key] = (order, ys)
        return ys

    def _jac_point(self, p):
        """The random point of jacobian() modulo p, each coordinate a dual with a unit gradient."""
        rng = session.current().rng("discrete orders", tuple(self.coords))
        m = len(self.cs)
        return [_Dual(rng.randint(2, p - 1), [int(i == j) for j in range(m)], p) for i in range(m)]

    def pin_values(self, p):
        """The pins num/den at the random point of jacobian() modulo p, as duals with the gradient over the
        coordinates; None at a pole."""
        cv = self._jac_point(p)
        zero = _Dual(0, [0] * len(self.cs), p)
        out = []
        for num, den in self.pins:
            d = _eval(den, cv, zero)
            if not d.v:
                return None
            out.append(_eval(num, cv, zero) / d)
        return out

    def rank_rows(self, cap, p, skip=(), pins=False):
        """(observable, order) of the Taylor coefficients whose gradients in the coordinates are independent at the
        random point of output_orders modulo the columns `skip`, as many as the other coordinates, lowest order first;
        None where fewer exist up to `cap`. With `pins`, the gradients of the pins come first, each row of a pin
        ("pin", i)."""
        # the rows up to an order are the first rows up to any higher one: the series grows by doubling up to `cap`
        n = min(cap, 8)
        while True:
            out = self._rank_rows(n, p, skip, pins)
            if out is not None or n >= cap or self.jacobian(n, p, 0) is None:
                return out
            n = min(cap, 2 * n)

    def _rank_rows(self, cap, p, skip, pins):
        ys = self.jacobian(cap, p, 0)
        if ys is None:
            return None
        m = len(self.cs)
        basis = [[int(i == c) for i in range(m)] for c in skip]
        piv, out = list(skip), []
        if len(piv) == m:
            return out
        rows = []
        if pins:
            pv = self.pin_values(p)
            if pv is None:
                return None
            # the pins of least degree first
            least = sorted(range(len(pv)), key=lambda i: _pin_degree(self.pins[i]))
            rows += [(("pin", i), list(pv[i].d)) for i in least]
        rows += [
            ((j, k), list(ys[j][k][0].d) if isinstance(ys[j][k], list) else list(ys[j][k].d))
            for k in range(cap + 1)
            for j in range(len(ys))
            if k < len(ys[j])
        ]
        for key, row in rows:
            for b, c in zip(basis, piv):
                if row[c]:
                    f = row[c]
                    row = [(x - f * y) % p for x, y in zip(row, b)]
            c = next((i for i, x in enumerate(row) if x), None)
            if c is None:
                continue
            inv = pow(row[c], -1, p)
            basis.append([x * inv % p for x in row])
            piv.append(c)
            out.append(key)
            if len(out) + len(skip) == m:
                return out
        return None

    def pinned_orders(self, cap, p):
        """Per observable the least order at which the gradients of the pins and of the Taylor coefficients up to the
        orders reach the number of coordinates, as output_orders() with the pins in the basis first; None where they do
        not up to `cap` or a pin has a pole at the point."""
        pv = self.pin_values(p)
        return None if pv is None else self.output_orders(cap, p, 0, [v.d for v in pv])

    def _output_orders(self, cap, p, gap, seed=()):
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

        for row in seed:
            adds(row)
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
        for row in seed:
            adds(row)
        orders = [0] * len(ys)
        for k in range(cap + 1):
            for j in range(len(ys)):
                if any([adds(v.d) for v in ys[j][k]]):
                    orders[j] = k
            if len(basis) == m:
                return orders
        return "short"

    def _jet_names(self, orders, order, gap):
        """{key: name} of the jet variables of the prolonged model; sets the orders `need` of a free power of a state
        at 0."""
        jets = {}
        for pi, part in enumerate(self.parts):
            kser = self._known_series(pi, order)
            sp = part.get("spow")
            if sp is not None:
                row0 = sum(len(seg["labels"]) for part_ in self.parts[:pi] for seg in part_["segs"])
                seg = part["segs"][0]
                sp["need"] = startpow.needs(
                    seg["F"], seg["G"], len(part["st"]), sp, orders[row0 : row0 + len(seg["G"])], part["known"]
                )
                jets.update(self._spow_jets(pi, order, kser))
                continue
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
                        # the start after an event is a separate jet where the event divides by the states, and
                        # at an anchor before the event
                        ev = seg["event"].get(i)
                        lo = 0 if seg["off"] is not None or (ev is not None and any(any(m) for m in ev[1])) else 1
                        for k in range(lo, order + 1):
                            for m in range(gap + 1):
                                jets[(pi, j, i, k, m)] = f"si_j{pi}_{j}_{i}_{k}_{m}"
        return jets

    def prolonged(self, orders, gap=0):
        """(names, equations, outputs, sat) up to the order of each observable in `orders` and `gap`: the jet
        variables after the coordinates, the equations of the model, the starts and the events as {exponents:
        Fraction} over coordinates and jets, the series (E, M) of each observable as E[l][a] and M[k][m] (Taylor
        coefficient, then coefficient in eps), and the factors that must not vanish."""
        order = max(orders)
        names = [str(c) for c in self.cs]
        jets = self._jet_names(orders, order, gap)
        # the jets of unknown inputs above those the outputs read at these orders start at zero
        zero_in = set()
        if self.inputs:
            from .uinputs import unread_jets

            zero_in = unread_jets(self, orders)
            jets = {key: v for key, v in jets.items() if (key[0], key[2]) not in zero_in or key[3] > 0}
        if len(names) + len(jets) > _FIBER_UNKNOWNS:
            raise FiberTooLargeError(len(names) + len(jets), "unknowns")
        R, *gens = ring(",".join(names + list(jets.values())), QQ)
        g = dict(zip(names + list(jets.values()), gens))
        cof = {str(c): g[str(c)] for c in self.cs}
        params = [g[str(p)] for p in self.params]
        pad = (0,) * len(jets)

        def in_ring(P):
            return R.from_dict({mm + pad: c for mm, c in P.items()}) if P else R.zero

        eqs, outs = [], []
        sat = [cof[str(c)] for c in self.cs]

        size = [0]

        def add_eq(e):
            cs = _coeffs(e)
            eqs.extend(cs)
            # the equations past the limit of a count are not built further
            size[0] += sum(_terms(c) for c in cs)
            if size[0] > _fiber_terms():
                raise FiberTooLargeError(size[0])

        def add_sat(e):
            e = _lead(e)
            if not e.is_ground:
                sat.append(e)

        row = 0
        for pi, part in enumerate(self.parts):
            kser = self._known_series(pi, order)
            xs = None
            sp = part.get("spow")
            for j, seg in enumerate(part["segs"]):
                if j == 0:
                    xs = []
                    for i in range(len(part["st"])):
                        v, num, den = seg["x0"][i]
                        if i in kser:
                            xs.append([R(spy.Rational(c.numerator, c.denominator)) for c in kser[i]])
                            continue
                        if sp is not None and i == sp["N"]:
                            xs.append(None)
                            continue
                        if (pi, i) in zero_in:
                            a0 = R.zero
                        elif v is None:
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
                        if sp is not None:
                            xs.append([a0] + [None] * order)
                            continue
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
                        at = (
                            _at_gap([w0] + ws, in_ring(seg["off"]), gap, R.one, R.zero)
                            if seg["off"] is not None
                            else w0
                        )
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
                if sp is not None:
                    # the layers of each index: jets where a term can be other than 0
                    L = order // sp["k"] + 1
                    o, z = _Eps([R.one] + [R.zero] * (L - 1)), _Eps([R.zero] * L)

                    def layers(key):
                        return _Eps([g[jets[key + (m,)]] if key + (m,) in jets else R.zero for m in range(L)])

                    for i in range(len(xs)):
                        if i == sp["N"]:
                            xs[i] = [z] * (order + 1)
                        elif i in kser:
                            xs[i] = [o * c for c in xs[i]]
                        else:
                            xs[i] = [o * xs[i][0]] + [layers((pi, 0, i, k)) for k in range(1, order + 1)]
                    Q = [o] + [layers(("q", pi, q)) for q in range(1, sp["need"][sp["N"]] - sp["k"] + 1)]
                    startpow.layered_equations(
                        seg["F"],
                        xs,
                        params,
                        sp["need"],
                        o,
                        z,
                        kser,
                        sp,
                        cof[str(self.cs[sp["alpha_index"]])],
                        Q,
                        add_eq,
                        add_sat,
                        _fiber_terms(),
                    )
                for i, (N, D) in enumerate(seg["F"]):
                    if (j == 0 and i in kser) or sp is not None:
                        continue
                    nser = _ser_poly(N, xs, params, order - 1, o, z, _fiber_terms()) if order else []
                    dser = _ser_poly(D, xs, params, order - 1, o, z, _fiber_terms()) if order else []
                    for k in range(order):
                        add_eq(sum((dser[l] * (k + 1 - l) * xs[i][k + 1 - l] for l in range(0, k + 1)), z) - nser[k])
                    add_sat(_ser_poly(D, xs, params, 0, o, z, _fiber_terms())[0])
                for M, E in seg["G"]:
                    n = orders[row]
                    row += 1
                    mser = _ser_poly(M, xs, params, n, o, z, _fiber_terms())
                    eser = _ser_poly(E, xs, params, n, o, z, _fiber_terms())
                    add_sat(eser[0])
                    outs.append(
                        (
                            [[_rdict(c) for c in _coeffs(e)] for e in eser],
                            [[_rdict(c) for c in _coeffs(m)] for m in mser],
                        )
                    )
        return list(jets.values()), [_rdict(e) for e in eqs if e], outs, [_rdict(f) for f in sat]


def _dict(e, syms):
    """A polynomial in `syms` as {exponents: Fraction}, in the sparse ring: the dense form recurses once per
    symbol, past the recursion limit with the jets of known inputs."""
    if not syms:
        c = spy.Rational(spy.expand(e))
        return {(): Fraction(int(c.p), int(c.q))} if c else {}
    return _rdict(_sparse_ring(tuple(syms)).from_expr(e))


_RINGS: dict = {}


def _sparse_ring(syms):
    """The sparse polynomial ring over Q in `syms`, cached."""
    if syms not in _RINGS:
        _RINGS[syms] = spy.polys.rings.ring(syms, spy.QQ)[0]
    return _RINGS[syms]


def _rdict(P):
    """A ring element as {exponents: Fraction}."""
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

    def __init__(self, series, orders, positive=True, gap=0, proj=None):
        self.series = series
        self.coords = list(series.coords)
        self.syms = [spy.Symbol(c) for c in self.coords]
        # the count modulo p is that of the distinct values of the first `proj` coordinates (all by default)
        self.proj = len(self.coords) if proj is None else int(proj)
        # one order per observable of each segment; a number is the same order for all
        self.orders = [int(orders)] * len(series.labels) if isinstance(orders, int) else [int(n) for n in orders]
        self.order = max(self.orders)
        # the coefficients in eps after an event; the state at the end of a gap needs the jets up to `gap`
        self.gap = min(int(gap), self.order) if series.has_events else 0
        self.positive = positive
        self.known = series.known
        self.names = [f"x{i}" for i in range(len(self.syms))]
        try:
            jets, static, outs, sat = series.prolonged(self.orders, self.gap)
        except FiberTooLargeError as exc:
            # a series past the limit: the equations have more terms, and no count is taken
            self.too_large, self.allnames, self.static, self.outs, self.sat = (
                _fiber_terms() + 1,
                list(self.names),
                [],
                [],
                [],
            )
            self.unknowns = exc.args[0] if exc.args[1:] == ("unknowns",) else None
            self.sep_mode = {}
            return
        self.allnames = self.names + jets
        # the monomials as (variable, exponent) pairs: the system has hundreds of variables and few terms
        self.static = [_sparse(P) for P in static]
        self.outs = [
            ([[_sparse(c) for c in row] for row in E], [[_sparse(c) for c in row] for row in M]) for E, M in outs
        ]
        self.sat = []
        for f in (_sparse(_clear(f)) for f in sat):
            if f not in self.sat:
                self.sat.append(f)
        # per characteristic, whether msolve gets the separating form (decided at the first call)
        self.sep_mode = {}

    def pin_polys(self, point, p=0):
        """num(x) den(point) - den(x) num(point) per pin num/den of the series, in the sparse form of the system."""
        out = []
        for num, den in self.series.pins:
            x = [Fraction(str(point[s])) for s in self.syms]
            a, b = _eval(num, x), _eval(den, x)
            if p and (a.denominator % p == 0 or b.denominator % p == 0):
                continue
            P = {}
            for mm, c in num.items():
                P[mm] = P.get(mm, 0) + c * b
            for mm, c in den.items():
                P[mm] = P.get(mm, 0) - c * a
            P = {mm: Fraction(c) for mm, c in P.items() if c}
            out.append(_sparse(P))
        return out

    def values(self, point, p=0):
        """The Taylor coefficients of the observables at `point` ({symbol: Rational}), modulo p with p, None at a
        pole."""
        return self.series.numeric(point, self.order, self.gap, p)

    def system(self, point, p=0, eliminate=False, extra=()):
        """{"blocks", "fixed"}: the msolve input of the points with the coefficients of `point`, over Q (p = 0) or
        GF(p). The variables that a linear equation in one of them fixes to a number are substituted, `fixed`
        holds the coordinates among those; the sparse polynomials `extra` join the equations. With `eliminate`, for
        a count only, so are the variables that an
        equation a*v + b*m sets to a monomial m (_monomial_eliminate()). The rest splits into blocks without a
        shared variable in an equation or a factor of the saturation, each {"names", "sep", "eqs"} with `sep` the
        coordinates of the block; the fiber is the product of the fibers of the blocks. None where the point has a
        pole or the system has no solution."""
        if getattr(self, "too_large", None):
            raise FiberTooLargeError(self.too_large)
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
        polys += [_clear(P, p) for P in self.pin_polys(point, p)] + [dict(P) for P in extra]
        polys = [P for P in polys if P]
        terms = sum(len(P) for P in polys)
        if eliminate and terms > _fiber_terms():
            self.too_large = max(terms, getattr(self, "too_large", 0))
            raise FiberTooLargeError(terms)
        sat = [{m: c % p if p else c for m, c in f.items()} for f in self.sat]
        red = _reduce(polys, sat, len(self.allnames), len(self.names), p)
        if red is None:
            return None
        polys, sat, fixed, gone = red
        if eliminate:
            # a coordinate outside the projection may leave for any monomial, as a jet does
            polys, sat, elim = _monomial_eliminate(polys, sat, getattr(self, "proj", len(self.names)), p)
            if any(list(P) == [()] for P in polys) or any(not f for f in sat):
                return None
            if elim:
                red = _reduce(polys, sat, len(self.allnames), len(self.names), p)
                if red is None:
                    return None
                polys, sat, fixed2, gone2 = red
                fixed, gone = {**fixed, **fixed2}, gone | gone2 | elim
        keep = [i for i in range(len(self.allnames)) if i not in fixed and i not in gone]
        # an unknown that is no coordinate and in no equation is free and leaves the count of the coordinates
        used = {v for P in polys + sat for mm in P for v, _ in mm}
        keep = [i for i in keep if i < len(self.names) or i in used]
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
            bp = [P for P in polys if {v for mm in P for v, _ in mm} & loc]
            bs = [f for f in sat if {v for mm in f for v, _ in mm} & loc]
            block = self._ms_block(idx, bp, bs, p)
            block.update({"idx": list(idx), "polys": bp, "sat": bs})
            blocks.append(block)
        return {"blocks": blocks, "fixed": {self.names[i]: v for i, v in fixed.items() if i < m}}

    def _ms_block(self, idx, polys, sat, p):
        """{"names", "sep", "eqs"}: the msolve input of the sparse equations `polys` in the variables `idx`, with the
        product of the factors `sat` inverted by si_w."""
        m = len(self.names)
        eqs = [_ms_sparse(_clear(P, p), self.allnames) for P in polys]
        fac = {(): 1}
        for f in sat:
            fac = _smul(fac, _clear(f, p), p)
        # points where a coordinate or a denominator at the start vanishes do not count; msolve reads no
        # brackets, so si_w multiplies each term
        eqs.append(_ms_sparse(fac, self.allnames, "si_w") + "-1")
        proj = getattr(self, "proj", m)
        return {
            "names": [self.allnames[i] for i in idx],
            "sep": [self.names[i] for i in idx if i < m],
            "proj": [self.names[i] for i in idx if i < proj],
            "eqs": eqs,
        }


def _monomial_eliminate(polys, sat, m, p=0):
    """(polys, sat, gone): the sparse system after substituting v = -b/a m for each equation a*v + b*m with a
    monomial m free of v, in turn, over Q (p = 0) or GF(p). A coordinate (index below `m`) leaves only for a
    monomial in coordinates: the coordinates of a point then fix the eliminated one, and distinct points stay
    distinct in the coordinates left. The substitution maps monomials to monomials, so no equation grows."""
    polys, sat, gone = [dict(P) for P in polys], [dict(f) for f in sat], set()

    def div(a, b):
        return a * pow(b, -1, p) % p if p else Fraction(a) / b

    def subst(P, v, c, mono):
        out = {}
        for mm, k in P.items():
            e = dict(mm).get(v)
            if e:
                k = k * pow(c, e, p) % p if p else k * Fraction(c) ** e
                d = {w: f for w, f in mm if w != v}
                for w, f in mono:
                    d[w] = d.get(w, 0) + e * f
                mm = tuple(sorted(d.items()))
            out[mm] = (out.get(mm, 0) + k) % p if p else out.get(mm, 0) + k
        return {mm: k for mm, k in out.items() if k}

    def target(P):
        if len(P) != 2:
            return None
        for (mv, a), (mo, b) in (tuple(P.items()), tuple(P.items())[::-1]):
            if len(mv) == 1 and mv[0][1] == 1:
                v = mv[0][0]
                if all(w != v for w, _ in mo) and (v >= m or all(w < m for w, _ in mo)):
                    return v, div(-b, a), mo
        return None

    while True:
        hit = next((t for P in polys for t in [target(P)] if t is not None), None)
        if hit is None:
            break
        v, c, mono = hit
        gone.add(v)
        polys = [Q for Q in (subst(P, v, c, mono) for P in polys) if Q]
        sat = [subst(f, v, c, mono) for f in sat]
    polys = [_clear(P, p) if not p else P for P in polys]
    sat = [_clear(f, p) if not p else f for f in sat]
    return polys, sat, gone


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
            P[tuple(zip(var[tstart[t] : tstart[t + 1]], ex[tstart[t] : tstart[t + 1]]))] = coef[t]
        out.append(P)
    return out


def _reduce_mod(polys, sat, nv, m0, p):
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
                    if (
                        v >= m0
                        and v not in insat
                        and count.get(v) == 1
                        and sum(1 for nn in P if any(w == v for w, _ in nn)) == 1
                    ):
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
    return sum((c * math.prod((xi**e for xi, e in zip(x, m) if e), start=zero + 1) for m, c in P.items()), zero)


def _msolve(names, p, eqs, flags, sep=()):
    """The parsed output of msolve on `eqs` in `names` + si_w over GF(p) (p = 0: Q), or None on failure or a system
    past the caps of work.msolve_too_large(). With `sep`, a last variable si_t is a linear form in these names with
    fixed coefficients: it separates distinct points, where si_w need not, so msolve parametrizes by it without
    trying other forms."""
    flags = list(flags) + ["-t", str(msolve._MSOLVE_THREADS)]
    if p:
        # the probabilistic sparse linear algebra of msolve over GF(p): the counts are at random points anyway
        flags += ["-l", "44"]
    names = list(names) + ["si_w"]
    eqs = list(eqs)
    if sep:
        eqs.append("si_t" + "".join(f"-{3 + 2 * i}*{n}" for i, n in enumerate(sep)))
        names.append("si_t")
    if work.msolve_too_large(names, eqs):
        _failed(kind="size")
        return None
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.ms"), os.path.join(td, "out.ms")
        with open(fi, "w") as fh:
            fh.write(",".join(names) + "\n" + str(p) + "\n" + ",\n".join(eqs) + "\n")
        r = work.run_msolve([msolve._msolve_binary(), "-f", fi, "-o", fo] + flags)
        txt = ""
        if os.path.exists(fo):
            with open(fo) as fh:
                txt = fh.read()
    if r.returncode != 0 or not txt.strip():
        _failed(r)
        return None
    txt = txt.strip().rstrip(":")
    out = parse_rational(txt) if p == 0 else ast.literal_eval(txt)
    if out and out[0] == 1:
        _failed(kind="dimension")
    return out


def parse_rational(txt):
    """msolve output over Q with every quotient of integers, a / 2^k or a / b, as an exact Fraction."""
    # a match starts only at the first digit of a number: otherwise each start inside a long integer scans the
    # rest of it, quadratic in the number of digits
    txt = re.sub(
        r"(?<!\d)(-?\d+)\s*/\s*(\d+)(?![\d^])",
        r"Fraction(\1, \2)",
        re.sub(r"(?<!\d)(-?\d+)\s*/\s*2\^(\d+)", r"Fraction(\1, 2**\2)", txt),
    )
    # the rational parametrization holds integers past the default limit of int()
    with _AllDigits():
        return eval(txt, {"Fraction": Fraction, "__builtins__": {}})


# ---- counts --------------------------------------------------------------------------------


def real_points(fib, point, precision=64):
    """The real points of the fiber of `point` as boxes [(lo, hi)] per coordinate, None on failure."""
    try:
        sysd = fib.system(point)
    except FiberTooLargeError:
        return None
    if sysd is None:
        return None
    per = []
    for block in sysd["blocks"]:
        at = {n: i for i, n in enumerate(block["names"])}
        for prec in (precision, 4 * precision, 16 * precision):
            data = _solve(fib, 0, block, ["-P", "0", "-p", str(prec)])
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
        # a fixed coordinate takes the fixed value on every point
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


def _solve(fib, p, block, flags):
    """_msolve() on a block of Fiber.system(), with the separating form si_t only where the first call on the block
    fails without it: msolve may then try several forms when si_w does not separate the points, and the dense form
    costs more where it does."""
    names, eqs, sep = block["names"], block["eqs"], block["sep"]
    key = (p == 0, tuple(names))
    mode = fib.sep_mode.get(key)
    if mode is not None or not sep:
        return _msolve(names, p, eqs, flags, sep if mode else ())
    plain = _msolve(names, p, eqs, flags)
    if plain is not None:
        fib.sep_mode[key] = False
        return plain
    with_sep = _msolve(names, p, eqs, flags, sep)
    fib.sep_mode[key] = with_sep is not None
    return with_sep


def complex_count(fib, point, p, exact=False, dims=None):
    """The number of points of the fiber over the algebraic closure of GF(p), None on failure: per block the number
    of standard monomials of the leading ideal of a Gröbner basis, the dimension of the quotient, without the
    parametrization of the points. With `exact`, msolve uses exact linear algebra and `dims` receives the dimension
    of the quotient of each block."""
    try:
        pt = {s: spy.Integer(_modp_rational(spy.Rational(v), p)) for s, v in point.items()}
        sysd = fib.system(pt, p, eliminate=True)
    except (ZeroDivisionError, FiberTooLargeError):
        return None
    if sysd is None:
        return None
    n = 1
    for block in sysd["blocks"]:
        if exact:
            k = _gb_count(block["names"], p, block["eqs"], block["proj"], exact=exact, dims=dims)
        else:
            k = _staged_count(fib, block["idx"], block["polys"], block["sat"], p)
        if not k:
            return None
        n *= k
    return n


def _vars(P):
    return {v for mm in P for v, _ in mm}


def _first_stage(polys, idx):
    """The variables of a closed stage of the equations: a proper subset A of `idx` whose equations that read only
    A match every variable of A, from the block triangular form of the incidence of equations and variables
    (Dulmage and Mendelsohn); the smallest such set that holds a cycle of the matching, None where there is none."""
    adj = [sorted(_vars(P)) for P in polys]
    mv, me = {}, {}
    # a maximum matching by augmenting paths, iterative
    for e0 in range(len(adj)):
        stack, seen, path = [(e0, iter(adj[e0]))], set(), []
        while stack:
            e, it = stack[-1]
            v = next((w for w in it if w not in seen), None)
            if v is None:
                stack.pop()
                if path:
                    path.pop()
                continue
            seen.add(v)
            path.append((e, v))
            if v not in mv:
                for e2, v2 in path:
                    mv[v2], me[e2] = e2, v2
                break
            stack.append((mv[v], iter(adj[mv[v]])))
    if len(mv) < len(idx):
        return None
    # the variable of an equation depends on the other variables it reads; strongly connected parts by Tarjan
    succ = {v: [w for w in adj[e] if w != v] for v, e in mv.items()}
    order, low, stack, onst, comps = {}, {}, [], set(), []
    for root in idx:
        if root in order:
            continue
        work = [(root, iter(succ[root]))]
        order[root] = low[root] = len(order)
        stack.append(root)
        onst.add(root)
        while work:
            v, it = work[-1]
            w = next(it, None)
            if w is not None:
                if w not in order:
                    order[w] = low[w] = len(order)
                    stack.append(w)
                    onst.add(w)
                    work.append((w, iter(succ[w])))
                elif w in onst:
                    low[v] = min(low[v], order[w])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == order[v]:
                c = []
                while True:
                    w = stack.pop()
                    onst.discard(w)
                    c.append(w)
                    if w == v:
                        break
                comps.append(c)
    best = None
    for c in comps:
        if len(c) < 2:
            continue
        clos, todo = set(c), list(c)
        while todo:
            for w in succ[todo.pop()]:
                if w not in clos:
                    clos.add(w)
                    todo.append(w)
        if len(clos) < len(idx) and (best is None or len(clos) < len(best)):
            best = clos
    return best


def _stage_points(fib, A, polys, sat, p):
    """The points of the equations `polys` in the variables A over the algebraic closure of GF(p) (p = 0: of Q), as
    {variable: value}, from the rational parametrization of msolve; None where it fails or a point is not in GF(p)
    (in Q)."""
    idx = sorted(A)
    block = fib._ms_block(idx, polys, sat, p)
    data = None
    for sep in ((), block["sep"]):
        data = _msolve(block["names"], p, block["eqs"], ["-P", "1"], sep)
        if data is not None:
            break
    if data is None or data[0] == -1:
        return [] if data is not None else None
    if data[0] != 0:
        return None
    _, _, deg, vnames, lin, par = data[1]
    if deg <= 0 or par[0] != 1:
        return None
    at = {fib.allnames[i]: i for i in idx}
    if p:
        rur = {
            "elim": [int(c) % p for c in par[1][0][1]],
            "den": [int(c) % p for c in par[1][1][1]],
            "par": [[int(c) % p for c in q[0][1]] for q in par[1][2]],
            "lin": [int(c) % p for c in lin],
            "vnames": list(vnames),
        }
        elim = rur["elim"]
        # every distinct root of the eliminating polynomial is a point; all of them in GF(p), or none is used
        sq = len(elim) - len(_gcd_mod(elim, [(k * elim[k]) % p for k in range(1, len(elim))], p))
        pts = _fiber_points_block(rur, p)
        if len(pts) != sq:
            return None
        return [{at[n]: v for n, v in pt.items() if n in at} for pt in pts]
    return _rational_points(par, lin, list(vnames), at)


def _rational_points(par, lin, vnames, at):
    """The points of a rational parametrization of msolve over Q, each coordinate -q_j(t) / (c_j den(t)) at a root t
    of the eliminating polynomial, where every distinct root is rational; None otherwise."""
    T = spy.Symbol("t")
    elim = spy.Poly(list(reversed([int(c) for c in par[1][0][1]])), T)
    sq = spy.sqf_part(elim)
    if sq.degree() > _STAGE_ROOTS:
        return None
    roots = [-f.all_coeffs()[1] / spy.Integer(f.all_coeffs()[0]) for f, _ in spy.factor_list(sq)[1] if f.degree() == 1]
    if len(roots) != sq.degree():
        return None

    def ev(coeffs, x):
        acc = Fraction(0)
        for c in reversed(coeffs):
            acc = acc * x + int(c)
        return acc

    out = []
    for r in roots:
        t = Fraction(int(r.p), int(r.q))
        dv = ev(par[1][1][1], t)
        if dv == 0:
            return None
        vals = {vnames[j]: -ev(q[0][1], t) / (int(q[1]) * dv) for j, q in enumerate(par[1][2])}
        last = len(vnames) - 1
        vals[vnames[last]] = (t - sum(int(lin[j]) * vals[vnames[j]] for j in range(last))) / int(lin[last])
        out.append({at[n]: v for n, v in vals.items() if n in at})
    return out


def _staged_count(fib, idx, polys, sat, p):
    """The count of a block modulo p (p = 0: over Q), as _block_count(), stage by stage: the points of a closed
    stage (_first_stage()) are found, each substituted into the other equations, and the counts of the rest add up
    over the points. The block is first counted at once within an eighth of the cap msolve_entries of
    work.CAPS; without a stage, or where a stage fails, it is counted at once within the whole cap."""
    m = len(fib.names)
    A = _first_stage(polys, idx)
    cap = work.cap("msolve_entries")
    if A is not None:
        # the block at once first: what is left after a stage is often small
        with work.local_caps(msolve_entries=cap / 8):
            k = _block_count(fib, idx, polys, sat, p)
        if k is not None:
            return k
        inA = [P for P in polys if _vars(P) <= A]
        satA = [f for f in sat if _vars(f) <= A]
        # over Q a stage first runs modulo a prime: rational points reduce to points in GF(p), so where the stage
        # has a point outside GF(p), the stage over Q is skipped
        if not p:
            with work.local_caps(msolve_entries=cap / 16):
                if _stage_points(fib, A, inA, satA, PRIMES[0]) is None:
                    A = None
    if A is not None:
        pts = _stage_points(fib, A, inA, satA, p)
    if A is None or pts is None:
        return _block_count(fib, idx, polys, sat, p)
    rest = [P for P in polys if not _vars(P) <= A]
    srest = [f for f in sat if not _vars(f) <= A]
    others = [i for i in idx if i not in A]
    total = 0
    for pt in pts:
        lin = [{((v, 1),): 1, (): (-x) % p if p else -x} for v, x in pt.items()]
        red = _reduce(lin + rest, srest, len(fib.allnames), m, p)
        if red is None:
            continue
        rp, rs, fixed, gone = red
        keep = [i for i in others if i not in fixed and i not in gone]
        if not keep:
            total += 1
            continue
        used = set().union(*[_vars(P) for P in rp]) if rp else set()
        if set(keep) - used:
            # a variable left without an equation: no finite count
            return None
        n = 1
        for sub in _components(keep, rp, rs):
            k = _staged_count(
                fib, sub, [P for P in rp if _vars(P) & set(sub)], [f for f in rs if _vars(f) & set(sub)], p
            )
            if k is None:
                return None
            n *= k
        total += n
    return total


def _block_count(fib, idx, polys, sat, p):
    """The count of a block at once: _gb_count() modulo p, over Q (p = 0) the degree of the rational
    parametrization of msolve, at least the number of distinct points."""
    b = fib._ms_block(idx, polys, sat, p)
    if p:
        return _gb_count(b["names"], p, b["eqs"], b["proj"])
    data = _solve(fib, 0, b, ["-P", "1"])
    if data is None or data[0] not in (0, -1):
        return None
    if data[0] == -1:
        return 0
    deg = int(data[1][2])
    if (
        len(b["proj"]) < len(b["names"])
        and getattr(fib, "proj", len(fib.names)) < len(fib.names)
        and deg <= _PROJ_POINTS
    ):
        # the count of a projection: the distinct values of the projected coordinates on the points
        k = _projected_count(data, b["proj"])
        return deg if k is None else k
    return deg


def _projected_count(data, proj):
    """The number of distinct values of the variables `proj` on the points of a rational parametrization of msolve
    over Q: the dimension of the subalgebra they generate in Q[t]/(w), w the square-free eliminating polynomial, an
    algebra of the functions on the points. None where the parametrization has another form."""
    _, _, deg, vnames, lin, par = data[1]
    if deg <= 0 or par[0] != 1:
        return None
    if not proj:
        return 1
    T = spy.Symbol("t")

    def poly(cs):
        return spy.Poly(list(reversed([spy.Integer(int(c)) for c in cs])) or [0], T, domain="QQ")

    w = spy.Poly(spy.sqf_part(poly(par[1][0][1]).as_expr()), T, domain="QQ")
    n = w.degree()
    try:
        dinv = poly(par[1][1][1]).invert(w)
    except (spy.polys.polyerrors.NotInvertible, ZeroDivisionError):
        return None
    vals = {vnames[j]: (-poly(q[0][1]) * dinv * spy.Rational(1, int(q[1]))).rem(w) for j, q in enumerate(par[1][2])}
    last = len(vnames) - 1
    if vnames[last] in proj:
        rest = spy.Poly(T, T, domain="QQ")
        for j in range(last):
            rest = rest - vals[vnames[j]] * int(lin[j])
        vals[vnames[last]] = (rest * spy.Rational(1, int(lin[last]))).rem(w)
    gens = [vals[v] for v in proj if v in vals]

    def vec(P):
        c = P.all_coeffs()[::-1]
        return [Fraction(int(x.p), int(x.q)) for x in c] + [Fraction(0)] * (n - len(c))

    # a reduced echelon basis: each row is zero at the pivots of the others
    rows = {}

    def add(P):
        v = vec(P)
        for piv, r in rows.items():
            if v[piv]:
                f = v[piv]
                v = [a - f * b for a, b in zip(v, r)]
        piv = next((i for i, a in enumerate(v) if a), None)
        if piv is None:
            return False
        v = [a / v[piv] for a in v]
        for q, r in rows.items():
            if r[piv]:
                f = r[piv]
                rows[q] = [a - f * b for a, b in zip(r, v)]
        rows[piv] = v
        return True

    basis = [spy.Poly(1, T, domain="QQ")]
    add(basis[0])
    k = 0
    while k < len(basis):
        for g in gens:
            P = (basis[k] * g).rem(w)
            if add(P):
                basis.append(P)
        k += 1
    return len(basis)


def _components(keep, polys, sat):
    """The variables `keep` split into the parts that no equation or factor joins."""
    parent = {i: i for i in keep}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for P in list(polys) + list(sat):
        vs = sorted(_vars(P) & parent.keys())
        for a, b in zip(vs, vs[1:]):
            parent[find(a)] = find(b)
    out = {}
    for i in keep:
        out.setdefault(find(i), []).append(i)
    return list(out.values())


def _gb_count(names, p, eqs, sep=(), cap=5000, exact=False, dims=None):
    """The number of points of the ideal of `eqs` in `names` + si_w over the algebraic closure of GF(p) with
    distinct values of the coordinates `sep`; None where it is not finite, the quotient above `cap`, or on failure.
    From the reduced Gröbner basis of msolve: the standard monomials span the quotient, the multiplication by a
    random linear form in `sep` acts on it, and the distinct eigenvalues of that matrix are the distinct values of
    the form on the points. A block without coordinates counts one point where it has any. With `exact`, msolve
    uses exact linear algebra, and `dims` receives the dimension of the quotient."""
    names = list(names) + ["si_w"]
    if work.msolve_too_large(names, eqs):
        _failed(kind="size")
        return None
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.ms"), os.path.join(td, "out.ms")
        with open(fi, "w") as fh:
            fh.write(",".join(names) + "\n" + str(p) + "\n" + ",\n".join(eqs) + "\n")
        r = work.run_msolve(
            [
                msolve._msolve_binary(),
                "-f",
                fi,
                "-o",
                fo,
                "-g",
                "2",
                "-l",
                "2" if exact else "44",
                "-t",
                str(msolve._MSOLVE_THREADS),
            ]
        )
        txt = ""
        if os.path.exists(fo):
            with open(fo) as fh:
                txt = fh.read()
    if r.returncode != 0:
        _failed(r)
        return None
    at = {n: i for i, n in enumerate(names)}
    body = "".join(line for line in txt.splitlines() if not line.startswith("#")).strip().rstrip(":").strip("[]")
    gb = [_gb_poly(t, at, p) for t in body.split(",") if t.strip()]
    if any(list(P) == [()] for P in gb):
        return 0
    lead = [next(iter(P)) for P in gb]
    basis = _standard_monomials([dict(m) for m in lead], len(names), cap)
    if basis is None:
        _failed(
            kind="dimension"
            if _standard_monomials([dict(m) for m in lead], len(names), 0, finite=True) is None
            else "quotient"
        )
        return None
    if dims is not None:
        dims.append(len(basis))
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


def _standard_monomials(lead, nv, cap, finite=False):
    """The monomials in nv variables that no monomial of `lead` ({variable: exponent}) divides, sparse; None where
    there are infinitely many or more than `cap`. With `finite`, [] where there are finitely many."""
    pure = {}
    for e in lead:
        if len(e) == 1:
            ((v, k),) = e.items()
            pure[v] = min(pure.get(v, k), k)
    if len(pure) < nv:
        return None
    if finite:
        return []
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
    """The points of the fiber at `pt` as exact boxes, the point and the images under `maps`, or None where an image
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
        if (
            here is None
            or there is None
            or [r for k, r in enumerate(here) if k not in fib.series.free_rows]
            != [r for k, r in enumerate(there) if k not in fib.series.free_rows]
        ):
            return None
    return [[(v, v) for v in t] for t in vals]


def domain_point(fib, rng):
    """A random rational point with every declared coordinate positive."""
    return {s: spy.Rational(rng.randint(1, 60), rng.randint(1, 12)) for s in fib.syms}


# ---- the rational univariate representation over GF(p) -------------------------------------


def _rur(fib, point, p):
    """{"deg", "elim", "den", "par", "lin", "vnames"} of the fiber of `point` (integers mod p), None on failure."""
    try:
        pt = {s: spy.Integer(_modp_rational(spy.Rational(v), p)) for s, v in point.items()}
    except ZeroDivisionError:
        return None
    try:
        sysd = fib.system(pt, p)
    except (ZeroDivisionError, FiberTooLargeError):
        return None
    if sysd is None:
        return None
    blocks = []
    for block in sysd["blocks"]:
        data = _solve(fib, p, block, ["-P", "1"])
        if data is None or data[0] != 0:
            return None
        _, _, deg, vnames, lin, par = data[1]
        if deg <= 0 or par[0] != 1:
            return None
        blocks.append(
            {
                "deg": int(deg),
                "elim": [int(c) % p for c in par[1][0][1]],
                "den": [int(c) % p for c in par[1][1][1]],
                "par": [[int(c) % p for c in q[0][1]] for q in par[1][2]],
                "lin": [int(c) % p for c in lin],
                "vnames": list(vnames),
            }
        )
    return {
        "deg": math.prod(b["deg"] for b in blocks),
        "blocks": blocks,
        "fixed": {n: int(v) % p for n, v in sysd["fixed"].items()},
    }


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


def _plain_mul(a, b, p):
    """The product of two polynomials over GF(p), lowest degree first."""
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] = (out[i + j] + x * y) % p
    return out


def _pmul(a, b, f, p):
    """a * b mod f over GF(p), lowest degree first."""
    return _pmod(_plain_mul(a, b, p), f, p)


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
    the fibers of the blocks: a variable of a block of k points has the polynomial over the block to the power
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
            P = np.array(
                [[sum(int(P[i, l]) * int(M[l, j]) for l in range(k)) % p for j in range(k)] for i in range(k)],
                dtype=object,
            )
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

    def __init__(self, fib, meter=None):
        self.fib, self.memo = fib, {}
        self.meter = meter or work.unlimited()
        self.rng = session.current().rng("discrete points", tuple(fib.coords))
        self.k = None
        self.pool, self.moved = [], {}
        self.checks = None

    def point(self):
        return tuple(self.rng.randint(2, 10**6) for _ in self.fib.syms)

    def points(self, n):
        """The first `n` points of one pool, shared by every coefficient."""
        while len(self.pool) < n:
            self.pool.append(self.point())
        return self.pool[:n]

    def moved_point(self, base, i):
        """`base` with coordinate i changed, the same for every coefficient."""
        if i not in self.moved:
            pt = list(base)
            pt[i] = self.rng.randint(2, 10**6)
            self.moved[i] = tuple(pt)
        return self.moved[i]

    def rur(self, pt, p):
        """The representation of the fiber at `pt` modulo p, None at a point of another size."""
        key = (pt, p)
        if key not in self.memo:
            if self.meter.spend():
                raise CapError()
            point = dict(zip(self.fib.syms, [spy.Integer(v) for v in pt]))
            rur = _rur(self.fib, point, p)
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
    """The exponent tuples in `nv` variables of total degree at most d."""
    return [m for t in range(d + 1) for m in _exps(nv, t)]


def _exps(nv, t):
    """The exponent tuples in `nv` variables of total degree t."""
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
    """The rational function from the coefficient vectors modulo the first primes, None where none is bounded."""
    rec = rat_recon(np.array(sols, dtype=np.int64).T, primes[: len(sols)])
    if any(dd == "0" for dd in rec["den"]):
        return None
    coef = [spy.Rational(int(n), int(dd)) for n, dd in zip(rec["num"], rec["den"])]
    num = sum((c * spy.Mul(*[s**e for s, e in zip(syms, m)]) for c, m in zip(coef[: len(mons)], mons)), spy.Integer(0))
    den = sum((c * spy.Mul(*[s**e for s, e in zip(syms, m)]) for c, m in zip(coef[len(mons) :], mons)), spy.Integer(0))
    return None if den == 0 else spy.cancel(num / den)


def _holds(oracle, sam, r, p):
    """Whether `r` matches `oracle` at two check points modulo p, apart from the pool."""
    if sam.checks is None:
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
    poly = Z**k + sum((c * Z**j for j, c in enumerate(coeffs)), spy.Integer(0))
    num = spy.fraction(spy.together(poly))[0]
    out = []
    for fac, _ in spy.factor_list(num, Z)[1]:
        if spy.degree(fac, Z) == 1:
            a, b = spy.Poly(fac, Z).all_coeffs()
            out.append(spy.cancel(-b / a))
    return out


def _labels(sam, roots, pt, p):
    """{label: fiber point} at `pt`: the label holds, per separating coordinate, the index of the root the value
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


def rational_maps(fib, meter=None):
    """(count, maps) of _rational_maps(), maps None where the points of the fiber reach the cap of `meter`."""
    try:
        return _rational_maps(fib, meter)
    except CapError:
        return None, None


def _rational_maps(fib, meter):
    """(count, maps): the size of the fiber and the rational maps sigma onto other points of it, None if unknown.

    Coordinates with the fewest arguments come first: their characteristic polynomials over the fiber are
    interpolated and factored until the rational roots separate the points; the other coordinates are then
    interpolated on each branch."""
    sam = _Sampler(fib, meter)
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
        # separated: no two points of rational branches share a label
        if lab is not None and len(lab) == sum(
            1
            for fp in sam.points_of(sam.points(1)[0], primes[1])
            if _on_roots(sam, roots, fp, sam.points(1)[0], primes[1])
        ):
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
    """Whether every separating coordinate of the fiber point `fp` takes one of the root values of that coordinate."""
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
            # a copy with a free start keeps the output only from the image of the free start, which is no coordinate
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


def candidates(series, order, gap, p, meter=None):
    """The permutations of the coordinates that keep the output coefficients up to `order` (and `gap` in eps) at a
    random point modulo p: the transpositions within a class of equal patterns, then the products of two
    transpositions that fail alone. Each as {symbol: image} over all coordinates; each permutation tested spends one
    unit of `meter`, and the search stops at the cap."""
    meter = meter or work.unlimited()
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
            if meter.spend():
                return _as_maps(series, found)
            (found if vals(swapped([(i, j)])) == base else failed).append([(i, j)])
    for a, b in itertools.combinations(failed, 2):
        if {*a[0]} & {*b[0]} or any(pats[a[0][0]] == pats[x] for x in b[0]):
            continue
        if meter.spend():
            break
        if vals(swapped(a + b)) == base:
            found.append(a + b)
    return _as_maps(series, found)


def _as_maps(series, found):
    """The transpositions `found` as {symbol: image} over all coordinates."""
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
    None. With the initial values `ic`, a state whose initial value is a separate coordinate, the state itself
    or a parameter found nowhere else, takes the image of that coordinate with every such coordinate read as the
    state. Another state is mapped to itself, or, where that fails, to the solution of h(phi, sigma) = h(x, theta)
    for the output h and the Lie derivatives of h when these equations are linear in phi; with `ic`, phi must map the
    initial value at theta to the initial value at sigma(theta)."""
    F = {spy.Symbol(k): model_sympify(v) for k, v in f.items()}
    X = [spy.Symbol(x) for x in states]
    H = [model_sympify(h) for h in g.values()]
    used = set().union(*[model_sympify(v).free_symbols for v in list(f.values()) + list(g.values())])
    own = {}
    for x in X if ic is not None else ():
        q = model_sympify(str(ic.get(str(x), x)))
        if q == x or (
            q.is_Symbol
            and q in sig
            and q not in used
            and sum(model_sympify(str(ic.get(str(y), y))) == q for y in X) == 1
        ):
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
        # a state that starts at 1 with a rate x*R(x, theta) maps to x^c where R(sigma) = c R for a rational c
        alt = dict(phi)
        for x in given:
            R = spy.cancel(F[x] / x)
            if x0.get(x) != 1 or R.has(x) or R == 0:
                continue
            c = spy.cancel(R.xreplace({**th, **phi}) / R)
            if c.is_Rational and c not in (0, 1):
                alt[x] = x**c
        if alt != phi and holds(alt):
            return alt
        blk = _block_map(F, H, X, x0, th)
        if blk is not None and holds(blk):
            return blk
        if not given or len(given) > _SOLVE_STATES:
            return None
        P = {x: spy.Dummy(str(x)) for x in given}
        img = {**th, **{x: phi[x] for x in X if x not in P}, **P}
        eqs, rows, cap = [], list(H), work.CAPS["state_map_terms"]
        for _ in range(len(given) + 1):
            # the terms of a row bound those of the equation from it, before the row is put over one denominator
            bound = max((_term_bound(h) for h in rows), default=0)
            if bound > cap:
                _CAPPED.add(
                    f"a map of the states was not solved from the outputs: their Lie derivatives expand to "
                    f"up to {bound:.3g} terms, past the cap state_map_terms of {cap:,}"
                )
                return None
            eqs += [spy.fraction(spy.together(h.xreplace(img) - h))[0] for h in rows]
            rows = [sum((spy.diff(h, y) * F[y] for y in F), spy.Integer(0)) for h in rows]
        bound = max((_term_bound(e) for e in eqs), default=0)
        if bound > cap:
            _CAPPED.add(
                f"a map of the states was not solved from the outputs: their equations expand to up to "
                f"{bound:.3g} terms, past the cap state_map_terms of {cap:,}"
            )
            return None
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


def _term_bound(e, memo=None):
    """An upper bound on the terms of the expansion of `e`, from the expression tree without expanding: sums add,
    products multiply, a power to an integer n takes the bound to the |n|-th power, as a denominator enters the
    numerator over one denominator, anything else is one term."""
    memo = {} if memo is None else memo
    if e in memo:
        return memo[e]
    if e.is_Add:
        out = sum(_term_bound(a, memo) for a in e.args)
    elif e.is_Mul:
        out = math.prod(_term_bound(a, memo) for a in e.args)
    elif e.is_Pow and e.exp.is_Integer:
        out = _term_bound(e.base, memo) ** abs(int(e.exp))
    else:
        out = 1
    memo[e] = out
    return out


def state_map_proof(parts, sig, states, positive=True):
    """_state_map_proof(), and for a model with unknown inputs, where that fails, a map of the states and the inputs
    (uinputs.input_state_map)."""
    out = _state_map_proof(parts, sig, states, positive)
    if out is None and any(seg.get("inputs") for part in parts for seg in part["segments"]):
        from .uinputs import input_state_map

        out = input_state_map(parts, sig, states)
    return out


def _state_map_proof(parts, sig, states, positive=True):
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
                jets = set(seg.get("inputs") or ())
                given = [v for k, v in (seg.get("ic") or {None: None}).items() if k not in jets]
                ic = seg["ic"] if j == 0 and None not in given else None
                phi = _state_map(seg["f"], seg["g"], ic, sig, states)
                if phi is None:
                    return None
                # the identities take the jets of unknown inputs as constants: phi must not read them
                if jets and any({str(v) for v in e.free_symbols} & jets for e in phi.values()):
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
        if e.has(spy.I) and spy.im(e.xreplace(fib.known).xreplace(pos)) != 0:
            # a map over Q(i) whose image is not real at a real point never meets the domain
            return False, [f"{e} is not real"]
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
    coordinates, parameters with simple images first: theta is the least of the orbit in that order."""
    syms = [spy.Symbol(c) for c in coords]
    order = sorted(
        syms,
        key=lambda x: (str(x) in set(states), sum(spy.count_ops(sig.get(x, x)) for sig in maps), coords.index(str(x))),
    )
    out = []
    for sig in maps:
        c = next(x for x in order if sig.get(x, x) != x)
        out.append(f"{c} < {sig[c]}")
    return list(dict.fromkeys(out))


def choose_order(series, positive, start, cap, floor=1, proj=None, lowest=False, raises=None):
    """(fiber, stable, count): the fiber at the orders of the observables that SIAN's rank criterion gives (else
    `start` for each), raised together by one until the count over the algebraic closure equals the count one step
    higher, at most to `cap` and at most `raises` times (_ORDER_RAISES by default). A fiber past the caps of msolve
    ends the search: a higher order has more terms. A count equal to `floor`, the size of an orbit of proven maps,
    ends it as well: a higher order only removes points, and the orbit stays. With `proj`, the counts are those of
    the values of the first `proj` coordinates; with `lowest`, each observable starts one order above the order of
    that observable in the rank criterion."""
    raises = _ORDER_RAISES if raises is None else raises
    crit = series.output_orders(cap, PRIMES[0])
    # at the orders of the rank criterion the system is square and has extra solutions; one order higher it is
    # smaller in count and faster to solve. Every observable starts at the highest of these orders: an observable
    # whose next coefficient adds no rank in the coordinates can still remove points, where the start is given
    base = (
        ([c + 1 for c in crit] if lowest else [max(crit) + 1] * len(crit))
        if crit
        else [max(1, start)] * len(series.labels)
    )
    rng = session.current().rng("discrete order", tuple(series.coords))
    point = {spy.Symbol(c): spy.Integer(rng.randint(2, 10**6)) for c in series.coords}
    prev = found = None
    for t in range(min(cap, raises) + 1):
        orders = [n + t for n in base]
        if max(orders) > cap:
            break
        # the gap order rises with the Lie order, from zero gaps on
        fib = Fiber(series, orders, positive, gap=t)
        if proj is not None:
            fib.proj = proj
        cc, sized = _sized_count(fib, point)
        if prev is not None and cc is not None and cc == prev[1]:
            return prev[0], True, cc
        if cc == floor:
            # a higher order adds equations, so the fiber there lies in this one and holds the orbit of the point
            return fib, True, cc
        if cc is None and (getattr(fib, "too_large", None) or sized):
            # a higher order has more terms
            return (found[0], False, found[1]) if found is not None else (fib, False, None)
        prev = (fib, cc)
        if cc is not None:
            found = prev
    if found is not None:
        return found[0], False, found[1]
    if prev is None:
        fib = Fiber(series, base, positive)
        if proj is not None:
            fib.proj = proj
        return fib, False, None
    return prev[0], False, None


def _sized_count(fib, point):
    """(count, sized): complex_count() modulo the first prime, and whether a system of it was past the caps of
    msolve."""
    before = set(_FAILURES)
    _FAILURES.discard("size")
    cc = complex_count(fib, point, PRIMES[0])
    sized = "size" in _FAILURES
    _FAILURES.update(before)
    return cc, sized


def pinned_count(series, positive, floor, names=None):
    """{"why", "orders", "count", "cert", "fiber", "functions"}: the count of the fiber of pins of the series, functions
    proven in the field of the output coefficients (`names` their texts), and of the output coefficients up to the
    orders at which their gradients reach the number of coordinates, raised by one at a time until the count modulo a
    prime at a random point is below twice `floor`, the orbit of the proven maps; then over Q at a random integer point
    with the bound of count_certificate(), which holds where that count is below twice `floor` (docs/method/global.rst,
    "The exact count"). The pins of least degree that raise the rank and as many more come first, then all of them.
    `cert` None where the route does not apply or no count is below twice the orbit."""
    out = {"why": None, "orders": None, "count": None, "cert": None, "fiber": None, "functions": None}
    cap = 2 * len(series.coords)
    if series.has_events or any(
        seg.get("free") or seg.get("on") is not None for part in series.parts for seg in part["segs"]
    ):
        out["why"] = "the count of the proven functions covers models without events, switches and free starts"
        return out
    crit = series.output_orders(cap, PRIMES[0])
    if crit is None:
        out["why"] = "the output coefficients do not reach the rank of the coordinates"
        return out
    every = list(series.pins)
    try:
        for use in _pin_sets(series, PRIMES[0]):
            series.pins = [every[i] for i in use]
            out = _pinned_once(series, positive, floor, crit, cap)
            out["functions"] = [names[i] for i in use] if names else None
            if (out["cert"] or {}).get("bound") is not None:
                break
    finally:
        series.pins = every
    return out


def _pin_sets(series, p):
    """Index lists of the pins to count: those of least degree whose gradients raise the rank at the random point of
    Series.jacobian() modulo p with as many more of least degree, then all; all where a pin has a pole there."""
    every = list(range(len(series.pins)))
    pv = series.pin_values(p)
    if pv is None:
        return [every]
    order = sorted(every, key=lambda i: (_pin_degree(series.pins[i]), sum(len(P) for P in series.pins[i])))
    basis, piv, first = [], [], []
    for i in order:
        row = list(pv[i].d)
        for b, c in zip(basis, piv):
            if row[c]:
                row = [(x - row[c] * y) % p for x, y in zip(row, b)]
        c = next((j for j, x in enumerate(row) if x), None)
        if c is not None:
            inv = pow(row[c], -1, p)
            basis.append([x * inv % p for x in row])
            piv.append(c)
            first.append(i)
    first += [i for i in order if i not in first][: len(first)]
    return [sorted(first), every] if len(first) < len(every) else [every]


def _pinned_once(series, positive, floor, crit, cap):
    """pinned_count() with the pins of the series as they are."""
    out = {"why": None, "orders": None, "count": None, "cert": None, "fiber": None}
    base = series.pinned_orders(cap, PRIMES[0])
    if base is None:
        out["why"] = "the proven functions and the output coefficients do not reach the rank of the coordinates"
        return out
    if max(base) >= max(crit):
        out["why"] = "the proven functions do not lower the orders of the rank criterion"
        return out
    rng = session.current().rng("discrete pinned", tuple(series.coords))
    point = {spy.Symbol(c): spy.Integer(rng.randint(2, 10**6)) for c in series.coords}
    fib = cc = None
    # one order above the criterion first, as in choose_order(): at the criterion the starts often have extra points;
    # at most to the order of the rank criterion of the outputs alone, one below the first count without pins
    for t in range(1, max(crit) + 1 - max(base)):
        fib = Fiber(series, [n + t for n in base], positive)
        cc, sized = _sized_count(fib, point)
        # a higher order has more terms
        if cc is not None and cc < 2 * floor or getattr(fib, "too_large", None) or sized:
            break
    out.update(orders=fib.orders, count=cc, fiber=fib)
    if cc is None or cc >= 2 * floor:
        out["why"] = (
            f"the count of the proven functions is {cc}, not below {2 * floor}"
            if cc is not None
            else "msolve gave no count of the proven functions"
        )
        return out
    out["cert"] = count_certificate(fib, floor, base, within=(floor, 2 * floor))
    if out["cert"]["bound"] is None:
        out["why"] = out["cert"]["why"]
    return out


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


def _candidate_stage(series, parts, states, positive, meter):
    """(maps, info) of the candidates: permutations of coordinates with equal patterns that keep the output modulo a
    prime, each with the state maps that prove it or None."""
    t0 = time.monotonic()
    crit = series.output_orders(2 * len(series.coords), PRIMES[0])
    if crit is None:
        return [], {"seconds": round(time.monotonic() - t0, 2), "tested": False}
    order = max(crit) + 1 + _DEEPER
    gap = _DEEPER if series.has_events else 0
    found = candidates(series, order, gap, PRIMES[0], meter)
    out = [(sig, None if series.start_powers else state_map_proof(parts, sig, states, positive)) for sig in found]
    return out, {
        "order": order,
        "found": len(found),
        "proven": sum(pf is not None for _, pf in out),
        "seconds": round(time.monotonic() - t0, 2),
        "permutations": meter.spent,
        "complete_search": not meter.over,
    }


def analyze(parts, coords, positive, states, order=None, start=None, points=_POINTS, inputs=(), experiments=None):
    """The discrete symmetries of a locally identifiable model, as a dict; `parts` as in Series, one per condition.
    First the candidates: permutations of coordinates whose output coefficients have the same pattern, kept modulo a
    prime and proven by a map of the states. Then the count: without `order`, the order rises from `start` (the Lie
    order of the rank) until the count over the algebraic closure is stable; after an event the gap order rises with
    it from zero. Where the short count finds none, the count of several experiments comes first
    (multiexp.count()); `experiments` True or a number of experiments takes it before the exact count, False never.
    Every stage ends at the work caps of work.CAPS; a stage that reaches one names it in the limits."""
    series = Series(parts, coords, positive)
    syms = [spy.Symbol(c) for c in coords]
    # a free power of a state at 0 has no state maps and no exact count through the Taylor coefficients alone
    spow = series.start_powers
    timing = {}
    limits = []
    _FAILURES.clear()
    _CAPPED.clear()
    caps = work.CAPS
    series.meter = work.Meter("the Taylor series of the outputs", caps["series_products"], "term products")
    cmeter = work.Meter("the search of candidates", caps["candidates"], "permutations")
    cand, cinfo = _candidate_stage(series, parts, states, positive, cmeter)
    timing["candidates"] = cinfo["seconds"]
    if cinfo.get("complete_search") is False:
        limits.append(f"the candidates were not all tested: {cmeter.reason()}")
    cc = None
    proven = [sig for sig, pf in cand if pf is not None]
    cgrp = group(proven, syms) if proven else None
    # the exact count first, with the group of the proven candidates: where it holds, the fiber is the orbit and
    # neither the count nor the reconstruction is needed
    from .fieldproof import exact_fiber

    proof, tried = None, None
    # the constants of the model that are no coordinates enter the exact count as known symbols, in the base field
    consts = sorted(str(x) for x in series.known)
    ex = Series(parts, coords + consts, positive) if consts and not spow else series
    ex_syms = syms + [spy.Symbol(c) for c in consts]
    floor = len(cgrp) if cgrp else 1
    quick, me = None, None
    forced = experiments is not None and experiments is not False and not order
    if forced:
        me = _experiments(parts, coords, series, positive, floor, experiments, timing)
    if _final(me, floor, proven, cand):
        pass
    elif not order and len(proven) == len(cand):
        crit = series.output_orders(2 * len(coords), PRIMES[0])
        # a short count first: a count above the orbit of the candidates shows maps the exact count needs
        t2 = time.monotonic()
        with work.quick_msolve():
            quick = choose_order(series, positive, int(start or 1), 2 * len(coords), floor=floor, raises=1)
        timing["count"] = round(time.monotonic() - t2, 2)
        if crit is not None and not spow and (quick[2] is None or quick[2] == floor) and not series.inputs:
            t3 = time.monotonic()
            tried = cgrp or [{s: s for s in syms}]
            proof = exact_fiber(ex, [max(crit) + 3] * len(crit), tried, ex_syms, inputs, consts)
            timing["exact"] = round(time.monotonic() - t3, 2)
    pinned = verified = None
    if proof is not None and not proof["exact"]:
        extra = []
        if proof.get("large") and len(proven) == len(cand):
            # a projection past the terms the exact search interpolates: a few of the coefficients, verified
            t3 = time.monotonic()
            verified = _verified(ex, ex_syms, proof["large"])
            extra = verified["functions"]
            timing["relation"] = round(time.monotonic() - t3, 2)
        # the functions proven in the field of the outputs take their value at every point of the fiber
        series.pins, texts = _pins(list(proof.get("proven") or []) + extra, series, names=True)
        if series.pins and len(proven) == len(cand):
            # where the search stops short of the coordinates: the count of the fiber of the proven functions
            t3 = time.monotonic()
            pinned = pinned_count(series, positive, floor, texts)
            timing["pinned"] = round(time.monotonic() - t3, 2)
            _add_relation_bound(pinned, verified)
    certified = pinned is not None and pinned["cert"] is not None and pinned["cert"]["bound"] is not None
    if proof is not None and proof["exact"]:
        fib, stable = Fiber(series, [max(crit) + 1] * len(crit), positive), None
    elif certified:
        fib, stable, cc = pinned["fiber"], None, floor
    elif order:
        fib, stable = Fiber(series, int(order), positive, gap=int(order)), None
    elif quick is not None and quick[2] is not None:
        fib, stable, cc = quick
    else:
        if me is None and experiments is not False:
            # one experiment gave no short count: several experiments come first
            me = _experiments(parts, coords, series, positive, floor, None, timing)
        if _final(me, floor, proven, cand):
            # the count is below twice the orbit of the proven maps, which divides the size of the fiber
            crit1 = series.output_orders(2 * len(coords), PRIMES[0])
            fib = Fiber(series, [max(crit1) + 1] * len(crit1) if crit1 else int(start or 1), positive)
            stable, cc, me["final"] = me["stable"], floor, True
        else:
            t2 = time.monotonic()
            # the limit of a count without result names the failures of this count
            _FAILURES.clear()
            fib, stable, cc = choose_order(series, positive, int(start or 1), 2 * len(coords), floor=floor)
            timing["count"] = round(timing.get("count", 0.0) + time.monotonic() - t2, 2)
            if cc is None and me is not None and me.get("count") is not None:
                # the count of the experiments stands for that of one, with no maps to reconstruct
                cc, me["used"] = me["count"], True
    k, maps, false = None, None, []
    if proof is not None and proof["exact"]:
        k, maps = len(tried), []
        timing["maps"] = 0.0
    elif certified:
        # below twice the orbit of the proven candidates, which divides the size of the fiber
        k, maps = floor, []
        timing["maps"] = 0.0
    elif me is not None and me.get("used"):
        # one experiment gave no count: the maps of the fiber of one are not reconstructed
        k = cc
        limits.append(
            "the count is that of several experiments; the maps of the fiber of one were not "
            "reconstructed, since msolve gave no count of it"
        )
    elif cc is not None and cgrp is not None and len(cgrp) == cc and len(proven) == len(cand):
        # the proven candidates account for every point counted: the fiber has no other map to reconstruct
        k, maps = cc, []
        timing["maps"] = 0.0
    elif order or cc is not None:
        t2 = time.monotonic()
        mmeter = work.Meter("the reconstruction of the maps", caps["map_points"], "points of the fiber")
        # a map of the fiber that does not keep the coefficients further up shows a fiber too large at this order:
        # the orders rise and the count is taken again
        for _ in range(_RAISES + 1):
            # a count of one has no maps to find
            k, maps = (1, []) if cc == 1 else rational_maps(fib, mmeter)
            k = cc if k is None else k
            if series.inputs:
                from .uinputs import preserved
            else:
                preserved = output_preserved
            false = [sig for sig in maps or [] if not preserved(fib, sig, deeper=_DEEPER)]
            if not false or order or mmeter.over:
                break
            fib, stable, cc = Fiber(series, [n + 2 for n in fib.orders], positive, gap=fib.gap + 2), False, None
        timing["maps"] = round(time.monotonic() - t2, 2)
        if false:
            limits.append(
                f"{len(false)} of the maps of the fiber at order {fib.order} change the coefficients up "
                f"to order {fib.order + _DEEPER}: the count is that of the equations up to order "
                f"{fib.order}, more than the values with the same output"
            )
            maps = [sig for sig in maps if all(sig is not f for f in false)]
        if maps is None and mmeter.over:
            limits.append(f"the closed forms of the maps were not found: {mmeter.reason()}")
    elif getattr(fib, "unknowns", None):
        limits.append(
            f"the equations of the fiber at order {fib.order} have {fib.unknowns} unknowns, more than the "
            f"{_FIBER_UNKNOWNS} a count takes"
        )
    elif getattr(fib, "too_large", None):
        limits.append(
            f"the equations of the fiber at order {fib.order} have more than {_fiber_terms()} terms, the "
            "most a count takes (switch FIBER_TERMS)"
        )
    else:
        limits.append(no_count_reason(fib.order))
    # the candidates join the maps of the fiber
    proofs = {}
    every = list(maps or [])
    # numerical monodromy proposes the maps that the reconstruction over Q misses (detection.monodromy)
    from .monodromy import proposals

    t2 = time.monotonic()
    mono, mono_maps = proposals(series, fib, k, every + [sig for sig, _ in cand])
    if mono is not None:
        timing["monodromy"] = round(time.monotonic() - t2, 2)
        if "cap" in str(mono.get("why") or ""):
            limits.append(f"the monodromy stopped: {mono['why']}")
    every += mono_maps
    for sig, pf in cand:
        same = next((m for m in every if _same_map(sig, m)), None)
        if same is None:
            every.append(sig)
            same = sig
        proofs[id(same)] = pf
    out = [_entry(sig, proofs, maps, mono_maps, fib, syms, parts, states, positive, spow) for sig in every]
    exact = [sig for sig, e in zip(every, out) if e["level"] == "exact"]
    grp = group(exact, syms)
    orbit = None if grp is None else len(grp)
    # the proven maps account for every point counted: the reconstruction of the fiber adds nothing
    complete = k is not None and not false and orbit == k and all(e["level"] == "exact" for e in out)
    # the count of several experiments: below twice the orbit of the exact maps, which divides the size of the
    # fiber, it leaves the orbit; `single` is a count of one experiment equal to the orbit
    route, single = None, complete and not (me or {}).get("final") and not (me or {}).get("used")
    if me is not None and (me.get("used") or me.get("final")):
        route = me
    elif (
        me is not None
        and me.get("count")
        and orbit
        and not false
        and orbit <= me["count"] < 2 * orbit
        and all(e["level"] == "exact" for e in out)
        and (forced or not complete)
    ):
        route, complete, k = me, True, orbit
    # the exact count: the field fixed by the exact maps inside the field of the output coefficients
    # again with the group of all exact maps, where the reconstruction found more of them
    if (
        grp is not None
        and not spow
        and not (proof and proof["exact"])
        and (tried is None or len(grp) != len(tried))
        and not (route is not None and complete)
        and not series.inputs
    ):
        t3 = time.monotonic()
        # the functions the first search proved in the field of the outputs hold for any group
        pre = (proof or {}).get("proven", [])
        proof = exact_fiber(ex, fib.orders, grp, ex_syms, inputs, consts, pre)
        timing["exact"] = round(timing.get("exact", 0.0) + time.monotonic() - t3, 2)
        if not proof["exact"]:
            series.pins = _pins(list(pre) + list(proof.get("proven") or []), series)
    if proof is not None and proof["exact"] and grp is not None:
        complete = True
        k = orbit
    counts = []
    if k is not None and not (me or {}).get("used"):
        counts = _domain_counts(fib, maps, k, points, coords, grp[1:] if complete else None)
    if k is None:
        k = next((c["complex"] for c in counts if c.get("complex")), None)
    cb = None
    if certified:
        cb = pinned["cert"]
    elif k is not None and not (proof and proof["exact"]):
        t4 = time.monotonic()
        if route is not None:
            # the experiments over Q: below twice the orbit, or at most their count modulo the prime
            cb = count_certificate(
                route["fiber"], route["count"], route["crit"], within=(orbit, 2 * orbit) if complete else None
            )
            if cb["bound"] is None and single:
                # the count of one experiment equals the orbit as well and may verify where the experiments do not
                one = count_certificate(fib, k, series.output_orders(2 * len(coords), PRIMES[0]))
                if one["bound"] is not None:
                    route, cb = None, one
        else:
            cb = count_certificate(fib, k, series.output_orders(2 * len(coords), PRIMES[0]))
        timing["verified"] = round(time.monotonic() - t4, 2)
    if k is not None and orbit is not None and orbit < k:
        for c, g in power_coordinates(series).items():
            limits.append(
                f"{c} enters only as {c}^{g}: the fiber holds {c} -> zeta*{c} with zeta^{g} = 1, maps "
                "over Q(zeta) that the reconstruction over Q does not give"
            )
    relevant = [e for e in out if e.get("preserves_domain") is not False and e["level"] != "open"]
    # the fiber is the orbit of the group: a map that keeps the domain keeps the point inside, one that never does
    # moves it out
    count_domain = None
    if complete and grp is not None and orbit == k:
        keeps = [domain_conditions(fib, g)[0] for g in grp[1:]]
        if all(x is not None for x in keeps):
            count_domain = 1 + sum(keeps)
    limits += sorted(_CAPPED)
    if series.capped is not None:
        limits.append(
            f"the Taylor coefficients of the outputs up to order {series.capped} were not formed: "
            f"{series.meter.reason()}; the stages that need the orders of the rank criterion were skipped"
        )
    info = {
        "order": fib.order,
        "orders": dict(zip(series.labels, fib.orders)),
        "gap_order": fib.gap,
        "order_stable": stable,
        "count": k,
        "count_domain": count_domain,
        "count_bound": cb and cb["bound"],
        "counts": counts,
        "limits": limits,
        "closed_forms": maps is not None,
        "count_final": complete,
        "fundamental_domain": fundamental_domain(
            coords, states, [{spy.Symbol(s): model_sympify(v) for s, v in e["map"].items()} for e in relevant]
        ),
        "candidates": cinfo,
        "timing": timing,
        "monodromy": mono,
        "certificate": certificate(out, orbit, k, counts, complete, cb, proof, route, _pinned_info(pinned)),
    }
    if me is not None:
        info["experiments"] = _route_info(me)
    if spow:
        _start_power_info(info, spow)
    return {"symmetries": out, "info": info}


def _entry(sig, proofs, maps, mono_maps, fib, syms, parts, states, positive, spow):
    """The result entry of the map `sig`: the state maps from `proofs` (by id) or proven here, and the stages that
    found it."""
    entry = {
        "map": {str(s): str(e) for s, e in sig.items() if e != s},
        "changes": [str(s) for s in syms if sig[s] != s],
    }
    pf = proofs[id(sig)] if id(sig) in proofs else None if spow else state_map_proof(parts, sig, states, positive)
    entry["level"] = "exact" if pf is not None else "checked"
    entry["found_by"] = [
        w
        for w, hit in (
            ("candidates", id(sig) in proofs),
            ("fiber", any(sig is m for m in maps or [])),
            ("monodromy", any(sig is m for m in mono_maps)),
        )
        if hit
    ]
    entry["state_maps"] = pf
    entry["preserves_domain"], entry["domain_conditions"] = domain_conditions(fib, sig)
    entry["fixed_set"] = fixed_set(sig)
    return entry


def _start_power_info(info, spow):
    """The free powers of states at 0 in the info and the certificate: the exponent of the point holds at every point
    of the fiber in the domain, proven or under the premises named."""
    cert = info["certificate"]
    info["start_powers"] = []
    premises = []
    for sp in spow:
        pf = sp["proof"]
        premises += pf["premises"]
        info["start_powers"].append(
            {
                "state": sp["state"],
                "exponent": sp["alpha"],
                "order": sp["k"],
                "leading": pf["leading"],
                "observable": pf["observable"],
                "index": pf["index"],
                "coefficient": pf["coefficient"],
                "premises": list(pf["premises"]),
            }
        )
        cert.setdefault("proven", []).append(
            f"every point of the fiber in the domain has the exponent {sp['alpha']} of the point, by the lowest "
            f"exponent other than an integer of {pf['observable']}"
        )
    info["limits"].append(
        "the series of " + ", ".join(f"{sp['state']}^{sp['alpha']}" for sp in spow) + " are in t and in the "
        "layers (t^k)^(m alpha), with the normalized power (x/c_x)^alpha (info['discrete']['start_powers'])"
    )
    if premises:
        cert["premise"] = "; ".join(premises)
        if cert.get("complete") in ("exact", "verified"):
            cert["complete"] = "axiom"
            cert["axiom"] = cert["premise"]


def _experiments(parts, coords, series, positive, floor, experiments, timing):
    """multiexp.count(), timed."""

    from . import multiexp

    t = time.monotonic()
    me = multiexp.count(parts, coords, series, positive, floor, experiments)
    timing["experiments"] = round(time.monotonic() - t, 2)
    return me


def _final(me, floor, proven, cand):
    """Whether the count of the experiments is below twice the orbit of the proven candidates, all proven: the
    orbit divides the size of the fiber, which the count bounds."""
    return me is not None and bool(me.get("count")) and len(proven) == len(cand) and floor <= me["count"] < 2 * floor


def _verified(series, syms, large):
    """fewcoef.verified_functions() with the threads of msolve for the kernels, within the cap verified_entries of
    work.CAPS on the entries of the matrices reduced."""
    from . import fewcoef, ioproj

    rng = session.current().rng("discrete verified", tuple(series.coords))
    log, threads = [], ioproj.THREADS[0]
    ioproj.THREADS[0] = max(1, int(msolve._MSOLVE_THREADS))
    try:
        out = fewcoef.verified_functions(series, syms, large, work.CAPS["verified_entries"], rng, log)
    finally:
        ioproj.THREADS[0] = threads
    out["log"] = log
    return out


def _add_relation_bound(pinned, verified):
    """Adds to the bound of the count the probabilities of the verified relation, where the functions of the
    relation are counted: that no relation exists on the support, and per function counted that it is not the
    coefficient."""
    if verified is None:
        return
    pinned["relation"] = {
        k: verified.get(k)
        for k in ("terms", "reference", "functions", "exist", "checks", "trials", "kernels", "degree", "height", "why")
    }
    cb = pinned["cert"]
    used = [f for f in pinned["functions"] or [] if f in verified["checks"]]
    if cb is None or cb["bound"] is None or not used:
        return
    extra = verified["exist"] + sum(verified["checks"][f] for f in used)
    pinned["relation_bound"] = extra
    cb["count_bound"] = cb["bound"]
    cb["bound"] = cb["bound"] + extra


def _pinned_info(pinned):
    """The part of the result of pinned_count() that the certificate reports, None where it did not run."""
    if pinned is None:
        return None
    cb = pinned["cert"] or {}
    fib = pinned["fiber"]
    out = {
        "functions": pinned["functions"],
        "orders": dict(zip(fib.series.labels, fib.orders)) if fib else None,
        "count_modp": pinned["count"],
        "count": cb.get("count"),
        "count_bound": cb.get("bound"),
        "count_bits": cb.get("bits"),
        "bezout": cb.get("bezout"),
        "why": pinned["why"],
    }
    if pinned.get("relation") is not None:
        # the bound of the count over Q alone, and that of the verified relation, whose sum is count_bound
        out.update(
            relation=pinned["relation"],
            relation_bound=pinned.get("relation_bound"),
            count_bound_q=cb.get("count_bound", cb.get("bound")),
        )
    return out


def _route_info(me):
    """The part of the result of multiexp.count() that info["discrete"]["experiments"] reports."""
    keys = ("why", "experiments", "orders", "count", "failed", "stable", "seconds", "first_integrals")
    return {k: me[k] for k in keys if k in me}


def _monic(P):
    """The polynomial {exponents: Fraction} over the coefficient of the largest exponent vector."""
    c = P[max(P)]
    return {m: v / c for m, v in P.items()}


def _pins(proven, series, names=False):
    """(num, den) in the coordinates of each proven function that reads coordinates, the known constants at the values
    of the series; with `names`, also the text of the function of each."""
    out, texts = [], []
    for f in dict.fromkeys(proven):
        e = spy.cancel(model_sympify(str(f)).xreplace(series.known))
        if not e.free_symbols or e.free_symbols - set(series.cs):
            continue
        num, den = spy.fraction(e)
        # f and a constant multiple of f are one equation
        P = tuple(_monic(_dict(x, series.cs)) for x in (num, den))
        if P not in out:
            out.append(P)
            texts.append(str(f))
    return (out, texts) if names else out


def power_coordinates(series):
    """{parameter: g}: the parameters that every right-hand side, observable and start reads only through powers
    with exponents divisible by g > 2; g = 2 is the sign map, a rational map."""
    out = {}
    for pi, p in enumerate(series.params):
        exps = []
        for part in series.parts:
            ns = len(part["st"])
            for seg in part["segs"]:
                for N, D in list(seg["F"]) + list(seg["G"]):
                    exps += [m[ns + pi] for P in (N, D) for m in P]
                if "x0" in seg:
                    k = series.cs.index(p)
                    exps += [m[k] for v in seg["x0"] if v[1] is not None for P in v[1:] for m in P]
        g = math.gcd(*[e for e in exps if e]) if any(exps) else 0
        if g > 2:
            out[str(p)] = g
    return out


def _domain_counts(fib, maps, k, points, coords, proven=None):
    """Per point of the domain the count over the algebraic closure and the real points in the domain. With the
    elements `proven` of a group of proven maps whose orbit has `k` points, the fiber at a point is the orbit of the
    point."""
    rng = session.current().rng("discrete domain", tuple(coords))
    pts = (
        [domain_point(fib, rng) for _ in range(points)]
        if isinstance(points, int)
        else [{spy.Symbol(str(k_)): spy.Rational(str(v)) for k_, v in dict(pt).items()} for pt in points]
    )

    def count_at(pt):
        if proven is not None:
            boxes = _boxes_of_maps(fib, pt, proven, check=False)
            if boxes is not None:
                return None, boxes
        cc = (
            complex_count(fib, {s: v for s, v in pt.items()}, PRIMES[0])
            if all(v.q % PRIMES[0] for v in pt.values())
            else None
        )
        # where the maps account for every point counted, the point and the images are the fiber, exactly
        boxes = _boxes_of_maps(fib, pt, maps) if maps is not None and cc == k == len(maps) + 1 else None
        return cc, boxes if boxes is not None else real_points(fib, pt)

    # msolve gains little from threads: the points run as processes of one thread each
    workers = max(1, min(int(msolve._MSOLVE_THREADS), len(pts)))
    with msolve.msolve_threads(1):
        if workers > 1:
            with ThreadPoolExecutor(workers) as ex:
                done = list(ex.map(count_at, pts))
        else:
            done = [count_at(pt) for pt in pts]
    counts = []
    for pt, (cc, boxes) in zip(pts, done):
        if boxes is None:
            counts.append(
                {"point": {str(s): str(v) for s, v in pt.items()}, "complex": cc, "real": None, "in_domain": None}
            )
            continue
        dom = [in_domain(fib, b) for b in boxes]
        counts.append(
            {
                "point": {str(s): str(v) for s, v in pt.items()},
                "complex": cc,
                "real": len(boxes),
                "in_domain": sum(d is True for d in dom),
                "undecided": sum(d is None for d in dom),
            }
        )
    return counts


def _sparse_deg(P):
    return max((sum(e for _, e in mm) for mm in P), default=0)


def _fiber_degrees(fib):
    """The total degrees in (theta, u) of the fiber equations of docs/method/global.rst, "The verified count": the
    equations of the model, the saturation, and the output equations with the Taylor coefficients over one denominator,
    whose degrees in theta come from the recursion of the series on degree bounds; None at a pole."""
    A = Atoms()
    try:
        outs = fib.series._outputs([Deg(A, 1) for _ in fib.series.cs], fib.order, 0, Deg.const(A, 1), Deg.const(A, 0))
    except ZeroDivisionError:
        outs = None
    if outs is None:
        return None
    # per Taylor coefficient the coefficients in eps or in the layers of a free power of a state at 0
    vals = [[list(y[k]) for k in range(n + 1)] for y, n in zip(outs, fib.orders)]
    dq, nums = common([v for row in vals for ks in row for v in ks])
    it = iter(nums)
    bnum = [[[next(it) for _ in ks] for ks in row] for row in vals]
    degs = [_sparse_deg(P) for P in fib.static] + [1 + sum(_sparse_deg(f) for f in fib.sat)]
    # a pin num(x) den(theta) - den(x) num(theta)
    degs += [2 * max(_sparse_deg(_sparse(n)), _sparse_deg(_sparse(d))) for n, d in fib.series.pins]
    for (E, M), b, n in zip(fib.outs, bnum, fib.orders):
        for k in range(n + 1):
            for m in range(len(M[k])):
                terms = [
                    _sparse_deg(E[l_][a]) + b[k - l_][m - a]
                    for l_ in range(k + 1)
                    for a in range(m + 1)
                    if a < len(E[l_]) and E[l_][a] and m - a < len(b[k - l_]) and b[k - l_][m - a] is not None
                ]
                if M[k][m]:
                    terms.append(_sparse_deg(M[k][m]) + dq)
                if terms:
                    degs.append(max(terms))
    return degs


class _AllDigits:
    """Integers of any length convert to text: the points of the verified count reach thousands of digits."""

    def __enter__(self):
        self.limit = sys.get_int_max_str_digits()
        sys.set_int_max_str_digits(0)

    def __exit__(self, *exc):
        sys.set_int_max_str_digits(self.limit)


def _x_bezout(fib):
    """The Bezout number of n output equations c(x) = c(theta), cleared, whose gradients in x are independent,
    and of w s(x) = 1, s the product of the coordinates and of the denominators at the start; None where the rows
    are not found (docs/method/global.rst, "The verified count")."""
    series = fib.series
    if series.start_powers:
        # the leading coefficient of the base of a free power is a factor of the saturation in the jets
        return None
    # a coordinate pinned to the value at the point is the equation x_i = theta_i, of degree one
    pinned = [
        i
        for i in range(len(series.cs))
        for n, d in series.pins
        if d == {(0,) * len(series.cs): 1} and n == {tuple(int(j == i) for j in range(len(series.cs))): 1}
    ]
    # the rows of the outputs, and the rows of the pins before those of the outputs: the smaller product
    got = [_row_degrees(series, series.rank_rows(2 * len(series.cs), PRIMES[0], skip=sorted(set(pinned))))]
    if series.pins:
        got.append(_row_degrees(series, series.rank_rows(2 * len(series.cs), PRIMES[0], pins=True)))
    got = [d for d in got if d is not None]
    if not got:
        return None
    degs = min(got, key=math.prod)
    # the saturation in x alone: the starts substituted into the denominators of the right-hand sides and the
    # observables
    sat = spy.prod(spy.Symbol(c) for c in series.coords)
    for part in series.parts:
        seg = part["segs"][0]
        start = [spy.sympify(v[0]) if v[0] is not None else spy.Symbol(str(x)) for x, v in zip(part["st"], seg["x0"])]
        gens = list(part["st"]) + list(series.params)
        vals = dict(zip(gens, start + list(series.params)))

        def ev(Pd):
            return sum(
                (
                    spy.Rational(c.numerator, c.denominator) * spy.prod([g**e for g, e in zip(gens, m)])
                    for m, c in Pd.items()
                ),
                spy.Integer(0),
            ).xreplace(vals)

        dens = (
            [spy.fraction(spy.together(v))[1] for v in start]
            + [ev(D) for _, D in seg["F"]]
            + [ev(E) for _, E in seg["G"]]
        )
        for d in dens:
            sat = sat * spy.fraction(spy.together(d))[0] * spy.fraction(spy.together(d))[1]
    syms = [spy.Symbol(c) for c in series.coords]
    try:
        ds = spy.Poly(spy.expand(sat), *syms).total_degree()
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    return (1 + ds) * math.prod(degs)


def _pin_degree(pin):
    """The degree of num(x) den(theta) - num(theta) den(x) in (theta, x) for a pin (num, den)."""
    return sum(max((sum(m) for m in P), default=0) for P in pin)


def _row_degrees(series, rows):
    """The degrees in (theta, x) of the equations of the rows of rank_rows(): num(x) den(theta) - num(theta) den(x)
    of an output coefficient or a pin; None without rows."""
    if rows is None:
        return None
    A = Atoms()
    top = max((k for j, k in rows if j != "pin"), default=0)
    try:
        outs = series._outputs([Deg(A, 1) for _ in series.cs], top, 0, Deg.const(A, 1), Deg.const(A, 0))
    except ZeroDivisionError:
        return None
    if outs is None:
        return None
    degs = []
    for j, k in rows:
        if j == "pin":
            degs.append(max(1, _pin_degree(series.pins[k])))
            continue
        v = outs[j][k][0]
        degs.append(max(1, v.dn + A.of(v.den)))
    return degs


def q_count(fib, point):
    with _AllDigits():
        return _q_count(fib, point)


def _q_count(fib, point):
    """The number of points of the fiber over the complex numbers at the rational `point`, from the rational
    parametrization of msolve over Q per block; None on failure or a block of positive dimension."""
    try:
        sysd = fib.system(point, eliminate=True)
    except FiberTooLargeError:
        return None
    if sysd is None:
        return 0
    n = 1
    for block in sysd["blocks"]:
        k = _staged_count(fib, block["idx"], block["polys"], block["sat"], 0)
        if k is None:
            return None
        n *= k
    return n


def count_certificate(fib, k, crit, within=None):
    """{"bound", "why", "bits", "count", "bezout"}: the count taken again over Q at a random integer point, and the
    probability that the fiber at a generic point has more than `k` points (docs/method/global.rst, "The verified
    count"); bound None where it does not apply, with the reason. With `within` = (lo, hi), the bound holds for the
    count over Q where lo <= count < hi, and `count` is that count."""
    out = {"bound": None, "why": None, "bits": None, "count": None, "bezout": None}
    series = fib.series
    if series.has_events or fib.gap:
        out["why"] = "the bound covers models without events"
        return out
    if any(seg.get("free") or seg.get("on") is not None for part in series.parts for seg in part["segs"]):
        out["why"] = "the bound covers models without switches and free starts"
        return out
    if series.inputs:
        from .uinputs import count_certificate as with_inputs

        return with_inputs(fib, k, out)
    if crit is None or any(o < c for o, c in zip(fib.orders, crit)):
        out["why"] = "the orders are below those of the rank criterion"
        return out
    degs = _fiber_degrees(fib)
    if degs is None:
        out["why"] = "the Taylor coefficients have a pole"
        return out
    D = math.prod(max(1, int(d)) for d in degs)
    nu = len(fib.allnames) + 1
    # the variety in the coordinates and w alone, where the Bezout number is smaller
    Dx = _x_bezout(fib)
    if Dx is not None and (len(fib.names) + 1 + 2 * Dx) * Dx < (nu + 2 * D) * D:
        D, nu = Dx, len(fib.names) + 1
    out["bezout"] = D
    deg = (nu + 2 * D) * D
    # logarithms of integers: the Bezout number can pass the range of a float
    bits = max(_MIN_BITS, math.ceil(math.log2(deg) - math.log2(_VERIFIED_AIM)))
    if bits > _VERIFIED_BITS:
        out["why"] = f"the point would need coordinates of {bits} bits for a bound below {_VERIFIED_AIM:g}"
        return out
    out["bits"] = bits
    rng = session.current().rng("verified count", tuple(fib.coords))
    pt = {s: spy.Integer(rng.randint(1, 2**bits)) for s in fib.syms}
    # msolve over Q per block, plain and then with the separating form
    k2 = q_count(fib, pt)
    out["count"] = k2
    if not k2:
        out["why"] = f"no finite count over Q at the random integer point of {bits} bits" + (
            f": {no_count_reason(fib.order)}" if k2 is None and "size" in _FAILURES else ""
        )
        return out
    if within is None and k2 != k:
        out["why"] = f"the count over Q at a random integer point is {k2}, not {k}"
        return out
    if within is not None and not within[0] <= k2 < within[1]:
        out["why"] = f"the count over Q at a random integer point is {k2}, not from {within[0]} to {within[1] - 1}"
        return out
    out["bound"] = 2.0 ** (math.log2(deg) - bits)
    return out


def certificate(entries, orbit, k, counts, complete, cb=None, exact=None, route=None, pinned=None):
    """What the result rests on: the invariance proven per map, the size of the orbit, and the count: exact by the
    field of the outputs (`exact`, of fieldproof.exact_fiber), verified with the error bound `cb` of
    count_certificate(), or under the axiom of the count. With `route`, of multiexp.count(), the count is that of
    several experiments, which bounds the fiber of one where no rational first integral exists. With `pinned`, of
    _pinned_info(), the count is that of the fiber of the functions proven in the field of the outputs and the output
    coefficients at low orders."""
    # a point of the domain whose fiber is read from the orbit has no count over the algebraic closure
    agree = [c["complex"] for c in counts if c.get("complex") is not None]
    cb = cb or {"bound": None, "why": None}
    exact = exact or {"exact": False, "why": None}
    bound = None if exact.get("exact") else cb.get("bound")
    out = {
        "invariance": [
            {"map": e["map"], "proven": e["level"] == "exact", "state_maps": e.get("state_maps")} for e in entries
        ],
        "identities": "f(phi(x), sigma(theta)) = Dphi(x) f(x, theta), h(phi(x), sigma(theta)) = h(x, theta) and "
        "phi(x0(theta)) = x0(sigma(theta)), in every segment and through every event, as identities "
        "of rational functions over Q, or over Q(i) for a map with complex coefficients, checked by "
        "cancellation",
        "orbit": orbit,
        "count": k,
        "prime": PRIMES[0],
        "counts_at_points": agree,
        "proven": [
            "the invariance of every map at level exact",
            f"the orbit of a generic point under the group of the exact maps has {orbit} points",
        ],
        "exact": exact,
        "count_bound": bound,
        "count_bound_why": None if exact.get("exact") else cb.get("why"),
        "count_bits": cb.get("bits"),
        "bezout": cb.get("bezout"),
    }
    if exact.get("exact"):
        out["proven"].append(
            "the field fixed by the exact maps lies in the field of the output coefficients, so "
            f"the fiber at a generic point is the orbit of {orbit} points"
        )
        out["complete"] = "exact"
        return out
    if pinned is not None:
        out["pinned"] = pinned
    used = pinned is not None and pinned["count_bound"] is not None and route is None
    if used:
        out["proven"].append(
            f"the {len(pinned['functions'])} functions under pinned lie in the field of the output coefficients (the "
            "relations and generators under exact"
            + (
                ", and for the functions under pinned.relation the relation of the projection there, verified"
                if pinned.get("relation_bound") is not None
                else ""
            )
            + "), so they take one value on the fiber, which lies in the solutions of these functions and the output "
            "coefficients up to the orders under pinned"
        )
        if complete:
            out["proven"].append(
                f"the orbit of {orbit} point{'' if orbit == 1 else 's'} divides the size of the "
                f"fiber, and a count below {2 * orbit} leaves the orbit"
            )
    if route is not None:
        K, fi = route["experiments"], route["first_integrals"]
        out["route"] = {"experiments": K, "orders": route["orders"], "count": route["count"], "first_integrals": fi}
        out["proven"].append(
            "no nonconstant rational first integral at a generic point: the eigenvalues of the Jacobian at an "
            f"equilibrium over {fi['field']} at one point of the parameters are linearly independent over Z "
            "(Cong, Llibre and Zhang 2011, Theorem 1), and a first integral at a generic point would give one there"
        )
        out["proven"].append(
            f"so the values of the parameters and the starts of the first of {K} experiments with shared parameters "
            "and free starts, over their fiber, are the fiber of one experiment"
        )
        if complete:
            out["proven"].append(
                f"the orbit of {orbit} point{'' if orbit == 1 else 's'} divides the size of the "
                f"fiber, and a count below {2 * orbit} leaves the orbit"
            )
    if bound is None:
        out["axiom"] = (
            "the number of solutions modulo the prime at a random point equals the number over the "
            "complex numbers at a generic point; no error bound is proven"
        )
        why = [w for w in (exact.get("why"), cb.get("why")) if w]
        out["count_bound_why"] = "; ".join(why) or None
    elif used:
        n = cb["count"]
        top = max(pinned["orders"].values())
        out["bounded"] = (
            f"the count of the proven functions and the output coefficients up to order {top} over Q "
            f"at a random integer point is {n}, so the fiber at a generic point has at most {n} "
            f"point{'' if n == 1 else 's'}, except with probability at most {bound:.3g}"
            + (
                f", of which {pinned['relation_bound']:.3g} for the verified relation"
                if pinned.get("relation_bound") is not None
                else ""
            )
        )
    elif route is not None:
        n = cb["count"]
        out["bounded"] = (
            f"the count of the {route['experiments']} experiments over Q at a random integer point is "
            f"{n}, at least the number of values of the parameters and the starts of the first there, so "
            f"the fiber of one experiment at a generic point has at most {n} point"
            f"{'' if n == 1 else 's'}, except with probability at most {bound:.3g}"
        )
    else:
        out["bounded"] = (
            f"the fiber at a generic point has at most {k} points, except with probability at most "
            f"{bound:.3g} (the count over Q at a random integer point)"
        )
    out["complete"] = (
        ("verified" if bound is not None else "axiom")
        if complete
        else ("no" if k is not None and orbit is not None and orbit < k else "open")
    )
    return out


def observed_states(f, g, events=()):
    """The states the observables depend on through the right-hand sides and the event values, in the order of
    `f`: a state outside never changes the output."""
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
    """{state: resting value} of a condition at rest, the inputs at zero: a state that is a coordinate holds a free
    resting value, a state without dynamics rests at the initial value, and the others solve the balances over
    Q(theta), where they have one rational solution with the most states away from zero. A ValueError where they
    have none, or where the Bezout number of the balances in the states is past the cap rest_bezout of
    work.CAPS, which SymPy is not given."""
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
    # a transcendental balance in the states has no rational closed form for sympy to find, only a long search
    odd = [str(x) for x, e in zip(U, bal) if not e.is_polynomial(*U)]
    if odd:
        from .rationalize import NotRationalError

        raise NotRationalError(
            "discrete() under equilibrate needs a resting state with one rational closed form in "
            f"the coordinates; the balances of {', '.join(odd)} are not rational in the states"
        )
    cap = work.CAPS["rest_bezout"]
    bezout = math.prod(max(1, spy.Poly(e, *U).total_degree()) for e in bal if e != 0)
    if bezout > cap:
        raise ValueError(
            f"discrete() under equilibrate does not solve the balances of {', '.join(map(str, U))}: "
            f"their Bezout number {bezout} in the states is past the cap rest_bezout of {cap}"
        )
    block = _largest_block(bal, U)
    cap = work.CAPS["rest_block"]
    if block > cap:
        raise ValueError(
            f"discrete() under equilibrate does not solve the balances of {', '.join(map(str, U))}: "
            f"their largest coupled block of {block} states is past the cap rest_block of {cap}"
        )
    sols = _solve_balances(bal, U)
    sols = [
        s for s in sols if set(s) >= set(U) and not any(v.has(spy.I) or v.free_symbols & set(U) for v in s.values())
    ]
    if sols:
        best = max(sum(s[x] != 0 for x in U) for s in sols)
        sols = [s for s in sols if sum(s[x] != 0 for x in U) == best]
    # an exponential of the coordinates is left to the rational form of the fiber (rationalize)
    if len(sols) != 1 or not all(
        _rational(
            sols[0][x].replace(lambda a: isinstance(a, spy.exp) and _rational(a.args[0]), lambda a: spy.Integer(2))
        )
        for x in U
    ):
        raise ValueError(
            "discrete() under equilibrate needs a resting state with one rational closed form in the "
            f"coordinates; the balances of {', '.join(map(str, U))} have {len(sols)} such solutions"
        )
    out.update({str(x): str(spy.cancel(sols[0][x])) for x in U})
    return out


def _largest_block(bal, U):
    """The states of the largest strongly connected block of the balances `bal`, each read as the balance of the
    state at the same place in `U`, an edge to every state it holds."""
    edges = [(x, y) for x, e in zip(U, bal) for y in e.free_symbols & set(U) if y != x]
    return max(len(c) for c in strongly_connected_components((U, edges)))


def _solve_balances(bal, U):
    """The solutions of the balances `bal` in the states `U` by SymPy; none where SymPy fails."""
    try:
        return spy.solve(bal, U, dict=True)
    except (NotImplementedError,) + SYMPY_FAILS as exc:
        sympy_fail(exc)
        return []


def _rational(e):
    """Whether `e` is a rational function, no root or other function."""
    return all(p.exp.is_Integer for p in e.atoms(spy.Pow)) and not e.atoms(spy.Function)
