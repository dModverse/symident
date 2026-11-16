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
from ..gfp import PRIMES, rat_recon, rref_modp
from ..sym.domain import declared, is_positive
from ..sym.gfpoly import _gf_roots
from ..sym.modp import _modp_rational
from ..sym.parse import model_sympify

# total degree past which a coefficient of a characteristic polynomial is not interpolated
_DEG_CAP = 6
# sample points of the real count in the domain
_POINTS = 5


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


class Series:
    """The model in prolonged form, as in SIAN: the derivatives of the states at the start are unknowns, tied by
    the coefficients of D(x) x' = N(x) and E(x) y = M(x) as power series in t."""

    def __init__(self, f, g, ic, coords, states):
        self.coords = list(coords)
        F = {k: model_sympify(v) for k, v in f.items()}
        G = [model_sympify(v) for v in g.values()]
        X0 = {k: model_sympify(v) for k, v in ic.items()}
        allsyms = set().union(*[e.free_symbols for e in list(F.values()) + G + list(X0.values())])
        self.st = [spy.Symbol(x) for x in f]
        cs = [spy.Symbol(c) for c in self.coords]
        self.params = [c for c in cs if c not in set(self.st)]
        rng = session.current().rng("discrete known", tuple(self.coords))
        # a symbol that is neither a state nor a coordinate is known: one value in a point and in the solutions
        self.known = {x: spy.Integer(rng.randint(2, 9)) for x in sorted(allsyms - set(self.st) - set(cs), key=str)}
        gens = self.st + self.params

        def frac(e):
            num, den = spy.fraction(spy.cancel(spy.together(e.xreplace(self.known))))
            return _dict(num, gens), _dict(den, gens)
        self.F = [frac(F[str(x)]) for x in self.st]
        self.G = [frac(e) for e in G]
        # the start of each state: a coordinate, a number, or a rational function of the coordinates
        self.x0 = []
        for x in self.st:
            v = spy.sympify(X0.get(str(x), x)).xreplace(self.known)
            if x not in cs and v == x:
                v = spy.Integer(rng.randint(2, 9))
            num, den = spy.fraction(spy.cancel(v))
            self.x0.append((v, _dict(num, cs), _dict(den, cs)))
        self.cs = cs
        # a state with a numeric start whose right-hand side reads only such states (the jets of a known input,
        # the clock) has a known series: it enters as numbers, not as unknowns
        ns = len(self.st)
        known = set()
        while True:
            more = {i for i in range(ns) if i not in known and self.x0[i][0].is_number and
                    all(not e or j in known for P in self.F[i] for m in P for j, e in enumerate(m[:ns])) and
                    all(not any(m[ns:]) for P in self.F[i] for m in P)}
            if not more:
                break
            known |= more
        self.known_states = known
        self._kser = {}

    def _known_series(self, order):
        """{state index: Taylor coefficients up to `order`} of the states with a known series, as Fractions."""
        if order not in self._kser:
            one, zero = Fraction(1), Fraction(0)
            def start(v):
                v = spy.Rational(v)
                return Fraction(int(v.p), int(v.q))
            xs = [[start(self.x0[i][0])] + [zero] * order if i in self.known_states else [zero] * (order + 1)
                  for i in range(len(self.st))]
            for k in range(order):
                for i in sorted(self.known_states):
                    N, D = self.F[i]
                    nser = _ser_poly(N, xs, [zero] * len(self.params), k, one, zero)
                    dser = _ser_poly(D, xs, [zero] * len(self.params), k, one, zero)
                    rest = sum((dser[l] * (k + 1 - l) * xs[i][k + 1 - l] for l in range(1, k + 1)), zero)
                    xs[i][k + 1] = (nser[k] - rest) / ((k + 1) * dser[0])
            self._kser[order] = {i: xs[i] for i in self.known_states}
        return self._kser[order]

    def _outputs(self, cvals, order, one, zero):
        """The Taylor coefficients of the observables up to `order`, the coordinates at `cvals` (in the arithmetic
        of one and zero), None at a pole."""
        val = dict(zip(self.cs, cvals))
        params = [val[s] for s in self.params]
        kser = self._known_series(order)
        xs = []
        for i, (_, num, den) in enumerate(self.x0):
            if i in kser:
                xs.append([one * c for c in kser[i]])
                continue
            d = _eval(den, cvals, zero)
            if d == 0:
                return None
            xs.append([_eval(num, cvals, zero) / d] + [zero] * order)
        # x_{k+1} from coefficient k of D(x) x' = N(x), with D at the start nonzero
        for k in range(order):
            for i, (N, D) in enumerate(self.F):
                if i in kser:
                    continue
                nser = _ser_poly(N, xs, params, k, one, zero)
                dser = _ser_poly(D, xs, params, k, one, zero)
                if dser[0] == 0:
                    return None
                rest = sum((dser[l] * (k + 1 - l) * xs[i][k + 1 - l] for l in range(1, k + 1)), zero)
                xs[i][k + 1] = (nser[k] - rest) / ((k + 1) * dser[0])
        out = []
        for M, E in self.G:
            mser = _ser_poly(M, xs, params, order, one, zero)
            eser = _ser_poly(E, xs, params, order, one, zero)
            if eser[0] == 0:
                return None
            y = []
            for k in range(order + 1):
                y.append((mser[k] - sum((eser[l] * y[k - l] for l in range(1, k + 1)), zero)) / eser[0])
            out.append(y)
        return out

    def numeric(self, point, order, p=0):
        """The Taylor coefficients of the observables at `point` ({symbol: Rational}) up to `order`, one list per
        observable, as Fractions or, with p, as integers modulo p; None at a pole."""
        cv = [Fraction(int(spy.Rational(point[s]).p), int(spy.Rational(point[s]).q)) for s in self.cs]
        if not p:
            return self._outputs(cv, order, Fraction(1), Fraction(0))
        try:
            out = self._outputs([_Dual(0, [], p) + c for c in cv], order, _Dual(1, [], p), _Dual(0, [], p))
        except ZeroDivisionError:
            return None
        return None if out is None else [[x.v for x in y] for y in out]

    def output_orders(self, cap, p):
        """Per observable the least order whose next Taylor coefficient adds no rank to the Jacobian in the
        coordinates, as in SIAN, at a random point modulo p; None where the rank stays below the number of
        coordinates up to `cap`."""
        rng = session.current().rng("discrete orders", tuple(self.coords))
        m = len(self.cs)
        cv = [_Dual(rng.randint(2, p - 1), [int(i == j) for j in range(m)], p) for i in range(m)]
        one, zero = _Dual(1, [0] * m, p), _Dual(0, [0] * m, p)
        try:
            ys = self._outputs(cv, cap, one, zero)
        except ZeroDivisionError:
            return None
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
                if adds(ys[j][k].d):
                    orders[j] = k
                elif k:
                    active.remove(j)
            if len(basis) == m or not active:
                break
        return orders if len(basis) == m else None

    def prolonged(self, orders):
        """(names, equations, outputs, sat) up to the order of each observable in `orders`: the jet variables after
        the coordinates, the equations of the model and the start as {exponents: Fraction} over coordinates and
        jets, the series (E, M) of each observable, and the factors that must not vanish."""
        order = max(orders)
        from sympy.polys.domains import QQ
        from sympy.polys.rings import ring
        names = [str(c) for c in self.cs]
        kser = self._known_series(order)
        jets = {}
        for i in range(len(self.st)):
            if i in kser:
                continue
            v = self.x0[i][0]
            lo = 0 if not (v.is_Symbol and v in self.cs) and not v.is_number else 1
            for k in range(lo, order + 1):
                jets[(i, k)] = f"si_j{i}_{k}"
        R, *gens = ring(",".join(names + list(jets.values())), QQ)
        g = dict(zip(names + list(jets.values()), gens))
        cof = {str(c): g[str(c)] for c in self.cs}
        params = [g[str(p)] for p in self.params]
        eqs = []
        xs = []
        for i in range(len(self.st)):
            v, num, den = self.x0[i]
            if i in kser:
                xs.append([R(spy.Rational(c.numerator, c.denominator)) for c in kser[i]])
                continue
            if (i, 0) in jets:
                a0 = g[jets[(i, 0)]]
                N0 = R.from_dict({m: c for m, c in num.items()}) if num else R.zero
                D0 = R.from_dict({m: c for m, c in den.items()})
                eqs.append(D0 * a0 - N0)
            elif v.is_number:
                a0 = R(spy.Rational(v))
            else:
                a0 = cof[str(v)]
            xs.append([a0] + [g[jets[(i, k)]] for k in range(1, order + 1)])
        sat = [cof[str(c)] for c in self.cs]
        for i, (N, D) in enumerate(self.F):
            if i in kser:
                continue
            nser = _ser_poly(N, xs, params, order - 1, R.one, R.zero) if order else []
            dser = _ser_poly(D, xs, params, order - 1, R.one, R.zero) if order else []
            for k in range(order):
                lhs = sum((dser[l] * (k + 1 - l) * xs[i][k + 1 - l] for l in range(0, k + 1)), R.zero)
                eqs.append(lhs - nser[k])
            d0 = _ser_poly(D, xs, params, 0, R.one, R.zero)[0]
            if not d0.is_ground:
                sat.append(d0)
        outs = []
        for (M, E), n in zip(self.G, orders):
            mser = _ser_poly(M, xs, params, n, R.one, R.zero)
            eser = _ser_poly(E, xs, params, n, R.one, R.zero)
            if not eser[0].is_ground:
                sat.append(eser[0])
            outs.append((eser, mser))
        return list(jets.values()), [_rdict(e) for e in eqs if e], \
            [([_rdict(e) for e in E], [_rdict(m) for m in M]) for E, M in outs], [_rdict(f) for f in sat]


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

    def __init__(self, series, orders, positive=True):
        self.series = series
        self.coords = list(series.coords)
        self.syms = [spy.Symbol(c) for c in self.coords]
        # one order per observable; a number is the same order for all
        self.orders = [int(orders)] * len(series.G) if isinstance(orders, int) else [int(n) for n in orders]
        self.order = max(self.orders)
        self.positive = positive
        self.known = series.known
        self.names = [f"x{i}" for i in range(len(self.syms))]
        jets, self.static, self.outs, sat = series.prolonged(self.orders)
        self.allnames = self.names + jets
        self.sat = [_clear(f) for f in sat]
        # per characteristic, whether msolve gets the separating form (decided at the first call)
        self.sep_mode = {}

    def values(self, point, p=0):
        """The Taylor coefficients of the observables at `point` ({symbol: Rational}), modulo p with p, None at a
        pole."""
        return self.series.numeric(point, self.order, p)

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
        for (E, M), b, n in zip(self.outs, vals, self.orders):
            for k in range(n + 1):
                P = {mm: -c for mm, c in M[k].items()}
                for l in range(k + 1):
                    for mm, c in E[l].items():
                        P[mm] = P.get(mm, 0) + c * b[k - l]
                polys.append(_clear({mm: c for mm, c in P.items() if c}, p))
        polys = [P for P in polys if P]
        sat = [{m: c % p if p else c for m, c in f.items()} for f in self.sat]
        out = _fix_linear(polys, sat, len(self.allnames), p)
        if out is None:
            return None
        polys, sat, fixed = out
        keep = [i for i in range(len(self.allnames)) if i not in fixed]
        # blocks: the components of the variables joined by an equation or a factor of the saturation
        parent = {i: i for i in keep}

        def find(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i
        for P in polys + sat:
            vs = sorted({i for m in P for i, e in enumerate(m) if e})
            for a, b_ in zip(vs, vs[1:]):
                parent[find(a)] = find(b_)
        comp = {}
        for i in keep:
            comp.setdefault(find(i), []).append(i)
        m = len(self.names)
        blocks = []
        for idx in comp.values():
            loc = {i: j for j, i in enumerate(idx)}
            names = [self.allnames[i] for i in idx]

            def restrict(P, idx=idx):
                return {tuple(mm[i] for i in idx): c for mm, c in P.items()}
            eqs = [_ms_dict(_clear(restrict(P), p), names) for P in polys
                   if {i for mm in P for i, e in enumerate(mm) if e} & set(loc)]
            fac = {(0,) * len(idx): 1}
            for f in sat:
                if {i for mm in f for i, e in enumerate(mm) if e} & set(loc):
                    fac = _pmul_dict(fac, _clear(restrict(f), p), p)
            # points where a coordinate or a denominator at the start vanishes do not count; msolve reads no
            # brackets, so si_w multiplies each term
            eqs.append(_ms_dict({mm + (1,): c for mm, c in fac.items()}, names + ["si_w"]) + "-1")
            blocks.append({"names": names, "sep": [self.names[i] for i in idx if i < m], "eqs": eqs})
        return {"blocks": blocks, "fixed": {self.names[i]: v for i, v in fixed.items() if i < m}}


def _pmul_dict(a, b, p):
    """The product of two polynomials {exponents: integer}, modulo p unless p is 0."""
    out = {}
    for ma, ca in a.items():
        for mb, cb in b.items():
            mm = tuple(x + y for x, y in zip(ma, mb))
            out[mm] = (out.get(mm, 0) + ca * cb) % p if p else out.get(mm, 0) + ca * cb
    return {mm: c for mm, c in out.items() if c}


def _fix_linear(polys, sat, nv, p):
    """(polys, sat, fixed): the variables that an equation a*v + b with numbers a, b fixes, substituted in turn in
    the equations and the factors `sat`; None where an equation becomes a nonzero number or a factor vanishes."""
    def div(a, b):
        return a * pow(b, -1, p) % p if p else Fraction(a) / b

    def subst(P, v, x):
        out = {}
        for m, c in P.items():
            e = m[v]
            if e:
                c = c * pow(x, e, p) % p if p else c * Fraction(x) ** e
                m = m[:v] + (0,) + m[v + 1:]
            out[m] = (out.get(m, 0) + c) % p if p else out.get(m, 0) + c
        return {m: c for m, c in out.items() if c}
    fixed = {}
    zero = (0,) * nv
    while True:
        hit = None
        for P in polys:
            vs = {i for m in P for i, e in enumerate(m) if e}
            if len(vs) == 1 and all(sum(m) <= 1 for m in P):
                v = vs.pop()
                a = P.get(zero[:v] + (1,) + zero[v + 1:], 0)
                hit = (v, div(-P.get(zero, 0), a))
                break
        if hit is None:
            return polys, sat, fixed
        v, x = hit
        fixed[v] = x
        polys = [subst(P, v, x) for P in polys]
        if any(list(P) == [zero] for P in polys):
            return None
        polys = [P for P in polys if P]
        sat = [subst(f, v, x) for f in sat]
        if any(not f for f in sat):
            return None
        # a factor that is a number vanishes nowhere
        sat = [f for f in sat if list(f) != [zero]]


def _eval(P, x, zero=Fraction(0)):
    return sum((c * math.prod((xi ** e for xi, e in zip(x, m) if e), start=zero + 1) for m, c in P.items()), zero)


def _ms_dict(P, names):
    """A polynomial {exponents: integer or Fraction} as msolve input, expanded: msolve reads no brackets."""
    out = []
    for m, c in P.items():
        mono = "*".join(n + (f"^{e}" if e > 1 else "") for n, e in zip(names, m) if e)
        out.append(f"{c}*{mono}" if mono else f"{c}")
    return "+".join(out).replace("+-", "-") or "0"


def _msolve(names, p, eqs, flags, timeout, sep=()):
    """The parsed output of msolve on `eqs` in `names` + si_w over GF(p) (p = 0: Q), or None on failure. With
    `sep`, a last variable si_t is a linear form in these names with fixed coefficients: it separates distinct
    points, where si_w need not, so msolve parametrizes by it without trying other forms."""
    from ..sym import msolve
    flags = list(flags) + ["-t", str(msolve._MSOLVE_THREADS)]
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
        txt = open(fo).read() if os.path.exists(fo) else ""
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
    """_msolve() on a block of Fiber.system(), with the separating form si_t where the first call on the block
    found it faster: without it, msolve may try several forms when si_w does not separate the points; with it, a
    dense form costs more."""
    import time
    names, eqs, sep = block["names"], block["eqs"], block["sep"]
    key = (p == 0, tuple(names))
    mode = fib.sep_mode.get(key)
    if mode is not None or not sep:
        return _msolve(names, p, eqs, flags, timeout, sep if mode else ())
    t0 = time.monotonic()
    plain = _msolve(names, p, eqs, flags, timeout)
    t1 = time.monotonic()
    with_sep = _msolve(names, p, eqs, flags, timeout, sep)
    fib.sep_mode[key] = with_sep is not None and (plain is None or time.monotonic() - t1 < t1 - t0)
    return with_sep if fib.sep_mode[key] else plain


def complex_count(fib, point, p, timeout):
    """The number of points of the fiber over the algebraic closure of GF(p), None on failure."""
    rur = _rur(fib, point, p, timeout)
    return None if rur is None else rur["deg"]


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


def charpolys(rur, p):
    """{variable: coefficients c_0..c_{K-1} of the monic characteristic polynomial} over the fiber, the product of
    the fibers of the blocks: a variable of a block of k points has its polynomial over the block to the power
    K/k; a fixed coordinate c has (Z - c)^K. None on failure."""
    K = rur["deg"]
    out = {}
    for b in rur["blocks"]:
        cp = _charpolys_block(b, p)
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


def _charpolys_block(rur, p):
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
            m[1] = charpolys(rur, p) or False
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


def output_preserved(fib, sig, points=3):
    """Whether the output coefficients agree at sigma(theta) and theta at random rational points."""
    rng = session.current().rng("discrete check", tuple(fib.coords))
    for _ in range(points):
        pt = {s: spy.Rational(rng.randint(1, 97), rng.randint(1, 13)) for s in fib.syms}
        img = {s: e.xreplace(pt) for s, e in sig.items()}
        a, b = fib.values(pt), fib.values(img)
        if a is None or b is None or any(x != y for x, y in zip(a, b)):
            return False
    return True


def state_map_proof(f, g, ic, sig, states):
    """Whether phi maps the model at theta onto the model at sigma(theta), over Q(x, theta). A state whose initial
    value is a coordinate of its own, the state itself or a parameter found nowhere else, takes the image of that
    coordinate with every such coordinate read as its state; another state is mapped to itself, and sigma must
    map its initial value."""
    F = {spy.Symbol(k): model_sympify(v) for k, v in f.items()}
    X = [spy.Symbol(x) for x in states]
    used = set().union(*[model_sympify(v).free_symbols for v in list(f.values()) + list(g.values())])
    own = {}
    for x in X:
        q = model_sympify(str(ic.get(str(x), x)))
        if q == x or (q.is_Symbol and q in sig and q not in used and
                      sum(model_sympify(str(ic.get(str(y), y))) == q for y in X) == 1):
            own[q] = x
    phi = {x: x for x in X}
    for q, x in own.items():
        phi[x] = sig.get(q, q).xreplace(own)
    th = {s: e for s, e in sig.items() if s not in set(X) and s not in own}
    both = {**th, **phi}
    for x in X:
        lhs = F[x].xreplace(both)
        rhs = sum((spy.diff(phi[x], y) * F[y] for y in X), spy.Integer(0))
        if spy.cancel(lhs - rhs) != 0:
            return False
    for x in F:
        if x not in set(X) and spy.cancel(F[x].xreplace(both) - F[x]) != 0:
            return False
    for h in g.values():
        h = model_sympify(h)
        if spy.cancel(h.xreplace(both) - h) != 0:
            return False
    # phi(x0(theta)) = x0(sigma(theta)) for the given initial values
    x0 = {x: model_sympify(str(ic.get(str(x), x))) for x in X}
    for x in X:
        if x in own.values():
            continue
        if spy.cancel(phi[x].xreplace(x0).xreplace(th) - x0[x].xreplace(th)) != 0:
            return False
    return True


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


def choose_order(series, positive, start, cap, timeout, budget=math.inf):
    """(fiber, stable): the fiber at the orders of the observables that SIAN's rank criterion gives (else `start`
    for each), raised together by one until the count over the algebraic closure equals the count one step higher,
    at most to `cap`, or within `budget` seconds."""
    import time
    t0 = time.monotonic()
    base = series.output_orders(cap, PRIMES[0]) or [max(1, start)] * len(series.G)
    rng = session.current().rng("discrete order", tuple(series.coords))
    point = {spy.Symbol(c): spy.Integer(rng.randint(2, 10 ** 6)) for c in series.coords}
    prev = None
    for t in range(cap + 1):
        orders = [n + t for n in base]
        if max(orders) > cap or (prev is not None and time.monotonic() - t0 > budget):
            break
        fib = Fiber(series, orders, positive)
        cc = complex_count(fib, point, PRIMES[0], timeout)
        if prev is not None and cc is not None and cc == prev[1]:
            return prev[0], True
        prev = (fib, cc)
    if prev is None:
        return Fiber(series, base, positive), False
    return prev[0], False


def analyze(f, g, ic, coords, positive, states, order=None, start=None, points=_POINTS, timeout=600,
            budget=300):
    """The discrete symmetries of a locally identifiable model with one condition, as a dict. Without `order`, the
    order rises from `start` (the Lie order of the rank) until the count over the algebraic closure is stable."""
    series = Series(f, g, ic, coords, states)
    if order:
        fib, stable = Fiber(series, int(order), positive), None
    else:
        fib, stable = choose_order(series, positive, int(start or 1), 2 * len(coords), timeout, budget)
    order = fib.order
    rng = session.current().rng("discrete domain", tuple(coords))
    counts = []
    pts = [domain_point(fib, rng) for _ in range(points)] if isinstance(points, int) else \
        [{spy.Symbol(str(k)): spy.Rational(str(v)) for k, v in dict(pt).items()} for pt in points]
    for pt in pts:
        boxes = real_points(fib, pt, timeout)
        cc = complex_count(fib, {s: v for s, v in pt.items()}, PRIMES[0], timeout) \
            if all(v.q % PRIMES[0] for v in pt.values()) else None
        if boxes is None:
            counts.append({"point": {str(s): str(v) for s, v in pt.items()}, "complex": cc, "real": None,
                           "in_domain": None})
            continue
        dom = [in_domain(fib, b) for b in boxes]
        counts.append({"point": {str(s): str(v) for s, v in pt.items()}, "complex": cc, "real": len(boxes),
                       "in_domain": sum(d is True for d in dom),
                       "undecided": sum(d is None for d in dom)})
    import time
    t0 = time.monotonic()
    k, maps = rational_maps(fib, timeout, budget)
    limits = []
    if maps is None and time.monotonic() - t0 > budget:
        limits.append(f"the closed forms of the maps were not found within the budget of {budget:g} s")
    if k is None:
        k = next((c["complex"] for c in counts if c.get("complex")), None)
    if k is None:
        limits.append("msolve found no finite count: the equations of the fiber have a component of positive "
                      f"dimension up to order {fib.order}, which the denominators and coordinates do not exclude")
    syms = list(fib.syms)
    out = []
    for sig in maps or []:
        entry = {"map": {str(s): str(e) for s, e in sig.items() if e != s},
                 "changes": [str(s) for s in syms if sig[s] != s]}
        if not output_preserved(fib, sig):
            entry["level"] = "open"
            out.append(entry)
            continue
        entry["level"] = "exact" if state_map_proof(f, g, ic, sig, [x for x in states if x in f]) else "checked"
        entry["preserves_domain"], entry["domain_conditions"] = domain_conditions(fib, sig)
        entry["fixed_set"] = fixed_set(sig)
        out.append(entry)
    relevant = [e for e in out if e.get("preserves_domain") is not False and e["level"] != "open"]
    info = {"order": order, "orders": dict(zip(list(g), fib.orders)), "order_stable": stable, "count": k,
            "counts": counts, "limits": limits,
            "closed_forms": maps is not None,
            "count_final": maps is not None and len(maps) + 1 == k and all(e["level"] == "exact" for e in out),
            "fundamental_domain": fundamental_domain(coords, states, [
                {spy.Symbol(s): model_sympify(v) for s, v in e["map"].items()} for e in relevant])}
    return {"symmetries": out, "info": info}
