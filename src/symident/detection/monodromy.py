"""Points of the fiber by numerical monodromy, certified by the Krawczyk operator: proposals of discrete maps.

The fiber of the output coefficients is overdetermined. A square subsystem of coefficients with independent
gradients is solved by parameter homotopies around random loops in the space of the values, starting from the
known point; the points that also keep the other coefficients are proposed as points of the fiber. Each is then
enclosed in a box with a unique zero of the square subsystem (docs/method/global.rst, "Monodromy and certified points").
The numbers only propose: a map read from the points counts once it is proven exactly."""

import itertools
from fractions import Fraction

import numpy as np
import sympy as spy

from .. import session
from ..gfp import PRIMES
from . import work

# step bounds of the tracker, in the homotopy parameter
_H_MAX, _H_MIN = 0.05, 1e-9
# a tracked point past this modulus has left for infinity
_FAR = 1e8
# relative tolerance of a point of the fiber on the coefficients outside the square subsystem
_TOL = 1e-7
# the most points of the square subsystem modulo a prime that monodromy is run on
_SQUARE = 200


class _Batch:
    """Values at a batch of points with the gradients over the coordinates, in complex floating point: the
    arithmetic of Series._outputs for many points at once."""
    __slots__ = ("v", "d")

    def __init__(self, v, d=None):
        self.v, self.d = v, d

    def _of(self, x):
        if isinstance(x, _Batch):
            return x
        return _Batch(np.full(self.v.shape, complex(float(x)) if isinstance(x, Fraction) else complex(x)))

    def __add__(self, x):
        x = self._of(x)
        d = self.d if x.d is None else x.d if self.d is None else self.d + x.d
        return _Batch(self.v + x.v, d)

    __radd__ = __add__

    def __neg__(self):
        return _Batch(-self.v, None if self.d is None else -self.d)

    def __sub__(self, x):
        return self + (-self._of(x))

    def __rsub__(self, x):
        return self._of(x) - self

    def __mul__(self, x):
        if not isinstance(x, _Batch):
            c = complex(float(x)) if isinstance(x, Fraction) else complex(x)
            return _Batch(self.v * c, None if self.d is None else self.d * c)
        if self.d is None:
            d = None if x.d is None else x.d * self.v[:, None]
        elif x.d is None:
            d = self.d * x.v[:, None]
        else:
            d = self.d * x.v[:, None] + x.d * self.v[:, None]
        return _Batch(self.v * x.v, d)

    __rmul__ = __mul__

    def __truediv__(self, x):
        x = self._of(x)
        if np.any(x.v == 0):
            raise ZeroDivisionError("division by zero at a point of the batch")
        q = self.v / x.v
        if x.d is None:
            return _Batch(q, None if self.d is None else self.d / x.v[:, None])
        d = -x.d * q[:, None] if self.d is None else self.d - x.d * q[:, None]
        return _Batch(q, d / x.v[:, None])

    def __rtruediv__(self, x):
        return self._of(x) / self

    def __pow__(self, e):
        if e == 0:
            return self._of(1)
        v = self.v ** e
        return _Batch(v, None if self.d is None else self.d * (e * self.v ** (e - 1))[:, None])

    def __eq__(self, x):
        return bool(np.any(self.v == self._of(x).v))

    __hash__ = None


def _flat(ys, orders, gap, free):
    """The coefficients of the observables as one list, coefficient k of observable j up to orders[j], each with
    the coefficients in eps up to `gap`; the rows of copies with a free start are left out."""
    out, index = [], []
    for j, y in enumerate(ys):
        if j in free:
            continue
        for k in range(min(orders[j], len(y) - 1) + 1):
            for m, c in enumerate(y[k][:gap + 1]):
                out.append(c)
                index.append((j, k, m))
    return out, index


def coefficients(series, X, orders, gap=0, jac=True):
    """(values (P, M), Jacobian (P, M, n) or None, index) of the output coefficients at the rows of X (P, n),
    complex; None where a point of the batch meets a pole."""
    X = np.atleast_2d(np.asarray(X, dtype=complex))
    P, n = X.shape
    eye = np.eye(n, dtype=complex)
    cv = [_Batch(X[:, i].copy(), np.repeat(eye[i][None, :], P, axis=0) if jac else None) for i in range(n)]
    one, zero = _Batch(np.ones(P, dtype=complex)), _Batch(np.zeros(P, dtype=complex))
    try:
        ys = series._outputs(cv, max(orders), gap, one, zero)
    except (ZeroDivisionError, FloatingPointError, OverflowError):
        return None
    if ys is None:
        return None
    rows, index = _flat(ys, orders, gap, series.free_rows)
    vals = np.stack([r.v if isinstance(r, _Batch) else np.full(P, complex(r)) for r in rows], axis=1)
    J = None
    if jac:
        J = np.stack([r.d if isinstance(r, _Batch) and r.d is not None else np.zeros((P, n), dtype=complex)
                      for r in rows], axis=1)
    return vals, J, index


class _System:
    """The fiber at a base point: the square subsystem `rows` of the coefficients, solved for given values, and all
    coefficients up to `orders`, which a point of the fiber keeps."""

    def __init__(self, series, orders, rows, base, gap=0):
        self.series, self.orders, self.gap = series, list(orders), gap
        self.n = len(series.cs)
        self.base = np.asarray([complex(float(Fraction(str(base[s])))) for s in series.cs])
        sub = [0] * len(orders)
        for j, k in rows:
            sub[j] = max(sub[j], k)
        self.sub_orders = sub
        got = coefficients(series, self.base, sub, 0, jac=False)
        if got is None:
            raise ZeroDivisionError("the base point is a pole")
        _, _, index = got
        pos = {(j, k): i for i, (j, k, m) in enumerate(index) if m == 0}
        self.rows = [pos[r] for r in rows]
        self.target = got[0][0, self.rows]
        full = coefficients(series, self.base, self.orders, gap, jac=False)
        if full is None:
            raise ZeroDivisionError("the base point is a pole")
        self.full = full[0][0]

    def square(self, X):
        """(values, Jacobian) of the square subsystem at the rows of X; None at a pole."""
        got = coefficients(self.series, X, self.sub_orders, 0)
        if got is None:
            return None
        v, J, _ = got
        return v[:, self.rows], J[:, self.rows, :]

    def residual(self, X):
        """The largest relative deviation of the coefficients up to `orders` at the rows of X from those at the
        base point, inf at a pole."""
        got = coefficients(self.series, X, self.orders, self.gap, jac=False)
        if got is None:
            return np.full(len(np.atleast_2d(X)), np.inf)
        dev = np.abs(got[0] - self.full[None, :]) / (1 + np.abs(self.full[None, :]))
        return dev.max(axis=1)


def _solve(J, r):
    """The solutions of the batch of linear systems J x = r, a row of NaN where one is singular."""
    try:
        return np.linalg.solve(J, r[..., None])[..., 0]
    except np.linalg.LinAlgError:
        out = np.full(r.shape, np.nan, dtype=complex)
        for i in range(len(r)):
            try:
                out[i] = np.linalg.solve(J[i], r[i])
            except np.linalg.LinAlgError:
                pass
        return out


def _eval_safe(F, X):
    """F at the rows of X, row by row where the batch meets a pole: (values, Jacobian, ok)."""
    got = F(X)
    if got is not None:
        return got[0], got[1], np.ones(len(X), dtype=bool)
    n = X.shape[1]
    v = np.full((len(X), n), np.nan, dtype=complex)
    J = np.full((len(X), n, n), np.nan, dtype=complex)
    ok = np.zeros(len(X), dtype=bool)
    for i in range(len(X)):
        g = F(X[i:i + 1])
        if g is not None:
            v[i], J[i], ok[i] = g[0][0], g[1][0], True
    return v, J, ok


def track(F, X0, wa, wb, meter=None, weight=1):
    """The end points at w = wb of the paths of F(x) = w from the points X0 at w = wa along the segment, by an
    Euler predictor and a Newton corrector with step control; a row of NaN where a path fails or stops at the cap
    of `meter`, which counts `weight` units per step of all paths at once."""
    meter = meter or work.unlimited()
    X = np.array(X0, dtype=complex)
    P = len(X)
    t = np.zeros(P)
    h = np.full(P, _H_MAX / 2)
    live = np.ones(P, dtype=bool)
    dw = wb - wa
    while True:
        act = np.where(live & (t < 1))[0]
        if not len(act) or meter.spend(weight):
            break
        x = X[act]
        hh = np.minimum(h[act], 1 - t[act])
        _, J, ok = _eval_safe(F, x)
        dx = _solve(np.where(ok[:, None, None], J, np.eye(J.shape[1])), np.repeat(dw[None, :], len(act), axis=0))
        xp = x + hh[:, None] * dx
        tp = t[act] + hh
        w = wa[None, :] + tp[:, None] * dw[None, :]
        good = ok & np.all(np.isfinite(xp), axis=1)
        first = None
        for _ in range(3):
            v, J, okc = _eval_safe(F, xp)
            step = _solve(np.where(okc[:, None, None], J, np.eye(J.shape[1])), np.where(okc[:, None], v - w, 0))
            good &= okc & np.all(np.isfinite(step), axis=1)
            sn = np.where(good, np.linalg.norm(np.where(np.isfinite(step), step, 0), axis=1), np.inf)
            if first is None:
                first = sn
            xp = np.where(good[:, None], xp - step, xp)
        scale = 1 + np.linalg.norm(np.where(np.isfinite(xp), xp, 0), axis=1)
        conv = good & (sn <= 1e-9 * scale) & (first <= 0.05 * scale)
        far = np.linalg.norm(np.where(np.isfinite(xp), xp, _FAR * 10), axis=1) > _FAR
        X[act[conv]] = xp[conv]
        t[act[conv]] = tp[conv]
        h[act[conv]] = np.minimum(h[act[conv]] * 1.6, _H_MAX)
        h[act[~conv]] /= 2
        dead = (h[act] < _H_MIN) | (conv & far)
        live[act[dead]] = False
    X[~live | (t < 1)] = np.nan
    return X


def _newton(F, X, w, steps=6):
    """Newton steps on F(x) = w from the rows of X, a row of NaN where a step fails."""
    for _ in range(steps):
        v, J, ok = _eval_safe(F, X)
        step = _solve(np.where(ok[:, None, None], J, np.eye(J.shape[1])), np.where(ok[:, None], v - w, 0))
        X = np.where((ok & np.all(np.isfinite(step), axis=1))[:, None], X - step, np.nan)
    return X


def _new(points, x, tol=1e-6):
    """Whether x differs from every one of `points` relative to tol."""
    s = 1 + np.linalg.norm(x)
    return all(np.linalg.norm(p - x) > tol * s for p in points)


def monodromy(system, rng, meter, target=None, stall=5, cap=400):
    """(fiber, square): the points of the square subsystem of `system` found by loops w0 -> w1 -> w2 -> w0 in the
    space of the values from the base point, and those among them that keep every coefficient up to the orders of
    the system. Stops after `stall` loops without a new point, at `cap` points, at `target` points of the fiber
    or at the cap of `meter` on the work of the steps of the paths (_step_work())."""
    n = system.n
    F = system.square
    w0 = system.target
    sols = [system.base.copy()]
    quiet = 0
    while quiet < stall and len(sols) < cap and not meter.over:
        scale = 1 + np.abs(w0)
        # nodes away from w0: a triangle through them winds around the branch locus more often
        w1 = scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        w2 = scale * (rng.standard_normal(n) + 1j * rng.standard_normal(n))
        X = np.array(sols)
        for a, b in ((w0, w1), (w1, w2), (w2, w0)):
            X = track(F, X, a, b, meter, _step_work(system))
            X = X[np.all(np.isfinite(X), axis=1)]
            if not len(X):
                break
        X = _newton(F, X, w0[None, :]) if len(X) else X
        added = 0
        for x in X:
            if np.all(np.isfinite(x)) and _new(sols, x):
                sols.append(x)
                added += 1
        quiet = 0 if added else quiet + 1
        if target is not None:
            res = system.residual(np.array(sols))
            if int(np.sum(res < _TOL)) >= target:
                break
    S = np.array(sols)
    res = system.residual(S)
    return S[res < _TOL], S


def _step_work(system):
    """The work of one step of the paths: four evaluations of the square subsystem with the Jacobian, each the
    Taylor series of the model up to the orders of the subsystem, (k + 1)^2 products per term at order k."""
    size = sum(len(N) + len(D) for part in system.series.parts for seg in part["segs"] for N, D in seg["F"])
    top = max(system.sub_orders) + 1
    return 4 * size * (system.n + 1) * sum((k + 1) ** 2 for k in range(top))


def rur_points(fib, point, digits=40):
    """The complex points of the fiber at the rational `point`, coordinates in the order of the series, from the
    rational parametrization of msolve over Q per block, the roots taken numerically; None on failure."""
    import mpmath

    from .discrete import FiberTooLargeError
    from .discrete import _solve as solve_block
    try:
        sysd = fib.system(point)
    except (FiberTooLargeError, ZeroDivisionError):
        return None
    if sysd is None:
        return None
    per = []
    for block in sysd["blocks"]:
        data = solve_block(fib, 0, block, ["-P", "1"])
        if data is None or data[0] != 0:
            return None
        _, _, deg, vn, lin, par = data[1]
        if deg <= 0 or par[0] != 1:
            return None
        elim, den, params = par[1][0][1], par[1][1][1], par[1][2]
        with mpmath.workdps(digits):
            def mp(c):
                c = Fraction(c)
                return mpmath.mpf(c.numerator) / c.denominator

            def ev(cs, t):
                return mpmath.polyval([mp(c) for c in reversed(cs)], t)

            roots = mpmath.polyroots([mp(c) for c in reversed(elim)], maxsteps=400, extraprec=4 * digits) \
                if len(elim) > 2 else [-mp(elim[0]) / mp(elim[1])]
            sols = []
            last = len(vn) - 1
            for t in roots:
                dv = ev(den, t)
                if dv == 0:
                    continue
                vals = {vn[j]: -ev(q[0][1], t) / (mp(q[1]) * dv) for j, q in enumerate(params)}
                vals[vn[last]] = (t - sum((mp(lin[j]) * vals[vn[j]] for j in range(last)), mpmath.mpf(0))) / \
                    mp(lin[last])
                sols.append({k: complex(v) for k, v in vals.items()})
        per.append(sols)
    out = []
    for combo in itertools.product(*per):
        merged = {k: v for sol in combo for k, v in sol.items()}
        merged.update({k: complex(float(v)) for k, v in sysd["fixed"].items()})
        if any(nm not in merged for nm in fib.names):
            return None
        out.append([merged[nm] for nm in fib.names])
    return np.array(out, dtype=complex).reshape(len(out), len(fib.names))


# ---- the Krawczyk operator in interval arithmetic --------------------------------------------

def _dn(x):
    return np.nextafter(x, -np.inf)


def _up(x):
    return np.nextafter(x, np.inf)


class _Rect:
    """Complex rectangles [rl, rh] + i [il, ih], arrays of one shape, with outward rounding: each floating-point
    operation is correctly rounded, so one step to the next float outward encloses the exact result."""
    __slots__ = ("rl", "rh", "il", "ih")

    def __init__(self, rl, rh, il, ih):
        self.rl, self.rh, self.il, self.ih = rl, rh, il, ih

    @staticmethod
    def point(z):
        z = np.asarray(z, dtype=complex)
        return _Rect(z.real.copy(), z.real.copy(), z.imag.copy(), z.imag.copy())

    @staticmethod
    def const(x, shape=()):
        try:
            f = float(x) if isinstance(x, (int, Fraction)) else None
        except OverflowError:
            return _Rect(np.full(shape, -np.inf), np.full(shape, np.inf), np.zeros(shape), np.zeros(shape))
        if f is not None and Fraction(f) == Fraction(x):
            return _Rect(np.full(shape, f), np.full(shape, f), np.zeros(shape), np.zeros(shape))
        if f is not None:
            return _Rect(np.full(shape, _dn(f)), np.full(shape, _up(f)), np.zeros(shape), np.zeros(shape))
        return _Rect.point(np.full(shape, complex(x)))

    def __add__(self, o):
        return _Rect(_dn(self.rl + o.rl), _up(self.rh + o.rh), _dn(self.il + o.il), _up(self.ih + o.ih))

    def __neg__(self):
        return _Rect(-self.rh, -self.rl, -self.ih, -self.il)

    def __sub__(self, o):
        return self + (-o)

    def __mul__(self, o):
        def mul(a, b, c, d):
            p = np.stack([a * c, a * d, b * c, b * d])
            bad = np.isnan(p).any(axis=0)
            return np.where(bad, -np.inf, _dn(p.min(axis=0))), np.where(bad, np.inf, _up(p.max(axis=0)))

        acl, ach = mul(self.rl, self.rh, o.rl, o.rh)
        bdl, bdh = mul(self.il, self.ih, o.il, o.ih)
        adl, adh = mul(self.rl, self.rh, o.il, o.ih)
        bcl, bch = mul(self.il, self.ih, o.rl, o.rh)
        return _Rect(_dn(acl - bdh), _up(ach - bdl), _dn(adl + bcl), _up(adh + bch))

    def scale(self, lo, hi):
        """The product with the real interval [lo, hi]."""
        return self * _Rect(lo, hi, np.zeros_like(lo), np.zeros_like(lo))

    def has_zero(self):
        return (self.rl <= 0) & (self.rh >= 0) & (self.il <= 0) & (self.ih >= 0)

    def inverse(self):
        def sq(a, b):
            lo = np.where((a <= 0) & (b >= 0), 0.0, _dn(np.minimum(a * a, b * b)))
            return lo, _up(np.maximum(a * a, b * b))

        cl, ch = sq(self.rl, self.rh)
        dl, dh = sq(self.il, self.ih)
        nl, nh = _dn(cl + dl), _up(ch + dh)
        if np.any(~(nl > 0)):
            raise ZeroDivisionError("an enclosure of a divisor holds zero")
        conj = _Rect(self.rl, self.rh, -self.ih, -self.il)
        return conj.scale(_dn(1 / nh), _up(1 / nl))

    def mag(self):
        """Upper bounds of the moduli, through |re| + |im|."""
        return _up(np.maximum(np.abs(self.rl), np.abs(self.rh)) + np.maximum(np.abs(self.il), np.abs(self.ih)))

    def __getitem__(self, i):
        return _Rect(self.rl[i], self.rh[i], self.il[i], self.ih[i])


def _sum(rects, shape):
    out = _Rect.const(0, shape)
    for r in rects:
        out = out + r
    return out


class _IDual:
    """An enclosure of a value with enclosures of the gradient over the coordinates: the arithmetic of
    Series._outputs over a box."""
    __slots__ = ("v", "d")
    n = 0

    def __init__(self, v, d=None):
        self.v, self.d = v, d

    def _of(self, x):
        return x if isinstance(x, _IDual) else _IDual(_Rect.const(x))

    def __add__(self, x):
        x = self._of(x)
        d = self.d if x.d is None else x.d if self.d is None else self.d + x.d
        return _IDual(self.v + x.v, d)

    __radd__ = __add__

    def __neg__(self):
        return _IDual(-self.v, None if self.d is None else -self.d)

    def __sub__(self, x):
        return self + (-self._of(x))

    def __rsub__(self, x):
        return self._of(x) - self

    def __mul__(self, x):
        x = self._of(x)
        if self.d is None:
            d = None if x.d is None else x.d * _bcast(self.v)
        elif x.d is None:
            d = self.d * _bcast(x.v)
        else:
            d = self.d * _bcast(x.v) + x.d * _bcast(self.v)
        return _IDual(self.v * x.v, d)

    __rmul__ = __mul__

    def __truediv__(self, x):
        x = self._of(x)
        r = x.v.inverse()
        q = self.v * r
        if x.d is None:
            return _IDual(q, None if self.d is None else self.d * _bcast(r))
        num = -(x.d * _bcast(q)) if self.d is None else self.d - x.d * _bcast(q)
        return _IDual(q, num * _bcast(r))

    def __rtruediv__(self, x):
        return self._of(x) / self

    def __pow__(self, e):
        out = self._of(1)
        for _ in range(e):
            out = out * self
        return out

    def __eq__(self, x):
        # a value whose enclosure holds zero cannot be divided by
        return bool((self - self._of(x)).v.has_zero())

    __hash__ = None


def _bcast(r):
    n = _IDual.n
    return _Rect(np.full(n, r.rl), np.full(n, r.rh), np.full(n, r.il), np.full(n, r.ih))


def _ivalues(series, cv, orders, gap, rows):
    """The coefficients `rows` as _IDual at the enclosures `cv` of the coordinates; None at a pole."""
    one, zero = _IDual(_Rect.const(1)), _IDual(_Rect.const(0))
    try:
        ys = series._outputs(cv, max(orders), gap, one, zero)
    except ZeroDivisionError:
        return None
    if ys is None:
        return None
    flat, _ = _flat(ys, orders, gap, series.free_rows)
    return [flat[r] if isinstance(flat[r], _IDual) else _IDual(_Rect.const(flat[r])) for r in rows]


def square_rows(series, orders, gap, x):
    """n rows of the coefficients up to `orders` with independent gradients at x, chosen by a pivoted QR
    decomposition of the Jacobian with rows of unit norm: the best conditioned square subsystem it finds."""
    import scipy.linalg as sl
    got = coefficients(series, np.asarray(x)[None, :], orders, gap)
    if got is None:
        return None
    A = got[1][0]
    nrm = np.linalg.norm(A, axis=1)
    keep = np.where(nrm > 0)[0]
    if len(keep) < A.shape[1]:
        return None
    _, _, piv = sl.qr((A[keep] / nrm[keep, None]).T, pivoting=True)
    return sorted(int(keep[i]) for i in piv[:A.shape[1]])


def krawczyk(series, orders, gap, rows, base, x, radius):
    """The box around x (complex, one entry per coordinate) with the given radius per coordinate, as
    [((re_lo, re_hi), (im_lo, im_hi))], if the Krawczyk operator proves a unique zero of the square subsystem
    c_rows(x') = c_rows(base) in it, else None; `rows` index the coefficients up to `orders` as coefficients()
    lists them, and `base` holds rationals."""
    n = len(series.cs)
    _IDual.n = n
    tb = _ivalues(series, [_IDual(_Rect.const(Fraction(str(base[s])))) for s in series.cs], orders, gap, rows)
    xm = np.asarray(x, dtype=complex)
    gm = _ivalues(series, [_IDual(_Rect.point(z)) for z in xm], orders, gap, rows)
    if tb is None or gm is None:
        return None
    g = [a.v - b.v for a, b in zip(gm, tb)]
    r = np.asarray(radius, dtype=float)
    box = _Rect(_dn(xm.real - r), _up(xm.real + r), _dn(xm.imag - r), _up(xm.imag + r))
    eye = np.eye(n)
    cv = [_IDual(box[i], _Rect(eye[i].copy(), eye[i].copy(), np.zeros(n), np.zeros(n))) for i in range(n)]
    JX = _ivalues(series, cv, orders, gap, rows)
    if JX is None or any(c.d is None for c in JX):
        return None
    # Y: a floating-point inverse of the Jacobian at the midpoint, with exact entries
    Jm = np.array([(c.d.rl + c.d.rh) / 2 + 1j * (c.d.il + c.d.ih) / 2 for c in JX])
    try:
        Y = np.linalg.inv(Jm)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(Y)):
        return None
    J = _Rect(np.array([c.d.rl for c in JX]), np.array([c.d.rh for c in JX]),
              np.array([c.d.il for c in JX]), np.array([c.d.ih for c in JX]))
    G = _Rect(np.array([v.rl for v in g]), np.array([v.rh for v in g]), np.array([v.il for v in g]),
              np.array([v.ih for v in g]))
    dX = box - _Rect.point(xm)
    for i in range(n):
        Yi = _Rect.point(Y[i])
        yg = _sum([(Yi[l:l + 1] * G[l:l + 1]) for l in range(n)], (1,))
        # row i of I - Y J(X)
        E = _Rect.const(0, (n,)) - _sum([Yi[l:l + 1] * J[l] for l in range(n)], (n,))
        E.rl[i], E.rh[i] = _dn(E.rl[i] + 1), _up(E.rh[i] + 1)
        # the row sum of |I - Y J(X)| weighted by the radii, below the radius: a contraction in the weighted norm
        q = float(_up(np.sum(E.mag() * r) * (1 + 4 * n * np.finfo(float).eps)))
        k = _Rect.point(xm[i:i + 1]) - yg + _sum([E[j:j + 1] * dX[j:j + 1] for j in range(n)], (1,))
        inside = k.rl[0] > box.rl[i] and k.rh[0] < box.rh[i] and k.il[0] > box.il[i] and k.ih[0] < box.ih[i]
        if not inside or not q < r[i]:
            return None
    return [((float(box.rl[i]), float(box.rh[i])), (float(box.il[i]), float(box.ih[i]))) for i in range(n)]


def certify(series, orders, gap, base, x):
    """The Krawczyk box of x on the square subsystem of square_rows(), trying radii from small to large; None if
    none is proven."""
    with np.errstate(all="ignore"):
        rows = square_rows(series, orders, gap, x)
        if rows is None:
            return None
        for r in (1e-10, 1e-7):
            box = krawczyk(series, orders, gap, rows, base, x, r * (1 + np.abs(x)))
            if box is not None:
                return box
    return None


def disjoint(a, b):
    """Whether two boxes of krawczyk() share no point: some coordinate has disjoint rectangles."""
    for (ra, ia), (rb, ib) in zip(a, b):
        if ra[1] < rb[0] or rb[1] < ra[0] or ia[1] < ib[0] or ib[1] < ia[0]:
            return True
    return False


# ---- maps read from the points ----------------------------------------------------------------

def _nice(z, den=12, tol=1e-7):
    """z as a Gaussian rational with denominators up to `den`, or None."""
    out = []
    for part in (z.real, z.imag):
        f = Fraction(part).limit_denominator(den)
        if abs(float(f) - part) > tol * (1 + abs(part)):
            return None
        out.append(f)
    return spy.Rational(out[0].numerator, out[0].denominator) + \
        spy.I * spy.Rational(out[1].numerator, out[1].denominator)


def _coordinate_forms(base, img, moved, syms):
    """The images c*theta_j^e (e = 1, -1) and c*theta_j + d of the base point that reproduce the value of each
    image coordinate, c and d Gaussian rationals; per coordinate a list of sympy expressions."""
    out = []
    for i, z in enumerate(img):
        forms = []
        for j, s in enumerate(syms):
            b = base[j]
            for e in (1, -1):
                c = _nice(z / b ** e)
                if c is not None and c != 0:
                    forms.append(c * s ** e)
            if j == i:
                d = _nice(z - b)
                if d is not None and d != 0:
                    forms.append(s + d)
        if not moved[i] and not forms:
            forms.append(syms[i])
        out.append(forms)
    return out


def read_maps(system, points, rng, meter, cap=64):
    """Maps sigma with sigma(base) at one of `points` whose image coordinates are c*theta_j^(+-1) or theta_i + d,
    that keep the coefficients at two further random points within the tolerance; sympy dicts."""
    syms = list(system.series.cs)
    base = system.base
    maps = []
    for x in points:
        moved = [abs(a - b) > 1e-9 * (1 + abs(b)) for a, b in zip(x, base)]
        if not any(moved):
            continue
        forms = _coordinate_forms(base, x, moved, syms)
        if any(not f for f in forms):
            continue
        # the identity first, then the simplest forms
        forms = [sorted(f, key=lambda e: (e != s, spy.count_ops(e))) for f, s in zip(forms, syms)]
        for combo in itertools.islice(itertools.product(*forms), cap):
            if meter.over:
                return maps
            sig = dict(zip(syms, combo))
            if any(all(spy.simplify(sig[s] - m[s]) == 0 for s in syms) for m in maps):
                break
            if _keeps(system, sig, rng):
                maps.append(sig)
                break
    return maps


def _keeps(system, sig, rng, points=2):
    """Whether the map `sig` keeps the coefficients of `system` at `points` random points within _TOL."""
    syms = list(system.series.cs)
    f = spy.lambdify(syms, [sig[s] for s in syms], "numpy")
    for _ in range(points):
        th = 0.5 + rng.random(len(syms))
        img = np.asarray(f(*th), dtype=complex)
        got = coefficients(system.series, np.array([th, img]), system.orders, system.gap, jac=False)
        if got is None:
            return False
        a, b = got[0]
        if np.max(np.abs(a - b) / (1 + np.abs(a))) > _TOL:
            return False
    return True


def propose(series, fib, meter=None, target=None):
    """{"points", "square", "source", "certified", "boxes", "base", "why"} and the maps read from the points
    (unproven): the points of the fiber at a random rational point, by monodromy where the square subsystem has
    at most _SQUARE points modulo a prime, and where it stops short of the count `target`, from the rational
    parametrization of msolve over Q; those enclosed in disjoint Krawczyk boxes. The paths stop at the cap
    monodromy_products of work.CAPS on the work of their steps (_step_work()), or of `meter`."""
    from .discrete import Fiber, complex_count
    meter = meter or work.Meter("the monodromy", work.CAPS["monodromy_products"], "term products")
    out = {"points": 0, "square": None, "source": None, "certified": 0, "boxes": [], "why": None}
    if series.free_rows:
        out["why"] = "copies with a free start on a switching surface"
        return out, []
    n = len(series.cs)
    orders = fib.orders
    rows = series.rank_rows(max(orders) + 2, PRIMES[0])
    if rows is None or len(rows) != n:
        out["why"] = "no square subsystem of independent coefficients"
        return out, []
    srng = session.current().rng("discrete monodromy", tuple(series.coords))
    rng = np.random.default_rng(srng.randint(0, 2 ** 32 - 1))
    base = {s: spy.Rational(srng.randint(5, 40), srng.randint(4, 12)) for s in series.cs}
    out["base"] = {str(s): str(v) for s, v in base.items()}
    try:
        with np.errstate(all="ignore"):
            system = _System(series, orders, rows, base, fib.gap)
    except ZeroDivisionError:
        out["why"] = "the random point is a pole"
        return out, []
    # the size of the square fiber decides whether monodromy can reach it
    pt = {s: spy.Integer(srng.randint(2, 10 ** 6)) for s in series.cs}
    with work.quick_msolve():
        est = complex_count(Fiber(series, system.sub_orders, fib.positive), pt, PRIMES[0])
    out["square"] = est
    pts = np.zeros((0, n), dtype=complex)
    if est is not None and est <= _SQUARE:
        with np.errstate(all="ignore"):
            pts, _ = monodromy(system, rng, meter, target)
        out["source"] = "monodromy"
    if meter.over:
        out["why"] = meter.reason()
    # without a count the fiber has no finite parametrization to take
    if target is not None and len(pts) < target and not meter.over:
        more = rur_points(fib, base)
        if more is not None and len(more):
            with np.errstate(all="ignore"):
                more = more[system.residual(more) < _TOL]
            add = [x for x in more if _new(list(pts), x)]
            if add:
                pts = np.vstack([pts] + [np.array(add)])
                out["source"] = "monodromy and parametrization" if out["source"] else "parametrization"
    out["points"] = len(pts)
    boxes = []
    for x in pts:
        if meter.over:
            break
        b = certify(series, orders, fib.gap, base, x)
        if b is not None and all(disjoint(b, c) for c in boxes):
            boxes.append(b)
    out["certified"], out["boxes"] = len(boxes), boxes
    with np.errstate(all="ignore"):
        maps = read_maps(system, pts, rng, meter)
    return out, maps


def proposals(series, fib, k, known, meter=None):
    """(info, maps): the result of propose() at the orders of the fiber `fib` and the proposed maps that are not in
    `known`, where the group of the known maps has fewer points than the count `k` or no count exists; (None, [])
    otherwise."""
    from .discrete import _same_map, group
    if getattr(fib, "too_large", None) or not series.cs:
        return None, []
    grp = group(known, series.cs) if known else [dict(zip(series.cs, series.cs))]
    if k is not None and grp is not None and len(grp) >= k:
        return None, []
    info, maps = propose(series, fib, meter, target=k)
    new = [m for m in maps if not any(_same_map(m, x) for x in known)]
    info["maps"] = [{str(s): str(e) for s, e in m.items() if e != s} for m in new]
    return info, new
