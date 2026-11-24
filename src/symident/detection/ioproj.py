"""Input-output projections for the exact count (docs/method/global.rst, "The exact count"): per observable the relation
of one derivative with the derivatives below a profile: the support found modulo a prime, the coefficients interpolated
in the parameters, the relation checked as an identity of polynomials and the coefficients shown to lie in the field of
the output coefficients."""

import ast
import math
import os
import tempfile
from fractions import Fraction

import numpy as np
import sympy as spy

from .. import _core
from ..gfp import all_primes, prime_run, sieve
from . import work

# the largest number of candidate monomials of one projection modulo a prime
CAP = 6000
# the largest number of terms of a projection whose coefficients are interpolated
TERMS = 1500
# the largest number of unknowns of one interpolation in the parameters
_UNKNOWNS = 6000
# the largest box of the numerators interpolated densely, all at once
_DENSE = 1500
# the most terms of a derivative whose next derivative a profile reads
_JET = 5000
# the largest total degree of a projection tried
_DEGREE = 14
# the most candidate monomials of a projection taken from bounds on the Newton polytope
POLYTOPE = 10000
# the threads of the kernels, set by the callers
THREADS = [1]


def _threads():
    return max(1, int(THREADS[0]))


def _inv(a, p):
    """Elementwise inverse modulo p of a nonzero int64 array."""
    return _pow(a, p - 2, p)


class _Polys:
    """Polynomials of the ring of the relations as term arrays, evaluated at many points modulo p in C++."""

    def __init__(self, polys, nv):
        exps, coefs, off = [], [], [0]
        for P in polys:
            for m, c in P.terms():
                exps.append(m)
                coefs.append((int(c.numerator), int(c.denominator)))
            off.append(len(exps))
        self.exps = np.array(exps, dtype=np.int32).reshape(-1, nv)
        self.coefs, self.off, self._res = coefs, np.array(off, dtype=np.int64), {}

    def __call__(self, pts, p):
        if p not in self._res:
            self._res[p] = np.array([a % p * pow(b, -1, p) % p for a, b in self.coefs], dtype=np.int64)
        return _core.io_poly_eval(self.exps, self._res[p], self.off, np.ascontiguousarray(pts, dtype=np.int64), p,
                                  _threads())


def _pow(a, e, p):
    """Elementwise power modulo p of an int64 array."""
    out = np.ones_like(a)
    b = a % p
    while e:
        if e & 1:
            out = out * b % p
        b = b * b % p
        e >>= 1
    return out


class Quo:
    """A quotient N / prod_k F_k^{e_k} of polynomials over a list F of factors shared by the functions of one model,
    kept without gcds."""
    __slots__ = ("numer", "exps")

    def __init__(self, numer, exps):
        self.numer, self.exps = numer, tuple(exps)


class Jets:
    """The derivatives of the observables along the flow as Quo over the factors Q, the product of the distinct
    denominators of the right-hand sides, and E_r, the denominator of observable r: with L(P) = P~ / Q for a polynomial
    P, the derivative of N / (Q^a E^b) is (N~ Q E - N (a Q~ E + b E~ Q)) / (Q^(a+2) E^(b+1))."""

    def __init__(self, rel):
        self.rel = rel
        R = rel.R.ring if hasattr(rel.R, "ring") else rel.R
        self.R = R
        gens = list(R.gens)
        self.X = [None if x is None else gens[rel.names.index(str(x))] for x in rel.X]

        def parts(q):
            n, d = (q.numer, q.denom) if hasattr(q, "numer") else (q, R.one)
            n, d = R(n) if n.ring != R else n, R(d) if d.ring != R else d
            if d.is_ground:
                return n * R.domain.quo(R.domain.one, d.LC), None
            c = d.LC
            return n * R.domain.quo(R.domain.one, c), d * R.domain.quo(R.domain.one, c)

        self._parts = parts
        fs = [None if f is None else parts(f) for f in rel.f]
        dens = []
        for f in fs:
            if f is not None and f[1] is not None and f[1] not in dens:
                dens.append(f[1])
        Q = R.one
        for d in dens:
            Q = Q * d
        self.Fh = []
        for f in fs:
            if f is None:
                self.Fh.append(None)
                continue
            n, d = f
            t = n
            for g in dens:
                if d is None or g != d:
                    t = t * g
            self.Fh.append(t)
        self.Q = None if not dens else Q
        self.Qt = None if self.Q is None else self.tilde(self.Q)
        self.fac = [] if self.Q is None else [self.Q]
        self._obs = {}

    def tilde(self, P):
        """P~ = Q L(P), L the derivative along the flow."""
        out = self.R.zero
        for x, F in zip(self.X, self.Fh):
            if x is not None and F is not None:
                d = P.diff(x)
                if d:
                    out = out + d * F
        return out

    def _factor(self, E):
        if E not in self.fac:
            self.fac.append(E)
        return self.fac.index(E)

    def observable(self, r, m):
        """The derivatives 0..m of observable r as Quo."""
        y = self.rel.y[r]
        key = (r, str(y))
        if key not in self._obs:
            M, E = self._parts(y)
            ie = None if E is None else self._factor(E)
            self._obs[key] = [[(M, 0, 0)], ie, None if E is None else self.tilde(E)]
        lst, ie, Et = self._obs[key]
        E = None if ie is None else self.fac[ie]
        while len(lst) <= m:
            N, a, b = lst[-1]
            Nt = self.tilde(N)
            if self.Q is None and E is None:
                lst.append((Nt, 0, 0))
            elif self.Q is None:
                lst.append((Nt * E - N * Et * b, 0, b + 1))
            elif E is None:
                lst.append((Nt * self.Q - N * self.Qt * a, a + 2, 0))
            else:
                lst.append((Nt * self.Q * E - N * (self.Qt * E * a + Et * self.Q * b), a + 2, b + 1))
        return [self.quo(N, a, ie, b) for N, a, b in lst[:m + 1]]

    def within(self, ms):
        """Whether each derivative of observable r up to order ms[r] is the derivative of one with at most _JET
        terms, computed up to the first that has more."""
        for r, m in enumerate(ms):
            for k in range(1, m + 1):
                if len(self.observable(r, k - 1)[k - 1].numer) > _JET:
                    return False
        return True

    def quo(self, N, a=0, ie=None, b=0):
        e = {}
        if a and self.Q is not None:
            e[0] = a
        if b and ie is not None:
            e[ie] = e.get(ie, 0) + b
        return Quo(N, sorted(e.items()))

    def state(self, k):
        return Quo(self.X[k], [])

    def denom(self, q):
        out = self.R.one
        for k, e in q.exps:
            out = out * self.fac[k] ** e
        return out


def jets_of(rel):
    """The Jets of `rel`, built once."""
    if not hasattr(rel, "_qjets"):
        rel._qjets = Jets(rel)
    return rel._qjets


class Base:
    """The base functions of a projection, Quo over the factors of their Jets, evaluated together at many points
    modulo p."""

    def __init__(self, rel, funcs):
        self.J = jets_of(rel)
        self.funcs = list(funcs)
        self.nv = len(rel.names)
        self.nf = len(self.J.fac)
        self.P = _Polys([q.numer for q in self.funcs] + list(self.J.fac), self.nv)

    def values(self, pts, p):
        """(rows kept, values): the points where no denominator vanishes and the base functions there."""
        V = self.P(pts, p)
        nb = len(self.funcs)
        num, F = V[:, :nb], V[:, nb:]
        used = sorted({k for q in self.funcs for k, _ in q.exps})
        ok = (F[:, used] != 0).all(axis=1) if used else np.ones(len(V), dtype=bool)
        num, F = num[ok], F[ok]
        den = np.ones_like(num)
        for v, q in enumerate(self.funcs):
            for k, e in q.exps:
                den[:, v] = den[:, v] * _pow(F[:, k], e, p) % p
        return ok, num * _inv(den, p) % p


def _monomials(nb, degree, top=None, cap_top=None):
    """Exponent vectors of total degree at most `degree`, the exponent of variable `top` at most `cap_top`, lowest
    degree first."""
    out = []
    for d in range(degree + 1):
        for m in _exps(nb, d):
            if top is None or m[top] <= cap_top:
                out.append(m)
    return out


def _exps(nb, d):
    if nb == 0:
        return [()] if d == 0 else []
    return [(a,) + r for a in range(d, -1, -1) for r in _exps(nb - 1, d - a)]


def _count(nb, degree, cap_top, cap2=None):
    """The number of exponent vectors of _monomials(nb, degree, top, cap_top), with `cap2` also those with the
    exponent of a second variable at most cap2."""
    if cap2 is None:
        return sum(math.comb(nb - 1 + degree - e, nb - 1) for e in range(min(degree, cap_top) + 1))
    return sum(math.comb(nb - 2 + degree - e - f, nb - 2) for e in range(min(degree, cap_top) + 1)
               for f in range(min(degree - e, cap2) + 1))


def _mono_matrix(B, mons, p):
    """The values of the monomials at the rows of B."""
    return _core.io_mono_matrix(np.ascontiguousarray(B, dtype=np.int64), np.array(mons, dtype=np.int32).reshape(
        len(mons), B.shape[1]), p, _threads())


class Projection:
    """A projection of the relations `rel`: the relation of one top function with the derivatives of each observable
    r up to order ms[r] and the input states read. For an observable, the top is the derivative of the order of the
    profile and ms the profile less one; for the start of a state, the top is the state, and `capped` = (variable,
    cap) bounds the degree of a derivative on top of the profile below that of the projection of the observable."""

    def __init__(self, rel, ms, pos, rng, top=None, extra=None):
        self.rel, self.ms, self.pos, self.rng = rel, list(ms), list(pos), rng
        J = jets_of(rel)
        funcs = [q for r, m in enumerate(self.ms) if m >= 0 for q in J.observable(r, m)] + \
            [J.state(k) for k in rel.inputs(max(self.ms))]
        if extra is not None:
            top, funcs = len(funcs), funcs + [J.state(extra)]
        self.base = Base(rel, funcs)
        self.nb = len(funcs)
        self.top = top
        self.mons = None
        self.capped = None

    @classmethod
    def output(cls, rel, h, i, pos, rng):
        """None where a derivative of the base is that of one with more than _JET terms."""
        ms = [k - 1 for k in h]
        ms[i] = h[i]
        if not jets_of(rel).within(ms):
            return None
        return cls(rel, ms, pos, rng, top=sum(m + 1 for m in ms[:i]) + h[i])

    @classmethod
    def state(cls, rel, h, k, pos, rng, lift=None):
        """With lift = (i, d), the base holds the derivative of observable i of order h[i], whose projection has
        degree d in it, at degree below d: a normal form modulo the projection. None as in output()."""
        ms = [x - 1 for x in h]
        if lift is not None:
            ms[lift[0]] = h[lift[0]]
        if not jets_of(rel).within(ms):
            return None
        pr = cls(rel, ms, pos, rng, extra=k)
        if lift is not None:
            pr.capped = (sum(m + 1 for m in ms[:lift[0]]) + h[lift[0]], lift[1] - 1)
        return pr

    def points(self, theta, p, m):
        """Base values at m random points of the states with the parameters at theta."""
        nv = self.base.nv
        out = []
        if not hasattr(self, "_gen"):
            self._gen = np.random.default_rng(self.rng.getrandbits(64))
        while sum(len(o) for o in out) < m:
            k = m - sum(len(o) for o in out) + 4
            pts = self._gen.integers(1, p, size=(k, nv), dtype=np.int64)
            pts[:, self.pos] = np.array(theta, dtype=np.int64) % p
            _, B = self.base.values(pts, p)
            out.append(B)
        return np.vstack(out)[:m]

    def search(self, theta, p, stop, cap=CAP, top_degree=None):
        """The exponent vectors of the support of the relation at the parameter point theta, of degree at most
        `top_degree` in the top function; None where no relation has at most `cap` candidate monomials or the
        kernel is not one vector. The boxes up to a quarter of the cap first, then the monomials inside the
        bounds on the Newton polytope (newton_bounds()), then the larger boxes."""
        pool = [np.zeros((0, self.nb), dtype=np.int64)]

        def kernel(box):
            mons = box if isinstance(box, list) else _monomials(self.nb, box[0], self.top, box[1])
            if self.capped is not None:
                mons = [m for m in mons if m[self.capped[0]] <= self.capped[1]]
            if len(pool[0]) < len(mons) + 8:
                pool[0] = np.vstack([pool[0], self.points(theta, p, len(mons) + 8 - len(pool[0]))])
            A = _mono_matrix(pool[0][:len(mons) + 8], mons, p)
            work.tick(A.shape)
            return mons, _core.io_kernel(A, p, _threads())

        def found(mons, v):
            S = [mons[j] for j in range(len(mons)) if v[j]]
            # a relation without the function on top is one of a smaller profile
            if not any(m[self.top] for m in S):
                return None
            self.mons = S
            return S

        # the largest boxes below a rising cap: one without a relation holds none of the smaller ones, and a
        # kernel of one vector is the relation
        done = set()
        for c in (cap // 4, None, cap // 2, cap):
            if c is None:
                if self.capped is not None or work.ticks() > stop:
                    continue
                mons = self.polytope(theta, p, stop)
                if self.newton and top_degree is not None and self.newton[_unit(self.nb, self.top)][1] > top_degree:
                    # the projection has a larger degree in the top function, and every relation is a multiple
                    return None
                if mons is None:
                    continue
                mons, ker = kernel(mons)
                if len(ker["free"]) == 1:
                    return found(mons, ker["basis"][0])
                continue
            boxes = [(D, dt) for D in range(1, _DEGREE + 1) for dt in range(1, min(D, top_degree or D) + 1)
                     if _count(self.nb, D, dt, None if self.capped is None else self.capped[1]) <= c]
            largest = [b for b in boxes if not any(o != b and o[0] >= b[0] and o[1] >= b[1] for o in boxes)]
            for b in largest:
                if b in done or work.ticks() > stop:
                    continue
                done.add(b)
                mons, ker = kernel(b)
                if len(ker["free"]) == 1:
                    return found(mons, ker["basis"][0])
                if not ker["free"]:
                    continue
                # several relations: the least box with one
                for box in sorted(x for x in boxes if x[0] <= b[0] and x[1] <= b[1]):
                    if work.ticks() > stop:
                        return None
                    mons, ker = kernel(box)
                    if len(ker["free"]) == 1:
                        return found(mons, ker["basis"][0])
                    if ker["free"]:
                        return None
                return None
        return None

    def weights(self):
        """Weights of the base functions for bounds on the Newton polytope: the order of each derivative plus one,
        one for the inputs and a state on top."""
        out = [k + 1 for m in self.ms if m >= 0 for k in range(m + 1)]
        return out + [1] * (self.nb - len(out))

    def polytope(self, theta, p, stop):
        """The monomials inside the bounds on the Newton polytope of the projection at the parameter point theta
        (newton_bounds()), at most POLYTOPE of them, the bounds in self.newton; None where they are not found."""
        self.newton = None
        sl = _Slice(self, theta, p, self.rng, stop)
        if not sl.ok:
            return None
        rng, nb = self.rng, self.nb
        order = self.weights()
        fixed = [tuple([1] * nb)] + [_unit(nb, v) for v in range(nb)] + \
            [tuple(order), tuple(max(order) + 1 - w for w in order)]
        bounds = newton_bounds(sl, fixed, stop)
        if bounds is None:
            return None
        mons = _inside(nb, bounds, POLYTOPE)
        # random weights while the candidates exceed the cap
        for _ in range(4):
            if mons is not None or work.ticks() > stop:
                break
            more = newton_bounds(sl, [tuple(rng.randrange(0, 4) for _ in range(nb)) for _ in range(8)], stop)
            if more is None:
                break
            bounds.update(more)
            mons = _inside(nb, bounds, POLYTOPE)
        self.newton = bounds
        return mons

    def vector(self, theta, p, mons=None):
        """The kernel vector on the monomials `mons` at the parameter point theta, last entry one; None where the
        kernel is not one vector with that entry nonzero."""
        mons = mons or self.mons
        B = self.points(theta, p, len(mons) + 6)
        A = _mono_matrix(B, mons, p)
        work.tick(A.shape)
        ker = _core.io_kernel(A, p, _threads())
        if ker["free"] != [len(mons) - 1]:
            return None
        return np.asarray(ker["basis"][0], dtype=np.int64)


def _unit(n, v):
    return tuple(1 if u == v else 0 for u in range(n))


def _gf_terms(P, sub, p):
    """{((variable, exponent), ...): coefficient} of the polynomial P modulo p, the variables of `sub` at their
    values."""
    out = {}
    for m, c in P.terms():
        v = int(c.numerator) % p * pow(int(c.denominator) % p, -1, p) % p
        key = []
        for g, e in enumerate(m):
            if not e:
                continue
            if g in sub:
                v = v * pow(sub[g], e, p) % p
            else:
                key.append((g, e))
        key = tuple(key)
        out[key] = (out.get(key, 0) + v) % p
    return {k: c for k, c in out.items() if c}


def _msolve_count(names, polys, p):
    """The number of distinct solutions over the algebraic closure of GF(p) of the polynomials (dicts over the
    indices of `names`), from the rational parametrization of msolve; None where the solutions are not finitely
    many, the system is past the caps of msolve (a tenth of msolve_entries) or msolve fails."""
    from ..sym.msolve import MsolveMissingError, _msolve_binary

    def text(d):
        out = []
        for key, c in d.items():
            mono = "*".join(names[g] + (f"^{e}" if e > 1 else "") for g, e in key)
            out.append(f"{c}*{mono}" if mono else f"{c}")
        return "+".join(out) or "0"

    try:
        binary = _msolve_binary()
    except MsolveMissingError:
        return None
    eqs = [text(d) for d in polys]
    if work.msolve_too_large(names, eqs):
        return None
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.ms"), os.path.join(td, "out.ms")
        with open(fi, "w") as fh:
            fh.write(",".join(names) + "\n" + str(p) + "\n" + ",\n".join(eqs) + "\n")
        # a count of the search takes a tenth of the cap of msolve, and the matrices of msolve count on the clock
        # of the search
        with work.local_caps(msolve_entries=work.cap("msolve_entries") / 10):
            r = work.run_msolve([binary, "-f", fi, "-o", fo, "-P", "1", "-t", str(_threads())])
        work.tick(int(r.entries))
        txt = open(fo).read().strip() if os.path.exists(fo) else ""
    if r.returncode != 0 or not txt:
        return None
    data = ast.literal_eval(txt.rstrip(":"))
    if data == [-1]:
        return 0
    if data[0] != 0 or data[1][2] <= 0:
        return None
    return int(data[1][2])


class _Slice:
    """The base functions of a projection at the parameter point theta modulo p as a polynomial system: the states
    read that give the Jacobian of the base functions rank nb - 1 stay unknown, the others take random values, each
    factor of the denominators is an unknown u_k, and w inverts the product of the factors."""

    def __init__(self, pr, theta, p, rng, stop):
        self.ok, self.p, self.stop = False, p, stop
        J = jets_of(pr.rel)
        funcs, nb = pr.base.funcs, pr.nb
        used = sorted({k for q in funcs for k, _ in q.exps})
        polys = [q.numer for q in funcs] + [J.fac[k] for k in used]
        nv = len(pr.rel.names)
        appear = sorted({g for P in polys for m in P.monoms() for g, e in enumerate(m) if e} - set(pr.pos))
        pt = [rng.randrange(1, p) for _ in range(nv)]
        for k, v in zip(pr.pos, theta):
            pt[k] = int(v) % p
        ok, vals = pr.base.values(np.array([pt], dtype=np.int64), p)
        if not ok[0] or len(appear) < nb - 1:
            return
        rows = _gradients(J, funcs, [J.R.gens[g] for g in appear], pt, p)
        if rows is None:
            return
        free = []
        for c in range(len(appear)):
            if _rank([[r[j] for r in rows] for j in free + [c]], p) == len(free) + 1:
                free.append(c)
            if len(free) == nb - 1:
                break
        if len(free) != nb - 1:
            return
        self.free = [appear[c] for c in free]
        sub = {g: pt[g] for g in range(nv) if g not in self.free}
        # unknowns: w, the factors, the free states, t, s
        self.names = ["w"] + [f"u{k}" for k in used] + [f"x{g}" for g in self.free] + ["t", "s"]
        at = {g: 1 + len(used) + j for j, g in enumerate(self.free)}
        self.u = {k: 1 + j for j, k in enumerate(used)}
        self.t, self.s = len(self.names) - 2, len(self.names) - 1

        def remap(d):
            return {tuple((at[g], e) for g, e in key): c for key, c in d.items()}

        self.N = [remap(_gf_terms(q.numer, sub, p)) for q in funcs]
        self.F = {k: remap(_gf_terms(J.fac[k], sub, p)) for k in used}
        self.dens = [tuple((self.u[k], e) for k, e in q.exps) for q in funcs]
        self.image = [int(v) for v in vals[0]]
        self.nb, self.used, self.rng = nb, used, rng
        self.ok = True
        self.fiber = self.count(self.image)
        self.ok = bool(self.fiber)

    def count(self, values, direc=None, weights=None, torus=False):
        """The number of points of the slice where the base functions take the values c_v + b_v t^(w_v), c the
        values, b the direction and w the weights, with t nonzero for `torus`; the values alone without a
        direction."""
        p = self.p
        eqs = []
        for v in range(self.nb):
            d = dict(self.N[v])
            den = self.dens[v]
            if values[v] % p:
                d[den] = (d.get(den, 0) - values[v]) % p
            if direc is not None and direc[v] % p:
                key = tuple(sorted(den + ((self.t, weights[v]),))) if weights[v] else den
                d[key] = (d.get(key, 0) - direc[v]) % p
            eqs.append({k: c for k, c in d.items() if c})
        for k in self.used:
            d = {m: (-c) % p for m, c in self.F[k].items()}
            key = ((self.u[k], 1),)
            d[key] = (d.get(key, 0) + 1) % p
            eqs.append({m: c for m, c in d.items() if c})
        eqs.append({tuple([(0, 1)] + [(self.u[k], 1) for k in self.used]): 1, (): p - 1})
        names = self.names[:self.t]
        if direc is not None:
            names = names + ["t"]
            if torus:
                names = names + ["s"]
                eqs.append({((self.t, 1), (self.s, 1)): 1, (): p - 1})
        return _msolve_count(names, eqs, p) if work.ticks() <= self.stop else None


def _gradients(J, funcs, gens, pt, p):
    """Per base function N / prod F_k^e_k the gradient in `gens` at the point pt modulo p, by (grad N - N sum_k
    e_k grad F_k / F_k) / prod F_k^e_k; None at a pole."""
    out = []
    for q in funcs:
        polys = [q.numer] + [q.numer.diff(x) for x in gens]
        for f, _ in q.exps:
            polys += [J.fac[f]] + [J.fac[f].diff(x) for x in gens]
        v = [int(x) for x in _Polys(polys, len(pt))(np.array([pt], dtype=np.int64), p)[0]]
        n, g = v[0], v[1:len(gens) + 1]
        den, at = 1, len(gens) + 1
        lg = [0] * len(gens)
        for _f, e in q.exps:
            F, dF = v[at], v[at + 1:at + 1 + len(gens)]
            at += 1 + len(gens)
            if not F:
                return None
            den = den * pow(F, e, p) % p
            iF = pow(F, -1, p)
            lg = [(a + e * b * iF) % p for a, b in zip(lg, dF)]
        iD = pow(den, -1, p)
        out.append([(a - n * b) * iD % p for a, b in zip(g, lg)])
    return out


def newton_bounds(sl, weights, stop):
    """{w: (low, high)}: per weight vector w the least and the largest of <w, m> over the support of the projection,
    modulo p. A curve c + b t^w with random c and b meets the image of the base functions where the projection P has
    the roots of P(c + b t^w), of degree high(w), each the image of the points of one fiber; the curve b t^w with t
    nonzero meets it in high(w) - low(w) points. Counts of points only propose: the support found from them is
    checked as every other. None where a count fails."""
    p, rng, nb, k = sl.p, sl.rng, sl.nb, sl.fiber
    out = {}
    for w in weights:
        if work.ticks() > stop:
            return None
        c = [rng.randrange(1, p) for _ in range(nb)]
        b = [rng.randrange(1, p) for _ in range(nb)]
        hi = sl.count(c, b, w)
        if hi is None or hi % k:
            return None
        hi //= k
        lo = 0
        if sum(1 for x in w if x) > 1 and hi:
            n = sl.count([0] * nb, b, w, torus=True)
            if n is None or n % k:
                return None
            lo = hi - n // k
        out[tuple(w)] = (lo, hi)
    return out


def _inside(nb, bounds, cap):
    """The exponent vectors m with low(w) <= <w, m> <= high(w) for the bounds {w: (low, high)}, lowest total degree
    first; None for more than cap of them. Every weight is nonnegative, so a partial sum above high(w) stays above."""
    W = np.array(list(bounds), dtype=np.int64).reshape(-1, nb)
    lo = np.array([b[0] for b in bounds.values()], dtype=np.int64)
    hi = np.array([b[1] for b in bounds.values()], dtype=np.int64)
    per = []
    for v in range(nb):
        caps = [h // w[v] for w, h in zip(W, hi) if w[v]]
        per.append(min(caps) if caps else 0)
    # the largest sum the variables from v on can add
    rest = np.zeros((nb + 1, len(W)), dtype=np.int64)
    for v in range(nb - 1, -1, -1):
        rest[v] = rest[v + 1] + W[:, v] * per[v]
    cur = np.zeros((1, 0), dtype=np.int64)
    S = np.zeros((1, len(W)), dtype=np.int64)
    for v in range(nb):
        e = np.arange(per[v] + 1, dtype=np.int64)
        S2 = (S[:, None, :] + e[None, :, None] * W[None, None, :, v]).reshape(-1, len(W))
        C2 = np.hstack([np.repeat(cur, len(e), axis=0), np.tile(e, len(cur))[:, None]])
        keep = (S2 <= hi).all(axis=1) & (S2 + rest[v + 1] >= lo).all(axis=1)
        cur, S = C2[keep], S2[keep]
        if len(cur) > 8 * cap:
            return None
    if len(cur) > cap or not len(cur):
        return None
    out = [tuple(int(x) for x in m) for m in cur]
    out.sort(key=lambda m: (sum(m), tuple(-x for x in m)))
    return out


# rational functions of one variable modulo p


def _poly_trim(a):
    while a and not a[-1]:
        a.pop()
    return a


def _newton(ts, vs, p):
    """Coefficients (lowest first) of the polynomial of degree < len(ts) through the points."""
    n = len(ts)
    c = list(vs)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            c[i] = (c[i] - c[i - 1]) * pow(ts[i] - ts[i - j], -1, p) % p
    out = [0] * n
    for i in range(n - 1, -1, -1):
        # out = out * (x - ts[i]) + c[i]
        out = [(a - ts[i] * b) % p for a, b in zip([0] + out[:-1], out)]
        out[0] = (out[0] + c[i]) % p
    return out


def _pdivmod(a, b, p):
    a = list(a)
    q = [0] * max(1, len(a) - len(b) + 1)
    inv = pow(b[-1], -1, p)
    for k in range(len(a) - len(b), -1, -1):
        c = a[k + len(b) - 1] * inv % p
        q[k] = c
        if c:
            for j, bj in enumerate(b):
                a[k + j] = (a[k + j] - c * bj) % p
    return _poly_trim(q), _poly_trim(a[:len(b) - 1])


def _psub(a, b, p):
    n = max(len(a), len(b))
    return _poly_trim([((a[i] if i < len(a) else 0) - (b[i] if i < len(b) else 0)) % p for i in range(n)])


def _pmul(a, b, p):
    if not a or not b:
        return []
    out = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                out[i + j] = (out[i + j] + x * y) % p
    return out


def ratrec1(ts, vs, p):
    """(numerator, denominator) of the rational function of one variable with these values, the denominator monic:
    the step of the extended Euclidean algorithm before the quotient of largest degree, at least three, so that the
    degrees leave three of the points over; None otherwise."""
    m = [1]
    for t in ts:
        m = _pmul(m, [(-t) % p, 1], p)
    r0, r1 = m, _poly_trim(_newton(ts, vs, p))
    if not r1:
        return [], [1]
    s0, s1 = [], [1]
    best, gap = None, 2
    while r1:
        q, r = _pdivmod(r0, r1, p)
        if len(q) - 1 > gap:
            best, gap = (r1, s1), len(q) - 1
        r0, r1 = r1, r
        s0, s1 = s1, _psub(s0, _pmul(q, s1, p), p)
    if best is None:
        return None
    r1, s1 = best
    c = pow(s1[-1], -1, p)
    return [x * c % p for x in r1], [x * c % p for x in s1]


def _pgcd(a, b, p):
    while b:
        a, b = b, _pdivmod(a, b, p)[1]
    c = pow(a[-1], -1, p)
    return [x * c % p for x in a]


def _plcm(a, b, p):
    return _pdivmod(_pmul(a, b, p), _pgcd(a, b, p), p)[0]


def _line(vec, base, direction, p, npts, start=1):
    """Per entry (numerator, denominator) of the restriction to the line base + t direction, t from `start`; None
    where not found."""
    ts, vals = [], []
    t = start - 1
    while len(ts) < npts:
        t += 1
        if t > 3 * npts:
            return None
        v = vec([(b + t * d) % p for b, d in zip(base, direction)], p)
        if v is not None:
            ts.append(t)
            vals.append(v)
    out = []
    for e in range(len(vals[0])):
        got = ratrec1(ts, [int(v[e]) for v in vals], p)
        if got is None:
            return None
        out.append(got)
    return out


def _common_denominator(fracs, p):
    """The least common multiple of the denominators of the (numerator, denominator) pairs."""
    L = [1]
    for _, d in fracs:
        L = _plcm(L, d, p)
    return L


def _degrees(fracs, p):
    """Degrees of the polynomials C_e over the line with e -> C_e / C_last of the entries, the common
    denominator cleared."""
    L = _common_denominator(fracs, p)
    out = []
    for n, d in fracs:
        q = _pdivmod(L, d, p)[0]
        out.append(len(_pmul(n, q, p)) - 1 if n else -1)
    return out


def _box(nv, total, per):
    """Exponent vectors with total degree at most `total` and degree at most per[v] in variable v."""
    out = []

    def rec(v, left, cur):
        if v == nv:
            out.append(tuple(cur))
            return
        for e in range(min(left, per[v]) + 1):
            cur.append(e)
            rec(v + 1, left - e, cur)
            cur.pop()
    rec(0, total, [])
    out.sort(key=lambda m: (sum(m), m))
    return out


def _vander(pts, mons, p):
    nv = len(mons[0]) if mons else 0
    X = np.array(pts, dtype=np.int64)
    top = [max(m[v] for m in mons) for v in range(nv)]
    pw = [[np.ones(len(pts), dtype=np.int64)] for _ in range(nv)]
    for v in range(nv):
        for _ in range(top[v]):
            pw[v].append(pw[v][-1] * X[:, v] % p)
    M = np.ones((len(pts), len(mons)), dtype=np.int64)
    for j, m in enumerate(mons):
        for v, e in enumerate(m):
            if e:
                M[:, j] = M[:, j] * pw[v][e] % p
    return M


def _ratrec(rows, primes):
    """Per column the rational number with these residues modulo the primes (one row each), by the Chinese remainder
    theorem and rational reconstruction with numerator and denominator below the square root of half the product;
    None where one has none."""
    M = math.prod(primes)
    lift = [(M // q) * pow(M // q, -1, q) for q in primes]
    bound = math.isqrt(M // 2)
    out = []
    for col in zip(*rows):
        a = sum(v * w for v, w in zip(col, lift)) % M
        r0, r1, s0, s1 = M, a, 0, 1
        while r1 > bound:
            qq = r0 // r1
            r0, r1 = r1, r0 - qq * r1
            s0, s1 = s1, s0 - qq * s1
        if s1 == 0 or abs(s1) > bound or math.gcd(r1, abs(s1)) != 1:
            return None
        out.append(Fraction(r1, s1))
    return out


def _primes():
    """The primes of the interpolation of coefficients: the engine primes, then lower ones up to 32."""
    head = all_primes()
    return head + [q for q in prime_run(32) if q not in head]


class _StoppedError(Exception):
    """The search reached the cap on the entries of the matrices reduced."""


class Coefficients:
    """The coefficients of a projection as rational functions of the parameters, N_e / D with one denominator,
    interpolated modulo several primes and rebuilt over Q; candidates only."""

    def __init__(self, proj, nparams, rng, stop):
        self.proj, self.n, self.rng, self.stop = proj, nparams, rng, stop
        self.why = None

    def _vec(self, order):
        """The kernel vector on the monomials in `order` (the normalizing one last) at a parameter point."""
        mons = [self.proj.mons[j] for j in order]

        def f(theta, p):
            if work.ticks() > self.stop:
                raise _StoppedError
            v = self.proj.vector(theta, p, mons)
            return None if v is None else v[:-1]

        return f

    def run(self):
        try:
            return self._run()
        except _StoppedError:
            self.why = "the end of the search"
            return None

    def _run(self):
        primes = _primes()
        p = primes[0]
        T = len(self.proj.mons)
        rng = self.rng
        # the degrees on a random line, the coefficients cleared of their common denominator, choose the normalizing
        # coefficient of least degree
        base = [rng.randrange(1, p) for _ in range(self.n)]
        direc = [rng.randrange(1, p) for _ in range(self.n)]
        order0 = list(range(T))
        npts = 24
        while True:
            fr = _line(self._vec(order0), base, direc, p, npts)
            if fr is not None and all(len(n) + len(d) + 2 <= npts - 4 for n, d in fr):
                break
            npts *= 2
            if npts > 400:
                self.why = "degrees above 200 on a line"
                return None
        degs = _degrees(fr + [([1], [1])], p)
        star = min(range(T), key=lambda j: (degs[j], -sum(self.proj.mons[j]), j))
        order = [j for j in range(T) if j != star] + [star]
        vec = self._vec(order)
        fr = _line(vec, base, direc, p, npts)
        if fr is None:
            self.why = "the line of the normalized coefficients"
            return None
        # the common denominator D: the least common multiple of the denominators on the line, and per parameter on
        # lines parallel to the axes
        dden = len(_common_denominator(fr, p)) - 1
        dnum = [len(nn) - 1 + dden - (len(d) - 1) if nn else -1 for nn, d in fr]
        per_den, per_num, pure = [], [[] for _ in fr], True
        for v in range(self.n):
            e = [0] * self.n
            e[v] = 1
            b = [rng.randrange(1, p) for _ in range(self.n)]
            fv = _line(vec, b, e, p, npts)
            if fv is None:
                self.why = "a line parallel to an axis"
                return None
            L = _common_denominator(fv, p)
            dv = len(L) - 1
            per_den.append(dv)
            pure = pure and not any(L[:-1])
            for k, (nn, d) in enumerate(fv):
                per_num[k].append(len(nn) - 1 + dv - (len(d) - 1) if nn else -1)
        Dco = {}
        if pure and sum(per_den) == dden:
            # a power of t on every axis: D is a monomial
            boxD = [tuple(per_den)]
            for q in primes:
                Dco[q] = np.array([1], dtype=np.int64)
            again = None
        else:
            # a random combination of the entries has the common denominator as denominator
            r = [rng.randrange(1, 1 << 20) for _ in fr]

            def comb(theta, q, vec=vec, r=r):
                v = vec(theta, q)
                return None if v is None else np.array([sum(int(a) * b for a, b in zip(v, r)) % q], dtype=np.int64)

            boxD = _box(self.n, dden, per_den)
            if len(boxD) > _UNKNOWNS // 4:
                self.why = f"{len(boxD)} unknowns of the denominator"
                return None
            beta = [rng.randrange(1, 1 << 20) for _ in range(self.n)]
            width = max(dnum) + dden + 6
            again = (comb, beta, width)
            for q in primes[:2]:
                got = self._denominator(comb, beta, width, boxD, q)
                if got is None:
                    self.why = self.why or "the denominator"
                    return None
                Dco[q] = got
        # the numerators over the denominator: in one box at once where it is small, else sparse per entry
        boxes = [_box(self.n, max(dnum[k], 0), [max(x, 0) for x in per_num[k]]) for k in range(len(fr))]
        allm = sorted(set().union(*[set(b) for b in boxes]), key=lambda m: (sum(m), m))

        def numerators(q):
            if len(allm) <= _DENSE:
                arr = self._numerators(vec, Dco[q], boxD, allm, len(fr), q)
                return None if arr is None else [dict(zip(allm, row)) for row in arr]
            return self._sparse(vec, Dco[q], boxD, boxes, q)

        num = {}
        for q in primes[:2]:
            num[q] = numerators(q)
            if num[q] is None:
                self.why = self.why or "the numerators"
                return None
        used = primes[:2]
        prev = None
        while True:
            # rational reconstruction of every coefficient from the primes used, the same without the last prime
            dco = self._lift({q: Dco[q] for q in used}, used)
            nco = self._lift_dicts([num[q] for q in used], used)
            if dco is not None and nco is not None and (dco, nco) == prev:
                break
            prev = None if dco is None or nco is None else (dco, nco)
            if len(used) == len(primes) or work.ticks() > self.stop:
                self.why = "the rational reconstruction of the coefficients"
                return None
            q = primes[len(used)]
            if q not in Dco:
                Dco[q] = self._denominator(*again, boxD, q)
                if Dco[q] is None:
                    return None
            num[q] = numerators(q)
            if num[q] is None:
                return None
            used = used + [q]
        self.order = order
        self.boxD = boxD
        self.den = {m: c for m, c in zip(boxD, dco) if c}
        self.nums = [{m: c for m, c in d.items() if c} for d in nco]
        return True

    @staticmethod
    def _lift(by, used):
        """The rationals with the residues `by[q]` modulo the primes `used`, as _ratrec()."""
        return _ratrec([[int(x) for x in by[q]] for q in used], used)

    @staticmethod
    def _lift_dicts(per_prime, used):
        """Per entry the coefficients over Q from their residues per prime, None where the supports differ or a
        coefficient has no bounded rational."""
        out = []
        for k in range(len(per_prime[0])):
            keys = sorted(per_prime[0][k])
            if any(sorted(d[k]) != keys for d in per_prime[1:]):
                return None
            if not keys:
                out.append({})
                continue
            got = _ratrec([[int(d[k][m]) for m in keys] for d in per_prime], used)
            if got is None:
                return None
            out.append(dict(zip(keys, got)))
        return out

    def _sparse(self, vec, dco, boxD, boxes, q):
        """Per entry {monomial: coefficient modulo q} of the numerators by Ben-Or and Tiwari: values at the points
        c_v b_v^j, b_v the first primes and c a random shift, one point for all entries."""
        bases = sieve(self.n)
        width = len(boxes)
        cap_len = 2 * TERMS + 2
        for _ in range(3):
            c = [self.rng.randrange(1, q) for _ in range(self.n)]
            cur, seqs, have, target, failed = list(c), [[] for _ in range(width)], 0, 16, False
            while not failed:
                while have < target:
                    v = vec(cur, q)
                    Dv = sum(int(a) * math.prod(pow(x, e, q) for x, e in zip(cur, m)) for m, a in zip(boxD, dco)) % q
                    if v is None or not Dv:
                        failed = True
                        break
                    for k in range(width):
                        seqs[k].append(int(v[k]) * Dv % q)
                    cur = [x * b % q for x, b in zip(cur, bases)]
                    have += 1
                if failed:
                    break
                if all(2 * int(_core.sym_b_morder(sq, q)) < have for sq in seqs):
                    break
                if have >= cap_len:
                    self.why = f"numerators with more than {TERMS} terms"
                    return None
                target = min(2 * have, cap_len)
            if failed:
                continue
            out = []
            for k in range(width):
                cand = np.array(boxes[k], dtype=np.int64).reshape(-1, self.n)
                res = _core.sym_mono_residues(cand, bases, q)
                r = _core.sym_sparse_poly(seqs[k], cand, res, q)
                if r.get("status") != "ok":
                    self.why = f"the sparse numerator of an entry ({r.get('status')})"
                    return None
                exps = np.asarray(r["exps"], dtype=np.int64).reshape(-1, self.n)
                row = {}
                for m, a in zip(exps, np.atleast_1d(r["coeffs"])):
                    m = tuple(int(x) for x in m)
                    shift = math.prod(pow(x, e, q) for x, e in zip(c, m)) % q
                    row[m] = int(a) * pow(shift, -1, q) % q
                out.append(row)
            return out
        self.why = "no points for the sparse numerators"
        return None

    def _samples(self, vec, q, m):
        """m random parameter points modulo q where `vec` has a value, and the values; (None, None) after many
        misses."""
        pts, vals = [], []
        tries = 0
        while len(pts) < m:
            tries += 1
            if tries > 3 * m + 20:
                return None, None
            x = [self.rng.randrange(1, q) for _ in range(self.n)]
            v = vec(x, q)
            if v is not None:
                pts.append(x)
                vals.append(v)
        return pts, np.array(vals, dtype=np.int64)

    def _denominator(self, comb, beta, width, boxD, q):
        """The coefficients of D on boxD modulo q with D(beta) = 1: on the line beta + t (theta - beta) the common
        denominator is L(t) / L(0), L the denominator of a random combination of the entries there."""
        pts, vals = [], []
        tries = 0
        while len(pts) < len(boxD) + 8:
            tries += 1
            if tries > 3 * len(boxD) + 30:
                return None
            x = [self.rng.randrange(1, q) for _ in range(self.n)]
            g = [(a - b) % q for a, b in zip(x, beta)]
            fr = _line(comb, [b % q for b in beta], g, q, width, start=0)
            if fr is None:
                continue
            _, L = fr[0]
            if not L[0]:
                continue
            pts.append(x)
            vals.append(sum(L) % q * pow(L[0], -1, q) % q)
        A = np.hstack([_vander(pts, boxD, q), (-np.array(vals, dtype=np.int64)[:, None]) % q])
        work.tick(A.shape)
        ker = _core.io_kernel(A, q, _threads())
        if ker["free"] != [len(boxD)]:
            self.why = "the denominator is not determined by the box of degrees"
            return None
        return np.asarray(ker["basis"][0][:len(boxD)], dtype=np.int64) % q

    def _numerators(self, vec, dco, boxD, allm, width, q):
        """The coefficients of every numerator on allm modulo q, one solve with the values times D."""
        U = len(allm)
        pts, vals = self._samples(vec, q, U + 8)
        if pts is None:
            return None
        Dv = (_vander(pts, boxD, q) * dco[None, :] % q).sum(axis=1) % q
        rhs = vals * Dv[:, None] % q
        V = _vander(pts, allm, q)
        A = np.hstack([V, (-rhs) % q])
        work.tick(A.shape)
        ker = _core.io_kernel(A, q, _threads())
        piv = ker["pivots"]
        if piv[:U] != list(range(U)):
            self.why = "the numerators are not determined by their box"
            return None
        # the vector with 1 on the column of a numerator holds the coefficients of that numerator in the first U entries
        basis = np.asarray(ker["basis"], dtype=np.int64)
        free = ker["free"]
        out = np.zeros((width, U), dtype=np.int64)
        for k in range(width):
            j = free.index(U + k) if U + k in free else None
            if j is None:
                return None
            out[k] = basis[j][:U] % q
        return out

    def expressions(self, params):
        """The coefficients as sympy expressions in `params`, in the order of the monomials of the projection."""
        def poly(co):
            return sum((spy.Rational(c.numerator, c.denominator) * spy.prod([s ** e for s, e in zip(params, m)])
                        for m, c in co.items()), spy.Integer(0))

        D = poly(self.den)
        out = [None] * len(self.order)
        for k, j in enumerate(self.order[:-1]):
            out[j] = spy.cancel(poly(self.nums[k]) / D)
        out[self.order[-1]] = spy.Integer(1)
        return out


def profiles(rel, unknown, p, rng, limit=6):
    """Parametric profiles: per observable the number h_r of derivatives in a transcendence basis of the
    derivatives of all of them over the states `unknown`, by the rank of their Jacobian at a random point modulo p;
    greedy in rounds and depth first, from each observable."""
    ny = len(rel.y)
    J = jets_of(rel)
    gens = [J.X[j] for j in unknown]
    pt = [rng.randrange(1, p) for _ in rel.names]
    memo = {}

    def row(r, k):
        if (r, k) not in memo:
            if k and len(J.observable(r, k - 1)[k - 1].numer) > _JET:
                memo[(r, k)] = None
            else:
                got = _gradients(J, [J.observable(r, k)[k]], gens, pt, p)
                memo[(r, k)] = None if got is None else got[0]
        return memo[(r, k)]

    out = []
    orders = [list(range(s, ny)) + list(range(s)) for s in range(ny)]
    for strategy in ("rounds", "depth"):
        for perm in orders:
            h, rows, rk = [0] * ny, [], 0
            active = list(perm)
            while active:
                for r in list(active):
                    if r not in active:
                        continue
                    while True:
                        if h[r] > len(gens):
                            active.remove(r)
                            break
                        nxt = row(r, h[r])
                        cand = rows + [nxt]
                        if nxt is not None and _rank(cand, p) > rk:
                            rows, rk = cand, rk + 1
                            h[r] += 1
                            if strategy == "rounds":
                                break
                        else:
                            active.remove(r)
                            break
            if h not in out:
                out.append(h)
            if len(out) >= limit:
                return out
    return out


def _rank(rows, p):
    """The rank of the rows modulo p."""
    A = np.array(rows, dtype=np.int64)
    work.tick(A.shape)
    ker = _core.io_kernel(A.T.copy(), p, 1)
    return A.shape[0] - len(ker["free"])


def _dot(a, b, p):
    """sum_i a_i b_i modulo p for int64 arrays of residues below 2^31, without overflow."""
    lo, hi = a & 0xFFFF, a >> 16
    return (int(np.dot(lo, b) % p) + (int(np.dot(hi, b) % p) << 16)) % p


class _Series:
    """Polynomials in the states with numeric coefficients, their Taylor coefficients along a trajectory computed
    order by order from those of the states, each monomial from one of lower degree."""

    def __init__(self, polys, vars_, vals, p, K):
        self.p, self.K = p, K
        self.vars = vars_
        zero = (0,) * len(vars_)
        self.mons, self.at = [zero], {zero: 0}
        self.polys = []
        for P in polys:
            terms = {}
            for m, c in P.terms():
                key = tuple(m[g] for g in vars_)
                v = int(c.numerator) % p * pow(int(c.denominator), -1, p) % p
                for g, e in enumerate(m):
                    if e and g not in vars_:
                        v = v * pow(vals[g], e, p) % p
                terms[key] = (terms.get(key, 0) + v) % p
                self._add(key)
            self.polys.append([(self.at[m], c) for m, c in terms.items() if c])
        self.ser = np.zeros((len(self.mons), K + 1), dtype=np.int64)
        self.ser[0, 0] = 1

    def _add(self, m):
        if m in self.at:
            return
        v = next(i for i, e in enumerate(m) if e)
        lower = m[:v] + (m[v] - 1,) + m[v + 1:]
        self._add(lower)
        self.at[m] = len(self.mons)
        self.mons.append(m)

    def order(self, X, k):
        """The coefficients of order k of the monomials, X[v] the series of variable v known up to order k."""
        p = self.p
        for j, m in enumerate(self.mons[1:], 1):
            v = next(i for i, e in enumerate(m) if e)
            par = self.at[m[:v] + (m[v] - 1,) + m[v + 1:]]
            self.ser[j, k] = _dot(self.ser[par, :k + 1], X[v][k::-1], p)

    def value(self, i, k):
        return sum(c * int(self.ser[j, k]) for j, c in self.polys[i]) % self.p


def trajectory(rel, values, p, K):
    """Taylor coefficients up to order K, modulo p, of the observables and of the known states along the trajectory of
    the first condition with the coordinates at `values` ({symbol: residue}); None at a pole."""
    J = jets_of(rel)
    seg = rel.part["segs"][0]
    names = rel.names
    vals = [0] * len(names)
    for s, v in values.items():
        if str(s) in names:
            vals[names.index(str(s))] = v % p
    kept = [j for j in range(rel.ns) if J.X[j] is not None]
    gidx = {j: names.index(str(rel.X[j])) for j in kept}
    vars_ = [gidx[j] for j in kept]
    known = set(rel.known)
    kser = _known_series(rel, values, p, K)
    if kser is None:
        return None
    X = np.zeros((len(kept), K + 1), dtype=np.int64)
    sub = {s: spy.Integer(int(v)) for s, v in values.items()}
    for a, j in enumerate(kept):
        if j in known:
            X[a] = kser[j]
            continue
        x0 = seg["x0"][j][0]
        num, den = (int(t) for t in _fraction(x0, sub))
        if den % p == 0:
            return None
        X[a, 0] = num % p * pow(den % p, -1, p) % p
    unknown = [a for a, j in enumerate(kept) if j not in known]
    Q = J.Q if J.Q is not None else J.R.one
    S = _Series([J.Fh[kept[a]] for a in unknown] + [Q], vars_, vals, p, K)
    q0 = None
    dser = {a: [] for a in unknown}
    qs = []
    for k in range(K):
        S.order(X, k)
        qs.append(S.value(len(unknown), k))
        if q0 is None:
            q0 = qs[0]
            if not q0:
                return None
            iq = pow(q0, -1, p)
        for n, a in enumerate(unknown):
            # x' Q = F: the coefficient k of x' from those below
            rest = sum(qs[i] * dser[a][k - i] for i in range(1, k + 1)) % p
            d = (S.value(n, k) - rest) * iq % p
            dser[a].append(d)
            X[a, k + 1] = d * pow(k + 1, -1, p) % p
    S.order(X, K)
    obs = []
    for r in range(len(rel.y)):
        M, E = J._parts(rel.y[r])
        T = _Series([M, E if E is not None else J.R.one], vars_, vals, p, K)
        for k in range(K + 1):
            T.order(X, k)
        m_ = [T.value(0, k) for k in range(K + 1)]
        e_ = [T.value(1, k) for k in range(K + 1)]
        if not e_[0]:
            return None
        ie = pow(e_[0], -1, p)
        y = []
        for k in range(K + 1):
            y.append((m_[k] - sum(e_[i] * y[k - i] for i in range(1, k + 1))) * ie % p)
        obs.append(y)
    states = {j: X[a] for a, j in enumerate(kept)}
    return obs, states


class _Terms:
    """A polynomial of the series as {exponents: Fraction}, read as terms."""

    def __init__(self, d):
        self.d = d

    def terms(self):
        return list(self.d.items())


def _integrate(polys, vars_, vals, starts, p, K):
    """Taylor coefficients up to order K of the states with x_i' D_i = N_i, polys = [N_1, ..., D_1, ...]; None at
    a pole."""
    n = len(starts)
    S = _Series(polys, vars_, vals, p, K)
    X = np.zeros((n, K + 1), dtype=np.int64)
    X[:, 0] = starts
    dser = [[] for _ in range(n)]
    dens = [[] for _ in range(n)]
    for k in range(K):
        S.order(X, k)
        for i in range(n):
            dens[i].append(S.value(n + i, k))
            if not dens[i][0]:
                return None
            rest = sum(dens[i][l] * dser[i][k - l] for l in range(1, k + 1)) % p
            d = (S.value(i, k) - rest) * pow(dens[i][0], -1, p) % p
            dser[i].append(d)
            X[i, k + 1] = d * pow(k + 1, -1, p) % p
    return X


def _known_series(rel, values, p, K):
    """{state: Taylor coefficients up to order K modulo p} of the states with a known series."""
    part, seg = rel.part, rel.part["segs"][0]
    known = sorted(part["known"])
    if not known:
        return {}
    ns = len(part["st"])
    params = [str(x) for x in rel.series.params]
    byname = {str(s): v for s, v in values.items()}
    vals = [0] * ns + [byname.get(x, 0) % p for x in params]
    starts = []
    for i in known:
        c = Fraction(str(seg["x0"][i][0]))
        starts.append(c.numerator % p * pow(c.denominator, -1, p) % p)
    polys = [_Terms(seg["F"][i][0]) for i in known] + [_Terms(seg["F"][i][1]) for i in known]
    X = _integrate(polys, known, vals, starts, p, K)
    return None if X is None else {i: X[a] for a, i in enumerate(known)}


def _fraction(expr, sub):
    """(numerator, denominator) of `expr` with the substitution `sub`, as sympy integers."""
    v = spy.sympify(expr).xreplace(sub)
    n, d = spy.fraction(spy.together(v))
    return spy.Integer(n), spy.Integer(d)


def taylor_rows(rel, ms, values, p, K):
    """The first K Taylor coefficients, along the trajectory of the first condition with the coordinates at
    `values`, of the derivatives y_r^(j), j <= ms[r], and of the input states read; None at a pole."""
    m = max(ms)
    got = trajectory(rel, values, p, m + K)
    if got is None:
        return None
    obs, states = got
    rows = []
    for r, mr in enumerate(ms):
        for j in range(mr + 1):
            c = obs[r]
            rows.append([c[k + j] * (math.factorial(k + j) // math.factorial(k)) % p for k in range(K)])
    for i in rel.inputs(m):
        rows.append([int(v) for v in states[i][:K]])
    return np.array(rows, dtype=np.int64)


def series_rank(rel, proj, mons, syms, rng, p):
    """The rank of the Taylor coefficients of the monomials along the trajectory of the first condition at a random
    point modulo p."""
    S = taylor_rows(rel, proj.ms, {s: rng.randrange(1, p) for s in syms}, p, len(mons) + 6)
    if S is None or S.shape[0] != proj.nb:
        return None
    ser = _core.io_series_monomials(S, np.array(mons, dtype=np.int32), p, _threads())
    work.tick(ser.shape)
    ker = _core.io_kernel(np.ascontiguousarray(ser.T), p, _threads())
    return len(mons) - len(ker["free"])


def combine(J, coeffs, quos):
    """(numerator, exponents T): sum_j coeffs_j quos_j over the common denominator prod_k F_k^T_k, each term times
    the powers of the factors it lacks; without gcds."""
    T = {}
    for q in quos:
        for k, x in q.exps:
            T[k] = max(T.get(k, 0), x)
    pw = {}
    total = J.R.zero
    for c, q in zip(coeffs, quos):
        if not c:
            continue
        term = c * q.numer
        have = dict(q.exps)
        for k, x in T.items():
            e = x - have.get(k, 0)
            if e:
                if (k, e) not in pw:
                    pw[(k, e)] = J.fac[k] ** e
                term = term * pw[(k, e)]
        total = total + term
    return total, T


def products(J, base, mons):
    """The monomials of the base functions as Quo."""
    pw = {}
    out = []
    for m in mons:
        num, ex = J.R.one, {}
        for v, e in enumerate(m):
            if not e:
                continue
            if (v, e) not in pw:
                pw[(v, e)] = base[v].numer ** e
            num = num * pw[(v, e)]
            for k, x in base[v].exps:
                ex[k] = ex.get(k, 0) + x * e
        out.append(Quo(num, sorted(ex.items())))
    return out


def identity(rel, proj, coef, params_pos):
    """Whether sum_j a_j m_j(base) is zero, the coefficients a_j = N_j / D and the base functions N_v / prod_k F_k^e_vk:
    the polynomial sum_j N_j prod_v N_v^m_jv prod_k F_k^(T_k - t_jk) in the ring of the states and coordinates, t_jk
    the exponent of F_k in the denominator of monomial j and T_k the largest; without gcds."""
    J = jets_of(rel)
    R = J.R
    ng = R.ngens
    mons = proj.mons

    def ring_poly(co):
        out = R.zero
        for m, c in co.items():
            e = [0] * ng
            for k, x in zip(params_pos, m):
                e[k] = x
            out = out + R({tuple(e): R.domain.convert(c.numerator) / R.domain.convert(c.denominator)})
        return out

    coeffs = [None] * len(mons)
    for k, j in enumerate(coef.order[:-1]):
        coeffs[j] = ring_poly(coef.nums[k])
    coeffs[coef.order[-1]] = ring_poly(coef.den)
    return combine(J, coeffs, products(J, proj.base.funcs, mons))[0] == 0


def coefficient_values(coef, theta, p):
    """The coefficients of the projection at the parameter point theta modulo p, in the order of the monomials;
    None where the denominator vanishes there."""
    def ev(co):
        s = 0
        for m, c in co.items():
            t = c.numerator % p * pow(c.denominator, -1, p)
            for x, e in zip(theta, m):
                if e:
                    t = t * pow(int(x), e, p) % p
            s += t
        return s % p

    D = ev(coef.den)
    if not D:
        return None
    iD = pow(D, -1, p)
    out = [0] * len(coef.order)
    for k, j in enumerate(coef.order[:-1]):
        out[j] = ev(coef.nums[k]) * iD % p
    out[coef.order[-1]] = 1
    return out
