"""State components of a parameter direction on the states it reaches, solved modulo a prime.

The identities [Y, F] = 0, Y(h) = 0 and xi(x0) = Y(x0) of a field Y are linear in the state components.
An ansatz of components on a support of states, grown along the readers of the states, is solved at a
random point modulo a prime; the components are then read over Q and proven by the identities."""

import random

import numpy as np
import sympy as spy
from sympy.polys.matrices import DomainMatrix

from ..errors import SYMPY_FAILS, InapplicableError, sympy_fail
from ..gfp import PRIMES, rref_modp
from .exact import _expr, _rhs, _rhs_of
from .linalg import _rational_reconstruct

# steps of growth of the support along the readers of the states in it, and the states past which no lift is
# sought
_GROWTH = 6
_SUPPORT_CAP = 200
# unknowns of one linear system
_UNKNOWN_CAP = 600
# sample points beyond the unknowns of an identity, so that the samples fix the identity
_EXTRA_POINTS = 4
# states of the support up to which quadratic components are tried
_QUADRATIC_STATES = 12
# unknowns up to which the solution is solved over the rational functions of the parameters
_SYMBOLIC_UNKNOWNS = 8


def _inverse(a, p):
    """Elementwise inverse of the residues `a` modulo p."""
    if np.any(a % p == 0):
        raise ZeroDivisionError("a pole at the sample point")
    out, b, e = np.ones_like(a), a % p, p - 2
    while e:
        if e & 1:
            out = out * b % p
        b = b * b % p
        e >>= 1
    return out


class _Residues:
    """Values modulo p of rational expressions at n points; states vary per point, other symbols hold one value."""

    def __init__(self, states, n, p, seed, high=None):
        self.states, self.n, self.p = states, n, p
        self.rng = random.Random(seed)
        self.high = high or p
        self.vals, self.memo = {}, {}

    def symbol(self, s):
        v = self.vals.get(s)
        if v is None:
            k = self.n if s in self.states else 1
            draw = [self.rng.randrange(2, self.high) for _ in range(k)]
            v = np.array(draw * (self.n if k == 1 else 1), dtype=np.int64) % self.p
            self.vals[s] = v
        return v

    def __call__(self, e):
        r = self.memo.get(e)
        if r is not None:
            return r
        p, n = self.p, self.n
        if e.is_Rational:
            if int(e.q) % p == 0:
                raise ZeroDivisionError("a denominator divisible by the prime")
            r = np.full(n, int(e.p) % p * pow(int(e.q), -1, p) % p, dtype=np.int64)
        elif e.is_Symbol:
            r = self.symbol(e)
        elif e.is_Add or e.is_Mul:
            r = np.zeros(n, dtype=np.int64) if e.is_Add else np.ones(n, dtype=np.int64)
            for a in e.args:
                r = (r + self(a)) % p if e.is_Add else r * self(a) % p
        elif e.is_Pow and e.exp.is_Integer:
            b, k = self(e.base), int(e.exp)
            if k < 0:
                b, k = _inverse(b, p), -k
            r = np.ones(n, dtype=np.int64)
            while k:
                if k & 1:
                    r = r * b % p
                b = b * b % p
                k >>= 1
        else:
            raise InapplicableError(f"not rational: {e}")
        self.memo[e] = r
        return r


def _ansatz(S, reads, readers, level):
    """Monomials, as tuples of (state, exponent), of the component of each state of the support: a constant and
    the state itself, at level 1 the states of the support next to it, at level 2 their products."""
    out = {}
    for i in sorted(S, key=str):
        near = sorted(((reads[i] | readers.get(i, set())) & S) - {i}, key=str)
        ms = [(), ((i, 1),)]
        if level >= 1:
            ms += [((j, 1),) for j in near]
        if level >= 2:
            nb = [i] + near
            ms += [
                ((a, 2),) if a == b else tuple(sorted(((a, 1), (b, 1)), key=lambda t: str(t[0])))
                for k, a in enumerate(nb)
                for b in nb[k:]
            ]
        out[i] = list(dict.fromkeys(ms))
    return out


class _Problem:
    """The identities of a lift with parameter components `theta` on the states of the regime `F`."""

    def __init__(self, theta, F, obs, x0):
        self.theta, self.F, self.obs, self.x0 = theta, F, obs, x0
        self.moved = set(theta)
        self.reads = {x: f.free_symbols for x, f in F.items()}
        self.readers = {}
        for x, r in self.reads.items():
            for s in r & F.keys():
                self.readers.setdefault(s, set()).add(x)
        self.diffs = {}

    def d(self, e, s):
        key = (e, s)
        if key not in self.diffs:
            self.diffs[key] = spy.diff(e, s)
        return self.diffs[key]

    def identities(self, S):
        """(kind, state, expression, the states of S it reads): [Y, F] per state of S or reading S or a parameter
        moved, Y(h) per output reading them, and the start of each state of S with an initial value."""
        out, touched = [], S | self.moved
        for l, f in self.F.items():
            if l in S or self.reads[l] & touched:
                out.append(("F", l, f, self.reads[l] & S))
        for h in self.obs:
            if h.free_symbols & touched:
                out.append(("h", None, h, h.free_symbols & S))
        for i in sorted(S, key=str):
            if i in self.x0:
                out.append(("x0", i, self.x0[i], {i}))
        return out

    def _at_x0(self, m):
        """The monomial m at the start, None where a state of m has no initial value."""
        if any(j not in self.x0 for j, _ in m):
            return None
        return spy.Mul(*[self.x0[j] ** e for j, e in m])

    def _theta_part(self, f, V):
        """Y(f) of the parameter components at the points of V."""
        p = V.p
        out = np.zeros(V.n, dtype=np.int64)
        for s in f.free_symbols & self.moved:
            out = (out + V(self.theta[s]) * V(self.d(f, s))) % p
        return out

    def rows(self, S, mons, cols, V):
        """(identity, {column: values at the points of V}, right-hand side) over the unknowns `cols`."""
        p, n = V.p, V.n
        mval = {}

        def mono(m):
            if m not in mval:
                r = np.ones(n, dtype=np.int64)
                for j, e in m:
                    for _ in range(e):
                        r = r * V(j) % p
                mval[m] = r
            return mval[m]

        def flux(m):
            r = np.zeros(n, dtype=np.int64)
            for j, e in m:
                r = (r + e * V(self.F[j]) % p * mono(m) % p * _inverse(V(j), p)) % p
            return r

        out = []
        for ident in self.identities(S):
            kind, l, f, ins = ident
            block = {}
            if kind == "x0":
                at = [self._at_x0(m) for m in mons[l]]
                if any(v is None for v in at):
                    continue
                for m, v in zip(mons[l], at):
                    if (l, m) in cols:
                        block[cols[(l, m)]] = V(v)
                out.append((ident, block, self._theta_part(f, V)))
                continue
            for i in ins:
                Ji = V(self.d(f, i))
                for m in mons[i]:
                    if (i, m) in cols:
                        block[cols[(i, m)]] = Ji * mono(m) % p
            if kind == "F" and l in S:
                for m in mons[l]:
                    if (l, m) in cols and m:
                        k = cols[(l, m)]
                        block[k] = (block.get(k, 0) - flux(m)) % p
            out.append((ident, block, (-self._theta_part(f, V)) % p))
        return out

    def solve(self, S, mons, p, seed):
        """The solution with free unknowns 0 as {(state, monomial): residue} and the point of the samples; None
        where no solution exists."""
        unknowns = [(i, m) for i in sorted(S, key=str) for m in mons[i]]
        cols = {u: k for k, u in enumerate(unknowns)}
        U = len(unknowns)
        width = max(
            (
                sum(len(mons[i]) for i in ins) + (len(mons[l]) if kind == "F" and l in S else 0)
                for kind, l, _, ins in self.identities(S)
            ),
            default=0,
        )
        V = _Residues(set(self.F), width + _EXTRA_POINTS, p, seed)
        blocks = []
        for ident, block, b in self.rows(S, mons, cols, V):
            if not block:
                if np.any(b % p):
                    return None, V
                continue
            ks = sorted(block)
            M = np.stack([block[k] for k in ks] + [b], axis=1) % p
            rr = rref_modp(M[:1] if ident[0] == "x0" else M, p)
            if not rr["rank"]:
                continue
            G = np.zeros((rr["rank"], U + 1), dtype=np.int64)
            G[:, ks] = rr["R"][: rr["rank"], :-1]
            G[:, U] = rr["R"][: rr["rank"], -1]
            blocks.append(G)
        if not blocks:
            return {}, V
        rr = rref_modp(np.concatenate(blocks, axis=0), p)
        piv = [int(c) for c in rr["piv"][: rr["rank"]]]
        if U in piv:
            return None, V
        return {unknowns[c]: int(rr["R"][r, U]) % p for r, c in enumerate(piv) if int(rr["R"][r, U]) % p}, V

    def factors(self):
        """Functions of the parameters by which the lift may change: the lift of c Y is c times the lift of Y."""
        out = [spy.Integer(1)]
        den = spy.Integer(1)
        for s, e in self.theta.items():
            out += [e, e / s]
            den = spy.lcm(den, spy.fraction(spy.together(e))[1])
        return list(dict.fromkeys(out + [1 / den]))

    def exact(self, S, mons, sol, V1):
        """The solution over Q: a number times one of the factors where the quotient does not change with the
        parameters, read from two primes, else an exact solve of the unknowns found; None where neither reads it."""
        keys = sorted(sol, key=str)
        p1, p2 = PRIMES[0], PRIMES[1]
        other, V2 = self.solve(S, mons, p1, 2)
        second, V3 = self.solve(S, mons, p2, 3)
        if other is not None and second is not None and set(other) == set(sol) == set(second):
            for c in self.factors():
                try:
                    a1, a2, a3 = (int(V(c)[0]) for V in (V1, V2, V3))
                except SYMPY_FAILS as exc:
                    sympy_fail(exc)
                    continue
                if not (a1 % p1 and a2 % p1 and a3 % p2):
                    continue
                k1 = {u: sol[u] * pow(a1, -1, p1) % p1 for u in keys}
                if k1 != {u: other[u] * pow(a2, -1, p1) % p1 for u in keys}:
                    continue
                out = {}
                for u in keys:
                    k3 = second[u] * pow(a3, -1, p2) % p2
                    q = _rational_reconstruct((k1[u] + p1 * ((k3 - k1[u]) * pow(p1, -1, p2) % p2)) % (p1 * p2), p1 * p2)
                    if q is None:
                        break
                    out[u] = q * c
                else:
                    return out
        return self.symbolic(S, mons, keys) if len(keys) <= _SYMBOLIC_UNKNOWNS else None

    def symbolic(self, S, mons, keys):
        """The unknowns `keys` over Q(theta) from as many independent identities at points with small states."""
        K = len(keys)
        cols = {u: k for k, u in enumerate(keys)}
        p = PRIMES[0]
        V = _Residues(set(self.F), K + _EXTRA_POINTS, p, 5, high=1000)
        rows = []
        for ident, block, b in self.rows(S, mons, cols, V):
            for t in range(1 if ident[0] == "x0" else V.n):
                v = np.zeros(K + 1, dtype=np.int64)
                for k, col in block.items():
                    v[k] = col[t]
                v[K] = b[t]
                rows.append((ident, t, v))
        if not rows:
            return None
        rr = rref_modp(np.stack([v[:K] for _, _, v in rows], axis=1), p)
        if rr["rank"] < K:
            return None
        Ms, bs = [], []
        for c in rr["piv"][:K]:
            ident, t, _ = rows[int(c)]
            sub = {s: spy.Integer(int(v[t])) for s, v in V.vals.items() if s in V.states}
            Ms.append([self._entry(ident, u, sub) for u in keys])
            bs.append(self._rhs(ident, sub))
        try:
            dA = DomainMatrix.from_Matrix(spy.Matrix(Ms).row_join(spy.Matrix(bs))).to_field()
            sol = dA.extract(list(range(K)), list(range(K))).lu_solve(dA.extract(list(range(K)), [K])).to_Matrix()
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            return None
        return {u: spy.cancel(sol[k]) for k, u in enumerate(keys)}

    def _entry(self, ident, u, sub):
        """The coefficient of the unknown u = (state, monomial) in the identity, at the states `sub`."""
        kind, l, f, _ = ident
        i, m = u
        if kind == "x0":
            return self._at_x0(m) if l == i else spy.Integer(0)
        mono = spy.Mul(*[j**e for j, e in m])
        e = self.d(f, i) * mono if i in f.free_symbols else spy.Integer(0)
        if kind == "F" and l == i and m:
            e -= sum((self.F[j] * spy.diff(mono, j) for j, _ in m), spy.Integer(0))
        return e.xreplace(sub)

    def _rhs(self, ident, sub):
        """The right-hand side of the identity, at the states `sub`."""
        kind, _, f, _ = ident
        y = sum((self.theta[s] * self.d(f, s) for s in f.free_symbols & self.moved), spy.Integer(0))
        return y if kind == "x0" else -y.xreplace(sub)


def support_lift(field, regime, ic_lines, states):
    """`field` with state components on the states the direction reaches, or None.

    The support starts at the states whose equation, start or output reads a parameter moved and grows by
    the states reading it; the components are polynomials of degree at most two in the states of the support.
    The identities of the lift decide it."""
    sts = {spy.Symbol(str(x)) for x in states}
    try:
        F = {x: f for x, f in _rhs_of(tuple(str(line) for line in regime["f"])).items() if x in sts}
        theta = {spy.Symbol(str(k)): _expr(str(v)) for k, v in dict(field).items()}
        theta = {k: v for k, v in theta.items() if v != 0}
        obs = [_expr(str(line).split("=", 1)[1] if "=" in str(line) else str(line)) for line in regime["g"]]
        x0 = {x: e for x, e in _rhs(ic_lines).items() if x in F}
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    if (
        not F
        or not theta
        or set(theta) & set(F)
        or any(v.free_symbols & set(F) for v in theta.values())
        or any(e.free_symbols & set(F) for e in x0.values())
    ):
        return None
    pb = _Problem(theta, F, obs, x0)
    S = {x for x in F if pb.reads[x] & pb.moved or (x in x0 and x0[x].free_symbols & pb.moved)}
    S |= {x for h in obs if h.free_symbols & pb.moved for x in h.free_symbols & set(F)}
    try:
        for _ in range(_GROWTH):
            for level in (0, 1, 2):
                if level == 2 and len(S) > _QUADRATIC_STATES:
                    break
                mons = _ansatz(S, pb.reads, pb.readers, level)
                if sum(len(v) for v in mons.values()) > _UNKNOWN_CAP:
                    break
                sol, V = pb.solve(S, mons, PRIMES[0], 1)
                if sol is None:
                    continue
                vals = pb.exact(S, mons, sol, V) if sol else None
                return None if vals is None else _lifted(field, vals)
            grow = set().union(*(pb.readers.get(s, set()) for s in S)) - S
            if not grow or len(S | grow) > _SUPPORT_CAP:
                return None
            S |= grow
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
    return None


def _lifted(field, vals):
    """`field` with the state components sum c m of the solution {(state, monomial): c}."""
    comp = {}
    for (i, m), c in vals.items():
        comp[i] = comp.get(i, spy.Integer(0)) + c * spy.Mul(*[j**e for j, e in m])
    out = dict(field)
    for i, e in comp.items():
        if e != 0:
            out[str(i)] = str(e)
    return out
