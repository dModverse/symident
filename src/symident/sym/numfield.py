"""Exact arithmetic in a number field Q[s]/(W(s)), W monic over Z, and the cycle types of the Frobenius elements of
the splitting field of a polynomial over it at primes of degree one (Dedekind)."""

import math
from fractions import Fraction

import sympy as spy
from sympy.polys.domains import ZZ
from sympy.polys.galoistools import gf_ddf_zassenhaus, gf_diff, gf_gcd, gf_monic


class NumberField:
    """Q[s]/(W(s)) for a monic irreducible W over Z; elements are tuples of Fractions, the coefficients of
    1, s, ..., s^(d-1)."""

    def __init__(self, w):
        """`w`: the integer coefficients of an irreducible polynomial in t, lowest degree first; s = lc(w) t makes it
        monic over Z."""
        w = [int(c) for c in w]
        while len(w) > 1 and w[-1] == 0:
            w.pop()
        self.d = len(w) - 1
        self.lc = w[-1]
        # W(s) = lc^(d-1) w(s / lc)
        self.W = [c * self.lc ** (self.d - 1 - k) for k, c in enumerate(w[:-1])] + [1] if self.d else [1]
        self.zero = (Fraction(0),) * max(self.d, 1)
        self.one = (Fraction(1),) + (Fraction(0),) * (max(self.d, 1) - 1)

    def __repr__(self):
        if self.d <= 1:
            return "Q"
        terms = [f"{c}*s^{k}" if k > 1 else f"{c}*s" if k == 1 else str(c) for k, c in enumerate(self.W) if c]
        return "Q[s]/(" + " + ".join(reversed(terms)).replace("+ -", "- ") + ")"

    def rational(self, c):
        return (Fraction(c),) + (Fraction(0),) * (max(self.d, 1) - 1)

    def from_t(self, coeffs):
        """The element of a polynomial in t (lowest degree first), with t = s / lc."""
        acc = [Fraction(0)] * max(len(coeffs), 1)
        for k, c in enumerate(coeffs):
            acc[k] = Fraction(c) / Fraction(self.lc) ** k
        return self._reduce(acc)

    def _reduce(self, a):
        d = self.d
        if d == 0:
            return (Fraction(a[0]) if a else Fraction(0),)
        if d == 1:
            # the field is Q, with s the rational root of W
            r = -Fraction(self.W[0])
            return (sum((c * r**k for k, c in enumerate(a)), Fraction(0)),)
        a = list(a) + [Fraction(0)] * max(0, d - len(a))
        for k in range(len(a) - 1, d - 1, -1):
            c = a[k]
            if c:
                for j in range(d):
                    a[k - d + j] -= c * self.W[j]
            a[k] = Fraction(0)
        return tuple(a[:d])

    def add(self, a, b):
        return tuple(x + y for x, y in zip(a, b))

    def sub(self, a, b):
        return tuple(x - y for x, y in zip(a, b))

    def neg(self, a):
        return tuple(-x for x in a)

    def scale(self, a, c):
        return tuple(x * c for x in a)

    def mul(self, a, b):
        if len(a) == 1:
            return (a[0] * b[0],)
        out = [Fraction(0)] * (len(a) + len(b) - 1)
        for i, x in enumerate(a):
            if x:
                for j, y in enumerate(b):
                    if y:
                        out[i + j] += x * y
        return self._reduce(out)

    def inv(self, a):
        if not any(a):
            raise ZeroDivisionError("inverse of zero in a number field")
        if len(a) == 1:
            return (1 / a[0],)
        s = spy.Symbol("s")
        A = spy.Poly(list(reversed([spy.Rational(x.numerator, x.denominator) for x in a])), s, domain="QQ")
        W = spy.Poly(list(reversed(self.W)), s, domain="QQ")
        r = A.invert(W).all_coeffs()
        return self._reduce([Fraction(int(c.p), int(c.q)) for c in reversed(r)])

    def is_zero(self, a):
        return not any(a)

    def residue(self, a, p, tau):
        """The image of `a` modulo the prime (p, s - tau); None where p divides a denominator."""
        acc = 0
        for c in reversed(a):
            if c.denominator % p == 0:
                return None
            acc = (acc * tau + c.numerator * pow(c.denominator, -1, p)) % p
        return acc

    def primes_of_degree_one(self, p):
        """The roots tau of W modulo p, one prime (p, s - tau) of degree one each, where W is square-free modulo p
        (so p is unramified in the field); [] otherwise."""
        if self.d <= 1:
            return [0]
        Wp = [c % p for c in reversed(self.W)]
        if gf_gcd(Wp, gf_diff(Wp, p, ZZ), p, ZZ) != [1]:
            return []
        return _roots_mod(self.W, p)


def _roots_mod(W, p):
    """The roots in GF(p) of the integer polynomial W, lowest degree first, zero included."""
    from .gfpoly import _gf_roots

    out = sorted(int(r) for r in _gf_roots([c % p for c in reversed(W)], p))
    return [0] + out if W[0] % p == 0 else out


def charpoly(K, A):
    """The coefficients of det(z I - A) over the number field K, highest degree first, by Faddeev and LeVerrier: only
    products in K and divisions by integers."""
    n = len(A)
    c = [K.one]
    M = [[K.zero] * n for _ in range(n)]
    for k in range(1, n + 1):
        # M_k = A M_{k-1} + c_{k-1} I, c_k = -tr(A M_k) / k
        AM = [[_dot(K, A[i], [M[r][j] for r in range(n)]) for j in range(n)] for i in range(n)]
        M = [[K.add(AM[i][j], c[-1]) if i == j else AM[i][j] for j in range(n)] for i in range(n)]
        tr = K.zero
        for i in range(n):
            tr = K.add(tr, _dot(K, A[i], [M[r][i] for r in range(n)]))
        c.append(K.scale(tr, Fraction(-1, k)))
    return c


def charpoly_mod(A, p):
    """The coefficients of det(z I - A) over GF(p), highest degree first, for an integer matrix A and p above the
    size of A (Faddeev and LeVerrier)."""
    n = len(A)
    c = [1]
    M = [[0] * n for _ in range(n)]
    for k in range(1, n + 1):
        AM = [[sum(A[i][r] * M[r][j] for r in range(n)) % p for j in range(n)] for i in range(n)]
        M = [[(AM[i][j] + c[-1]) % p if i == j else AM[i][j] for j in range(n)] for i in range(n)]
        tr = sum(A[i][r] * M[r][i] for i in range(n) for r in range(n)) % p
        c.append(-tr * pow(k, -1, p) % p)
    return c


def _dot(K, row, col):
    acc = K.zero
    for a, b in zip(row, col):
        if any(a) and any(b):
            acc = K.add(acc, K.mul(a, b))
    return acc


def cycle_type(coeffs, p):
    """The degrees of the irreducible factors over GF(p) of a monic polynomial (coefficients mod p, highest degree
    first), in decreasing order; None where it is not square-free modulo p."""
    f = gf_monic([int(c) % p for c in coeffs], p, ZZ)[1]
    if len(f) <= 2:
        return [1] * (len(f) - 1)
    if gf_gcd(f, gf_diff(f, p, ZZ), p, ZZ) != [1]:
        return None
    out = []
    for g, d in gf_ddf_zassenhaus(f, p, ZZ):
        out += [int(d)] * ((len(g) - 1) // int(d))
    return sorted(out, reverse=True)


def order_of(ctype):
    """The order of a permutation of this cycle type."""
    return math.lcm(*ctype) if ctype else 1


def rank_over_q(vectors):
    """The rank of a list of rational vectors over Q, by exact elimination."""
    rows = [list(v) for v in vectors]
    rank, cols = 0, len(rows[0]) if rows else 0
    for c in range(cols):
        piv = next((i for i in range(rank, len(rows)) if rows[i][c]), None)
        if piv is None:
            continue
        rows[rank], rows[piv] = rows[piv], rows[rank]
        for i in range(len(rows)):
            if i != rank and rows[i][c]:
                f = rows[i][c] / rows[rank][c]
                rows[i] = [x - f * y for x, y in zip(rows[i], rows[rank])]
        rank += 1
    return rank
