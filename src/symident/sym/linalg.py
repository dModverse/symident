"""Exact linear algebra: GF(p) elimination, CRT, rational reconstruction, integer kernels."""

import math

import numpy as np
import sympy as spy

from .parse import _PRIMES


def _rref_mod_p(A, p):
    """Vectorized int64 Gauss-Jordan over GF(p); returns (R, pivots). Eliminates
    only rows nonzero in the pivot column (the scaling matrix is sparse) by a rank-1
    update; products < p^2 < 2^62 fit int64."""
    A = (np.asarray(A, dtype=np.int64) % p)
    nrows, ncols = A.shape
    pivots = []
    r = 0
    for c in range(ncols):
        nz = np.nonzero(A[r:, c])[0]
        if nz.size == 0:
            continue
        piv = r + int(nz[0])
        if piv != r:
            A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), p - 2, p)
        # the pivot row is zero left of c, so the updates start at c
        A[r, c:] = (A[r, c:] * inv) % p
        col = A[:, c].copy()
        col[r] = 0
        nzr = np.nonzero(col)[0]
        if nzr.size:
            A[nzr, c:] = (A[nzr, c:] - np.outer(col[nzr], A[r, c:])) % p
        pivots.append(c)
        r += 1
        if r == nrows:
            break
    return A, pivots


def _rational_reconstruct(a, m):
    """Recover n/d with a == n*d^{-1} (mod m), |n|,|d| bounded by sqrt(m/2)."""
    a %= m
    if a == 0:
        return spy.Integer(0)
    bound = math.isqrt(m // 2)
    r0, r1 = m, a
    s0, s1 = 0, 1
    while r1 > bound:
        q = r0 // r1
        r0, r1 = r1, r0 - q * r1
        s0, s1 = s1, s0 - q * s1
    if s1 == 0 or abs(s1) > bound:
        return None
    return spy.Rational(r1, s1)


def sym_rat_recon_big(residues, primes):
    """Arbitrary-precision per-row CRT + rational reconstruction, free of the u128
    cap (4 primes) of the C++ sym_rat_recon. Each row of `residues` is one coefficient
    across `primes`; returns {'num', 'den'} as decimal strings, den '0' where no lift
    exists. For coefficients with logarithms whose height exceeds the 4-prime bound."""
    from sympy.ntheory.modular import crt
    mods = [int(p) for p in primes]
    M = 1
    for m in mods:
        M *= m
    rows = [[int(x) for x in row] for row in np.asarray(residues, dtype=object)]
    num, den = [], []
    for row in rows:
        res = [r % m for r, m in zip(row, mods)]
        x, _ = crt(mods, res)
        val = _rational_reconstruct(int(x), int(M))
        if val is None:
            num.append("0")
            den.append("0")
        else:
            val = spy.Rational(val)
            num.append(str(int(val.p)))
            den.append(str(int(val.q)))
    return {'num': num, 'den': den}


def _crt_nullspace(reduce_mod_p, ncols):
    """Multi-prime GF(p) + CRT + rational-reconstruction nullspace, shared by the
    two exact nullspace routines below. `reduce_mod_p(p)` yields the matrix mod p.
    Primes whose pivot set differs from the first (unlucky rank drop) are skipped;
    the first four agreeing ones are lifted to Q. Returns the unvalidated basis as
    sympy column vectors (callers validate), or None if no residue lifts."""
    ref_pivots = None
    free = None
    residues = {}
    mods = []
    # the first four primes reduce in parallel threads (the eliminations are numpy
    # array operations, which release the GIL); further primes only on a pivot clash
    first = list(_PRIMES[:4])
    try:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(first)) as ex:
            pre = dict(zip(first, ex.map(lambda q: _rref_mod_p(reduce_mod_p(q), q), first)))
    except Exception:
        pre = {}
    for p in _PRIMES:
        R, pivots = pre[p] if p in pre else _rref_mod_p(reduce_mod_p(p), p)
        if ref_pivots is None:
            ref_pivots = pivots
            pivset = set(pivots)
            free = [c for c in range(ncols) if c not in pivset]
        elif pivots != ref_pivots:
            continue
        mods.append(p)
        for ki in range(len(ref_pivots)):
            for f in free:
                residues.setdefault((ki, f), []).append(int(R[ki, f]) % p)
        if len(mods) >= 4:
            break
    if not mods:
        return None

    from sympy.ntheory.modular import crt
    exact = {}
    for key, res in residues.items():
        x, Mmod = crt(mods, res)
        val = _rational_reconstruct(int(x), int(Mmod))
        if val is None:
            return None
        exact[key] = val

    basis = []
    for f in free:
        v = [spy.Integer(0)] * ncols
        v[f] = spy.Integer(1)
        for ki, c in enumerate(ref_pivots):
            v[c] = -exact[(ki, f)]
        basis.append(spy.Matrix(v))
    return basis


def _modular_nullspace(M):
    """Exact nullspace of sympy Matrix M via _crt_nullspace, validated symbolically.
    Returns list of sympy column vectors, or None if reconstruction/validation
    fails (caller falls back to sympy)."""
    nrows, ncols = M.shape
    if nrows == 0:
        return [spy.Matrix([1 if j == c else 0 for j in range(ncols)])
                for c in range(ncols)]
    Aint = []
    for i in range(nrows):
        row = [spy.Rational(M[i, j]) for j in range(ncols)]
        L = 1
        for r in row:
            L = spy.ilcm(L, r.q)
        Aint.append([int(r * L) for r in row])

    basis = _crt_nullspace(lambda p: [[x % p for x in row] for row in Aint], ncols)
    if basis is None:
        return None
    for v in basis:
        if not (M * v).is_zero_matrix:
            return None
    return basis


def _modular_nullspace_int(rows_int, ncols):
    """Integer-matrix analogue of _modular_nullspace (scaling determining matrix),
    in numpy int with no per-entry sympy. Returns sympy column vectors, or None if
    reconstruction or validation fails."""
    A0 = np.asarray(rows_int, dtype=np.int64)
    basis = _crt_nullspace(lambda p: A0 % p, ncols)
    if basis is None:
        return None

    # exact validation over the integers: clear denominators per basis vector, then
    # A0 @ W == 0 in one object-dtype matmul (exact big-int, no overflow)
    if basis:
        cols = []
        for v in basis:
            L = 1
            for e in v:
                L = spy.ilcm(L, spy.Rational(e).q)
            cols.append([int(spy.Rational(e) * L) for e in v])
        W = np.array(cols, dtype=object).T
        if np.any(A0.astype(object).dot(W) != 0):
            return None
    return basis


def exact_int_kernel(rows, ncols):
    """Primitive integer basis of the kernel of an integer matrix, as plain int
    lists: exact_nullspace + the _scaling_gens content clearing."""
    basis = exact_nullspace([[int(x) for x in r] for r in rows], int(ncols),
                           integer=True)
    out = []
    for v in basis:
        L = 1
        for e in v:
            L = spy.ilcm(L, spy.Rational(e).q)
        w = [int(spy.Rational(e) * L) for e in v]
        G = 0
        for e in w:
            G = spy.igcd(G, abs(e))
        if G > 1:
            w = [e // G for e in w]
        out.append(w)
    return out


def exact_nullspace(rows, ncols, integer=False):
    """Nullspace basis as sympy column vectors. integer=True (integer rows) tries
    the numpy-int GF(p) + CRT path first, falling back to sympy."""
    if not rows:
        return [spy.Matrix([1 if j == c else 0 for j in range(ncols)])
                for c in range(ncols)]
    if integer:
        basis = _modular_nullspace_int(rows, ncols)
        if basis is not None:
            return basis
    M = spy.Matrix([[spy.sympify(x) for x in row] for row in rows])
    # the modular solver needs a rational matrix; transcendental constants
    # (e.g. log(10) from a log10 observable) route to the exact sympy nullspace
    if all(bool(e.is_rational) for e in M):
        basis = _modular_nullspace(M)
        if basis is not None:
            return basis
    return M.nullspace()


def _rank_mod(rows, p):
    """Rank of a list of rows (lists of ints) over GF(p)."""
    m = [list(r) for r in rows]
    rank, ncol = 0, (len(m[0]) if m else 0)
    for c in range(ncol):
        piv = next((i for i in range(rank, len(m)) if m[i][c] % p), None)
        if piv is None:
            continue
        m[rank], m[piv] = m[piv], m[rank]
        inv = pow(m[rank][c] % p, p - 2, p)
        m[rank] = [x * inv % p for x in m[rank]]
        for i in range(len(m)):
            if i != rank and m[i][c] % p:
                fct = m[i][c]
                m[i] = [(a - fct * b) % p for a, b in zip(m[i], m[rank])]
        rank += 1
    return rank
