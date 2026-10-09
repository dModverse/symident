"""Linear algebra over GF(p), sample points and gauges of the residual directions."""

import contextlib
import math
import time

import numpy as np

from . import _core

# four primes below 1518500213, msolve's bound for FGLM; a list, so that use_primes()
# can switch every module that imported it
PRIMES = [1518500183, 1518500173, 1518500171, 1518500143]
# disjoint from the reconstruction primes; certifies a direction at a fresh evaluation
VERIFY_PRIME = 1518500141
_VERIFY = [VERIFY_PRIME]


def verify_prime():
    """The prime of the fresh-point checks."""
    return _VERIFY[0]


class NeedPrimesError(Exception):
    """Raised when some engine prime does not hold a root of the model: a rational constant
    under a square root that is no square, or an odd root degree sharing a factor with
    p - 1. The analysis reruns under use_primes(constants, degrees)."""

    def __init__(self, constants=(), degrees=()):
        super().__init__("roots of degree %s and square roots of %s" % (
            ", ".join(map(str, degrees)) or "-", ", ".join(map(str, constants)) or "-"))
        self.constants = list(constants)
        self.degrees = list(degrees)


def holds_roots(constants, q, degrees=()):
    """Whether every (c, k), c a positive rational, has k nested square roots modulo the
    prime q, and every odd degree is prime to q - 1."""
    if any(math.gcd(int(d), q - 1) != 1 for d in degrees):
        return False
    for c, k in constants:
        v = int(c.p) * pow(int(c.q), q - 2, q) % q
        for _ in range(k):
            v = sqrt_mod(v, q)
            if not v:
                return False
    return True


@contextlib.contextmanager
def use_primes(constants=(), degrees=()):
    """Five primes below the default ones that hold every root of the model (see
    holds_roots): four for the analysis and one for the checks."""
    from sympy import Rational, prevprime
    consts = [(Rational(c), int(k)) for c, k in constants]
    found, q = [], PRIMES[0] + 1
    while len(found) < 5:
        q = prevprime(q)
        if holds_roots(consts, q, degrees):
            found.append(q)
    saved, saved_v = list(PRIMES), _VERIFY[0]
    PRIMES[:] = found[:4]
    _VERIFY[0] = found[4]
    try:
        yield
    finally:
        PRIMES[:] = saved
        _VERIFY[0] = saved_v


def sieve(n):
    """The first `n` primes."""
    limit = 200
    while True:
        is_p = np.ones(limit + 1, dtype=bool)
        is_p[:2] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                is_p[i * i::i] = False
        pr = np.nonzero(is_p)[0]
        if len(pr) >= n:
            return [int(x) for x in pr[:n]]
        limit *= 2


def random_point(n, seed=1):
    """Residues below the smallest prime by the MINSTD generator, deterministic in seed."""
    m = 2147483647
    x = (1234567 * int(seed) + 89) % m
    if x == 0:
        x = 1
    out = np.zeros(n, dtype=np.int64)
    pmin = min(PRIMES)
    for i in range(n):
        x = (48271 * x) % m
        r = x % pmin
        out[i] = 1 if r == 0 else r
    return out


class Pool:
    """Distinct primes as generic evaluation coordinates, grown on demand (1-based k)."""

    def __init__(self):
        self.cache = sieve(1000)

    def __call__(self, k):
        k = np.atleast_1d(np.asarray(k, dtype=np.int64))
        if k.size and int(k.max()) > len(self.cache):
            self.cache = sieve(max(2 * len(self.cache), int(k.max())))
        return np.array([self.cache[int(i) - 1] for i in k], dtype=np.int64)


def null_residues(res, free_col, p):
    """Kernel vector of the reduced rows `res` with a one in the free column."""
    nz = int(res["dim"])
    v = np.zeros(nz, dtype=np.int64)
    v[free_col] = 1
    R = np.asarray(res["R"], dtype=np.int64).reshape(-1, nz)
    piv = np.atleast_1d(np.asarray(res["pivots"], dtype=np.int64))
    for ri, pc in enumerate(piv):
        v[pc] = (p - R[ri, free_col]) % p
    return v


def null_basis_rows(res, free_cols, p):
    """The kernel vectors of `null_residues` for all `free_cols`, one per row."""
    nz = int(res["dim"])
    fc = np.asarray(free_cols, dtype=np.int64)
    out = np.zeros((len(fc), nz), dtype=np.int64)
    if not len(fc):
        return out
    out[np.arange(len(fc)), fc] = 1
    R = np.asarray(res["R"], dtype=np.int64).reshape(-1, nz)
    piv = np.atleast_1d(np.asarray(res["pivots"], dtype=np.int64))
    if len(piv):
        out[:, piv] = ((p - R[:len(piv)][:, fc]) % p).T
    return out


def in_span(M, x, nz, p):
    """Whether x lies in the column span of the nz-row matrix M over GF(p)."""
    M = np.asarray(M, dtype=np.int64).reshape(nz, -1)
    if M.shape[1] == 0:
        return False
    return _core.sym_solve_mod(M, np.asarray(x, dtype=np.int64), p) is not None


def mulmod(a, b, p):
    """Elementwise product of `a` and `b` mod `p`."""
    a = np.asarray(a, dtype=np.int64) % p
    b = np.asarray(b, dtype=np.int64) % p
    return (a * b) % p


def invmod(a, p):
    """The inverse of `a` mod the prime `p`."""
    return pow(int(a) % p, p - 2, p)


def rref_modp(M, p):
    """RREF over GF(p) with the zero rows kept: R, 0-based pivots, rank."""
    M = np.asarray(M)
    nr, nc = M.shape if M.ndim == 2 else (0, 0)
    if nr == 0 or nc == 0:
        return {"R": np.zeros((nr, nc), dtype=np.int64), "piv": np.zeros(0, dtype=np.int64),
                "rank": 0}
    rr = _core.sym_rref_mod(np.asarray(M, dtype=np.float64) % p, p)
    R = np.zeros((nr, nc), dtype=np.int64)
    rank = int(rr["rank"])
    if rank > 0:
        R[:rank, :] = np.asarray(rr["R"], dtype=np.int64).reshape(rank, nc)
    return {"R": R, "piv": np.atleast_1d(np.asarray(rr["piv"], dtype=np.int64)), "rank": rank}


def phys_rref(B, phys_cols, nz, p):
    """RREF of B preferring the columns phys_cols (0-based) as pivots."""
    B = np.asarray(B, dtype=np.int64)
    if B.shape[0] == 0:
        return {"R": B, "piv": np.zeros(0, dtype=np.int64)}
    phys = sorted(int(c) for c in phys_cols)
    ord_ = phys + sorted(set(range(nz)) - set(phys))
    rr = rref_modp(B[:, ord_], p)
    Rback = np.zeros((rr["R"].shape[0], nz), dtype=np.int64)
    Rback[:, ord_] = rr["R"]
    return {"R": Rback, "piv": np.array([ord_[i] for i in rr["piv"]], dtype=np.int64)}


def matinv_modp(B, p):
    """k-by-k inverse over GF(p) by Gauss-Jordan; None if singular."""
    B = np.asarray(B, dtype=np.int64) % p
    k = B.shape[0]
    A = np.concatenate([B, np.eye(k, dtype=np.int64)], axis=1)
    for col in range(k):
        nzr = np.nonzero(A[col:, col] % p)[0]
        if not len(nzr):
            return None
        pr = col + int(nzr[0])
        if pr != col:
            A[[col, pr]] = A[[pr, col]]
        A[col] = mulmod(A[col], invmod(A[col, col], p), p)
        for r in range(k):
            if r != col:
                f = A[r, col] % p
                if f:
                    A[r] = (A[r] - mulmod(A[col], f, p)) % p
    return A[:, k:]


def reduce_mod_rows(S):
    """Reducer of rows modulo the RREF of the scaling lattice S, memoised per prime."""
    S = np.asarray(S, dtype=np.int64)
    cache = {}

    def red(R, p):
        R = np.array(R, dtype=np.int64) % p
        if S.shape[0] == 0:
            return R
        Sr = cache.get(p)
        if Sr is None:
            Sr = rref_modp(S, p)
            cache[p] = Sr
        for j, pc in enumerate(Sr["piv"]):
            fac = R[:, pc] % p
            for i in range(R.shape[0]):
                if fac[i]:
                    R[i] = (R[i] - mulmod(Sr["R"][j], fac[i], p)) % p
        return R
    return red


def gauge_from_rows(residual_free, rows_fn, ref_rows, P, k, nz):
    """Pin the k pivot columns of the reference rows to the identity."""
    raw = {"anchors": list(residual_free), "residue_fns": [None] * k}
    if ref_rows is None:
        return raw
    rr = rref_modp(ref_rows, P)
    if rr["rank"] < k:
        return raw
    dp = rr["piv"]

    def make(i):
        def fn(rp, p, zvals=None):
            R = rows_fn(rp, p, zvals)
            if R is None:
                return None
            Binv = matinv_modp(R[:, dp], p)
            if Binv is None:
                return None
            w = np.zeros(nz, dtype=np.int64)
            for l in range(k):
                w = (w + mulmod(R[l], Binv[i, l], p)) % p
            return w
        return fn
    return {"anchors": [int(x) for x in dp], "residue_fns": [make(i) for i in range(k)]}


def canon_gauge(residual_free, scal_rows, P, nz, sc):
    """Residual directions reduced modulo the scalings, 1 on own pivot, 0 on the others'."""
    k = len(residual_free)
    if k <= 1:
        return {"anchors": list(residual_free), "residue_fns": [None] * k}
    reduce_scal = reduce_mod_rows(scal_rows)

    def resid(rp, p, zvals=None):
        return reduce_scal(np.stack([null_residues(rp, fc, p) for fc in residual_free]), p)
    return gauge_from_rows(residual_free, resid, resid(sc["ref"], P), P, k, nz)


def logcoord_gauge(residual_free, scal_rows, P, nz, sc, zvals0):
    """canon_gauge in log coordinates: eta_i = xi_i / z_i."""
    k = len(residual_free)
    if k == 0:
        return {"anchors": [], "residue_fns": []}
    scal_rows = np.asarray(scal_rows, dtype=np.int64).reshape(-1, nz)
    W = np.zeros((scal_rows.shape[0], nz), dtype=np.int64)
    for c in range(nz):
        zc = int(zvals0[c]) % P
        if zc == 0:
            continue
        wc = mulmod(scal_rows[:, c], invmod(zc, P), P)
        W[:, c] = np.where(wc > P / 2, wc - P, wc)
    reduce_scal = reduce_mod_rows(W)

    def eta_rows(rp, p, zvals):
        R = np.stack([null_residues(rp, fc, p) for fc in residual_free])
        for c in range(nz):
            zc = int(zvals[c]) % p
            if zc == 0:
                if np.any(R[:, c] % p):
                    return None
                continue
            R[:, c] = mulmod(R[:, c], invmod(zc, p), p)
        return reduce_scal(R, p)
    return gauge_from_rows(residual_free, eta_rows, eta_rows(sc["ref"], P, zvals0), P, k, nz)


def minsupport_gauge(residual_free, scal_rows, P, nz, sc, free_cols, support_cap=6,
                     cand_cap=20000, max_secs=20):
    """Residual directions by minimal support over small column subsets."""
    k = len(residual_free)
    raw = {"anchors": list(residual_free), "residue_fns": [None] * k}
    if k == 0:
        return raw
    deadline = time.monotonic() + max_secs
    scal_rows = np.asarray(scal_rows, dtype=np.int64).reshape(-1, nz)
    free_cols = list(free_cols)

    def basis_rows(rp, p):
        return np.stack([null_residues(rp, fc, p) for fc in free_cols])

    def row_comb(c, B, p):
        w = np.zeros(nz, dtype=np.int64)
        for l in range(len(c)):
            if c[l] % p:
                w = (w + mulmod(B[l], c[l], p)) % p
        return w

    def left_null(B, keep, p):
        nr = B.shape[0]
        M = B[:, keep]
        rr = rref_modp(np.concatenate([M % p, np.eye(nr, dtype=np.int64)], axis=1), p)
        z = np.nonzero(np.sum(rr["R"][:, :M.shape[1]] % p != 0, axis=1) == 0)[0]
        if not len(z):
            return None
        return rr["R"][z[0], M.shape[1]:] % p

    def supported(B, S, p):
        c = left_null(B, sorted(set(range(nz)) - set(S)), p)
        if c is None:
            return None
        return row_comb(c, B, p)

    def is_scaling(v):
        return in_span(scal_rows.T, v, nz, P)

    B0 = basis_rows(sc["ref"], P)
    cols = sorted({int(c) for fc in residual_free
                   for c in np.nonzero(B0[free_cols.index(fc)] % P)[0]})
    if len(cols) < 2:
        return raw
    n_cols = len(cols)
    budget = int(cand_cap)

    def next_combo(idx, n, s):
        idx = list(idx)
        i = s - 1
        while i >= 0 and idx[i] == n - s + i:
            i -= 1
        if i < 0:
            return None
        idx[i] += 1
        for j in range(i + 1, s):
            idx[j] = idx[j - 1] + 1
        return idx

    found = []
    sel = np.zeros((nz, 0), dtype=np.int64)
    it = 0
    for s in range(2, min(n_cols, support_cap) + 1):
        if len(found) >= k or budget <= 0:
            break
        idx = list(range(s))
        while True:
            if len(found) >= k or budget <= 0:
                break
            it += 1
            if it % 256 == 0 and time.monotonic() > deadline:
                budget = 0
                break
            S = [cols[i] for i in idx]
            idx = next_combo(idx, n_cols, s)
            budget -= 1
            spanned = any(all(c in S for c in fc["supp"]) for fc in found)
            if not spanned:
                v = supported(B0, S, P)
                if v is not None and not is_scaling(v):
                    base = np.concatenate([scal_rows.T, sel], axis=1)
                    if not in_span(base, v, nz, P):
                        supp = [int(c) for c in np.nonzero(v % P)[0]]
                        found.append({"supp": supp, "anchor": supp[0], "S": S})
                        sel = np.concatenate([sel, v.reshape(nz, 1)], axis=1)
            if idx is None:
                break
    if not found:
        return raw

    def make(fc):
        S, anchor = fc["S"], fc["anchor"]

        def fn(rp, p, zvals=None):
            v = supported(basis_rows(rp, p), S, p)
            if v is None or v[anchor] % p == 0:
                return None
            v = mulmod(v, invmod(v[anchor] % p, p), p)
            if zvals is None:
                return v
            za = int(zvals[anchor]) % p
            if za == 0:
                return None
            eta = np.zeros(nz, dtype=np.int64)
            for c in S:
                zc = int(zvals[c]) % p
                if zc == 0:
                    if v[c] % p:
                        return None
                    continue
                eta[c] = int(mulmod(v[c], mulmod(za, invmod(zc, p), p), p))
            return eta
        fn.pinned_support = True
        return fn
    return {"anchors": [fc["anchor"] for fc in found], "residue_fns": [make(fc) for fc in found],
            "vectors": sel}


def bigmod(s, p):
    """An integer given as a decimal string, modulo p."""
    return int(str(s)) % p


def sqrt_mod(a, p):
    """The smaller square root of a modulo the odd prime p, as the kernel takes it; None
    for a non-residue."""
    from sympy.ntheory import sqrt_mod as _sqrt
    r = _sqrt(int(a) % p, p)
    return None if r is None else min(r, p - r)


def tape_fn(tp, n_leaves):
    """Evaluator of a straight-line tape (0 const, 1 add, 2 mul, 3 inv, 4 sqrt, 5 odd root
    of degree b) modulo p."""
    op = [int(x) for x in np.atleast_1d(tp["op"])]
    a = [int(x) for x in np.atleast_1d(tp["a"])]
    b = [int(x) for x in np.atleast_1d(tp["b"])]
    cnum = [str(x) for x in _strs(tp["cnum"])]
    cden = [str(x) for x in _strs(tp["cden"])]
    out = [int(x) for x in np.atleast_1d(tp["out"])]
    cache = {}

    def fn(point, p):
        cv = cache.get(p)
        if cv is None:
            cv = {}
            for i, o in enumerate(op):
                if o == 0:
                    d = int(cden[i]) % p
                    cv[i] = None if d == 0 else (int(cnum[i]) % p) * pow(d, p - 2, p) % p
            cache[p] = cv
        if any(v is None for v in cv.values()):
            return None
        v = [0] * (n_leaves + len(op))
        for L in range(n_leaves):
            v[L] = int(point[L]) % p
        for i, o in enumerate(op):
            s = n_leaves + i
            if o == 0:
                v[s] = cv[i]
            elif o == 1:
                v[s] = (v[a[i]] + v[b[i]]) % p
            elif o == 2:
                v[s] = v[a[i]] * v[b[i]] % p
            elif o == 4:
                r = sqrt_mod(v[a[i]], p)
                if not r:
                    return None
                v[s] = r
            elif o == 5:
                if math.gcd(b[i], p - 1) != 1 or v[a[i]] == 0:
                    return None
                v[s] = pow(v[a[i]], pow(b[i], -1, p - 1), p)
            else:
                if v[a[i]] == 0:
                    return None
                v[s] = pow(v[a[i]], p - 2, p)
        return np.array([v[o] for o in out], dtype=np.int64)
    return fn


def _strs(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return list(x)


def exp_rows(rel, rel_fn, znames, point, p):
    """Relation rows e_W - W log(r) grad(tau/L) of the exponential leaves at a point."""
    vals = rel_fn(point, p)
    if vals is None:
        return None
    W = [str(w) for w in _strs(rel["W"])]
    Ws = list(dict.fromkeys(W))
    zi = {z: i for i, z in enumerate(znames)}
    M = np.zeros((len(Ws), len(znames)), dtype=np.int64)
    for r, w in enumerate(Ws):
        M[r, zi[w]] = 1
    zc = [zi.get(str(z)) for z in _strs(rel["z"])]
    for k, c in enumerate(zc):
        if c is None:
            continue
        r = Ws.index(W[k])
        M[r, c] = (M[r, c] - int(vals[k])) % p
    return M


def mono_table(nvar, degree):
    """Exponent vectors of nvar variables with total degree <= degree, by degree."""
    if nvar == 0:
        return np.zeros((1, 0), dtype=np.int64)

    def gen(n, d):
        if n == 1:
            return [[e] for e in range(d + 1)]
        out = []
        for e in range(d + 1):
            for s in gen(n - 1, d - e):
                out.append([e] + s)
        return out
    rows = gen(nvar, degree)
    rows.sort(key=lambda r: sum(r))
    return np.array(rows, dtype=np.int64)


def rat_recon(residues, primes):
    """Per-row CRT over `primes` and rational reconstruction as {'num', 'den'} strings,
    den '0' where no bounded rational exists; in arbitrary precision where the kernel
    has no 128-bit integers."""
    if getattr(_core, "has_int128", True):
        return _core.sym_rat_recon(residues, primes)
    from .sym.linalg import sym_rat_recon_big
    return sym_rat_recon_big(residues, primes)
