"""Truncated power series over GF(p): nullspaces, division and Pade approximants."""

import numpy as np

from .. import _core
from ..gfp import invmod, mulmod


def series_null(S, nz, N, p):
    """Series nullspace of stacked series rows over GF(p)[t]/t^N, one vector per free column."""
    S = np.asarray(S, dtype=np.int64)

    def Mk(k):
        return S[:, np.arange(nz) * N + k]
    M0 = Mk(0)
    rr = _core.sym_rref_mod(M0.astype(np.float64), p)
    piv = [int(x) for x in np.atleast_1d(rr["piv"])]
    R = np.asarray(rr["R"], dtype=np.int64).reshape(len(piv), nz) if piv else np.zeros((0, nz), dtype=np.int64)
    free = [c for c in range(nz) if c not in set(piv)]
    Ms = [Mk(k) for k in range(1, N)]
    A = M0
    out = []
    for fc in free:
        V = np.zeros((nz, N), dtype=np.int64)
        V[fc, 0] = 1
        for i, pc in enumerate(piv):
            V[pc, 0] = (p - R[i, fc]) % p
        okv = True
        for k in range(1, N):
            b = np.zeros(M0.shape[0], dtype=np.int64)
            for j in range(1, k + 1):
                b = (b + mat_vec_mod(Ms[j - 1], V[:, k - j], p)) % p
            x = _core.sym_solve_mod(A, (p - b) % p, p)
            if x is None:
                okv = False
                break
            V[:, k] = np.asarray(x, dtype=np.int64) % p
        out.append(V if okv else None)
    return {"free": free, "piv": piv, "V": out}


def mat_vec_mod(M, v, p):
    """The product `M @ v` mod `p`."""
    out = np.zeros(M.shape[0], dtype=np.int64)
    for c in np.nonzero(v)[0]:
        out = (out + mulmod(M[:, c], v[c], p)) % p
    return out


def pade_mod(s, dn, dd, p):
    """Pade approximant P/Q of the series s with Q(0) = 1; None if singular or inconsistent."""
    s = [int(x) for x in s]
    N = len(s)
    if dn + dd + 1 > N:
        return None
    q = [1]
    if dd > 0:
        A = np.zeros((dd, dd), dtype=np.int64)
        b = np.zeros(dd, dtype=np.int64)
        for r in range(1, dd + 1):
            i = dn + r
            for j in range(1, dd + 1):
                A[r - 1, j - 1] = s[i - j] if i - j >= 0 else 0
            b[r - 1] = (p - s[i]) % p
        qq = _core.sym_solve_mod(A, b, p)
        if qq is None:
            return None
        q = [1] + [int(x) % p for x in np.atleast_1d(qq)]

    def conv(i):
        acc = 0
        for j in range(len(q)):
            if i - j >= 0:
                acc = (acc + q[j] * s[i - j]) % p
        return acc
    P = [conv(i) for i in range(dn + 1)]
    for i in range(dn + 1, N):
        if conv(i) != 0:
            return None
    return {"P": P, "Q": q}


def ser_div(a, b, p):
    """Quotient of the truncated power series `a` by `b` mod `p`."""
    N = len(a)
    q = [0] * N
    ib = invmod(b[0], p)
    for k in range(N):
        acc = int(a[k])
        for j in range(1, k + 1):
            acc = (acc - int(b[j]) * q[k - j]) % p
        q[k] = acc % p * ib % p
    return q


def pade_at(pq, t, p):
    """The Pade approximant `pq` at `t` mod `p`, or None at a pole."""
    def ev(cf):
        acc = 0
        for c in reversed(cf):
            acc = (acc * t + c) % p
        return acc
    qv = ev(pq["Q"])
    if qv == 0:
        return None
    return ev(pq["P"]) * invmod(qv, p) % p
