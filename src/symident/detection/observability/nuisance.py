"""Nuisance coordinates: kernel results taken modulo the directions that lie within them alone."""

import numpy as np

from ... import _core


def _rref(M, p):
    rr = _core.sym_rref_mod(M.astype(np.float64), p)
    rank = int(rr["rank"])
    R = np.asarray(rr["R"], dtype=np.int64).reshape(rank, M.shape[1])
    return R, (np.atleast_1d(rr["piv"]).astype(np.int64) if rank else np.zeros(0, dtype=np.int64))


def quotient(o, cols, p, drop=()):
    """The kernel result `o` modulo the directions within the columns `cols`, its vectors zero on `drop`."""
    if not cols or o is None or not o.get("ok"):
        return o
    nz, r = int(o["dim"]), int(o["rank"])
    R = np.asarray(o["R"], dtype=np.int64).reshape(r, nz)
    piv = set()
    if r:
        _, pv = _rref(R[:, cols], p)
        piv = {int(x) for x in pv}
    free = [c for j, c in enumerate(cols) if j not in piv]
    E = np.zeros((len(free), nz), dtype=np.int64)
    E[np.arange(len(free)), free] = 1
    R = np.vstack([R, E])
    drop = [c for c in drop if c in set(cols)]
    if drop:
        # the rows that hold no dropped column once those are eliminated first
        rest = [c for c in range(nz) if c not in set(drop)]
        Rd, pv = _rref(R[:, drop + rest], p)
        keep = Rd[pv >= len(drop)][:, len(drop):]
        R = np.zeros((keep.shape[0] + len(drop), nz), dtype=np.int64)
        R[:keep.shape[0], rest] = keep
        R[np.arange(keep.shape[0], R.shape[0]), drop] = 1
    R, pv = _rref(R, p)
    out = dict(o, R=R, pivots=pv, rank=int(R.shape[0]), nuisance_pinned=len(free))
    if o.get("rank_s") is not None and o.get("S") is not None:
        out.update(_series_quotient(o, cols, free, p))
    return out


def _series_rank(S, nz, N, p, cols=()):
    return int(_core.sym_series_rank(S, nz, N, p, list(cols))["rank"])


def _series_quotient(o, cols, free, p):
    """The series rows and rank of `o` with the directions within `cols` pinned by unit rows off a column basis."""
    nz, N = int(o["dim"]), int(o["N"])
    S = np.asarray(o["S"], dtype=np.int64).reshape(-1, nz * N)
    want = int(o["rank_s"]) + len(cols) - (_series_rank(S, nz, N, p, cols) if S.shape[0] else 0)

    def pinned(fr):
        E = np.zeros((len(fr), nz * N), dtype=np.int64)
        E[np.arange(len(fr)), np.asarray(fr, dtype=np.int64) * N] = 1
        return np.vstack([S, E])
    if _series_rank(pinned(free), nz, N, p) != want:
        basis = []
        for c in cols:
            if S.shape[0] and _series_rank(S, nz, N, p, basis + [c]) > len(basis):
                basis.append(c)
        free = [c for c in cols if c not in set(basis)]
    sr = _core.sym_series_rank(pinned(free), nz, N, p, [])
    return {"S": sr["S"], "rank_s": int(sr["rank"])}
