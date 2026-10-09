"""Leaves that move each kernel entry, and a sample bank shared by the dense fits."""

import math

import numpy as np

from .. import state
from ..gfp import PRIMES, null_residues
from .closedform import expired
from .sparse import _ints, _pivots_eq


def _optimistic(residue_fn):
    return getattr(residue_fn, "pinned_support", False) or getattr(residue_fn, "narrow", False)


def direction_relevance(f, ref, pivots, z_slots, n_leaves, point0, rel_probe, residue_fn):
    """Base nullspace vector, its support columns and the leaves moving each entry."""
    P = PRIMES[0]
    zvals0 = None if z_slots is None else np.asarray(point0)[z_slots]

    def nv(rp, p, zvals=zvals0):
        return null_residues(rp, f, p) if residue_fn is None else residue_fn(rp, p, zvals)
    optimistic = _optimistic(residue_fn)
    base_nv = nv(ref, P)
    if base_nv is None:
        base_nv = null_residues(ref, f, P)
    support_cols = [int(c) for c in np.nonzero(base_nv)[0] if int(c) != f]
    relevant = []
    rel_by_entry = [[] for _ in support_cols]
    for li in range(n_leaves):
        pr = rel_probe[li] if li < len(rel_probe) else None
        rp = pr["rp"] if pr is not None else None
        zvp = pr["zvals"] if pr is not None else None
        nvp = nv(rp, P, zvp) if _pivots_eq(rp, pivots) else None
        if nvp is None:
            if optimistic:
                continue
            relevant.append(li)
            for r in rel_by_entry:
                r.append(li)
            continue
        if np.any(nvp != base_nv):
            relevant.append(li)
        for i, c in enumerate(support_cols):
            if nvp[c] != base_nv[c]:
                rel_by_entry[i].append(li)
    return {"base_nv": base_nv, "support_cols": support_cols, "relevant": relevant,
            "rel_by_entry": rel_by_entry}


def dense_need(rel_by_entry, ctrl):
    """Sample count the dense interpolation needs, and whether any entry is dense."""
    def is_dense(r):
        return 1 <= len(r) <= ctrl["relevance_cap"]
    dense_rel = max([0] + [len(r) for r in rel_by_entry if is_dense(r)])
    any_dense = any(is_dense(r) for r in rel_by_entry)
    max_need = 2 * math.comb(dense_rel + ctrl["degree_cap"], dense_rel) - 1 + ctrl["sample_slack"] \
        if any_dense else 0
    return {"max_need": int(max_need), "any_dense": any_dense}


def build_shared_bank(union, need_pts, point0, pool, pool_next, kbatch, pivots, nt_used, n_p, ctrl=None):
    """Pivot-consistent points over the union of relevant leaves, kernels per point and prime."""
    nu = len(union)
    U = np.zeros((0, nu), dtype=np.int64)
    points, rps = [], []
    tries = 0
    bd = state.env("BANKDIAG")
    n_solve_fail = n_piv_mismatch = n_good = 0
    while len(points) < need_pts and tries < 30 * need_pts:
        if not points and tries >= min(2 * need_pts, 48):
            break
        if ctrl is not None and expired(ctrl):
            return {"ok": False, "pool_next": pool_next}
        chunk = max(1, min(need_pts - len(points) + 5, 30 * need_pts - tries))
        if not points:
            chunk = min(chunk, 24)
        if ctrl is not None and ctrl.get("deadline") is not None:
            chunk = min(chunk, 96)
        tries += chunk
        uv = []
        for _ in range(chunk):
            uv.append(pool(pool_next + np.arange(nu)))
            pool_next += nu
        cand = []
        for u in uv:
            pt = np.array(point0, dtype=np.int64)
            pt[union] = u
            cand.append(pt)
        res = kbatch([c for c in cand for _ in range(n_p)], list(PRIMES[:n_p]) * chunk, nt_used)
        for ci in range(chunk):
            if len(points) >= need_pts:
                break
            good = True
            rpc = []
            for j in range(n_p):
                rp = res[ci * n_p + j]
                if rp is None or not rp.get("ok"):
                    good = False
                    n_solve_fail += 1
                    break
                if _ints(rp["pivots"]) != _ints(pivots):
                    good = False
                    n_piv_mismatch += 1
                    break
                rpc.append(rp)
            if not good:
                continue
            n_good += 1
            U = np.vstack([U, np.asarray(uv[ci]).reshape(1, -1)])
            points.append(cand[ci])
            rps.append(rpc)
    if bd:
        print(f"[bankdiag] union={nu} need_pts={need_pts} tries={tries} good={n_good} "
              f"solve_fail={n_solve_fail} piv_mismatch={n_piv_mismatch}")
    if len(points) < need_pts:
        return {"ok": False, "pool_next": pool_next}
    return {"ok": True, "U": U, "points": points, "rps": rps, "union": list(union),
            "pool_next": pool_next}
