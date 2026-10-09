"""Verification of reconstructed directions at fresh points and primes."""

import numpy as np

from .. import _core
from ..gfp import mulmod, null_residues, verify_prime
from .closedform import eval_modq
from .forward import _free_cols


def verify_direction(entry, f, znames, leaf_names, point0, nt_used, kcall, residue_fn=None):
    """Reconstructed direction against the null vector at the base point modulo the verification prime."""
    if entry.get("vector") is None:
        return True
    q = verify_prime()
    rq = kcall(point0, q, int(nt_used))
    if not rq.get("ok"):
        return False
    aq = null_residues(rq, f, q) if residue_fn is None else residue_fn(rq, q, None)
    if aq is None:
        return False
    env = {nm: int(point0[i]) for i, nm in enumerate(leaf_names)}
    for nm, ex in entry["vector"].items():
        if nm not in znames:
            continue
        col = znames.index(nm)
        got = eval_modq(ex, env, q)
        if got is None:
            continue
        if got % q != int(aq[col]) % q:
            return False
    return True


def verify_in_nullspace(entry, f, znames, leaf_names, point0, nt_used, kcall, pool, pool_next, nz,
                        dir_call=None, tries=4):
    """Gauge-independent check that a direction lies in the nullspace at a fresh point.

    The components on the free columns fix the kernel vector, so they must evaluate; the
    others are compared where they evaluate (resting-state coordinates are not leaves).
    """
    if entry.get("vector") is None:
        return True
    q = verify_prime()
    for k in range(tries):
        pt = pool(pool_next + k * (len(point0) + 1) + np.arange(len(point0)))
        rq = kcall(pt, q, int(nt_used))
        if not rq.get("ok"):
            continue
        env = {nm: int(pt[i]) for i, nm in enumerate(leaf_names)}
        v = np.zeros(nz, dtype=np.int64)
        v[f] = 1
        known = np.ones(nz, dtype=bool)
        for nm, ex in entry["vector"].items():
            if nm not in znames:
                continue
            col = znames.index(nm)
            got = eval_modq(ex, env, q)
            if got is None:
                known[col] = False
            else:
                v[col] = got % q
        free = _free_cols(rq, nz)
        if not free:
            return False
        if dir_call is not None and known.all():
            ok = dir_call(pt, q, int(nt_used), v)
            if ok is not None:
                return ok
        basis = np.stack([null_residues(rq, fcol, q) for fcol in free]) % q
        if known.all() or all(known[c] for c in free):
            recon = np.zeros(nz, dtype=np.int64)
            for j, fcol in enumerate(free):
                if v[fcol] % q:
                    recon = (recon + mulmod(basis[j], int(v[fcol]) % q, q)) % q
            return bool(np.all(((recon - v) % q == 0)[known]))
        # the free columns do not all evaluate: membership of the known part in the kernel
        return _core.sym_solve_mod(basis[:, known].T, v[known] % q, q) is not None
    return False


def verify_perprime(entry, f, znames, leaf_names, point0, nt_used, kcall, pivots, pool, pool_next,
                    nz, rel_leaves=None, tries=80):
    """Strict check for the coupled steady-state path: fresh points with the same pivots."""
    if entry.get("vector") is None:
        return False
    q = verify_prime()
    pn = pool_next
    pert_idx = list(range(len(point0))) if rel_leaves is None else list(rel_leaves)
    piv0 = [int(x) for x in np.atleast_1d(pivots)]
    for _ in range(tries):
        pt = np.array(point0, dtype=np.int64)
        if pert_idx:
            pt[pert_idx] = pool(pn + np.arange(len(pert_idx)))
        pn += len(pert_idx) + 1
        rq = kcall(pt, q, int(nt_used))
        if not pert_idx and not rq.get("ok"):
            return False
        if not rq.get("ok") or [int(x) for x in np.atleast_1d(rq["pivots"])] != piv0:
            continue
        env = {nm: int(pt[i]) for i, nm in enumerate(leaf_names)}
        v = np.zeros(nz, dtype=np.int64)
        v[f] = 1
        bad = False
        for nm, ex in entry["vector"].items():
            if nm not in znames:
                continue
            got = eval_modq(ex, env, q)
            if got is None:
                bad = True
                break
            v[znames.index(nm)] = got % q
        if bad:
            continue
        recon = np.zeros(nz, dtype=np.int64)
        for fcol in _free_cols(rq, nz):
            vf = int(v[fcol]) % q
            if vf:
                recon = (recon + mulmod(null_residues(rq, fcol, q), vf, q)) % q
        return bool(np.all((recon - v) % q == 0))
    return False


def in_kernel(rq, v, known, nz, q):
    """Whether a kernel vector of `rq` modulo `q` agrees with `v` on the columns `known`."""
    free = _free_cols(rq, nz)
    if not free:
        return not np.any(np.asarray(v)[known] % q)
    basis = np.stack([null_residues(rq, fcol, q) for fcol in free]) % q
    return _core.sym_solve_mod(basis[:, known].T, np.asarray(v)[known] % q, q) is not None
