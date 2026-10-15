"""Verification of reconstructed directions at fresh points and primes."""

import re

import numpy as np

from .. import _core, session
from ..gfp import mulmod, null_residues, random_point, verify_prime
from .closedform import eval_modq
from .forward import _free_cols

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def verify_direction(entry, f, znames, leaf_names, point0, nt_used, kcall, residue_fn=None, tries=4):
    """Reconstructed direction against the null vector of the gauge modulo the verification
    prime, at the first of `tries` fresh seeded points where the kernel exists. A component
    over the leaves that does not evaluate there fails the check; components that name
    other coordinates, such as resting states, are compared where they evaluate."""
    if entry.get("vector") is None:
        return True
    q = verify_prime()
    leaves = set(leaf_names)
    for k in range(tries):
        pt = random_point(len(point0), session.current().seed("verify direction", f, k))
        rq = kcall(pt, q, int(nt_used))
        if not rq.get("ok"):
            continue
        aq = null_residues(rq, f, q) if residue_fn is None else residue_fn(rq, q, None)
        if aq is None:
            continue
        env = {nm: int(pt[i]) for i, nm in enumerate(leaf_names)}
        for nm, ex in entry["vector"].items():
            if nm not in znames:
                continue
            got = eval_modq(ex, env, q)
            if got is None:
                if set(_IDENT.findall(str(ex))) <= leaves:
                    return False
                continue
            if got % q != int(aq[znames.index(nm)]) % q:
                return False
        return True
    return False


def verify_in_nullspace(entry, f, znames, leaf_names, point0, nt_used, kcall, pool, pool_next, nz,
                        dir_call=None, tries=6, need=2):
    """Gauge-independent check that a direction lies in the nullspace at `need` fresh points
    drawn uniformly from the seed, out of `tries`.

    The components on the free columns fix the kernel vector, so they must evaluate; the
    others are compared where they evaluate (resting-state coordinates are not leaves).
    """
    if entry.get("vector") is None:
        return True
    q = verify_prime()
    passed = 0
    for k in range(tries):
        if passed >= need:
            return True
        pt = random_point(len(point0), session.current().seed("verify kernel", f, pool_next, k))
        ok_ = _in_nullspace_at(entry, f, znames, leaf_names, pt, nt_used, kcall, nz, dir_call, q)
        if ok_ is False:
            return False
        passed += ok_ is True
    return passed >= need


def _in_nullspace_at(entry, f, znames, leaf_names, pt, nt_used, kcall, nz, dir_call, q):
    """The check of verify_in_nullspace() at one point: True, False, or None where the kernel
    does not exist there."""
    rq = kcall(pt, q, int(nt_used))
    if not rq.get("ok"):
        return None
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
            return bool(ok)
    basis = np.stack([null_residues(rq, fcol, q) for fcol in free]) % q
    if known.all() or all(known[c] for c in free):
        recon = np.zeros(nz, dtype=np.int64)
        for j, fcol in enumerate(free):
            if v[fcol] % q:
                recon = (recon + mulmod(basis[j], int(v[fcol]) % q, q)) % q
        return bool(np.all(((recon - v) % q == 0)[known]))
    # the free columns do not all evaluate: membership of the known part in the kernel
    return _core.sym_solve_mod(basis[:, known].T, v[known] % q, q) is not None


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
