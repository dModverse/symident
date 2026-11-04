"""Verification of reconstructed directions at fresh points and primes."""

import re

import numpy as np

from .. import _core, session
from ..gfp import mulmod, null_residues, uniform_point, verify_prime
from .closedform import eval_modq
from .forward import _free_cols

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def verify_direction(entry, f, znames, leaf_names, point0, nt_used, kcall, residue_fn=None, tries=4):
    """Whether the direction matches the gauge null vector modulo the verification prime at a fresh seeded point."""
    if entry.get("vector") is None:
        return True
    q = verify_prime()
    leaves = set(leaf_names)
    for k in range(tries):
        pt = uniform_point(len(point0), q, session.current().seed("verify direction", f, k))
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
            on_leaves = set(_IDENT.findall(str(ex))) <= leaves
            got = eval_modq(ex, env, q) if on_leaves else None
            if got is None:
                if on_leaves:
                    return False
                continue
            if got % q != int(aq[znames.index(nm)]) % q:
                return False
        return True
    return False


def verify_in_nullspace(entry, f, znames, leaf_names, point0, nt_used, kcall, pool, pool_next, nz,
                        dir_call=None, tries=6, need=2, cols=None):
    """Gauge-independent check that a direction lies in the nullspace at `need` of `tries` fresh seeded points."""
    if entry.get("vector") is None:
        return True
    q = verify_prime()
    passed = 0
    for k in range(tries):
        pt = uniform_point(len(point0), q, session.current().seed("verify kernel", f, pool_next, k))
        ok_ = _in_nullspace_at(entry, f, znames, leaf_names, pt, nt_used, kcall, nz, dir_call, q, cols)
        if ok_ is False:
            return False
        # the error bound does not cover a check on part of the stated entries
        if ok_ == "partial":
            entry["partial_check"] = True
        passed += ok_ in (True, "partial")
        if passed >= need:
            session.note("draws", k + 1)
            return True
    return False


def _in_nullspace_at(entry, f, znames, leaf_names, pt, nt_used, kcall, nz, dir_call, q, cols=None):
    """The check of verify_in_nullspace() at one point: True, False, "partial" when only the entries it can
    evaluate pass, or None where the kernel does not exist."""
    rq = kcall(pt, q, int(nt_used))
    if not rq.get("ok"):
        return None
    env = {nm: int(pt[i]) for i, nm in enumerate(leaf_names)}
    v = np.zeros(nz, dtype=np.int64)
    v[f] = 1
    known = np.ones(nz, dtype=bool)
    if cols is not None:
        # the entry states the columns `cols`; off them a kernel vector may take any value
        known[:] = False
        known[list(cols) + [f]] = True
    stated = known.copy()
    for nm, ex in entry["vector"].items():
        if nm not in znames:
            continue
        col = znames.index(nm)
        got = eval_modq(ex, env, q) if set(_IDENT.findall(str(ex))) <= set(env) else None
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
        ok = bool(np.all(((recon - v) % q == 0)[known]))
        return "partial" if ok and not known[stated].all() else ok
    # the free columns do not all evaluate: membership of the known part in the kernel
    if _core.sym_solve_mod(basis[:, known].T, v[known] % q, q) is None:
        return False
    return "partial" if not known[stated].all() else True


def in_kernel(rq, v, known, nz, q):
    """Whether a kernel vector of `rq` modulo `q` agrees with `v` on the columns `known`."""
    free = _free_cols(rq, nz)
    if not free:
        return not np.any(np.asarray(v)[known] % q)
    basis = np.stack([null_residues(rq, fcol, q) for fcol in free]) % q
    return _core.sym_solve_mod(basis[:, known].T, np.asarray(v)[known] % q, q) is not None
