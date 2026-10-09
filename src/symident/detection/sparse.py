"""Sparse interpolation of kernel entries: Laurent monomials, Ben-Or-Tiwari and general rational fits."""

import math

import numpy as np

from .. import _core
from ..gfp import PRIMES, mono_table, null_residues, rat_recon, sieve
from .closedform import expired, mono_string, poly_string


def _ints(x):
    return [int(v) for v in np.atleast_1d(x)] if x is not None else []


def _pivots_eq(rp, pivots):
    return rp is not None and bool(rp.get("ok")) and _ints(rp["pivots"]) == _ints(pivots)


def laurent_candidates(nvar, d_num, d_den):
    """Distinct exponent vectors of a monomial of degree <= `d_num` over one of degree <= `d_den`."""
    num = mono_table(nvar, d_num)
    den = mono_table(nvar, d_den)
    cand = np.vstack([num - den[d] for d in range(den.shape[0])])
    _, idx = np.unique(cand, axis=0, return_index=True)
    return cand[np.sort(idx)]


def bot_reconcile(per_prime):
    """Rational coefficients of a sparse polynomial from its fits per prime, or None when the supports differ."""
    np_ = len(per_prime)

    def key(m):
        return [",".join(str(int(x)) for x in row) for row in np.asarray(m).reshape(len(m), -1)]
    k1 = key(per_prime[0]["exps"])
    t = len(k1)
    coef = np.zeros((t, np_), dtype=np.int64)
    coef[:, 0] = _ints(per_prime[0]["coeffs"])
    for pj in range(1, np_):
        kp = key(per_prime[pj]["exps"])
        if len(kp) != t or set(kp) != set(k1):
            return None
        cp = _ints(per_prime[pj]["coeffs"])
        coef[:, pj] = [cp[kp.index(k)] for k in k1]
    rec = rat_recon(coef, list(PRIMES[:np_]))
    if any(d == "0" for d in rec["den"]):
        return None
    return {"exps": np.asarray(per_prime[0]["exps"], dtype=np.int64), "num": rec["num"],
            "den": rec["den"]}


def laurent_assemble(per_prime, reli, leaf_names, nvar):
    """The Laurent polynomial from the fits per prime as a string over the leaves `reli`, or None."""
    rc = bot_reconcile(per_prime)
    if rc is None:
        return None
    exp_rows = rc["exps"].reshape(-1, nvar)
    mu = np.array([max(0, -int(exp_rows[:, c].min())) for c in range(nvar)], dtype=np.int64)
    vars_ = [leaf_names[r] for r in reli]
    num_str = poly_string(rc["num"], rc["den"], exp_rows + mu, vars_)
    den_str = mono_string(mu, vars_)
    return num_str if den_str == "1" else f"({num_str})/({den_str})"


def _default_kbatch(kcall):
    def kb(pl, pv, Nt):
        return [kcall(pp, pr, Nt) for pp, pr in zip(pl, pv)]
    return kb


def sparse_entry(reli, support_col, f, point0, leaf_names, nt_used, kcall, pivots, residue_fn=None,
                 ctrl=None, z_slots=None, kbatch=None):
    """One wide entry by sparse Laurent interpolation; None if not a bounded Laurent polynomial."""
    if kbatch is None:
        kbatch = _default_kbatch(kcall)
    nvar = len(reli)
    bases = sieve(nvar)
    np_ = len(PRIMES)
    max_len = 2 * ctrl["term_cap"] + 2
    seqs = [[] for _ in range(np_)]
    cur = [[1] * nvar for _ in range(np_)]
    have = 0
    while True:
        if expired(ctrl):
            return None
        target = min(16 if have == 0 else 2 * have, max_len)
        pl, pv, slot = [], [], []
        for pj in range(np_):
            p = PRIMES[pj]
            for k in range(have, target):
                pt = np.array(point0, dtype=np.int64)
                pt[reli] = cur[pj]
                pl.append(pt)
                pv.append(p)
                slot.append((pj, k))
                cur[pj] = [c * b % p for c, b in zip(cur[pj], bases)]
        if expired(ctrl):
            return None
        res = kbatch(pl, pv, nt_used)
        for i, pt in enumerate(pl):
            rp, p = res[i], pv[i]
            if not _pivots_eq(rp, pivots):
                return None
            zv = None if z_slots is None else pt[z_slots]
            nvp = null_residues(rp, f, p) if residue_fn is None else residue_fn(rp, p, zv)
            if nvp is None:
                return None
            pj, k = slot[i]
            while len(seqs[pj]) <= k:
                seqs[pj].append(0)
            seqs[pj][k] = int(nvp[support_col])
        have = target
        if 2 * int(_core.sym_b_morder(seqs[0], PRIMES[0])) < have:
            break
        if have >= max_len:
            return None
    for d_den in range(ctrl["laurent_deg_den"] + 1):
        for d_num in range(1, ctrl["laurent_deg_num"] + 1):
            if math.comb(nvar + d_num, nvar) * math.comb(nvar + d_den, nvar) > ctrl["laurent_cand_cap"]:
                continue
            cand_m = laurent_candidates(nvar, d_num, d_den)
            per_prime = []
            ok_all = True
            for pj in range(np_):
                mono_res = _core.sym_mono_residues(cand_m, bases, PRIMES[pj])
                r = _core.sym_sparse_poly(seqs[pj], cand_m, mono_res, PRIMES[pj])
                if r.get("status") != "ok" or int(r["nterms"]) == 0:
                    ok_all = False
                    break
                per_prime.append(r)
            if not ok_all:
                continue
            out = laurent_assemble(per_prime, reli, leaf_names, nvar)
            if out is not None:
                return out
    return None


def general_rational_entry(reli, support_col, f, point0, leaf_names, nt_used, kcall, pivots,
                           residue_fn=None, ctrl=None, z_slots=None, kbatch=None):
    """One wide entry with a general denominator by Cauchy fits along rays and Ben-Or/Tiwari."""
    if kbatch is None:
        kbatch = _default_kbatch(kcall)
    nvar = len(reli)
    np_ = len(PRIMES)
    bases = sieve(nvar)
    s = np.asarray(point0, dtype=np.int64)[reli]

    def sample_many(rv_list, p_vec):
        pl = []
        for i, rv in enumerate(rv_list):
            pt = np.array(point0, dtype=np.int64)
            pt[reli] = np.asarray(rv, dtype=np.int64) % p_vec[i]
            pl.append(pt)
        res = kbatch(pl, p_vec, nt_used)
        out = []
        for i, pt in enumerate(pl):
            rp, p = res[i], p_vec[i]
            if not _pivots_eq(rp, pivots):
                out.append(None)
                continue
            zv = None if z_slots is None else pt[z_slots]
            nvp = null_residues(rp, f, p) if residue_fn is None else residue_fn(rp, p, zv)
            out.append(None if nvp is None else int(nvp[support_col]))
        return out

    def ray_samples(u_list, p_vec, nt):
        rv, pv = [], []
        for i, u in enumerate(u_list):
            v = (np.asarray(u, dtype=np.int64) - s) % p_vec[i]
            for tj in range(nt + 1):
                rv.append((s + tj * v) % p_vec[i])
                pv.append(p_vec[i])
        out = sample_many(rv, pv)
        return [out[i * (nt + 1):(i + 1) * (nt + 1)] for i in range(len(u_list))]

    def from_ray(rv, p, d_n, d_d):
        tn = list(range(d_n + d_d + 5))
        rv = rv[:len(tn)]
        if any(x is None for x in rv):
            return None
        res = _core.sym_cauchy_eval(tn, rv, d_n, d_d, p)
        if res.get("status") != "ok":
            return None
        return (int(res["N"]), int(res["D"]))
    P1 = PRIMES[0]
    u0 = [(b * b + 3) % P1 for b in bases]
    deg = None
    ray0 = ray_samples([u0], [P1], ctrl["general_deg_num"] + ctrl["general_deg_den"] + 4)[0]
    for tot in range(2, ctrl["general_deg_num"] + ctrl["general_deg_den"] + 1):
        if expired(ctrl):
            return None
        for a in range(max(1, tot - ctrl["general_deg_den"]), min(tot - 1, ctrl["general_deg_num"]) + 1):
            b_d = tot - a
            if b_d < 1 or b_d > ctrl["general_deg_den"]:
                continue
            if from_ray(ray0, P1, a, b_d) is not None:
                deg = (a, b_d)
                break
        if deg is not None:
            break
    if deg is None:
        return None
    d_n, d_d = deg
    if math.comb(nvar + d_n, nvar) + math.comb(nvar + d_d, nvar) > ctrl["laurent_cand_cap"]:
        return None
    cand_n = mono_table(nvar, d_n)
    cand_d = mono_table(nvar, d_d)
    max_len = 2 * ctrl["term_cap"] + 2
    Nseq = [[] for _ in range(np_)]
    Dseq = [[] for _ in range(np_)]
    cur = [[1] * nvar for _ in range(np_)]
    have = 0
    while True:
        if expired(ctrl):
            return None
        target = min(16 if have == 0 else 2 * have, max_len)
        u_l, p_v, slot = [], [], []
        for pj in range(np_):
            p = PRIMES[pj]
            for k in range(have, target):
                u_l.append(list(cur[pj]))
                p_v.append(p)
                slot.append((pj, k))
                cur[pj] = [c * b % p for c, b in zip(cur[pj], bases)]
        if expired(ctrl):
            return None
        rays = ray_samples(u_l, p_v, d_n + d_d + 4)
        for i in range(len(u_l)):
            e = from_ray(rays[i], p_v[i], d_n, d_d)
            if e is None:
                return None
            pj, k = slot[i]
            while len(Nseq[pj]) <= k:
                Nseq[pj].append(0)
                Dseq[pj].append(0)
            Nseq[pj][k], Dseq[pj][k] = e
        have = target
        if 2 * int(_core.sym_b_morder(Nseq[0], P1)) < have and 2 * int(_core.sym_b_morder(Dseq[0], P1)) < have:
            break
        if have >= max_len:
            return None

    def fit_poly(seq_list, cand_m):
        per_prime = []
        for pj in range(np_):
            mr = _core.sym_mono_residues(cand_m, bases, PRIMES[pj])
            r = _core.sym_sparse_poly(seq_list[pj], cand_m, mr, PRIMES[pj])
            if r.get("status") != "ok" or int(r["nterms"]) == 0:
                return None
            per_prime.append(r)
        return bot_reconcile(per_prime)
    Nrc = fit_poly(Nseq, cand_n)
    if Nrc is None:
        return None
    Drc = fit_poly(Dseq, cand_d)
    if Drc is None:
        return None
    vars_ = [leaf_names[r] for r in reli]
    num_s = poly_string(Nrc["num"], Nrc["den"], Nrc["exps"].reshape(-1, nvar), vars_)
    den_s = poly_string(Drc["num"], Drc["den"], Drc["exps"].reshape(-1, nvar), vars_)
    if den_s == "0":
        return None
    return num_s if den_s == "1" else f"({num_s})/({den_s})"
