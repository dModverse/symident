"""Pinned monomials and fits per prime for directions under a resting state."""

import numpy as np

from .. import _core
from ..gfp import PRIMES, invmod, mono_table, mulmod, null_residues, rat_recon
from ..rexpr import sym_sort
from .closedform import expired, poly_string, simplify
from .relevance import direction_relevance
from .sparse import _pivots_eq


def pinned_monomials(f, pivots, znames, leaf_names, point0, nt_used, kcall, nv, base_nv,
                     support_cols, rel_by_entry, rel_probe, max_exp=6):
    """Each entry of a support-pinned direction as a Laurent monomial; None otherwise."""
    P = PRIMES[0]

    def find_exp(base, target, p):
        target %= p
        if target == 1:
            return 0
        acc = 1
        for a in range(1, max_exp + 1):
            acc = acc * base % p
            if acc == target:
                return a
        acc, bi = 1, invmod(base, p)
        for a in range(1, max_exp + 1):
            acc = acc * bi % p
            if acc == target:
                return -a
        return None
    entries = {}
    for i, ci in enumerate(support_cols):
        reli = rel_by_entry[i]
        e0 = int(base_nv[ci]) % P
        if e0 == 0:
            return None
        exps = []
        for l in reli:
            pr = rel_probe[l]
            if pr is None or not _pivots_eq(pr["rp"], pivots):
                return None
            ev = nv(pr["rp"], P, pr["zvals"])
            if ev is None:
                return None
            bval = int(point0[l]) % P
            pval = int(pr["pertval"]) % P
            if bval == 0 or pval == 0:
                return None
            a = find_exp(pval * invmod(bval, P) % P, int(ev[ci]) % P * invmod(e0, P) % P, P)
            if a is None:
                return None
            exps.append(a)

        def mono_denom(pj):
            d = 1
            for l, a in zip(reli, exps):
                bv = int(point0[l]) % pj
                d = d * (pow(bv, a, pj) if a >= 0 else pow(invmod(bv, pj), -a, pj)) % pj
            return d
        use_primes = [P]
        resid = [e0 * invmod(mono_denom(P), P) % P]
        for pj in PRIMES[1:]:
            rp = kcall(point0, pj, nt_used)
            if not _pivots_eq(rp, pivots):
                continue
            nvp = nv(rp, pj)
            if nvp is None:
                continue
            use_primes.append(pj)
            resid.append(int(nvp[ci]) % pj * invmod(mono_denom(pj), pj) % pj)
        rec = rat_recon(np.array([resid], dtype=np.int64), use_primes)
        if rec["den"][0] == "0":
            return None
        coef = rec["num"][0] if rec["den"][0] == "1" else f"{rec['num'][0]}/({rec['den'][0]})"
        num, den = [], []
        for l, a in zip(reli, exps):
            if a == 0:
                continue
            v = leaf_names[l]
            term = v if abs(a) == 1 else f"{v}^{abs(a)}"
            (num if a > 0 else den).append(term)
        num_s = "*".join(([coef] if coef != "1" or not num else []) + num)
        expr = num_s if not den else f"({num_s})/({'*'.join(den)})"
        entries[znames[ci]] = simplify(expr)
    entries[znames[f]] = "1"
    return {"support": sym_sort(list(entries)), "vector": entries, "type": "general",
            "closed_form": True}


def _fallback(pool_next, znames, base_nv, reason):
    return {"pool_next": pool_next,
            "entry": {"support": sym_sort([znames[i] for i in np.nonzero(base_nv)[0]]),
                      "type": "general", "closed_form": False, "reason": reason}}


_TIMEOUT = "reconstruction time budget (reconst_control(timeout=)) exceeded"


def _lift_const(vals, prs):
    rec = rat_recon(np.array([vals], dtype=np.int64), list(prs))
    if rec["den"][0] == "0":
        return None
    return rec["num"][0] if rec["den"][0] == "1" else f"{rec['num'][0]}/{rec['den'][0]}"


def interpolate_perprime(f, ref, pivots, znames, z_slots, leaf_names, n_leaves, point0, pool,
                         pool_next, nt_used, kcall, kbatch, rel_probe, ctrl, aux_leaves=(),
                         phys_cols=None):
    """Per-prime fits over the points each prime solves, coefficients lifted by CRT."""
    n_p = len(PRIMES)
    aux = set(aux_leaves)

    def nv(rp, p):
        return null_residues(rp, f, p)
    rel = direction_relevance(f, ref, pivots, z_slots, n_leaves, point0, rel_probe, None)
    base_nv, support_cols, rel_by_entry = rel["base_nv"], rel["support_cols"], rel["rel_by_entry"]
    if phys_cols is not None:
        pc = set(phys_cols)
        keep = [i for i, c in enumerate(support_cols) if c in pc]
        support_cols = [support_cols[i] for i in keep]
        rel_by_entry = [rel_by_entry[i] for i in keep]
    relevant = sorted({x for r in rel_by_entry for x in r})
    if expired(ctrl):
        return _fallback(pool_next, znames, base_nv, _TIMEOUT)

    def ex(v):
        return [x for x in v if x not in aux]
    max_rel_phys = max([0] + [len(ex(e)) for e in rel_by_entry])
    if len(ex(relevant)) > ctrl["relevance_cap_dir"] or max_rel_phys > ctrl["relevance_cap_sparse"]:
        return _fallback(pool_next, znames, base_nv,
                         f"direction couples {len(ex(relevant))} parameters (an entry up to {max_rel_phys})")

    def const_entry(col):
        vals, prs = [], []
        for pj in PRIMES:
            rp = kcall(point0, pj, nt_used)
            if not _pivots_eq(rp, pivots):
                continue
            v = nv(rp, pj)
            if v is None:
                continue
            vals.append(int(v[col]))
            prs.append(pj)
        if not prs:
            return None
        return _lift_const(vals, prs)
    if not relevant:
        entries = {}
        for c in support_cols:
            e = const_entry(c)
            if e is None:
                return _fallback(pool_next, znames, base_nv, "a constant entry could not be lifted")
            entries[znames[c]] = e
        entries[znames[f]] = "1"
        return {"pool_next": pool_next, "entry": {"support": sym_sort(list(entries)),
                                                  "vector": entries, "type": "general",
                                                  "closed_form": True, "relevant_leaves": []}}
    nrel = len(relevant)
    bank_u = [np.zeros((0, nrel), dtype=np.int64) for _ in range(n_p)]
    bank_r = [np.zeros((0, len(support_cols)), dtype=np.int64) for _ in range(n_p)]
    tries_p = [0] * n_p
    dead_p = [False] * n_p
    box = {"pool_next": pool_next}

    def ensure_prime(j, need):
        p = PRIMES[j]
        while bank_u[j].shape[0] < need and not dead_p[j]:
            if expired(ctrl):
                break
            have = bank_u[j].shape[0]
            chunk = max(1, min(need - have + 4, 128))
            uv = []
            for _ in range(chunk):
                uv.append(pool(box["pool_next"] + np.arange(nrel)))
                box["pool_next"] += nrel
            cand = []
            for u in uv:
                pt = np.array(point0, dtype=np.int64)
                pt[relevant] = u
                cand.append(pt)
            res = kbatch(cand, [p] * chunk, nt_used)
            for ci in range(chunk):
                rp = res[ci]
                if not _pivots_eq(rp, pivots):
                    continue
                vv = nv(rp, p)
                if vv is None:
                    continue
                bank_u[j] = np.vstack([bank_u[j], uv[ci].reshape(1, -1)])
                bank_r[j] = np.vstack([bank_r[j], vv[support_cols].reshape(1, -1)])
            tries_p[j] += chunk
            if bank_u[j].shape[0] == 0 and tries_p[j] >= 8 * need + 32:
                dead_p[j] = True
                break
            if tries_p[j] >= 80 * need + 256:
                break
        return bank_u[j].shape[0] >= need
    entries = {}
    for i, c in enumerate(support_cols):
        reli = rel_by_entry[i]
        if not reli:
            e = const_entry(c)
            if e is None:
                return _fallback(box["pool_next"], znames, base_nv, "a constant entry could not be lifted")
            entries[znames[c]] = e
            continue
        cols_i = [relevant.index(r) for r in reli]
        fitted = None
        for degree in range(ctrl["degree_cap"] + 1):
            if expired(ctrl):
                return _fallback(box["pool_next"], znames, base_nv, _TIMEOUT)
            mons = mono_table(len(reli), degree)
            need = 2 * mons.shape[0] - 1 + ctrl["sample_slack"]
            if need > ctrl["perprime_cap"]:
                break
            live = [j for j in range(n_p) if not dead_p[j] and ensure_prime(j, need)]
            if len(live) < ctrl["perprime_min_primes"]:
                continue
            raw, ref_free, ok_all = [], None, True
            for j in live:
                su = bank_u[j][:need][:, cols_i]
                rv = bank_r[j][:need, i]
                fit = _core.sym_fit_rational(su, mons, rv, PRIMES[j])
                if fit.get("status") != "ok":
                    ok_all = False
                    break
                raw.append(np.asarray(fit["coeffs"], dtype=np.int64))
                if ref_free is None:
                    ref_free = int(fit["free_col"])
            if not ok_all:
                continue
            coef_res = np.zeros((2 * mons.shape[0], len(live)), dtype=np.int64)
            for jj, j in enumerate(live):
                pj = PRIMES[j]
                d = int(raw[jj][ref_free]) % pj
                if d == 0:
                    ok_all = False
                    break
                coef_res[:, jj] = mulmod(raw[jj], invmod(d, pj), pj)
            if not ok_all:
                continue
            rec = rat_recon(coef_res, [PRIMES[j] for j in live])
            if any(d == "0" for d in rec["den"]):
                continue
            n_mon = mons.shape[0]
            fitted = {"num_n": rec["num"][:n_mon], "num_d": rec["den"][:n_mon],
                      "den_n": rec["num"][n_mon:], "den_d": rec["den"][n_mon:], "mons": mons}
            break
        if fitted is None:
            return _fallback(box["pool_next"], znames, base_nv,
                             "a joint entry could not be fit per-prime within degree_cap/perprime_cap; "
                             "raise them or the direction is not a bounded-degree rational")
        vars_ = [leaf_names[r] for r in reli]
        num_str = poly_string(fitted["num_n"], fitted["num_d"], fitted["mons"], vars_)
        den_str = poly_string(fitted["den_n"], fitted["den_d"], fitted["mons"], vars_)
        expr = num_str if den_str == "1" else f"({num_str})/({den_str})"
        entries[znames[c]] = simplify(expr)
    entries[znames[f]] = "1"
    return {"pool_next": box["pool_next"], "entry": {"support": sym_sort(list(entries)),
                                                     "vector": entries, "type": "general",
                                                     "closed_form": True, "relevant_leaves": relevant}}
