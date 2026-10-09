"""Interpolation of one residual direction, entry by entry, as rational functions."""

import numpy as np

from ..gfp import PRIMES, mono_table, null_residues
from ..rexpr import sym_sort
from ..sym.recast import drop_monomial_content, exp_backsub_expr, trig_full_angle
from .closedform import expired, poly_string, simplify
from .perprime import _TIMEOUT, _fallback, _lift_const, pinned_monomials
from .relevance import dense_need, direction_relevance
from .sparse import _default_kbatch, _pivots_eq, general_rational_entry, sparse_entry


def interpolate_direction(f, ref, pivots, znames, z_slots, leaf_names, n_leaves, point0, pool,
                          pool_next, nt_used, kcall, rel_probe=None, residue_fn=None, ctrl=None,
                          kbatch=None, shared_bank=None, fast_only=False, aux_leaves=()):
    """One residual direction as rational entries over the leaves that move each entry."""
    from .closedform import reconstruct_entry
    if kbatch is None:
        kbatch = _default_kbatch(kcall)
    zvals0 = None if z_slots is None else np.asarray(point0)[z_slots]

    def zvals_of(pt):
        return None if z_slots is None else np.asarray(pt)[z_slots]

    def nv(rp, p, zvals=zvals0):
        return null_residues(rp, f, p) if residue_fn is None else residue_fn(rp, p, zvals)
    rel = direction_relevance(f, ref, pivots, z_slots, n_leaves, point0, rel_probe, residue_fn)
    base_nv, support_cols = rel["base_nv"], rel["support_cols"]
    relevant, rel_by_entry = rel["relevant"], rel["rel_by_entry"]
    box = {"pool_next": pool_next, "const_base": None}

    def fallback(reason=None):
        return _fallback(box["pool_next"], znames, base_nv, reason)
    if expired(ctrl):
        return fallback(_TIMEOUT)

    def const_entry(col):
        if box["const_base"] is None:
            kr = kbatch([point0] * len(PRIMES), list(PRIMES), nt_used)
            box["const_base"] = [nv(kr[k], PRIMES[k]) if _pivots_eq(kr[k], pivots) else None
                                 for k in range(len(PRIMES))]
        res = []
        for nvp in box["const_base"]:
            if nvp is None:
                return None
            res.append(int(nvp[col]))
        return _lift_const(res, PRIMES)
    if not relevant:
        entries = {}
        for c in support_cols:
            e = const_entry(c)
            if e is None:
                return fallback("a constant entry could not be lifted from its residues")
            entries[znames[c]] = e
        entries[znames[f]] = "1"
        return {"pool_next": box["pool_next"], "entry": {"support": sym_sort(list(entries)),
                                                         "vector": entries, "type": "general",
                                                         "closed_form": True, "fit_how": "constant"}}
    if getattr(residue_fn, "pinned_support", False) and rel_probe is not None:
        mono = pinned_monomials(f, pivots, znames, leaf_names, point0, nt_used, kcall, nv, base_nv,
                                support_cols, rel_by_entry, rel_probe)
        if mono is not None:
            return {"pool_next": box["pool_next"], "entry": mono}
        if fast_only:
            return fallback("pinned entry is not a bounded-degree monomial")
    aux = set(aux_leaves)

    def ex(v):
        return [x for x in v if x not in aux]
    rel_phys = ex(relevant)
    max_rel_phys = max([0] + [len(ex(e)) for e in rel_by_entry])
    if len(rel_phys) > ctrl["relevance_cap_dir"] or max_rel_phys > ctrl["relevance_cap_sparse"]:
        return fallback(f"direction couples {len(rel_phys)} parameters (an entry up to {max_rel_phys}), "
                        f"above relevance_cap_dir={ctrl['relevance_cap_dir']} / relevance_cap_sparse="
                        f"{ctrl['relevance_cap_sparse']}; raise the relevant cap")
    nrel = len(relevant)
    n_p = len(PRIMES)
    dn = dense_need(rel_by_entry, ctrl)
    sample_u = np.zeros((0, nrel), dtype=np.int64)
    rstore = [np.zeros((0, n_p), dtype=np.int64) for _ in support_cols]
    if dn["any_dense"]:
        max_need = dn["max_need"]
        if shared_bank is not None:
            n_acc = shared_bank["U"].shape[0]
            if n_acc < max_need:
                return fallback("the shared bank holds fewer points than this direction needs")
            un = list(shared_bank["union"])
            sample_u = shared_bank["U"][:, [un.index(r) for r in relevant]]
            rstore = [np.zeros((n_acc, n_p), dtype=np.int64) for _ in support_cols]
            for a in range(n_acc):
                zvc = zvals_of(shared_bank["points"][a])
                for j in range(n_p):
                    nvp = nv(shared_bank["rps"][a][j], PRIMES[j], zvc)
                    if nvp is None:
                        return fallback("shared-bank residue extraction failed")
                    for i, c in enumerate(support_cols):
                        rstore[i][a, j] = nvp[c]
        else:
            tries = 0
            rows_u, rows_r = [], [[] for _ in support_cols]
            while len(rows_u) < max_need and tries < 30 * max_need:
                if expired(ctrl):
                    return fallback(_TIMEOUT)
                if not rows_u and tries >= 2 * max_need:
                    return fallback("no valid steady state under perturbation of the relevant leaves")
                # the hit rate so far cannot fill the need within the budget: stop early
                if tries >= 2 * max_need and len(rows_u) + len(rows_u) / tries * (30 * max_need - tries) < max_need:
                    return fallback("dense sampling could not collect enough pivot-consistent points; "
                                    "raise sample_slack or probe_retries")
                chunk = max(1, min(max_need - len(rows_u) + ctrl["sample_slack"], 30 * max_need - tries))
                if ctrl.get("deadline") is not None:
                    chunk = min(chunk, 96)
                tries += chunk
                uv = []
                for _ in range(chunk):
                    uv.append(pool(box["pool_next"] + np.arange(nrel)))
                    box["pool_next"] += nrel
                cand = []
                for u in uv:
                    pt = np.array(point0, dtype=np.int64)
                    pt[relevant] = u
                    cand.append(pt)
                res = kbatch([c for c in cand for _ in range(n_p)], list(PRIMES) * chunk, nt_used)
                for ci in range(chunk):
                    if len(rows_u) >= max_need:
                        break
                    good = True
                    rrow = np.zeros((len(support_cols), n_p), dtype=np.int64)
                    zvc = zvals_of(cand[ci])
                    for j in range(n_p):
                        rp = res[ci * n_p + j]
                        if not _pivots_eq(rp, pivots):
                            good = False
                            break
                        nvp = nv(rp, PRIMES[j], zvc)
                        if nvp is None:
                            good = False
                            break
                        rrow[:, j] = np.asarray(nvp)[support_cols]
                    if not good:
                        continue
                    rows_u.append(uv[ci])
                    for i in range(len(support_cols)):
                        rows_r[i].append(rrow[i])
            if len(rows_u) < max_need:
                return fallback("dense sampling could not collect enough pivot-consistent points; "
                                "raise sample_slack or probe_retries")
            sample_u = np.vstack(rows_u)
            rstore = [np.vstack(r) for r in rows_r]
    entries = {}
    how = []
    for i, c in enumerate(support_cols):
        reli = rel_by_entry[i]
        if not reli:
            e = const_entry(c)
            if e is None:
                return fallback("a constant entry could not be lifted from its residues")
            entries[znames[c]] = e
            how.append("constant")
            continue
        if len(reli) > ctrl["relevance_cap"]:
            e = sparse_entry(reli, c, f, point0, leaf_names, nt_used, kcall, pivots, residue_fn,
                             ctrl, z_slots, kbatch)
            how.append("sparse Laurent")
            if e is None:
                e = general_rational_entry(reli, c, f, point0, leaf_names, nt_used, kcall, pivots,
                                           residue_fn, ctrl, z_slots, kbatch)
                how[-1] = "sparse rational"
            if e is None and expired(ctrl):
                return fallback(_TIMEOUT)
            if e is None:
                return fallback(f"an entry couples {len(reli)} variables and the sparse fit hit its caps "
                                f"(laurent_deg_num={ctrl['laurent_deg_num']}, general_deg_num="
                                f"{ctrl['general_deg_num']}, general_deg_den={ctrl['general_deg_den']}, "
                                f"term_cap={ctrl['term_cap']}); raise these, or raise relevance_cap="
                                f"{ctrl['relevance_cap']} to use the dense fit")
            entries[znames[c]] = simplify(e)
            continue
        cols_i = [relevant.index(r) for r in reli]
        vars_i = [leaf_names[r] for r in reli]
        fitted = None
        for degree in range(ctrl["degree_cap"] + 1):
            mons = mono_table(len(reli), degree)
            need = 2 * mons.shape[0] - 1 + ctrl["sample_slack"]
            rec = reconstruct_entry(sample_u[:need][:, cols_i], mons, rstore[i][:need], list(PRIMES))
            if rec is not None:
                fitted = (rec, mons)
                break
        if fitted is None:
            return fallback(f"an entry exceeded degree_cap={ctrl['degree_cap']}; raise it")
        rec, mons = fitted
        num_str = poly_string(rec["num_coef_n"], rec["num_coef_d"], mons, vars_i)
        den_str = poly_string(rec["den_coef_n"], rec["den_coef_d"], mons, vars_i)
        expr = num_str if den_str == "1" else f"({num_str})/({den_str})"
        entries[znames[c]] = simplify(expr)
        how.append("dense")
    entries[znames[f]] = "1"
    return {"pool_next": box["pool_next"],
            "entry": {"support": sym_sort(list(entries)), "vector": entries, "type": "general",
                      "closed_form": True, "fit_how": ", ".join(dict.fromkeys(how))}}


def exp_backsub(vector, back, atoms=None):
    """Components in the exponential and half-angle leaves back in the model symbols."""
    nm = [str(x) for x in back["names"]]
    vl = [str(x) for x in back["values"]]
    out = {}
    for k, x in vector.items():
        try:
            v = exp_backsub_expr(str(x), nm, vl)
            out[k] = v if isinstance(v, str) else str(x)
        except Exception:
            out[k] = str(x)
    skip = [str(a) for a in (atoms or [])]
    for tidy in (trig_full_angle, drop_monomial_content):
        try:
            v = tidy(out, skip)
        except Exception:
            v = None
        if v is not None:
            out = {k: str(v[k]) for k in out}
    return out
