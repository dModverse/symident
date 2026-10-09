"""Curved blocks: invariant search per block, fixed parameters, block verification."""

import re

import numpy as np
import sympy as spy

from ..gfp import Pool
from ..rexpr import sym_sort
from ..sym.invariants import verify_invariants
from .darboux import _moved_vars, darboux, exp_factors, int_factor, separable
from .expr import gen_prep, group_by, module_reduce, red_locals, red_sympify
from .invariants import (
    monomial_invariants,
    poly_invariants,
    qr_rank,
    rational_invariants,
)
from .scaling import coords_of, scaling_block
from .sections import translation_compress, unmoved_compress
from .sign import sgn


def block_corank(preps):
    """Number of moved coordinates minus the rank of the block directions at a numeric point."""
    from ..rexpr import eval_numeric
    moved, vars_ = _moved_vars(preps)
    pt = dict(zip(vars_, [float(x) for x in Pool()(np.arange(1, len(vars_) + 1) + 5)]))
    Xi = np.array([[eval_numeric(pr["comps"][v], pt) if v in pr["comps"] else 0.0 for v in moved]
                   for pr in preps], dtype=float)
    return len(moved) - qr_rank(Xi, tol=1e-9)


def independent_set(invs, vars_, target, moved=None, keep_order=False):
    """A functionally independent subset of invariants by a numeric Jacobian rank test."""
    if moved is None:
        moved = vars_
    if not invs or target <= 0:
        return []
    pool = np.array([float(x) for x in Pool()(np.arange(1, len(vars_) + 1) + 13)])
    if any("exp(" in s for s in invs):
        pool = 1 + pool / (pool.max() + 1)
    pt = dict(zip(vars_, pool))
    loc = {v: spy.Symbol(v) for v in vars_}

    def grad(iv):
        try:
            e = spy.sympify(iv.replace("^", "**"), locals=dict(loc, **red_locals(iv)))
        except Exception:
            return None
        out = []
        subs = {spy.Symbol(k): v for k, v in pt.items()}
        for v in moved:
            try:
                out.append(float(spy.N(spy.diff(e, spy.Symbol(v)).subs(subs))))
            except Exception:
                return None
        return np.array(out)
    J = np.zeros((0, len(moved)))
    sel = []
    cand = list(dict.fromkeys(invs))
    if not keep_order:
        cand = sorted(cand, key=len)
    for iv in cand:
        if len(sel) >= target:
            break
        gr = grad(iv)
        if gr is None or not np.all(np.isfinite(gr)):
            continue
        Jc = np.vstack([J, gr])
        if qr_rank(Jc, tol=1e-9) > J.shape[0]:
            J = Jc
            sel.append(iv)
    return sel


def solve_block(preps, d_poly, d_darboux, d_exp, verbose=False, sep=True):
    """The invariant stages on one module-reduced sub-block."""
    moved, vars_ = _moved_vars(preps)
    target = block_corank(preps)
    certs, found, stage_ = [], [], "monomial"
    moved_re = re.compile(r"(?<![0-9A-Za-z_.])(" + "|".join(re.escape(m) for m in moved) +
                          r")(?![0-9A-Za-z_.])")
    sgn_cache = {}

    def definite(iv):
        if iv in sgn_cache:
            return sgn_cache[iv]
        try:
            e = red_sympify(iv.replace("^", "**"), red_locals(iv))
            v = sgn(e) != 0
        except Exception:
            v = False
        sgn_cache[iv] = v
        return v

    def pick(cand):
        cand = list(dict.fromkeys(c for c in cand if moved_re.search(c)))
        is_exp = ["exp(" in c for c in cand]
        tier = [3 if ie else (1 if definite(c) else 2) for c, ie in zip(cand, is_exp)]
        order = sorted(range(len(cand)), key=lambda i: (tier[i], len(cand[i])))
        return independent_set([cand[i] for i in order], vars_, target, moved, keep_order=True)

    def short():
        return len(found) < target or not all(definite(f) for f in found)

    def better(got):
        return len(got) > len(found) or (len(got) == len(found) and
                                         sum(map(definite, got)) > sum(map(definite, found)))
    mono = monomial_invariants(preps)
    certs.append(mono["cert"])
    if mono["ok"]:
        found = pick(mono["invariants"])
    if short():
        poly = poly_invariants(preps, d_poly)
        certs.append(poly["cert"])
        if poly["ok"]:
            got = pick(found + poly["invariants"])
            if got != found:
                if better(got):
                    stage_ = "polynomial"
                found = got
    if sep and short():
        s = separable(preps)
        certs.append(s["cert"])
        if s["ok"]:
            got = pick(found + s["invariants"])
            if got != found:
                if better(got):
                    stage_ = "separable"
                found = got
    elif not sep:
        certs.append("separable-characteristics stage skipped (separable=False)")
    if short():
        rat = rational_invariants(preps, d_poly)
        certs.append(rat["cert"])
        if rat["ok"]:
            got = pick(found + rat["invariants"])
            if got != found:
                if better(got):
                    stage_ = "rational"
                found = got
    dar = None
    if short():
        dar = darboux(preps, d_darboux, extactic_ok=len(found) < target)
        certs.append(dar["cert"])
        if dar["ok"]:
            got = pick(found + dar["invariants"])
            if got != found:
                if better(got):
                    stage_ = "darboux"
                found = got
    if short() and dar is not None:
        if d_exp > 0:
            ex = exp_factors(preps, dar, d_exp)
            certs.append(ex["cert"])
            if ex["ok"]:
                got = pick(found + ex["invariants"])
                if got != found:
                    if better(got):
                        stage_ = "exp"
                    found = got
            if len(found) < target and len(preps) == 1 and len(moved) == 2:
                itf = int_factor(preps, dar, ex["factors"])
                certs.append(itf["cert"])
                if itf["ok"]:
                    found = found + [itf["invariants"][0]]
                    stage_ = "intfactor"
        else:
            certs.append("exp stage skipped (d_exp = 0)")
    reason = None if len(found) >= target else (
        f"found {len(found)} of {target} independent invariants (poly/rational degree <= {d_poly}, "
        f"Darboux degree <= {d_darboux}, exp numerator degree <= {d_exp}); gauge pinning needs the full set")
    return {"stage": stage_, "invariants": found, "target": target, "certificates": certs,
            "reason": reason}


def fixed_reduce(preps, fixed):
    """Combinations of a curved group that leave every fixed coordinate unmoved."""
    k = len(preps)
    cols = sym_sort(list(dict.fromkeys(c for pr in preps for c in pr["support"])))
    fixed_cols = [c for c in fixed if c in cols]
    if not fixed_cols or not k:
        return {"preps": preps, "sources": [[i] for i in range(k)], "removed": 0, "fixed_cols": []}
    loc = red_locals([x for pr in preps for x in pr["comps"].values()])
    E = [{cc: red_sympify(pr["comps"].get(cc, "0"), loc) for cc in cols} for pr in preps]
    sources = [[i] for i in range(k)]
    used_pivot = [False] * k
    for fc in fixed_cols:
        piv = None
        for r in range(k):
            if not used_pivot[r] and str(E[r][fc]) != "0":
                piv = r
                break
        if piv is None:
            continue
        for j in range(k):
            if j == piv or str(E[j][fc]) == "0":
                continue
            m = spy.cancel(E[j][fc] / E[piv][fc])
            for c2 in cols:
                E[j][c2] = spy.cancel(E[j][c2] - m * E[piv][c2])
            sources[j] = sorted(set(sources[j]) | set(sources[piv]))
        used_pivot[piv] = True
    out_p, out_s = [], []
    for j in [i for i in range(k) if not used_pivot[i]]:
        comps = {cc: str(E[j][cc]) for cc in cols}
        pr = gen_prep({"generator": {c: v for c, v in comps.items() if v != "0"}})
        if not pr["ok"] or not pr["support"]:
            continue
        out_p.append(pr)
        out_s.append(sources[j])
    return {"preps": out_p, "sources": out_s, "removed": sum(used_pivot),
            "removed_idx": [i for i in range(k) if used_pivot[i]], "fixed_cols": fixed_cols}


def curved(syms, idx, labels, fixed, d_poly, d_darboux, d_exp, verbose=False, sep=True, pins=None):
    """Curved directions: prep, group, fixed elimination, compressions, module reduction, search."""
    blocks = []

    def unresolved(labs, support, reason):
        return {"labels": labs, "type": "curved", "support": support, "stage": "none",
                "status": "unresolved", "invariants": [], "certificates": [], "reason": reason,
                "module_combos": None, "transversal": None, "survivor_meaning": None, "pins": None}
    scal_labs = {labels[i] for i, d in enumerate(syms) if d.get("type") == "scaling"}

    def kind_of(labs):
        return "scaling" if labs and all(l in scal_labs for l in labs) else "curved"
    has_gen = [i for i in idx if syms[i].get("generator") is not None]
    for i in idx:
        if syms[i].get("generator") is None:
            blocks.append(unresolved([labels[i]], coords_of(syms[i]),
                                     "no closed-form generator; re-run the detection with reconstruct=True"))
    gi = has_gen
    preps = [gen_prep(syms[i], pins) for i in gi]
    for w, pr in enumerate(preps):
        if not pr["ok"]:
            blocks.append(unresolved([labels[gi[w]]], coords_of(syms[gi[w]]), pr.get("reason")))
    keep = [w for w, pr in enumerate(preps) if pr["ok"]]
    gi = [gi[w] for w in keep]
    preps = [preps[w] for w in keep]
    if not gi:
        return blocks
    for grp in group_by([pr["support"] for pr in preps]):
        glab = [labels[gi[g]] for g in grp]
        fx = fixed_reduce([preps[g] for g in grp], fixed)
        fix_cert = [f"fixed-coordinate elimination: {fx['removed']} of {len(grp)} direction(s) cannot "
                    f"combine away from {{{', '.join(fx['fixed_cols'])}}}"] if fx["removed"] > 0 else []
        fixed_labs = []
        if fx["removed"] > 0:
            used = {s for src in fx["sources"] for s in src}
            fixed_labs = [glab[i] for i in fx["removed_idx"] if i not in used]
        if fixed_labs or not fx["preps"]:
            fl = fixed_labs if fx["preps"] else glab
            blocks.append({"labels": fl, "type": "curved", "kind": kind_of(fl),
                           "support": sym_sort(list(dict.fromkeys(c for g in grp for c in preps[g]["support"]))),
                           "stage": "fixed", "status": "fixed", "removed_by_fixed": fx["removed"],
                           "invariants": [], "certificates": fix_cert, "reason": None,
                           "module_combos": None, "transversal": None, "survivor_meaning": None,
                           "pins": None, "gauge_note": "removed entirely by the user's fixed coordinates"})
        if not fx["preps"]:
            continue
        glab_k = [glab[s[0]] if len(s) == 1 else "(" + "+".join(glab[i] for i in s) + ")"
                  for s in fx["sources"]]
        try:
            tc = translation_compress(fx["preps"])
        except Exception:
            tc = None
        p1 = fx["preps"] if tc is None else tc["preps"]
        try:
            uc = unmoved_compress(p1)
        except Exception:
            uc = None
        p2 = p1 if uc is None else uc["preps"]

        def decomp(pr):
            if uc is not None:
                pr = uc["back_prep"](pr)
            if tc is not None:
                pr = tc["back_prep"](pr)
            return pr
        mr = module_reduce(p2, glab_k)
        subs = group_by([pr["support"] for pr in mr["preps"]])
        # the sub-blocks share the group's labels: each label goes to one sub-block, the
        # sub-blocks with fewer sources first
        src_of = [{s for j in sub for li in mr["labels_of"][j] for s in fx["sources"][li]} for sub in subs]
        taken = set()
        for k in sorted(range(len(subs)), key=lambda k: len(src_of[k])):
            own = sym_sort([glab[s] for s in src_of[k] if glab[s] not in taken])[:len(subs[k])]
            taken.update(own)
            src_of[k] = own
        for sub, sub_labs in zip(subs, src_of):
            supp = sym_sort(list(dict.fromkeys(c for j in sub for c in mr["preps"][j]["support"])))
            if len(supp) == 1:
                # a direction along one coordinate: its orbit is that axis, pinned like a scaling
                blocks.append(scaling_block(np.array([[1]], dtype=np.int64), supp, sub_labs, fixed))
                continue
            sol = solve_block([mr["preps"][j] for j in sub], d_poly, d_darboux, d_exp, verbose, sep)
            face_inv = None
            if sol["invariants"]:
                if uc is not None:
                    face_inv = sol["invariants"] if tc is None else tc["back"](sol["invariants"])
                    sol["invariants"] = uc["back"](sol["invariants"])
                    sol["certificates"].append("searched in the unmoved monomial(s) " +
                                               "; ".join(uc["note"]) +
                                               " (constants along every orbit), invariants substituted back")
                if tc is not None:
                    sol["invariants"] = tc["back"](sol["invariants"])
                    sol["certificates"].append("searched in the translation-group sum(s) " +
                                               "; ".join(tc["note"]) + ", invariants substituted back")
            preps_orig = [mr["preps"][j] for j in sub]
            if uc is not None or tc is not None:
                preps_orig = [decomp(pr) for pr in preps_orig]
            blocks.append({"labels": sub_labs, "type": "curved", "kind": kind_of(sub_labs),
                           "support": sym_sort(list(dict.fromkeys(c for pr in preps_orig for c in pr["support"]))),
                           "stage": sol["stage"], "target": sol["target"],
                           "status": "invariant_only" if sol["invariants"] else "unresolved",
                           "removed_by_fixed": fx["removed"] if fx["removed"] > 0 and not fixed_labs else None,
                           "invariants": sol["invariants"],
                           "certificates": fix_cert + sol["certificates"], "reason": sol["reason"],
                           "module_combos": mr["combos"], "transversal": None,
                           "survivor_meaning": None, "pins": None, "preps": preps_orig,
                           "face_invariants": face_inv,
                           "face_back": uc["back"] if uc is not None else None})
    return blocks


def verify_blocks(blocks):
    """X(I) = 0 for every invariant and every generator, exactly."""
    def chk(inv, preps):
        try:
            return [bool(x) for x in verify_invariants([s.replace("^", "**") for s in inv],
                                                         [dict(pr["comps"]) for pr in preps])]
        except Exception:
            return [False] * len(inv)
    ident = re.compile(r"[A-Za-z._][A-Za-z0-9._]*")
    for bi, b in enumerate(blocks):
        if b["type"] == "scaling":
            if b.get("Wres") is not None and b.get("inv_exps") is not None and \
                    b["inv_exps"].shape[1] and b["Wres"].shape[0]:
                cols = b["Wcols"]
                sel = [cols.index(c) for c in b["support"]]
                if np.any(b["Wres"][:, sel] @ b["inv_exps"] != 0):
                    raise RuntimeError("internal error: scaling invariant not "
                                       "in the weight kernel.")
                b["certificates"].append("verified: integer weight annihilation, exact")
            continue
        if not b.get("invariants") or b.get("preps") is None:
            continue
        keep = chk(b["invariants"], b["preps"])
        if not all(keep):
            b["certificates"].append(f"verification DROPPED {sum(not k for k in keep)} invariant(s): X(I) != 0")
            b["invariants"] = [i for i, k in zip(b["invariants"], keep) if k]
            if not any(keep):
                b["status"] = "unresolved"
        else:
            b["certificates"].append("verified: X(I) = 0 exactly (sympy cancel) for every generator")
        inv = b["invariants"]
        inv_syms = {m for s in inv for m in ident.findall(s)}
        other = [pr for j, ob in enumerate(blocks) if j != bi for pr in (ob.get("preps") or [])]
        if not other or not inv:
            continue
        other = [pr for pr in other if set(pr["support"]) & inv_syms]
        if not other:
            b["certificates"].append("verified: X(I) = 0 for every direction of every other block too, "
                                     "none of which moves a symbol of these invariants")
            continue
        keep = chk(inv, other)
        if not all(keep):
            b["certificates"].append(f"verification DROPPED {sum(not k for k in keep)} invariant(s): "
                                     "X(I) != 0 for a direction of another block")
            b["invariants"] = [i for i, k in zip(inv, keep) if k]
            if not any(keep):
                b["status"] = "unresolved"
        else:
            b["certificates"].append("verified: X(I) = 0 for every direction of every other block too")
    return blocks
