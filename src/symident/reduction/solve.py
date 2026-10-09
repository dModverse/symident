"""Solving the invariant map for the carriers q_k."""

import itertools
import re

import numpy as np
import sympy as spy

from .. import state
from ..gfp import Pool
from ..rexpr import sym_sort
from .expr import free_syms, red_locals, red_sympify, tri_solve
from .invariants import qr_rank
from .sections import inv_summands, match_carriers, section_cands, section_monotone
from .sign import (
    DOMAIN,
    domain_name,
    entry_class,
    is_real,
    pos_form,
    pos_form_real,
    positive_set,
    real_form,
    sgn,
    shift_fix,
    shifted_meaning,
    time_up,
    translates,
    translation_gauge,
)


def _q_names(n, coords, start):
    out, k = [], start
    for _ in range(n):
        while True:
            k += 1
            if f"q_{k}" not in set(coords):
                break
        out.append(f"q_{k}")
    return out


def _str_keys(d):
    return {str(k): v for k, v in d.items()}


def solve_invariants(b, pins, coords=(), inv_start=0, prefer_real=False, split_pair=None):
    """Invariants set equal to fresh outer parameters q_k and solved, certified positive."""
    out = {"pins": None, "meaning": None, "solved": False}
    if not b.get("invariants"):
        return out
    coords = list(coords)
    loc = red_locals(list(b["invariants"]) + b["support"] + list(pins) + list(pins.values()))
    try:
        Ies = [red_sympify(iv.replace("^", "**"), loc) for iv in b["invariants"]]
    except Exception:
        return out
    Ies2 = []
    for e in Ies:
        try:
            Ies2.append(spy.powsimp(spy.cancel(spy.together(e)), combine="exp"))
        except Exception:
            Ies2.append(e)
    Ies = Ies2
    inv_n = _q_names(len(Ies), coords, inv_start)

    def degs_in(p, v):
        try:
            pp = spy.Poly(p, spy.Symbol(v))
            return sorted({int(m[0]) for m in pp.monoms()})
        except Exception:
            return None
    pos = positive_set()
    adm = []
    for Ie in Ies:
        cand = list(b["support"])
        if coords:
            cand = sorted(cand, key=lambda c: -(coords.index(c) + 1 if c in coords else 0))
        if prefer_real:
            cand = sorted(cand, key=lambda c: (pos is not None and c in set(pos)) or pos is None)
        exp_a = list(Ie.atoms(spy.exp))
        if len(exp_a) > 1:
            return out
        G = exp_a[0].args[0] if exp_a else None
        R = spy.cancel(Ie / exp_a[0]) if exp_a else Ie
        frac_r = spy.fraction(spy.cancel(spy.together(R)))
        g_syms = set(free_syms(G)) if G is not None else set()
        r_syms = set(free_syms(R))

        def power_ok(frac, v):
            dn, dd = degs_in(frac[0], v), degs_in(frac[1], v)
            if dn is None or dd is None:
                return False
            return len(set(dn + dd) - {0}) == 1
        frac_g = spy.fraction(spy.cancel(spy.together(G))) if G is not None else None
        ok = [((v not in r_syms) and power_ok(frac_g, v)) if v in g_syms else power_ok(frac_r, v) for v in cand]
        a = [v for v, k in zip(cand, ok) if k]
        if not a:
            return out
        adm.append(a)
    carriers = match_carriers(adm)
    if carriers is None:
        return out
    tmp_n = [f"si_red_c{l + 1}" for l in range(len(Ies))]
    real_tmp = [t for t, Ie in zip(tmp_n, Ies) if sgn(Ie) != 1]
    old_rc = DOMAIN["real_carriers"]
    DOMAIN["real_carriers"] = real_tmp
    try:
        return _solve_invariants_inner(b, pins, coords, Ies, inv_n, carriers, tmp_n, real_tmp, loc,
                                       split_pair, out)
    finally:
        DOMAIN["real_carriers"] = old_rc


def _solve_invariants_inner(b, pins, coords, Ies, inv_n, carriers, tmp_n, real_tmp, loc, split_pair, out):
    best = best_root = None
    eqs = [Ies[l] - spy.Symbol(tmp_n[l]) for l in range(len(Ies))]
    try:
        sol_carr = [_str_keys(d) for d in spy.solve(eqs, [spy.Symbol(v) for v in carriers], dict=True)]
    except Exception:
        sol_carr = None
    gauge_all = [v for v in b["support"] if v not in carriers]
    pos = positive_set()
    g_pos = gauge_all if pos is None else [v for v in gauge_all if v in set(pos)]
    keep_sets = []
    if len(gauge_all) > 2 or len(g_pos) < len(gauge_all):
        for k in range(1, min(2, len(g_pos)) + 1):
            keep_sets += [list(c) for c in itertools.combinations(g_pos, k)]
    if len(gauge_all) <= 2:
        keep_sets.append(list(gauge_all))
    if len(gauge_all) > 2 and len(g_pos) == len(gauge_all):
        keep_sets = [list(gauge_all)] + keep_sets
    ks2 = []
    for k in keep_sets:
        if k not in ks2:
            ks2.append(k)
    keep_sets = ks2

    def pin_val(v):
        return "0" if is_real(v) else "1"

    def keep_ok(k):
        pr = [v for v in gauge_all if v not in k and is_real(v)]
        kr = [v for v in k if is_real(v)]
        rest = [v for v in gauge_all if v not in k]
        return (not pr or all(translates(b, v, rest) for v in pr)) and \
            not any(translates(b, v, gauge_all) for v in kr)
    keep_sets = [k for k in keep_sets if keep_ok(k)]
    pins_in = dict(pins)
    eqs_in = eqs
    red_diag = state.env("REDDIAG")
    ident_re = re.compile(r"[A-Za-z._][A-Za-z0-9._]*")

    def subs_of(vals):
        return [(spy.Symbol(v), e) for v, e in vals.items()]
    for gauge0 in keep_sets:
        if best_root is not None or time_up():
            break
        pinned1 = [v for v in gauge_all if v not in gauge0]
        pins = dict(pins_in, **{v: pin_val(v) for v in pinned1})
        pin1 = [(spy.Symbol(v), spy.Integer(int(pin_val(v)))) for v in pinned1]
        eqs = [e.subs(pin1) for e in eqs_in] if pin1 else eqs_in
        if not (len(gauge0) >= 1 and not any("exp(" in s for s in b["invariants"])):
            continue
        scal_pin_pairs = [(spy.Symbol(nm), red_sympify(v, loc)) for nm, v in pins.items()]
        all_unk = list(carriers) + list(gauge0)
        inv_syms = list(dict.fromkeys(s for Ie in Ies for s in free_syms(Ie)))
        sec_vars = [v for v in sym_sort(list(dict.fromkeys(
            b["support"] + ([s for s in inv_syms if s in set(coords)] if coords else inv_syms))))
            if v not in pinned1]
        summands = [inv_summands(Ie) for Ie in Ies]
        share_sets = []
        if len(gauge0) > 2:
            share_sets = [[[sm[0], sj] for sj in sm[1:]] for sm in summands if len(sm) == len(gauge0) + 1]
            cands = []
            for st in share_sets:
                for pr in st:
                    if pr not in cands:
                        cands.append(pr)
        else:
            cands = section_cands(sec_vars, summands)
        if split_pair is not None and len(split_pair) == 2 and all(x in sec_vars for x in split_pair):
            cands = [["*".join(split_pair), "1"]] + cands

        def touches(pr):
            txt = " ".join(pr)
            return any(re.search(r"(?<![0-9A-Za-z_.])" + re.escape(v) + r"(?![0-9A-Za-z_.])", txt)
                       for v in b["support"])
        cands = [c for c in cands if touches(c)]
        rows = None

        def pt_rank(M):
            M = np.asarray(M, dtype=float)
            return min(qr_rank(M[:, p::2], tol=1e-9) for p in (0, 1))
        if b.get("preps") is not None and cands:
            sec_loc = red_locals([x for g in b["preps"] for x in g["comps"].values()] +
                                [x for c in cands for x in c])
            xi_of = [{v: red_sympify(x, sec_loc) for v, x in g["comps"].items()} for g in b["preps"]]
            eval_vars = sym_sort(list(dict.fromkeys(sec_vars + [v for g in b["preps"] for v in g["vars"]])))
            poolv = np.array([float(x) for x in Pool()(np.arange(1, 2 * len(eval_vars) + 1) + 17)])
            poolv = 1 + poolv / (poolv.max() + 1)
            pts = [dict(zip(eval_vars, poolv[:len(eval_vars)])),
                   dict(zip(eval_vars, poolv[len(eval_vars):]))]
            if pos is not None:
                for v in eval_vars:
                    if v not in set(pos):
                        pts[1][v] = -pts[1][v]
            need = 40 if len(gauge0) == 1 else 10
            keep, rws = [], []
            for pr in cands[:200]:
                row = section_monotone(pr, b["preps"], xi_of, sec_loc, pts)
                if row is None:
                    continue
                keep.append(pr)
                row = np.array(row, dtype=float)
                rws.append(row / np.max(np.abs(row)))
                if len(keep) >= 3 * need:
                    break
                if len(keep) >= need and pt_rank(np.vstack(rws)) >= len(gauge0):
                    break
            cands = keep
            rows = np.vstack(rws) if rws else None
        if len(gauge0) > 2:
            ok_keys = [" = ".join(pr) for pr in cands]
            sets = []
            for st in share_sets:
                ks = [" = ".join(pr) for pr in st]
                if all(k in ok_keys for k in ks) and \
                        (rows is None or pt_rank(rows[[ok_keys.index(k) for k in ks]]) >= len(gauge0)):
                    sets.append(st)
        elif len(gauge0) == 1:
            sets = [[c] for c in cands]
        elif len(cands) > 1:
            n_c = min(len(cands), 10) if rows is None else len(cands)
            sets = [[cands[i], cands[j]] for i, j in itertools.combinations(range(n_c), 2)
                    if rows is None or pt_rank(rows[[i, j]]) >= len(gauge0)]
        else:
            sets = []
        carr_br = []
        if sol_carr is not None:
            for br in sol_carr:
                cv = {v: br.get(v) for v in carriers}
                if any(e is None for e in cv.values()):
                    continue
                if pin1:
                    cv = {v: spy.cancel(e.subs(pin1)) for v, e in cv.items()}
                    if any(e.has(spy.zoo, spy.nan) for e in cv.values()):
                        continue
                carr_br.append({"vals": cv, "subs": subs_of(cv)})
        if pin1 and sol_carr and not carr_br:
            continue

        def branches_of(sec_eqs, order):
            if order == "sections" or not carr_br:
                return tri_solve(sec_eqs + eqs, all_unk)
            res = []
            for cb in carr_br:
                try:
                    sub = [spy.together(e.subs(cb["subs"])) for e in sec_eqs]
                except Exception:
                    continue
                for gb in tri_solve(sub, gauge0):
                    if len(gb) < len(gauge0):
                        continue
                    gp = subs_of(gb)
                    d = {v: e.subs(gp) for v, e in cb["vals"].items()}
                    d.update({v: gb[v] for v in gauge0})
                    res.append(d)
            return res

        def verified(vals, sec_eqs):
            pairs = [(spy.Symbol(v), vals[v]) for v in all_unk]

            def vanishes(e):
                try:
                    return spy.simplify(e.subs(pairs)).is_zero is True
                except Exception:
                    return None
            for e in eqs + sec_eqs:
                try:
                    fr = spy.fraction(spy.together(e))
                except Exception:
                    return False
                if vanishes(fr[1]) is not False or vanishes(fr[0]) is not True:
                    return False
            return True

        def sec_positive(st):
            sy = {m for pr in st for x in pr for m in ident_re.findall(x)}
            return bool(sy) and (pos is None or sy <= set(pos))
        for st in sets[:40]:
            for order in ("carriers", "sections"):
                if best_root is not None or time_up():
                    continue
                if order == "sections" and not carr_br:
                    continue
                sec_eqs = [red_sympify(pr[0], loc) - red_sympify(pr[1], loc) for pr in st]
                sol2 = branches_of(sec_eqs, order)
                if not sol2:
                    continue
                for br in sol2:
                    es, ok_br = {}, True
                    for v in all_unk:
                        e0 = br.get(v)
                        if e0 is None or set(all_unk) & set(free_syms(e0)):
                            ok_br = False
                            break
                        es[v] = spy.cancel(e0.subs(scal_pin_pairs))
                        if es[v].has(spy.zoo, spy.nan, spy.oo, spy.S.NegativeInfinity):
                            ok_br = False
                            break
                    if not ok_br:
                        continue
                    if any(sgn(e) == -1 for e in es.values()):
                        continue
                    sh = shift_fix(es, tmp_n, Ies)
                    real_br = [t for t in real_tmp if t not in sh["shift"]]
                    ent, has_root, partial = {}, False, False
                    for v in all_unk:
                        e = None
                        if is_real(v) and (v in carriers or sec_positive(st)):
                            e = real_form(sh["es"][v], real_br)
                        if e is None:
                            e = pos_form_real(sh["es"][v], real_br)
                        if e is None:
                            e = pos_form(sh["es"][v])
                            if e is None:
                                if red_diag:
                                    print(f"[reddiag]   {v} = {sh['es'][v]} not certified")
                                ok_br = False
                                break
                            partial = True
                        cls = entry_class(e)
                        if cls == "no":
                            ok_br = False
                            break
                        if cls == "root":
                            has_root = True
                        ent[v] = str(e).replace("**", "^")
                    if not ok_br or not verified(br, sec_eqs):
                        continue
                    for l in range(len(carriers)):
                        ent = {k: re.sub(r"\b" + tmp_n[l] + r"\b", inv_n[l], x) for k, x in ent.items()}
                    cand = dict(out)
                    cand["pins"] = dict(ent, **{v: pins[v] for v in pinned1})
                    cand["gauge"] = list(gauge0) + pinned1
                    secs = []
                    for pr in st:
                        try:
                            r = spy.fraction(spy.cancel(red_sympify(pr[0], loc) / red_sympify(pr[1], loc)))
                            a_ = str(r[0]).replace("**", "^")
                            bq = str(r[1]).replace("**", "^")
                            secs.append(f"{bq} = 1" if a_ == "1" else f"{a_} = {bq}")
                        except Exception:
                            secs.append(f"{pr[0]} = {pr[1]}")
                    cand["section"] = secs + [f"{v} = {pins[v]}" for v in pinned1]
                    cand["root_note"] = ("a solved entry contains a square root; the branch is the "
                                         "certified positive one") if has_root else None
                    cand["meaning"] = shifted_meaning(b["invariants"], inv_n, tmp_n, sh)
                    cand["shifted"] = [inv_n[tmp_n.index(t)] for t in sh["shift"]]
                    cand["inv_names"] = inv_n
                    cand["solved"] = True
                    cand["coverage"] = "partial" if partial else "total"
                    cand["carrier_domain"] = {inv_n[l]: ("real" if tmp_n[l] in real_br else "positive")
                                              for l in range(len(tmp_n))}
                    if not partial and not has_root:
                        return cand
                    if not partial and best_root is None:
                        best_root = cand
                    if partial and best is None:
                        best = cand
    pins = pins_in
    eqs = eqs_in
    gauge0 = gauge_all
    if sol_carr:
        gauge = [v for v in b["support"] if v not in carriers]
        relax = len(sol_carr) == 1 and translation_gauge(b, gauge)
        pin_str = [{v: ("0" if relax and is_real(v) else "1") for v in gauge}]
        if real_tmp and gauge:
            pin_str.append({v: "1 + " + " + ".join(f"{t}**2" for t in real_tmp) for v in gauge})
        pin_loc = red_locals(list(pins) + list(pins.values()) + [x for gp in pin_str for x in gp.values()])
        for gp in pin_str:
            all_pins = dict(pins, **gp)
            pin_pairs = [(spy.Symbol(nm), red_sympify(v, pin_loc)) for nm, v in all_pins.items()]
            for sol in sol_carr:
                es = [sol.get(c) for c in carriers]
                ok_pin = all(e is not None for e in es)
                if ok_pin:
                    for _ in carriers:
                        dirty = False
                        for l in range(len(carriers)):
                            fs = set(free_syms(es[l]))
                            hit = [l2 for l2, c in enumerate(carriers) if c in fs and l2 != l]
                            if hit:
                                dirty = True
                                es[l] = es[l].subs([(spy.Symbol(carriers[l2]), es[l2]) for l2 in hit])
                        if not dirty:
                            break
                sh = None
                pinned = {}
                if ok_pin:
                    for l, c in enumerate(carriers):
                        if c in free_syms(es[l]):
                            ok_pin = False
                            break
                        pinned[c] = spy.cancel(es[l].subs(pin_pairs))
                    if ok_pin and any(sgn(e) == -1 for e in pinned.values()):
                        ok_pin = False
                    if ok_pin:
                        sh = shift_fix(pinned, tmp_n, Ies)
                        pinned = sh["es"]
                entries, root, partial = {}, False, False
                real_pin = real_tmp if sh is None else [t for t in real_tmp if t not in sh["shift"]]
                if ok_pin:
                    for c in carriers:
                        e = real_form(pinned[c], real_pin) if relax and is_real(c) else None
                        if e is None:
                            e = pos_form_real(pinned[c], real_pin)
                        if e is None:
                            e = pos_form(pinned[c])
                            if e is None:
                                ok_pin = False
                                break
                            partial = True
                        cls = entry_class(e)
                        if cls == "no":
                            ok_pin = False
                            break
                        if cls == "root":
                            root = True
                        entries[c] = str(e).replace("**", "^")
                if ok_pin:
                    def to_inv(v):
                        out2 = {}
                        for k, x in v.items():
                            for l in range(len(carriers)):
                                x = re.sub(r"\b" + tmp_n[l] + r"\b", inv_n[l], x)
                            out2[k] = x.replace("**", "^")
                        return out2
                    cand = dict(out)
                    cand["pins"] = dict(to_inv(entries), **to_inv(gp))
                    cand["gauge"] = gauge
                    cand["meaning"] = shifted_meaning(b["invariants"], inv_n, tmp_n, sh)
                    cand["shifted"] = [inv_n[tmp_n.index(t)] for t in (sh["shift"] if sh else {})]
                    cand["inv_names"] = inv_n
                    cand["solved"] = True
                    cand["root_note"] = ("a solved entry contains a square root; the branch is the "
                                         "certified positive one") if root else None
                    cand["coverage"] = "partial" if partial else "total"
                    cand["carrier_domain"] = {inv_n[l]: ("real" if tmp_n[l] in real_pin else "positive")
                                              for l in range(len(tmp_n))}
                    if not partial and not root:
                        return cand
                    if not partial and best_root is None:
                        best_root = cand
                    if partial and best is None:
                        best = cand
    if best_root is not None:
        return best_root
    if best is not None:
        return best
    if time_up():
        out["reason"] = "the chart search stopped at reduce(timeout=)"
        return out
    tried = ""
    if any("exp(" in s for s in b["invariants"]):
        tried = " (skipped, a transcendental invariant is in the set)"
    elif len(gauge0) < 1:
        tried = " (skipped, no gauge coordinate)"
    elif len(gauge0) > 2:
        tried = f" with {len(gauge0)} gauge coordinates, the surplus pinned"
    out["reason"] = (f"no gauge section with entries certified positive on the whole {domain_name()} "
                     f"(tried: monomial balances{tried}, then the constant pin)")
    return out
