"""Darboux polynomials, exponential factors and integrating factors."""

import math
import re

import numpy as np
import sympy as spy

from .. import _core
from ..gfp import Pool, mulmod
from ..rexpr import sym_sort
from ..sym.invariants import (
    darboux_cofactors,
    integrating_factor_integral,
    verify_invariants,
)
from ..sym.modp import eval_rational_batch
from .expr import eval_batch, free_syms, red_locals, red_sympify
from .invariants import eta_rows, modular_nullspace, poly_string
from .scaling import int_kernel, mono_string
from .sign import positive_set, sgn

EXTACTIC_CAP = 12


def apply_x(xi_exprs, f, loc):
    """The generator with components `xi_exprs` applied to `f`, expanded."""
    out = spy.Integer(0)
    for v, xi in xi_exprs.items():
        out = out + xi * spy.diff(f, red_sympify(v, loc))
    return spy.expand(out)


def subs_rational(e, pt):
    """`e` at the integer point `pt`, cancelled, as a string."""
    pairs = [(spy.Symbol(n), spy.Integer(int(v))) for n, v in pt.items()]
    return str(spy.cancel(e.subs(pairs)))


def cofactor_rows(cols, vars_, n_gen, n_mon):
    """Exact integer sampling rows of a cofactor matrix, one per (generator, point)."""
    rows, per_gen = [], [0] * n_gen
    for g in range(n_gen):
        exprs = [str(cl[g]) for cl in cols]
        pts = [[int(x) for x in Pool()(np.arange(1, len(vars_) + 1) + 2 * t + 7 * (g + 1))]
               for t in range(1, n_mon + 1)]
        try:
            v = eval_rational_batch(exprs, list(vars_), pts)
        except Exception:
            continue
        for t in range(n_mon):
            try:
                num = [int(x) for x in v["num"][t]]
                den = [int(x) for x in v["den"][t]]
            except (TypeError, ValueError):
                continue
            if any(d == 0 for d in den):
                continue
            L = 1
            for d in set(den):
                L = L * d // math.gcd(L, d)
            row = [n * (L // d) for n, d in zip(num, den)]
            if max(abs(x) for x in row) < 2 ** 53:
                rows.append(row)
                per_gen[g] += 1
    if any(x == 0 for x in per_gen):
        return []
    return rows


def _paren(s):
    return f"({s})" if re.search(r"[+*/ -]", s) else s


def darboux(preps, d_darboux, extactic_ok=True):
    """Rational invariants from Darboux polynomials (extactic route)."""
    from ..gfp import mono_table
    moved = sym_sort(list(dict.fromkeys(v for pr in preps for v in pr["support"])))
    vars_ = sym_sort(list(dict.fromkeys(moved + [v for pr in preps for v in pr["vars"]])))
    loc = red_locals([x for pr in preps for x in pr["comps"].values()] + vars_)
    xi_of = [{v: red_sympify(x, loc) for v, x in pr["comps"].items()} for pr in preps]
    cands = [red_sympify(v, loc) for v in moved]
    for xs in xi_of:
        for xi in xs.values():
            try:
                fl = spy.factor_list(xi)
                cands += [pair[0] for pair in fl[1]]
            except Exception:
                pass
    Nbasis = math.comb(len(moved) + d_darboux, d_darboux)
    Dmax = max([1] + [3 if pr["degree"] is None else int(pr["degree"]) for pr in preps])
    entry_deg = d_darboux + (Nbasis - 1) * (Dmax - 1)
    extactic = extactic_ok and d_darboux >= 1 and Nbasis <= EXTACTIC_CAP and entry_deg <= EXTACTIC_CAP
    degenerate = False
    if extactic:
        expts = mono_table(len(moved), int(d_darboux))
        basis = [red_sympify(mono_string(expts[k], moved), loc) for k in range(expts.shape[0])]
        N = len(basis)
        rws = [basis]
        for j in range(N - 1):
            rws.append([apply_x(xi_of[0], f, loc) for f in rws[j]])
        D = spy.Matrix(rws).det(method="berkowitz")
        nonzero = False
        for t in (1, 2):
            pt = dict(zip(vars_, [int(x) for x in Pool()(np.arange(1, len(vars_) + 1) + 11 * t)]))
            try:
                v = subs_rational(D, pt)
            except Exception:
                v = None
            if v is not None and v != "0":
                nonzero = True
                break
        if not nonzero:
            degenerate = True
        else:
            fl = spy.factor_list(D)
            cands += [pair[0] for pair in fl[1]]
    dc = darboux_cofactors([str(c) for c in cands], [dict(pr["comps"]) for pr in preps])
    Ps = [red_sympify(s, loc) for s in dc["Ps"]]
    lams = [[red_sympify(s, loc) for s in lr] for lr in dc["lams"]]
    if degenerate:
        coverage = "extactic degenerate (E identically 0): coordinate and xi factors only"
    elif extactic:
        coverage = f"extactic complete for factor degree <= {d_darboux} in the moved coordinates"
    elif not extactic_ok:
        coverage = "extactic skipped (invariant count already met): coordinate and xi factors only"
    else:
        coverage = (f"extactic skipped (moved-coordinate basis {Nbasis}, projected entry degree "
                    f"{entry_deg}, cap {EXTACTIC_CAP}): coordinate and xi factors only")
    cert_base = f"Darboux stage ({len(Ps)} candidate factors; {coverage})"

    def fail(cert):
        return {"ok": False, "invariants": [], "cert": cert, "Ps": Ps, "lams": lams,
                "coverage": coverage}
    if len(Ps) < 2:
        return fail(f"no rational invariant from Darboux factors; {cert_base}")
    n_mon = math.comb(len(vars_) + max(1, d_darboux), len(vars_)) + 3
    rows = cofactor_rows(lams, vars_, len(preps), n_mon)
    if not rows:
        return fail(f"cofactor sampling failed (inconclusive); {cert_base}")
    K = int_kernel(np.array(rows, dtype=object))
    if K.shape[1] == 0:
        return fail(f"no rational invariant with Darboux factors of degree <= {d_darboux}; {cert_base}")
    Pstr = [str(P).replace("**", "^") for P in Ps]
    inv = []
    for j in range(K.shape[1]):
        parts = []
        for l in np.nonzero(K[:, j])[0]:
            base = _paren(Pstr[l])
            parts.append(base if K[l, j] == 1 else f"{base}^{int(K[l, j])}")
        inv.append("*".join(parts))
    return {"ok": True, "invariants": inv, "cert": cert_base, "Ps": Ps, "lams": lams,
            "coverage": coverage}


EXP_WORK_CAP = 3250


EXP_SIZE_CAP = 250


def _moved_vars(preps):
    moved = sym_sort(list(dict.fromkeys(v for pr in preps for v in pr["support"])))
    vars_ = sym_sort(list(dict.fromkeys(moved + [v for pr in preps for v in pr["vars"]])))
    return moved, vars_


def exp_factors(preps, darb, d_exp):
    """Exponential factors exp(g/h) and invariants combining them with Darboux factors."""
    from ..gfp import mono_table
    moved, vars_ = _moved_vars(preps)
    K = len(preps)
    D = max([1] + [1 if pr["degree"] is None else int(pr["degree"]) for pr in preps])
    loc = red_locals([x for pr in preps for x in pr["comps"].values()] + vars_)
    xi_of = [{v: red_sympify(x, loc) for v, x in pr["comps"].items()} for pr in preps]
    zero = spy.Integer(0)
    h_c = [{"h": spy.Integer(1), "cof": [zero] * K, "deg": 0}]
    h_c2 = []
    for j, P in enumerate(darb["Ps"]):
        try:
            dj = int(spy.total_degree(P))
        except Exception:
            dj = None
        if dj is None or dj < 1:
            continue
        h_c.append({"h": P, "cof": darb["lams"][j], "deg": dj})
        h_c2.append({"h": spy.expand(P * P), "cof": [spy.Integer(2) * l for l in darb["lams"][j]],
                     "deg": 2 * dj})
    h_c = h_c + h_c2
    m_idx = [vars_.index(m) for m in moved]
    expts_gs = []
    for dg in range(1, max(1, int(d_exp)) + 1):
        e = mono_table(len(vars_), dg)
        expts_gs.append(e[(e[:, m_idx].sum(axis=1) > 0) | (e.sum(axis=1) == 0)])
    factors, seen = [], set()
    size_skipped, capped, work = 0, False, 0
    for hc in h_c:
        for dg in reversed(range(len(expts_gs))):
            expts_g = expts_gs[dg]
            Ng = expts_g.shape[0]
            deg_mu = min(D - 1, (dg + 1) + D - 1 - hc["deg"])
            expts_mu = mono_table(len(vars_), deg_mu) if deg_mu >= 0 else np.zeros((0, len(vars_)), dtype=np.int64)
            Nmu = expts_mu.shape[0]
            n = Ng + K * Nmu
            if n <= EXP_SIZE_CAP:
                break
        if n > EXP_SIZE_CAP:
            size_skipped += 1
            continue
        if work + n > EXP_WORK_CAP:
            capped = True
            break
        work += n
        h_str = str(hc["h"])
        lam_str = [str(c) for c in hc["cof"]]

        def rows_at(pt, p, expts_g=expts_g, expts_mu=expts_mu, Ng=Ng, Nmu=Nmu, n=n,
                   h_str=h_str, lam_str=lam_str):
            eta = eta_rows(preps, vars_, pt, p)
            if eta is None:
                return None
            hl = eval_batch([h_str] + lam_str, pt, p)
            if any(x is None for x in hl):
                return None
            ptv = [int(pt[v]) % p for v in vars_]
            mv_g = np.asarray(_core.sym_mono_residues(expts_g, ptv, p), dtype=np.int64)
            mv_mu = np.asarray(_core.sym_mono_residues(expts_mu, ptv, p), dtype=np.int64) if Nmu else None
            inner = (expts_g @ eta.T) % p
            mu_neg = (p - mulmod(mv_mu, hl[0], p)) % p if Nmu else None
            out = []
            for g in range(K):
                gcols = (mulmod(mv_g, inner[:, g], p) - mulmod(mv_g, hl[1 + g], p)) % p
                mucols = np.zeros(K * Nmu, dtype=np.int64)
                if Nmu:
                    mucols[g * Nmu:(g + 1) * Nmu] = mu_neg
                out.append(np.concatenate([gcols, mucols]))
            return np.stack(out)
        ns = modular_nullspace(rows_at, n, vars_)
        if ns["basis"] is None or ns["basis"].shape[1] == 0:
            continue
        for jc in range(ns["basis"].shape[1]):
            gvec = ns["basis"][:Ng, jc]
            if not np.any(gvec):
                continue
            g_expr = red_sympify(poly_string(gvec, expts_g, vars_), loc)
            q = spy.cancel(g_expr / hc["h"])
            if q.is_number:
                continue
            if hc["deg"] > 0:
                try:
                    if q.is_polynomial():
                        continue
                except Exception:
                    pass
            mu = []
            for g in range(K):
                if not Nmu:
                    mu.append(zero)
                    continue
                mv = ns["basis"][Ng + g * Nmu:Ng + (g + 1) * Nmu, jc]
                mu.append(zero if not np.any(mv) else red_sympify(poly_string(mv, expts_mu, vars_), loc))
            ok_all = True
            for g in range(K):
                Xg = apply_x(xi_of[g], g_expr, loc)
                if str(spy.cancel(spy.together(Xg - g_expr * hc["cof"][g] - mu[g] * hc["h"]))) != "0":
                    ok_all = False
                    break
            if not ok_all:
                continue
            key = str(q)
            if key in seen:
                continue
            seen.add(key)
            factors.append({"g": g_expr, "h": hc["h"], "mu": mu})
    cert_base = (f"exp stage (numerator degree <= {d_exp} over {len(h_c)} denominator candidate(s)"
                 + (f", capped at {EXP_WORK_CAP} unknowns" if capped else "")
                 + (f", {size_skipped} skipped over the {EXP_SIZE_CAP}-unknown budget" if size_skipped else "")
                 + f"): {len(factors)} exponential factor(s)")
    if not factors:
        return {"ok": False, "invariants": [], "factors": [],
                "cert": f"no exponential factor with numerator degree <= {d_exp}; {cert_base}"}
    J = len(darb["Ps"])
    cols = list(darb["lams"]) + [f["mu"] for f in factors]
    n_mon = math.comb(len(vars_) + max(1, d_exp + D - 1), len(vars_)) + 3
    rows = cofactor_rows(cols, vars_, K, n_mon)
    if not rows:
        return {"ok": False, "invariants": [], "factors": factors,
                "cert": f"cofactor sampling failed (inconclusive); {cert_base}"}
    Kk = int_kernel(np.array(rows, dtype=object))
    use_col = [j for j in range(Kk.shape[1]) if np.any(Kk[J:J + len(factors), j] != 0)]
    if not use_col:
        return {"ok": False, "invariants": [], "factors": factors,
                "cert": f"no invariant combining the exponential factors; {cert_base}"}
    Pstr = [str(P).replace("**", "^") for P in darb["Ps"]]
    pos = positive_set()
    inv = []
    for j in use_col:
        w = Kk[:, j]
        parts = []
        for l in np.nonzero(w[:J])[0]:
            base = _paren(Pstr[l])
            parts.append(base if w[l] == 1 else f"{base}^{int(w[l])}")
        arg = spy.Integer(0)
        for e2 in np.nonzero(w[J:J + len(factors)])[0]:
            arg = arg + spy.Integer(int(w[J + e2])) * factors[e2]["g"] / factors[e2]["h"]
        arg = spy.cancel(spy.together(arg))
        arg_str = str(arg).replace("**", "^")
        s_arg = sgn(arg) if not parts else 0
        real_arg = not parts and pos is not None and bool(set(free_syms(arg)) - set(pos))
        if s_arg == 1 or (s_arg == 0 and real_arg):
            inv.append(arg_str)
        elif s_arg == -1:
            inv.append(str(spy.cancel(-arg)).replace("**", "^"))
        else:
            inv.append("*".join(parts + [f"exp({arg_str})"]))
    return {"ok": True, "invariants": inv, "cert": cert_base, "factors": factors}


def int_factor(preps, darb, ex_factors):
    """A Darboux-form integrating factor and the first integral by quadrature."""
    moved, vars_ = _moved_vars(preps)
    loc = red_locals([x for pr in preps for x in pr["comps"].values()] + vars_)
    xi = {v: red_sympify(preps[0]["comps"][v], loc) for v in moved}
    div_x = spy.expand(sum(spy.diff(xi[v], spy.Symbol(v)) for v in moved))
    J, E = len(darb["Ps"]), len(ex_factors)
    cols = list(darb["lams"]) + [f["mu"] for f in ex_factors] + [[div_x]]
    n_mon = math.comb(len(vars_) + 2, len(vars_)) + 3
    rows = cofactor_rows(cols, vars_, 1, n_mon)
    cert_base = f"integrating-factor stage ({J} Darboux + {E} exponential factor(s))"
    if not rows:
        return {"ok": False, "invariants": [], "cert": f"cofactor sampling failed (inconclusive); {cert_base}"}
    Kk = int_kernel(np.array(rows, dtype=object))
    last = J + E
    use_col = [j for j in range(Kk.shape[1]) if Kk[last, j] != 0]
    if not use_col:
        return {"ok": False, "invariants": [], "cert": f"no Darboux-form integrating factor; {cert_base}"}
    Pstr = [str(P).replace("**", "^") for P in darb["Ps"]]
    inv = []
    for j in use_col:
        w = Kk[:, j]
        w_l = int(w[last])
        parts = [f"{_paren(Pstr[l])}^({int(w[l])}/{w_l})" for l in np.nonzero(w[:J])[0]]
        exp_part = []
        if E and np.any(w[J:J + E] != 0):
            arg = spy.Integer(0)
            for e2 in np.nonzero(w[J:J + E])[0]:
                arg = arg + spy.Rational(int(w[J + e2]), w_l) * ex_factors[e2]["g"] / ex_factors[e2]["h"]
            exp_part = [f"exp({spy.cancel(spy.together(arg))})"]
        Mstr = "*".join(parts + exp_part) or "1"
        try:
            Iv = integrating_factor_integral(Mstr.replace("^", "**"), str(xi[moved[0]]),
                                              str(xi[moved[1]]), moved[0], moved[1])
        except Exception:
            Iv = None
        if Iv is not None:
            inv.append(str(Iv).replace("**", "^"))
            break
    if not inv:
        return {"ok": False, "invariants": [], "cert": f"integrating factor found, quadrature failed; {cert_base}"}
    return {"ok": True, "invariants": inv, "cert": f"{cert_base}: solved by quadrature"}


def separable(preps):
    """Invariants of generators whose characteristic system decouples into quadratures."""
    moved = sym_sort(list(dict.fromkeys(v for pr in preps for v in pr["support"])))
    loc = red_locals([x for pr in preps for x in pr["comps"].values()] + moved)
    xi_of = [{v: red_sympify(x, loc) for v, x in pr["comps"].items()} for pr in preps]
    mset = set(moved)
    sep_idx = []
    for g, pr in enumerate(preps):
        nm = list(pr["comps"])
        if len(nm) >= 2 and all(not ((set(free_syms(xi_of[g][v])) & mset) - {v}) for v in nm):
            sep_idx.append(g)
    if not sep_idx:
        return {"ok": False, "invariants": [], "cert": "no generator has a separable characteristic system"}
    cand = []
    for g in sep_idx:
        nm = list(preps[g]["comps"])
        G = {}
        for v in nm:
            try:
                G[v] = spy.integrate(spy.Integer(1) / xi_of[g][v], red_sympify(v, loc))
            except Exception:
                G[v] = None
        nm = [v for v in nm if G[v] is not None]
        if len(nm) < 2:
            continue
        for j in range(1, len(nm)):
            Dd = G[nm[0]] - G[nm[j]]
            try:
                e = spy.simplify(spy.exp(Dd)) if Dd.atoms(spy.log) else spy.cancel(spy.together(Dd))
                cand.append(str(e).replace("**", "^"))
            except Exception:
                pass
    cand = list(dict.fromkeys(c for c in cand if c and c != "1"))
    cert = (f"separable-characteristics stage ({len(sep_idx)} of {len(preps)} generator(s) separable, "
            f"{len(cand)} quadrature candidate(s))")
    if not cand:
        return {"ok": False, "invariants": [], "cert": f"no closed-form quadrature; {cert}"}
    try:
        keep = [bool(x) for x in verify_invariants([c.replace("^", "**") for c in cand],
                                                     [dict(pr["comps"]) for pr in preps])]
    except Exception:
        keep = None
    if keep is None or not any(keep):
        return {"ok": False, "invariants": [],
                "cert": f"quadrature integrals are not invariants of every generator; {cert}"}
    return {"ok": True, "invariants": [c for c, k in zip(cand, keep) if k],
            "cert": f"{cert}: {sum(keep)} verified"}
