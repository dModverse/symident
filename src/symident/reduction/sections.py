"""Face sections and carriers of the invariant map."""

import itertools
import math
import re

import numpy as np
import sympy as spy

from ..rexpr import sym_sort
from .darboux import apply_x
from .expr import first_subsets, free_syms, red_locals, red_sympify, tri_solve
from .scaling import mono_string
from .sign import _num_val, positive_set, sgn


def mono_parts(m):
    """Coefficient and exponents of the monomial string `m`, or None unless it is a monomial with positive powers."""
    coef, ex = 1, {}
    for f in [x.strip() for x in m.split("*")]:
        if re.match(r"^[0-9]+$", f):
            coef *= int(f)
            continue
        sp = f.split("^")
        if len(sp) > 2 or not re.match(r"^[A-Za-z.][0-9A-Za-z._]*$", sp[0]):
            return None
        try:
            k = int(sp[1]) if len(sp) == 2 else 1
        except ValueError:
            return None
        if k < 1:
            return None
        ex[sp[0]] = ex.get(sp[0], 0) + k
    if not ex:
        return None
    return {"coef": coef, "ex": ex}


def balance_products(p, q):
    """The pairs [p0*q0, p1*q1] and [p0*q1, p1*q0] of monomials, common factors cancelled, where they
    differ; [] unless all four are monomials."""
    parts = [mono_parts(x) for x in list(p) + list(q)]
    if any(x is None for x in parts):
        return []

    def mul(u, v):
        ex = dict(u["ex"])
        for n, k in v["ex"].items():
            ex[n] = ex.get(n, 0) + k
        return {"coef": u["coef"] * v["coef"], "ex": ex}

    def join(u):
        ex = {k: v for k, v in u["ex"].items() if v != 0}
        if not ex:
            return str(u["coef"])
        nms = sym_sort(list(ex))
        mono = mono_string([ex[n] for n in nms], nms)
        return mono if u["coef"] == 1 else f"{u['coef']}*{mono}"
    out = []
    for flip in (False, True):
        l = mul(parts[0], parts[3 if flip else 2])
        r = mul(parts[1], parts[2 if flip else 3])
        for n in list(dict.fromkeys(list(l["ex"]) + list(r["ex"]))):
            g = min(l["ex"].get(n, 0), r["ex"].get(n, 0))
            if g > 0:
                l["ex"][n] -= g
                r["ex"][n] -= g
        ls, rs = join(l), join(r)
        if ls != rs:
            out.append([ls, rs])
    return out


def section_cands(support, extra=()):
    """Monomial balance candidates m1 = m2 for a curved block's gauge section."""
    def key(pr):
        return " = ".join(sym_sort(list(pr)))
    head, groups = [], []
    for tm in extra:
        tm = list(dict.fromkeys(tm))
        if len(tm) < 2:
            continue
        g = [[a, b] for a, b in itertools.combinations(tm, 2)]
        groups.append(g)
        head += g
    if len(groups) > 1:
        for ia, ib in itertools.combinations(range(len(groups)), 2):
            A, B = groups[ia], groups[ib]
            for u in A[:3]:
                for v in B[:3]:
                    head += balance_products(u, v)
            if len(head) >= 40:
                break
    seen_k, h2 = set(), []
    for pr in head:
        k = key(pr)
        if k not in seen_k:
            seen_k.add(k)
            h2.append(pr)
    head = h2
    mons = [(m, 1) for m in support]
    if len(support) > 1:
        mons += [(f"{a}*{b}", 2) for a, b in itertools.combinations(support, 2)]
    if len(mons) < 2:
        return head
    pairs = list(itertools.combinations(range(len(mons)), 2))
    out = [[mons[i][0], mons[j][0]] for i, j in pairs]
    order = sorted(range(len(pairs)), key=lambda k: mons[pairs[k][0]][1] + mons[pairs[k][1]][1])
    out = [out[k] for k in order]
    head_k = {key(pr) for pr in head}
    return head + [pr for pr in out if key(pr) not in head_k]


def inv_summands(Ie):
    """The distinct unsigned terms of the numerator of `Ie`, sorted; [] for fewer than two."""
    try:
        num = spy.expand(spy.fraction(spy.together(Ie))[0])
        tm = [str(x) for x in spy.Add.make_args(num)]
    except Exception:
        return []
    if len(tm) < 2:
        return []
    return sym_sort(list(dict.fromkeys(re.sub(r"^-", "", t.replace("**", "^")) for t in tm)))


def section_monotone(pr, preps, xi_of, loc, pts):
    """Values of the numerator of X(log(m1/m2)) at the points, None unless sign-pure."""
    from ..rexpr import eval_numeric
    ms = [red_sympify(m.replace("^", "**"), loc) for m in pr]
    row, moved = [], False
    for g in range(len(preps)):
        Xm = [apply_x(xi_of[g], m, loc) for m in ms]
        try:
            d = spy.expand(Xm[0] * ms[1] - ms[0] * Xm[1])
        except Exception:
            return None
        if d.is_zero:
            row += [0.0, 0.0]
            continue
        if d.is_number:
            v = _num_val(d)
            if v is None:
                return None
            moved = True
            row += [v, v]
            continue
        try:
            cf = [float(spy.Float(c)) for c in spy.Poly(d).coeffs()]
        except Exception:
            return None
        if not cf or not (all(c > 0 for c in cf) or all(c < 0 for c in cf)):
            return None
        moved = True
        dstr = str(d).replace("**", "^")
        vals = []
        for pt in pts:
            v = eval_numeric(dstr, pt)
            vals.append(v if v is not None and math.isfinite(v) else math.copysign(1.0, cf[0]))
        row += vals
    if not moved:
        return None
    return row


def model_denominators(exprs):
    """The distinct non-constant factors of the denominators of `exprs`."""
    if not exprs:
        return []
    exprs = [str(x).replace("^", "**") for x in exprs]
    loc = red_locals(exprs)
    out, keys = [], set()
    for x in exprs:
        try:
            den = spy.fraction(spy.together(red_sympify(x, loc)))[1]
        except Exception:
            continue
        if not free_syms(den):
            continue
        for fct in spy.Mul.make_args(den):
            if fct.is_Pow:
                fct = fct.base
            if free_syms(fct) and str(fct) not in keys:
                keys.add(str(fct))
                out.append(fct)
    return out


def denominator_syms(exprs):
    """The symbols in the denominators of `exprs`."""
    return list(dict.fromkeys(s for d in model_denominators(exprs) for s in free_syms(d)))


def _bad(e):
    return e is None or e.has(spy.zoo, spy.nan, spy.oo)


def face_section(b, pins, dens=(), max_sets=200):
    """Face section: pin some coordinates to 0, certified to meet every orbit once."""
    out = {"solved": False}
    if not b.get("invariants") or any(re.search(r"exp\(|log\(", s) for s in b["invariants"]):
        return out
    if b.get("face_invariants") is not None and len(b["face_invariants"]) == len(b["invariants"]) \
            and callable(b.get("face_back")):
        b2 = {k: v for k, v in b.items() if k not in ("invariants", "face_invariants", "face_back")}
        b2["invariants"] = b["face_invariants"]
        res = face_section(b2, pins, dens, max_sets)
        if res.get("solved"):
            nm = list(res["meaning"])
            res["meaning"] = dict(zip(nm, b["face_back"](list(res["meaning"].values()))))
        return res
    supp = b["support"]
    r = len(supp) - len(b["invariants"])
    if r < 1 or (b.get("target") is not None and len(b["invariants"]) < b["target"]):
        return out
    z_cand = [s for s in supp if s not in pins]
    if positive_set() is not None:
        z_cand = [s for s in z_cand if s in set(positive_set())]
    if len(z_cand) < r:
        return out
    loc = red_locals(list(b["invariants"]) + supp + list(pins) + list(pins.values()) +
                     [f"{s}_si_f" for s in supp])
    try:
        Ies = [red_sympify(iv.replace("^", "**"), loc) for iv in b["invariants"]]
    except Exception:
        return out
    if pins:
        pin_pairs = [(spy.Symbol(nm), red_sympify(v, loc)) for nm, v in pins.items()]
        Ies = [e.subs(pin_pairs) for e in Ies]

    def zero_ok(sub):
        for Ie in Ies:
            try:
                e = spy.cancel(Ie.subs(sub))
            except Exception:
                return False
            if e.is_zero or _bad(e):
                return False
        return True
    z_cand = [v for v in z_cand if zero_ok([(spy.Symbol(v), spy.Integer(0))])]
    if len(z_cand) < r:
        return out
    sets = first_subsets(len(z_cand), r, max_sets)
    z_ord = list(reversed(z_cand))
    conditional = []
    pos = positive_set()
    for s in sets:
        Z = [z_ord[i] for i in s]
        sub_z = [(spy.Symbol(v), spy.Integer(0)) for v in Z]
        if not zero_ok(sub_z):
            continue
        ok_d = True
        for d in dens:
            try:
                e = spy.expand(d.subs(sub_z))
            except Exception:
                ok_d = False
                break
            if e.is_zero:
                ok_d = False
                break
        if not ok_d:
            continue
        U = [v for v in supp if v not in Z]
        face = sub_z + [(spy.Symbol(v), spy.Symbol(f"{v}_si_f")) for v in U]
        eqs, ok_z = [], True
        for Ie in Ies:
            try:
                Ief = spy.together(Ie.subs(face))
            except Exception:
                Ief = None
            if _bad(Ief):
                ok_z = False
                break
            eqs.append(spy.together(Ief - Ie))
        if not ok_z:
            continue
        brs = tri_solve(eqs, [f"{v}_si_f" for v in U])
        good, n_open = None, 0
        pos_u = [True] * len(U) if pos is None else [v in set(pos) for v in U]
        for br in brs:
            if len(br) < len(U):
                continue
            vals = [spy.cancel(br[f"{v}_si_f"]) for v in U]
            if any(_bad(e) or any(x.endswith("_si_f") for x in free_syms(e)) for e in vals):
                continue
            back = [(spy.Symbol(f"{v}_si_f"), vals[i]) for i, v in enumerate(U)]
            ok_eq = True
            for e in eqs:
                try:
                    fr = spy.fraction(spy.together(e.subs(back)))
                    if not (spy.cancel(fr[0]).is_zero and not spy.cancel(fr[1]).is_zero):
                        ok_eq = False
                        break
                except Exception:
                    ok_eq = False
                    break
            if not ok_eq:
                continue
            sg_all = [sgn(e) for e in vals]
            if any(x == -1 for x in sg_all):
                continue
            if all(x == 1 for x in sg_all) and good is None:
                good = vals
                continue
            n_open += 1
            sg = [1 if not pos_u[i] else x for i, x in enumerate(sg_all)]
            if any(x == -1 for x in sg):
                continue
            if len(brs) == 1:
                cond = []
                for i, e in enumerate(vals):
                    if sg[i] != 0:
                        continue
                    hit = None
                    for k, Ie in enumerate(Ies):
                        try:
                            q = sgn(spy.cancel(e / Ie))
                        except Exception:
                            q = 0
                        if q != 0:
                            hit = (k, q)
                            break
                    cond.append(hit)
                if cond and all(c is not None for c in cond):
                    if len({c[0] for c in cond}) == 1 and len({c[1] for c in cond}) == 1:
                        conditional.append({"face": Z, "inv": cond[0][0], "sign": cond[0][1]})
        if good is None or n_open > 0:
            continue
        id_face = True
        for i, v in enumerate(U):
            try:
                e = spy.cancel(good[i].subs(sub_z) - spy.Symbol(v))
            except Exception:
                id_face = False
                break
            if not e.is_zero:
                id_face = False
                break
        if not id_face:
            continue
        res = dict(out)
        res.update(solved=True, pins={z: "0" for z in Z}, gauge=Z, section=[f"{z} = 0" for z in Z],
                   face=True, coverage="total", carrier_domain={}, inv_names=[], shifted=[],
                   meaning={v: str(spy.factor(e)).replace("**", "^") for v, e in zip(U, good)})
        return res
    for a in conditional:
        for c2 in conditional:
            if a["inv"] == c2["inv"] and a["sign"] == 1 and c2["sign"] == -1:
                out["split"] = {"invariant": str(Ies[a["inv"]]).replace("**", "^"),
                                "positive": [f"{z} = 0" for z in a["face"]],
                                "negative": [f"{z} = 0" for z in c2["face"]]}
                return out
    return out


def face_carriers(sol, coords, inv_start):
    """`sol` with each coordinate the face moves pinned to a fresh carrier q_<k> holding its value there."""
    m = sol["meaning"]
    moved = [u for u, v in m.items() if re.sub(r"\s", "", v) != u and u in set(coords)]
    m = {u: m[u] for u in moved}
    if not moved:
        sol["meaning"] = m
        return sol
    qn, k = [], inv_start
    for _ in moved:
        while True:
            k += 1
            if f"q_{k}" not in set(coords):
                break
        qn.append(f"q_{k}")
    sol["pins"] = dict(sol["pins"], **dict(zip(moved, qn)))
    sol["inv_names"] = qn
    sol["meaning"] = dict(zip(qn, [m[u] for u in moved]))
    sol["carrier_domain"] = {q: "positive" for q in qn}
    return sol


def prep_degree(comps):
    """Maximal total degree of the components `comps`, or None."""
    loc = red_locals(list(comps.values()))
    degs = []
    for x in comps.values():
        try:
            degs.append(int(spy.total_degree(red_sympify(str(x).replace("^", "**"), loc))))
        except Exception:
            pass
    return max(degs) if degs else None


def translation_compress(preps):
    """A moved coordinate entering together with unmoved ones only through their sum."""
    supp = list(dict.fromkeys(c for pr in preps for c in pr["support"]))
    vars_ = list(dict.fromkeys(v for pr in preps for v in pr["vars"]))
    unmoved = [v for v in vars_ if v not in set(supp)]
    if not unmoved:
        return None
    loc = red_locals([x for pr in preps for x in pr["comps"].values()] + vars_)
    E = [{k: red_sympify(str(x).replace("^", "**"), loc) for k, x in pr["comps"].items()} for pr in preps]

    def dz(e, a, b):
        return spy.expand(spy.diff(e, spy.Symbol(a)) - spy.diff(e, spy.Symbol(b))).is_zero

    def seen(v):
        return any(v in free_syms(e) for ex in E for e in ex.values())
    groups, used = {}, set()
    for v in supp:
        if not seen(v):
            continue
        mem = [w for w in unmoved if w not in used and
               all(dz(e, v, w) for ex in E for e in ex.values())]
        if mem:
            groups[v] = mem
            used |= set(mem)
    if not groups:
        return None
    taken = set(vars_) | set(loc)
    U = {}
    for k, v in enumerate(groups, start=1):
        u = f"si_u{k}"
        while u in taken:
            u += "_"
        U[v] = u
    fwd = [(spy.Symbol(v), spy.Symbol(U[v]) - sum(spy.Symbol(w) for w in groups[v])) for v in groups]
    all_mem = {w for m in groups.values() for w in m}
    new_preps = []
    for i, pr in enumerate(preps):
        ex = {k: spy.expand(e.subs(fwd)) for k, e in E[i].items()}
        left = {s for e in ex.values() for s in free_syms(e)}
        if all_mem & left:
            raise RuntimeError("translation group did not separate")
        keys = [U.get(k, k) for k in pr["comps"]]
        comps = dict(zip(keys, [str(e) for e in ex.values()]))
        pr = dict(pr, comps=comps, support=keys, vars=sym_sort(list(dict.fromkeys(keys + list(left)))),
                  degree=prep_degree(comps))
        new_preps.append(pr)
    bwd = [(spy.Symbol(U[v]), sum((spy.Symbol(x) for x in [v] + groups[v]), spy.Integer(0))) for v in groups]
    extra_names = list(U.values()) + list(groups) + [w for m in groups.values() for w in m]

    def back(invs):
        out = []
        for iv in invs:
            l2 = red_locals([iv] + extra_names)
            e = spy.sympify(iv.replace("^", "**"), locals=l2).subs(bwd)
            out.append(str(spy.factor(spy.together(e))).replace("**", "^"))
        return out
    key_back = {u: v for v, u in U.items()}

    def back_prep(pr):
        l2 = red_locals(list(pr["comps"].values()) + extra_names)
        comps = {key_back.get(k, k): str(spy.expand(spy.sympify(str(x).replace("^", "**"), locals=l2).subs(bwd)))
                 for k, x in pr["comps"].items()}
        keys = list(comps)
        vs = {s for x in comps.values() for s in free_syms(spy.sympify(x, locals=l2))}
        return dict(pr, comps=comps, support=keys, vars=sym_sort(list(dict.fromkeys(keys + list(vs)))),
                    degree=prep_degree(comps))
    return {"preps": new_preps, "back": back, "back_prep": back_prep,
            "note": [" + ".join([v] + groups[v]) for v in groups]}


def unmoved_compress(preps):
    """Unmoved coordinates entering only through a few monomials, searched as single symbols."""
    supp = list(dict.fromkeys(c for pr in preps for c in pr["support"]))
    vars_ = list(dict.fromkeys(v for pr in preps for v in pr["vars"]))
    unmoved = sym_sort([v for v in vars_ if v not in set(supp)])
    if len(unmoved) < 2:
        return None
    loc = red_locals([x for pr in preps for x in pr["comps"].values()] + vars_)
    usyms = [spy.Symbol(u) for u in unmoved]
    terms = []
    for pr in preps:
        tp = []
        for x in pr["comps"].values():
            P = spy.Poly(red_sympify(str(x).replace("^", "**"), loc), *usyms)
            tp.append([{"e": [int(a) for a in t[0]], "c": t[1]} for t in P.terms()])
        terms.append(tp)
    all_e = []
    for tp in terms:
        for tl in tp:
            for t in tl:
                if t["e"] not in all_e:
                    all_e.append(t["e"])
    E = np.array([e for e in all_e if sum(e) > 0], dtype=np.int64)
    if E.size == 0:
        return None

    def gcdv(v):
        g = 0
        for x in v:
            g = math.gcd(g, abs(int(x)))
        return g
    prim = []
    for row in E:
        p_ = list(row // gcdv(row))
        if p_ not in prim:
            prim.append(p_)
    prim = sorted(prim, key=sum)
    B = []

    def coef_of(v, B):
        v = np.asarray(v, dtype=float)
        if not B:
            return [] if np.all(v == 0) else None
        Bt = np.array(B, dtype=float).T
        sol, *_ = np.linalg.lstsq(Bt, v, rcond=None)
        rs = np.round(sol)
        if np.any(np.abs(sol - rs) > 1e-9) or np.any(rs < 0) or np.any(np.abs(Bt @ rs - v) > 1e-9):
            return None
        return [int(x) for x in rs]
    for p_ in prim:
        if coef_of(p_, B) is None:
            B.append(p_)
    if len(B) > 3 or len(B) >= int(np.sum(E.sum(axis=0) > 0)):
        return None
    if any(coef_of(E[i], B) is None for i in range(E.shape[0])):
        return None
    taken = set(vars_) | set(loc)
    W = []
    for k in range(1, len(B) + 1):
        w = f"si_w{k}"
        while w in taken:
            w += "_"
        W.append(w)
    wsyms = [spy.Symbol(w) for w in W]
    used_u = [unmoved[j] for j in range(len(unmoved)) if E[:, j].sum() > 0]
    new_preps = []
    for i, pr in enumerate(preps):
        comps = {}
        for j, key in enumerate(pr["comps"]):
            e = spy.Integer(0)
            for tm in terms[i][j]:
                k = coef_of(tm["e"], B)
                mono = spy.Integer(1)
                for ws, kk in zip(wsyms, k or []):
                    mono = mono * spy.Pow(ws, spy.Integer(kk))
                e = e + tm["c"] * mono
            comps[key] = str(spy.expand(e))
        vs = set(pr["support"]) | (set(pr["vars"]) - set(used_u))
        for x in comps.values():
            vs |= set(free_syms(red_sympify(x, red_locals([x] + W))))
        new_preps.append(dict(pr, comps=comps, degree=prep_degree(comps), vars=sym_sort(list(vs))))
    Bm = np.array(B, dtype=np.int64)
    bwd = []
    for k in range(len(B)):
        e = spy.Integer(1)
        for j in np.nonzero(Bm[k] > 0)[0]:
            e = e * spy.Pow(usyms[j], spy.Integer(int(Bm[k, j])))
        bwd.append((wsyms[k], e))

    def back(invs):
        out = []
        for iv in invs:
            l2 = red_locals([iv] + W + unmoved)
            e = spy.sympify(iv.replace("^", "**"), locals=l2).subs(bwd)
            out.append(str(spy.factor(spy.together(e))).replace("**", "^"))
        return out

    def back_prep(pr):
        l2 = red_locals(list(pr["comps"].values()) + W + unmoved)
        comps = {k: str(spy.expand(spy.sympify(str(x).replace("^", "**"), locals=l2).subs(bwd)))
                 for k, x in pr["comps"].items()}
        vs = set(pr["support"]) | {s for x in comps.values() for s in free_syms(spy.sympify(x, locals=l2))}
        return dict(pr, comps=comps, vars=sym_sort(list(vs)), degree=prep_degree(comps))
    note = []
    for k in range(len(B)):
        parts = [unmoved[j] if Bm[k, j] == 1 else f"{unmoved[j]}^{int(Bm[k, j])}"
                 for j in np.nonzero(Bm[k] > 0)[0]]
        note.append("*".join(parts))
    return {"preps": new_preps, "back": back, "back_prep": back_prep, "note": note}


def match_carriers(adm):
    """One distinct carrier per entry from its admissible list in `adm`, or None."""
    n = len(adm)
    pick = [None] * n

    def go(l, used):
        if l >= n:
            return True
        for v in adm[l]:
            if v in used:
                continue
            pick[l] = v
            if go(l + 1, used | {v}):
                return True
        return False
    return pick if go(0, frozenset()) else None
