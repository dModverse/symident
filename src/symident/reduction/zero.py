"""Compatibility with states held at zero."""

import re

import sympy as spy

from ..rexpr import sym_sort
from .expr import first_subsets, free_syms, red_locals, red_sympify, tri_solve
from .sign import sgn, sgn_poly


def ineq_str(e):
    """`e` > 0 as positive terms > negated negative terms, or None when all terms share a sign."""
    try:
        ex = spy.expand(e)
        tm = [str(x).replace("**", "^") for x in spy.Add.make_args(ex)]
    except Exception:
        return None
    if not tm:
        return None
    neg = [t.startswith("-") for t in tm]
    if all(neg) or not any(neg):
        return None
    return (" + ".join(t for t, n in zip(tm, neg) if not n) + " > " +
            " + ".join(re.sub(r"^-", "", t) for t, n in zip(tm, neg) if n))


def condition(e):
    """Sign of `e` on the domain, else the inequality under which it is positive; both None when undecided."""
    na = {"sign": None, "cond": None}
    try:
        fr = spy.fraction(spy.cancel(spy.together(e)))
    except Exception:
        return na
    try:
        if spy.expand(fr[0]).is_zero:
            return {"sign": 0, "cond": None}
    except Exception:
        pass
    try:
        p = spy.expand(fr[0] * fr[1])
        fl = spy.factor_list(p)
    except Exception:
        return na
    s = sgn_poly(fl[0])
    if s == 0:
        return na
    keep = []
    for f, k in fl[1]:
        try:
            k = int(k)
        except Exception:
            return na
        if k % 2 == 0:
            continue
        sf = sgn(f)
        if sf == 1:
            continue
        if sf == -1:
            s = -s
            continue
        keep.append(f)
    if not keep:
        return {"sign": s, "cond": None}
    q = spy.Integer(1)
    for f in keep:
        q = q * f
    if s < 0:
        q = q * spy.Integer(-1)
    sq = sgn_poly(q)
    if sq != 0:
        return {"sign": sq, "cond": None}
    return {"sign": None, "cond": ineq_str(q)}


def zero_compat(b, fixed=(), verbose=False):
    """Which coordinate sets the orbit can drive to zero with nothing diverging."""
    supp = [v for v in b["support"] if v not in set(fixed)]
    invs = b.get("invariants") or []
    if b.get("status") == "fixed" or not supp or not invs:
        return None
    loc = red_locals(list(invs) + supp)
    try:
        Ies = [red_sympify(iv.replace("^", "**"), loc) for iv in invs]
        fr = [spy.fraction(spy.cancel(spy.together(e))) for e in Ies]
    except Exception:
        return None
    zero = spy.Integer(0)
    pv = {v: f"si_zero{i + 1}" for i, v in enumerate(supp)}
    certain = b.get("target") is None or len(invs) >= b["target"]
    comp_loc = red_locals([x for pr in b["preps"] for x in pr["comps"].values()]) if b.get("preps") else None

    def darboux_(v):
        if b.get("preps") is None:
            return True
        for pr in b["preps"]:
            if v not in pr["comps"]:
                continue
            try:
                e = red_sympify(pr["comps"][v], comp_loc)
                if not spy.expand(e.subs([(spy.Symbol(v), zero)])).is_zero:
                    return False
            except Exception:
                return False
        return True

    def face_solve(eqs, unk):
        if not unk:
            return [{}]
        if len(eqs) >= len(unk):
            return tri_solve(eqs, unk)
        out = []
        for cols in first_subsets(len(unk), len(eqs), 12):
            out += tri_solve(eqs, [unk[c] for c in cols])
        return out

    def face_eqs(Z):
        other = [u for u in supp if u not in Z]
        subs_v = [(spy.Symbol(u), spy.Symbol(pv[u])) for u in other] + [(spy.Symbol(u), zero) for u in Z]
        unk = [pv[u] for u in other]
        eqs, dens = [], []
        for num, den in fr:
            try:
                E = spy.expand(num.subs(subs_v) * den - num * den.subs(subs_v))
            except Exception:
                return None
            if E.is_zero:
                continue
            if not set(unk) & set(free_syms(E)):
                return "blocked"
            eqs.append(E)
            dens.append(den.subs(subs_v))
        return {"eqs": eqs, "dens": dens, "unk": unk, "other": other}

    def analyse(Z):
        f = face_eqs(Z)
        if f is None:
            return {"verdict": "unknown"}
        if f == "blocked":
            return {"verdict": "no"}
        try:
            sols = face_solve(f["eqs"], f["unk"])
        except Exception:
            sols = []
        grow, unclear = [], False
        for br in sols:
            free = [br.get(u) is None for u in f["unk"]]
            vals = {u: (spy.Symbol(u) if br.get(u) is None else br[u]) for u in f["unk"]}
            pairs = [(spy.Symbol(u), vals[u]) for u in f["unk"]]

            def vanishes(e):
                try:
                    return spy.simplify(e.subs(pairs)).is_zero is True
                except Exception:
                    return False
            if not all(vanishes(e) for e in f["eqs"]):
                continue
            cnd, dead, opn = [], [], False
            for i, u in enumerate(f["unk"]):
                s = condition(vals[u])
                if s["sign"] == 1:
                    continue
                if s["sign"] is not None:
                    dead.append(f["other"][i])
                    continue
                free_u = [w for w, fr_ in zip(f["unk"], free) if fr_]
                if s["cond"] is None or any(re.search(r"\b" + w + r"\b", s["cond"]) for w in free_u):
                    opn = True
                    continue
                if s["cond"] not in cnd:
                    cnd.append(s["cond"])
            if dead:
                grow = list(dict.fromkeys(grow + dead))
                continue
            if any(vanishes(d) for d in f["dens"]):
                continue
            if opn:
                unclear = True
                continue
            parts = [f"{z} = 0" for z in Z] + \
                [f"{o} = " + ("(free)" if fr_ else str(vals[u]).replace("**", "^"))
                 for o, u, fr_ in zip(f["other"], f["unk"], free)]
            at = ", ".join(parts)
            for u, p in pv.items():
                at = re.sub(r"\b" + p + r"\b", f"{u}'", at)
            return {"verdict": "if" if cnd else "yes", "condition": " & ".join(cnd), "at": at}
        if grow:
            return {"verdict": "grow", "grow": grow}
        return {"verdict": "unknown" if unclear or not sols else "no"}
    rows, seen = [], set()
    for v0 in supp:
        Z = [v0]
        while True:
            a = analyse(Z)
            if a["verdict"] != "grow":
                break
            Z2 = sym_sort(list(dict.fromkeys(Z + a["grow"])))
            if len(Z2) == len(Z):
                a["verdict"] = "no"
                break
            Z = Z2
        key = ", ".join(Z)
        if key in seen:
            continue
        seen.add(key)
        if b.get("face") and key == ", ".join(sym_sort(b.get("transversal") or [])):
            continue
        rows.append({"coordinates": key, "verdict": a["verdict"],
                     "limit": any(darboux_(z) for z in Z) if a["verdict"] in ("yes", "if") else None,
                     "certain": certain or a["verdict"] == "no",
                     "condition": a.get("condition") or "", "at": a.get("at") or ""})
    if b.get("face") and b.get("transversal"):
        Z = sym_sort(b["transversal"])
        sm = b.get("survivor_meaning") or {}
        pins_inv = {v: k for k, v in (b.get("pins") or {}).items()}
        parts = [f"{z} = 0" for z in Z] + [f"{pins_inv.get(cn, cn)} = {m}" for cn, m in sm.items()]
        rows = [{"coordinates": ", ".join(Z), "verdict": "yes", "limit": False, "certain": True,
                 "condition": "", "at": ", ".join(parts)}] + rows
    return rows or None
