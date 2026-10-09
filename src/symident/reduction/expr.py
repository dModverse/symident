"""Expressions of the reduction: parsing, generator preparation, module reduction and small
algebraic solvers."""

import numpy as np
import sympy as spy

from ..gfp import PRIMES, Pool, invmod, mulmod
from ..rexpr import sym_sort
from ..sym.invariants import module_reduce_replay
from ..sym.modp import eval_rational_mod_batch
from .scaling import _ID, rank_modp


def red_locals(strings):
    """SymPy symbols for the identifiers in `strings`, as locals for sympify."""
    if isinstance(strings, str):
        strings = [strings]
    return {x: spy.Symbol(x) for s in strings for x in _ID.findall(str(s)) if x}


_SYMP = {}


def red_sympify(x, locals_):
    """`x` sympified with `locals_`, memoised."""
    key = str(x)
    hit = _SYMP.get(key)
    if hit is not None:
        return hit
    val = spy.sympify(key, locals=locals_)
    if len(_SYMP) > 4096:
        _SYMP.clear()
    _SYMP[key] = val
    return val


def free_syms(e):
    """The sorted names of the free symbols of `e`."""
    return sym_sort([str(s) for s in e.free_symbols])


def gen_prep(d, pins=None):
    """A curved generator with denominators cleared, as polynomial components."""
    out = {"ok": False, "comps": None, "support": None, "vars": None, "degree": None}
    comp0 = {k: str(v).replace("^", "**") for k, v in d["generator"].items()}
    pins = {k: v for k, v in (pins or {}).items()
            if not (k in comp0 and comp0[k] != "0")}
    loc = red_locals(list(comp0.values()) + list(pins.values()))
    try:
        exprs = {k: red_sympify(x, loc) for k, x in comp0.items()}
    except Exception:
        out["reason"] = "component not parseable by sympy"
        return out
    if pins:
        try:
            subs = [(spy.Symbol(nm), red_sympify(str(v).replace("^", "**"), loc)) for nm, v in pins.items()]
            exprs = {k: e.subs(subs) for k, e in exprs.items()}
        except Exception:
            out["reason"] = "gauge not substitutable"
            return out
    L = spy.Integer(1)
    for e in exprs.values():
        L = spy.lcm(L, spy.fraction(spy.cancel(spy.together(e)))[1])
    exprs = {k: spy.expand(spy.cancel(e * L)) for k, e in exprs.items()}
    for e in exprs.values():
        try:
            if not e.is_polynomial():
                out["reason"] = "component not a rational function"
                return out
        except Exception:
            out["reason"] = "component not a rational function"
            return out
    comps = {k: str(e) for k, e in exprs.items()}
    nz = [k for k in comps if comps[k] != "0"]
    out["ok"] = True
    out["comps"] = {k: comps[k] for k in nz}
    out["support"] = nz
    out["vars"] = sym_sort(list(dict.fromkeys(nz + [s for k in nz for s in free_syms(exprs[k])])))
    degs = []
    for k in nz:
        try:
            degs.append(int(spy.total_degree(exprs[k])))
        except Exception:
            pass
    out["degree"] = max(degs) if degs else None
    return out


def group_by(sets):
    """Index groups of `sets`, joined transitively where sets share an element."""
    k = len(sets)
    if not k:
        return []
    parent = list(range(k))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(k - 1):
        for j in range(i + 1, k):
            if set(sets[i]) & set(sets[j]):
                parent[find(j)] = find(i)
    groups = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(i)
    return [groups[g] for g in sorted(groups)]


def reduce_schedule(V, p):
    """Pivots (row, column) of a greedy elimination of `V` mod `p` with low fill-in, and the final fill."""
    V = np.array(V, dtype=np.int64)
    k, n = V.shape
    used = [False] * k
    sched = []
    for _ in range(k):
        nz_r = np.sum(V != 0, axis=1)
        nz_c = np.sum(V != 0, axis=0)
        best = None
        for r in [i for i in range(k) if not used[i]]:
            for cc in np.nonzero(V[r])[0]:
                if nz_c[cc] < 2:
                    continue
                pinv = invmod(int(V[r, cc]), p)
                delta = 0
                for j in np.nonzero(V[:, cc])[0]:
                    if j == r:
                        continue
                    m = int(V[j, cc]) * pinv % p
                    delta += int(np.sum((V[j] - mulmod(V[r], m, p)) % p != 0)) - int(nz_r[j])
                if best is None or delta < best[2]:
                    best = (r, int(cc), delta)
        if best is None:
            break
        r, cc, _ = best
        piv = int(V[r, cc])
        for j in range(k):
            if j != r and V[j, cc] != 0:
                m = int(V[j, cc]) * invmod(piv, p) % p
                V[j] = (V[j] - mulmod(V[r], m, p)) % p
        used[r] = True
        sched.append((r, cc))
    return {"sched": sched, "fill": int(np.sum(V != 0))}


def eval_batch(exprs, pt, p):
    """Rational expressions at one integer point mod p; None where a denominator vanishes."""
    if not exprs:
        return []
    try:
        v = eval_rational_mod_batch([str(e) for e in exprs], list(pt.keys()),
                                    [[int(x) for x in pt.values()]], int(p))
    except Exception:
        return [None] * len(exprs)
    out = [int(x) for x in np.asarray(v[0]).ravel()]
    return [None if x < 0 else x for x in out]


def module_reduce(preps, labels):
    """Module reduction of a curved block over the function field, replayed exactly."""
    k = len(preps)
    cols = sym_sort(list(dict.fromkeys(c for pr in preps for c in pr["support"])))
    vars_ = sym_sort(list(dict.fromkeys(v for pr in preps for v in pr["vars"])))
    if k < 2:
        return {"preps": preps, "labels_of": [[i] for i in range(k)], "combos": []}
    has = [[j for j, c in enumerate(cols) if c in pr["comps"]] for pr in preps]
    scheds = []
    for p in PRIMES[:2]:
        pool = Pool()
        vals0 = pool(np.arange(1, len(vars_) + 1) + 7)
        pt = dict(zip(vars_, [int(x) for x in vals0]))
        exprs = [preps[r]["comps"][cols[c]] for r in range(k) for c in has[r]]
        vals = [0 if x is None else x for x in eval_batch(exprs, pt, p)]
        V = np.zeros((k, len(cols)), dtype=np.int64)
        off = 0
        for r in range(k):
            for c in has[r]:
                V[r, c] = vals[off]
                off += 1
        scheds.append(reduce_schedule(V, p))
    sched = min(scheds, key=lambda s: s["fill"])["sched"]
    rr = module_reduce_replay([dict(pr["comps"]) for pr in preps], list(cols),
                               [[int(r) + 1, int(c) + 1] for r, c in sched])
    cand = [{"pr": preps[j], "src_idx": [j], "combo": None} for j in range(k)]
    for j in range(k):
        comps = {kk: v for kk, v in (rr["rows"][j] or {}).items()}
        if not comps:
            continue
        pr = gen_prep({"generator": comps})
        if not pr["ok"] or not pr["support"]:
            continue
        Crow = [str(x) for x in rr["combo"][j]]
        src_idx = [i for i, x in enumerate(Crow) if x != "0"]
        if len(src_idx) == 1 and src_idx[0] == j and Crow[j] == "1":
            continue
        combo = " + ".join(f"({Crow[i]})*{labels[i]}" for i in src_idx) + f"  [support {len(pr['support'])}]"
        cand.append({"pr": pr, "src_idx": src_idx, "combo": combo})
    p = PRIMES[0]
    pool = Pool()
    pt = dict(zip(vars_, [int(x) for x in pool(np.arange(1, len(vars_) + 1) + 23)]))

    def vrow(pr):
        out = np.zeros(len(cols), dtype=np.int64)
        hs = [j for j, c in enumerate(cols) if c in pr["comps"]]
        if hs:
            v = eval_batch([pr["comps"][cols[j]] for j in hs], pt, p)
            for j, x in zip(hs, v):
                out[j] = 0 if x is None else x
        return out
    order = sorted(range(len(cand)), key=lambda j: len(cand[j]["pr"]["support"]))
    acc = np.zeros((0, len(cols)), dtype=np.int64)
    out_preps, labels_of, combos = [], [], []
    for j in order:
        if acc.shape[0] == k:
            break
        aug = np.vstack([acc, vrow(cand[j]["pr"])])
        if rank_modp(aug, p) > acc.shape[0]:
            acc = aug
            out_preps.append(cand[j]["pr"])
            labels_of.append(cand[j]["src_idx"])
            if cand[j]["combo"] is not None:
                combos.append(cand[j]["combo"])
    return {"preps": out_preps, "labels_of": labels_of, "combos": combos}


def roots_in(e, v):
    """Roots in `v` of the linear and quadratic factors of the numerator of `e`."""
    sv = spy.Symbol(v)
    try:
        num = spy.fraction(spy.together(e))[0]
        fl = spy.factor_list(num, sv)
    except Exception:
        return []
    out = []
    for fm in fl[1]:
        f = fm[0]
        try:
            dg = int(spy.Poly(f, sv).degree())
        except Exception:
            continue
        if dg < 1 or dg > 2:
            continue
        try:
            out += list(spy.solve(f, sv))
        except Exception:
            pass
    return out


def first_subsets(n, k, m):
    """The first m k-subsets of range(n) in lexicographic order (0-based)."""
    if k > n or k < 0:
        return []
    if k == 0:
        return [[]]
    idx = list(range(k))
    out = []
    while True:
        out.append(list(idx))
        if len(out) >= m:
            break
        i = k - 1
        while i >= 0 and idx[i] == n - k + i:
            i -= 1
        if i < 0:
            break
        idx[i] += 1
        for j in range(i + 1, k):
            idx[j] = idx[j - 1] + 1
    return out


def tri_solve(sys_, vars_):
    """Triangular elimination: roots of one equation in one unknown, substituted, recursed."""
    if not vars_:
        return [{}]
    for i, eq in enumerate(sys_):
        fs = set(free_syms(eq))
        for v in vars_:
            if v not in fs:
                continue
            roots = roots_in(eq, v)
            if not roots:
                continue
            out = []
            for r in roots:
                try:
                    rest = [spy.together(e.subs([(spy.Symbol(v), r)])) for j, e in enumerate(sys_) if j != i]
                except Exception:
                    continue
                for s in tri_solve(rest, [x for x in vars_ if x != v]):
                    s = dict(s)
                    s[v] = spy.cancel(r.subs([(spy.Symbol(k), x) for k, x in s.items()])) if s else r
                    out.append(s)
            if out:
                return out
    return []
