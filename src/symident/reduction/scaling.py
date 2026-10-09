"""Scaling directions gauged through the integer weight lattice: transversals and blocks."""

import itertools
import math
import re
import warnings

import numpy as np

from .. import state
from ..gfp import PRIMES, rref_modp
from ..rexpr import sym_sort
from ..sym.linalg import exact_int_kernel


def _canon(syms):
    return [d for d in syms]


def red_coordinates(obj):
    """The coordinates of the detection result, else those moved by some direction, with a warning."""
    co = obj["info"].get("coordinates")
    if co:
        return [str(c) for c in co]
    warnings.warn("this result has no coordinate list; the emitted "
                  "trafo covers only coordinates appearing in some direction.", stacklevel=2)
    return sym_sort(list(dict.fromkeys(c for d in obj["symmetries"] for c in coords_of(d))))


def coords_of(d):
    """The coordinates of direction `d`: its generator entries, else its support."""
    return list(d["generator"]) if d.get("generator") is not None else list(d.get("support") or [])


def weight_rows(syms):
    """Integer weight rows of the scalings; symbolic weights are listed apart."""
    rows, symbolic, wlist = [], [], []
    for i, d in enumerate(syms):
        if not (d.get("type") == "scaling" and d.get("weights") is not None):
            continue
        try:
            w = [float(x) for x in d["weights"].values()]
            bad = any(x != round(x) for x in w)
        except (TypeError, ValueError):
            bad = True
        if bad:
            symbolic.append(i)
        else:
            rows.append(i)
            wlist.append({k: int(float(v)) for k, v in d["weights"].items()})
    cols = list(dict.fromkeys(k for w in wlist for k in w))
    W = np.zeros((len(wlist), len(cols)), dtype=np.int64)
    for r, w in enumerate(wlist):
        for k, v in w.items():
            W[r, cols.index(k)] = v
    return {"W": W, "cols": cols, "rows": rows, "symbolic": symbolic}


def rank_modp(M, p=PRIMES[0]):
    """Rank of `M` mod `p`."""
    M = np.asarray(M)
    if M.size == 0 or M.ndim != 2 or M.shape[0] == 0 or M.shape[1] == 0:
        return 0
    return rref_modp(np.asarray(M, dtype=np.int64) % p, p)["rank"]


def int_kernel(M):
    """Primitive integer kernel basis of an integer matrix (columns)."""
    M = np.asarray(M, dtype=object)
    nc = M.shape[1] if M.ndim == 2 else 0
    if nc == 0:
        return np.zeros((0, 0), dtype=np.int64)
    ker = exact_int_kernel([[int(x) for x in row] for row in M], nc)
    if not ker:
        return np.zeros((nc, 0), dtype=np.int64)
    return np.array([[int(x) for x in k] for k in ker], dtype=np.int64).T


def primitive_rows(M):
    """`M` with each row divided by the gcd of its entries."""
    M = np.asarray(M, dtype=np.int64)
    if M.shape[0] == 0:
        return M
    out = M.copy()
    for i in range(M.shape[0]):
        nzv = [abs(int(x)) for x in M[i] if x != 0]
        g = 0
        for x in nzv:
            g = math.gcd(g, x)
        if g:
            out[i] = M[i] // g
    return out


def mono_string(a, vars_):
    """The monomial with exponents `a` in `vars_` as a string, "1" for the constant."""
    terms = [(v if int(e) == 1 else f"{v}^{int(e)}") for e, v in zip(a, vars_) if int(e) != 0]
    return "*".join(terms) if terms else "1"


def scaling_fixed(W, cols, fixed):
    """The scalings of `W` that leave `fixed` unmoved, with the count removed, the redundant fixed
    coordinates and the combinations taken."""
    S = [c for c in fixed if c in cols]
    k = W.shape[0]
    if not S:
        return {"Wres": W, "removed": 0, "redundant": [], "combo": np.eye(k, dtype=np.int64)}
    Sidx = [cols.index(c) for c in S]
    C = int_kernel(W[:, Sidx].T)
    Wres = primitive_rows(C.T @ W) if C.shape[1] else W[:0]
    red, acc_idx = [], []
    for c, ci in zip(S, Sidx):
        if rank_modp(W[:, acc_idx + [ci]]) == rank_modp(W[:, acc_idx]):
            red.append(c)
        else:
            acc_idx.append(ci)
    return {"Wres": Wres, "removed": k - C.shape[1], "redundant": red, "combo": C}


def certify_transversal(W, cols, T):
    """Integer matrix expressing `W` through its columns `T`, or None unless `T` is a transversal."""
    A = W[:, [cols.index(t) for t in T]].astype(float)
    if A.shape[0] != len(T):
        return None
    try:
        M = np.linalg.solve(A, W.astype(float))
    except np.linalg.LinAlgError:
        return None
    Mr = np.round(M)
    if np.max(np.abs(M - Mr)) > 1e-7:
        return None
    Mr = Mr.astype(np.int64)
    if not np.array_equal(W[:, [cols.index(t) for t in T]] @ Mr, W):
        return None
    return Mr


def transversal(Wres, cols, fixed, allowed=None):
    """A certified transversal, the admissible family and the survivor meanings."""
    k = Wres.shape[0]
    supp = [c for j, c in enumerate(cols) if np.any(Wres[:, j] != 0) and c not in set(fixed)]
    supp_all = list(supp)
    if allowed is not None:
        supp = [c for c in supp if c in set(allowed)]
    out = {"T": None, "M": None, "admissible": None, "matroid": None, "survivor_meaning": None}
    if not k:
        return out
    occ = {c: int(np.sum(Wres[:, cols.index(c)] != 0)) for c in supp}
    ordc = sorted(supp, key=lambda c: (occ[c], cols.index(c)))
    enumerate_ = len(supp) <= 30 and math.comb(len(supp), k) <= 200
    if enumerate_:
        combos = [list(c) for c in itertools.combinations(supp, k)]
        adm = [T for T in combos if rank_modp(Wres[:, [cols.index(t) for t in T]]) == k]
        out["admissible"] = adm
        # reduce() asks for a later certified transversal when it collects alternative charts
        skip = int(state.get("transversal_skip") or 0)
        certified = []
        for T in adm:
            M = certify_transversal(Wres, cols, T)
            if M is not None:
                certified.append((T, M))
                if len(certified) > skip:
                    break
        if certified:
            out["T"], out["M"] = certified[min(skip, len(certified) - 1)]
    else:
        T = []
        for cn in ordc:
            if len(T) == k:
                break
            if rank_modp(Wres[:, [cols.index(t) for t in T + [cn]]]) > len(T):
                T.append(cn)
        if len(T) == k:
            M = certify_transversal(Wres, cols, T)
            if M is not None:
                out["T"], out["M"] = T, M
        ref = rref_modp(Wres[:, [cols.index(c) for c in supp]] % PRIMES[0], PRIMES[0])
        out["matroid"] = [[supp[j] for j in np.nonzero(ref["R"][r])[0]] for r in range(ref["rank"])]
    if out["T"] is not None:
        Tn = out["T"]
        sm = {}
        for j in supp_all:
            if j in Tn:
                continue
            ex = [1] + [-int(out["M"][i, cols.index(j)]) for i in range(len(Tn))]
            sm[j] = mono_string(ex, [j] + Tn)
        out["survivor_meaning"] = sm
    return out


def scaling_block(W, cols, labels, fixed, allowed=None):
    """One scaling block: fixed coordinates, invariant lattice, transversal, pins."""
    fx = scaling_fixed(W, cols, fixed)
    Wres = fx["Wres"]
    support = [c for j, c in enumerate(cols) if np.any(W[:, j] != 0)]
    blk = {"labels": labels, "type": "scaling", "support": support, "stage": "transversal",
           "module_combos": None, "removed_by_fixed": fx["removed"],
           "redundant_fixed": fx["redundant"], "certificates": [], "reason": None}
    if Wres.shape[0] == 0:
        blk.update(status="fixed", stage="fixed", invariants=[], transversal=None,
                   gauge_note="removed entirely by the user's fixed coordinates")
        return blk
    K = int_kernel(Wres[:, [cols.index(c) for c in support]])
    blk["inv_exps"] = K
    blk["invariants"] = [mono_string(K[:, j], support) for j in range(K.shape[1])]
    tv = transversal(Wres, cols, fixed, allowed)
    blk.update(transversal=tv["T"], admissible=tv["admissible"], matroid=tv["matroid"],
               survivor_meaning=tv["survivor_meaning"], Wres=Wres, Wcols=cols)
    if tv["T"] is not None:
        blk["status"] = "reduced"
        blk["gauge_note"] = ("each transversal coordinate may be pinned to any nonzero constant "
                             "(representative uses 1); the family ranges over all admissible "
                             "transversals")
        blk["pins"] = {t: "1" for t in tv["T"]}
        blk["certificates"].append("transversal certified exactly: W[,T] invertible with "
                                   "integer W[,T]^-1 W")
    else:
        blk["status"] = "invariant_only"
        blk["reason"] = "no transversal with an integer (Laurent) absorption matrix"
        blk["pins"] = None
    return blk


def components(W):
    """Index groups of the rows of `W`, joined transitively where rows share a nonzero column."""
    k = W.shape[0]
    if not k:
        return []
    parent = list(range(k))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for i in range(k - 1):
        for j in range(i + 1, k):
            if np.any((W[i] != 0) & (W[j] != 0)):
                parent[find(j)] = find(i)
    groups = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(i)
    return [groups[g] for g in sorted(groups)]


_ID = re.compile(r"(?<![0-9.A-Za-z_])[A-Za-z_][A-Za-z0-9_.]*(?![A-Za-z0-9_.])(?!\s*\()")
