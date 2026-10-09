"""Gauge choice: one coordinate per scaling fixed, chosen so the general directions stay narrow."""

import fnmatch
import time
import warnings

import numpy as np

from .. import _core, state
from ..rexpr import eval_numeric


def gauge_state(kernel, coords):
    """Row echelon form of the kernel basis mod p, or None without a kernel over all of `coords`."""
    if kernel is None or kernel.get("N") is None or not np.asarray(kernel["N"]).size:
        return None
    if not all(c in kernel["coords"] for c in coords):
        return None
    N = np.asarray(kernel["N"], dtype=np.int64)
    p = kernel["p"]
    rr = _core.sym_rref_mod((N.T % p).astype(np.float64), p)
    return {"R": np.asarray(rr["R"], dtype=np.float64).reshape(-1, N.shape[0]),
            "piv": np.atleast_1d(rr["piv"]).astype(np.int64), "p": p,
            "coords": list(kernel["coords"])}


def outside_span(W, chosen_idx):
    """Whether each column of `W` lies outside the span of the columns `chosen_idx`."""
    if not chosen_idx:
        return np.any(W != 0, axis=0)
    A = W[:, chosen_idx]
    coef, *_ = np.linalg.lstsq(A, W, rcond=None)
    r = W - A @ coef
    return np.sqrt((r ** 2).sum(axis=0)) > 1e-7 * np.maximum(1, np.sqrt((W ** 2).sum(axis=0)))


def gauge_shape(kernel, coords, count=None):
    """The coordinates each remaining kernel direction moves, as a function of the fixed set."""
    if kernel is None or kernel.get("N") is None or not np.asarray(kernel["N"]).size:
        return None
    if not all(c in kernel["coords"] for c in coords):
        return None
    N = np.asarray(kernel["N"], dtype=np.int64)
    p = kernel["p"]
    B = (N.T % p).astype(np.float64)
    names = list(kernel["coords"])

    def shape(cs):
        cs = list(cs)
        rest = [c for c in names if c not in set(cs)]
        order = cs + rest
        rr = _core.sym_rref_mod(B[:, [names.index(c) for c in order]], p)
        rk = int(rr["rank"])
        if not rk:
            return []
        R = np.asarray(rr["R"]).reshape(rk, len(order))
        free = np.atleast_1d(rr["piv"]) >= len(cs)
        cols = [i for i, c in enumerate(order) if count is None or c in set(count)]
        return [int(x) for x in (R[free][:, cols] != 0).sum(axis=1)]
    return shape


def _coords_of(d):
    return list(d["generator"]) if d.get("generator") is not None else list(d.get("support") or [])


def gauge_suggest(syms, kernel, coordinates, pref=None, fixed=()):
    """The suggested scaling gauge with the sizes of the general directions it leaves, and the plain choice
    for comparison; None without both scalings and general directions."""
    scal = [d for d in syms if d["type"] == "scaling"]
    gen = [d for d in syms if d["type"] != "scaling"]
    if not scal or not gen:
        return None
    shape = gauge_shape(kernel, list(coordinates), count=list(coordinates))
    if shape is None:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        best = choose_gauge(scal, None, fixed=fixed, kernel=kernel)
        plain = choose_gauge(scal, pref, fixed=fixed,
                             avoid=list(dict.fromkeys(c for d in gen for c in _coords_of(d))))
    return {"gauge": best, "sizes": sorted(shape(best)), "plain": plain,
            "plain_sizes": sorted(shape(plain))}


def choose_gauge(scal, pref, avoid=(), fixed=(), kernel=None):
    """A basis of the scaling weights, one coordinate per scaling, in preference order."""
    if not scal:
        return []
    fixed = set(fixed)
    coords = [c for c in dict.fromkeys(k for d in scal for k in d["weights"]) if c not in fixed]
    W = np.zeros((len(scal), len(coords)))
    for i, d in enumerate(scal):
        for k, v in d["weights"].items():
            if k in coords:
                W[i, coords.index(k)] = eval_numeric(str(v))
    pref = list(pref or [])
    pref_rank = [len(pref) + 1] * len(coords)
    for i in reversed(range(len(pref))):
        for j, c in enumerate(coords):
            if fnmatch.fnmatchcase(c, pref[i]):
                pref_rank[j] = i + 1
    avoid = set(avoid)
    order = sorted(range(len(coords)), key=lambda j: (pref_rank[j], coords[j] in avoid, j))
    chosen = []
    st = gauge_state(kernel, coords)
    if st is not None:
        while len(chosen) < W.shape[0]:
            ci = [coords.index(c) for c in chosen]
            os_ = outside_span(W, ci)
            ok = [j for j in range(len(coords)) if coords[j] not in chosen and os_[j]]
            if not ok:
                break
            top = min(pref_rank[j] for j in ok)
            cand = [coords[j] for j in ok if pref_rank[j] == top]
            score = _core.sym_gauge_scores(st["R"], st["piv"], st["p"],
                                         [st["coords"].index(c) for c in cand])
            k = cand[int(np.argmin(np.asarray(score)))]
            fx = _core.sym_gauge_fix(st["R"], st["piv"], st["p"], st["coords"].index(k))
            st["R"] = np.asarray(fx["R"], dtype=np.float64)
            st["piv"] = np.atleast_1d(fx["piv"]).astype(np.int64)
            chosen.append(k)
    else:
        for j in order:
            cand = chosen + [coords[j]]
            if np.linalg.matrix_rank(W[:, [coords.index(c) for c in cand]], tol=1e-7) > len(chosen):
                chosen = cand
            if len(chosen) == W.shape[0]:
                break
    if len(chosen) < W.shape[0]:
        warnings.warn(f"the scaling weights have rank {len(chosen)} for "
                      f"{W.shape[0]} scalings; the gauge fixes {len(chosen)}.", stacklevel=2)
    return chosen


def gauged_detection(args, pref, fixed, reconstruct, verbose):
    """Scalings first, then the general directions in a gauge of the scalings."""
    from .core import symmetry_detection
    args = dict(args, gauge_preference=False, verbose=False)
    t0 = time.monotonic()
    a1 = dict(args, reconstruct=False)
    with state.options(keep_kernel=True, stacked_quick=True):
        s1 = symmetry_detection(**a1)

    def split(s):
        return ([d for d in s["symmetries"] if d["type"] == "scaling"],
                [d for d in s["symmetries"] if d["type"] != "scaling"])
    scal, gen = split(s1)
    if s1["info"].get("rank_proven") is None and not s1["info"].get("lie_certified") and \
            (not (gen and reconstruct) or s1["info"].get("stacked")):
        with state.options(keep_kernel=True, stacked_first=False):
            s1 = symmetry_detection(**a1)
        scal, gen = split(s1)
    gauge = choose_gauge(scal, pref, avoid=list(dict.fromkeys(s for d in gen for s in d["support"])),
                         fixed=list(fixed or []), kernel=s1["info"]["kernel"] if gen else None)
    if pref and gen:
        coords_i = list(s1["info"]["coordinates"])
        shape = gauge_shape(s1["info"]["kernel"], coords_i, count=coords_i)
        sug = gauge_suggest(s1["symmetries"], s1["info"]["kernel"], coords_i, fixed=list(fixed or []))
        if shape is not None and sug is not None and max(shape(gauge) or [0]) > max(sug["sizes"] or [0]):
            warnings.warn(f"the preferred gauge ({', '.join(gauge)}) leaves "
                          f"general directions on up to {max(shape(gauge))} coordinates; fixing "
                          f"{', '.join(sug['gauge'])} leaves at most {max(sug['sizes'])}, which "
                          "reduce() handles faster.", stacklevel=2)
    res = s1
    s1info = dict(s1["info"])
    res["info"] = dict(s1info, kernel=None, sat_score=None, stacked=None)
    if gen and reconstruct:
        a2 = dict(args, fixed=list(dict.fromkeys(list(fixed or []) + gauge)))
        s1ok = bool(s1info.get("lie_certified")) or s1info.get("rank_proven") is not None
        hint = None
        if s1info.get("lie_block_orders"):
            hint = {"block_orders": [int(x) for x in s1info["lie_block_orders"]],
                    "nt_used": int(s1info["lie_order_used"]), "sat_score": s1info.get("sat_score"),
                    "Mtot": int(s1info.get("gap_order_used") or 0), "certified": s1ok}
        with state.options(order_hint=hint):
            s2 = symmetry_detection(**a2)
        res["symmetries"] = scal + s2["symmetries"]
        res["info"]["verification"] = s2["info"].get("verification")
        if not s1info.get("lie_certified") and s1info.get("rank_proven") is None:
            res["rank"] = int(s2["rank"])
            res["identifiable"] = res["rank"] == res["dim"]
        if res["info"].get("rank_proven") is None and s2["info"].get("rank_proven") is not None:
            res["rank"] = int(s2["rank"])
            res["identifiable"] = res["rank"] == res["dim"]
            res["info"]["rank_proven"] = int(s2["info"]["rank_proven"] + len(scal))
            res["info"]["rank_proof"] = s2["info"].get("rank_proof")
            res["info"]["lie_order_used"] = s2["info"].get("lie_order_used")
    res["gauge"] = gauge
    res["info"]["settings"] = dict(res["info"]["settings"], gauge_preference=pref)
    res["info"]["elapsed"] = time.monotonic() - t0
    return res
