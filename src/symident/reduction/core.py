"""The reduction of a detection result: scalings, curved blocks, charts and the result."""

import math
import re
import time
import warnings

import numpy as np

from .. import state
from ..gfp import PRIMES
from ..rexpr import sym_sort
from ..sym.charts import (
    exp_chart_back,
    exp_chart_generators,
    log_chart_back,
    log_chart_generators,
)
from .blocks import curved, verify_blocks
from .scaling import (
    components,
    coords_of,
    rank_modp,
    red_coordinates,
    scaling_block,
    weight_rows,
)
from .sections import face_carriers, face_section, model_denominators
from .sign import DOMAIN, domain_name, positive_set, positive_set_of, set_positive
from .solve import solve_invariants
from .zero import zero_compat


def assemble_trafo(blocks, coords):
    """The trafo of `coords`: the identity with the pins of the blocks."""
    vals = {c: c for c in coords}
    for b in blocks:
        if b.get("pins"):
            for k, v in b["pins"].items():
                if k in vals:
                    vals[k] = v
    return vals


def log_chart(obj, coords):
    """The result in the chart X = b^theta, with the name maps."""
    gens = [None if d.get("generator") is None else
            {k: str(v).replace("^", "**") for k, v in d["generator"].items()} for d in obj["symmetries"]]
    if not gens or all(g is None for g in gens):
        return None
    try:
        ch = log_chart_generators(gens, list(coords))
    except Exception:
        return None
    if ch is None:
        return None
    th = [str(m["theta"]) for m in ch["map"]]
    X = [str(m["X"]) for m in ch["map"]]
    bs = [str(m["base"]) for m in ch["map"]]

    def ren(v):
        if v is None:
            return v
        return [X[th.index(x)] if x in th else x for x in v]
    o = dict(obj, symmetries=[dict(d) for d in obj["symmetries"]], info=dict(obj["info"]))
    for i, d in enumerate(o["symmetries"]):
        g = ch["gens"][i]
        if g is None:
            d["support"] = ren(d.get("support"))
            continue
        d["generator"] = {k: str(v).replace("**", "^") for k, v in g.items()}
        d["support"] = sym_sort(ren(d.get("support")))
        d.pop("complete_generator", None)
    o["info"]["coordinates"] = ren(list(coords))
    return {"object": o, "theta": th, "X": X, "base": bs, "ren": ren}


def exp_chart(obj, coords, positive):
    """The detection result in the chart L = log(v) with the name maps, or None without such a chart."""
    gens = [None if d.get("generator") is None else
            {k: str(v).replace("^", "**") for k, v in d["generator"].items()} for d in obj["symmetries"]]
    if not gens or all(g is None for g in gens):
        return None
    pos = True if positive is True else ([] if positive is False else list(positive))
    try:
        ch = exp_chart_generators(gens, list(coords), pos)
    except Exception:
        return None
    if ch is None:
        return None
    v = [str(m["v"]) for m in ch["map"]]
    L = [str(m["L"]) for m in ch["map"]]

    def ren(x):
        if x is None or isinstance(x, bool):
            return x
        return [L[v.index(y)] if y in v else y for y in x]
    o = dict(obj, symmetries=[dict(d) for d in obj["symmetries"]], info=dict(obj["info"]))
    for i, d in enumerate(o["symmetries"]):
        g = ch["gens"][i]
        if g is not None:
            d["generator"] = {k: str(x).replace("**", "^") for k, x in g.items()}
            d.pop("complete_generator", None)
        if any(s in set(v) for s in d.get("support") or []):
            d["weights"] = None
            d["type"] = "general"
        d["support"] = sym_sort(ren(d.get("support") or []))
    o["info"]["coordinates"] = ren(list(coords))
    return {"object": o, "v": v, "L": L, "ren": ren}


def _chart_back(res, names_from, names_to, back):
    def unren(x):
        if x is None:
            return x
        return [names_to[names_from.index(y)] if y in names_from else y for y in x]

    def entries(x):
        if not x:
            return x
        return {unren([k])[0]: back(v, k if k in names_from else None) for k, v in x.items()}
    res = dict(res)
    res["coordinates"] = unren(res.get("coordinates"))
    res["fixed"] = unren(res.get("fixed"))
    if res.get("trafo") is not None:
        res["trafo"] = entries(res["trafo"])
    blocks = []
    for b in res["blocks"]:
        b = dict(b)
        b["support"] = unren(b.get("support"))
        b["transversal"] = unren(b.get("transversal"))
        if b.get("pins"):
            b["pins"] = entries(b["pins"])
        if b.get("invariants"):
            b["invariants"] = [back(x) for x in b["invariants"]]
        if b.get("survivor_meaning"):
            b["survivor_meaning"] = {unren([k])[0]: back(v) for k, v in b["survivor_meaning"].items()}
        blocks.append(b)
    res["blocks"] = blocks
    return res


def exp_back(res, ec):
    """The reduction result `res` mapped back from the exp chart `ec`."""
    def back(x, solve_for=None):
        return str(exp_chart_back(str(x).replace("^", "**"), ec["v"], ec["L"], solve_for)).replace("**", "^")
    return _chart_back(res, ec["L"], ec["v"], back)


def log_back(res, lc):
    """The reduction result `res` mapped back from the log chart `lc`."""
    def back(x, solve_for=None):
        return str(log_chart_back(str(x).replace("^", "**"), lc["X"], lc["theta"], lc["base"],
                                   solve_for)).replace("**", "^")
    return _chart_back(res, lc["X"], lc["theta"], back)


_SUB = "₀₁₂₃₄₅₆₇₈₉"


def ordered(obj):
    """Directions grouped by type with labels X1, X2, ..."""
    syms = obj.get("symmetries") or []
    if not syms:
        return {"syms": [], "labels": []}
    rank = {"scaling": 0, "general": 1}
    order = sorted(range(len(syms)), key=lambda i: (rank.get(syms[i]["type"], 2), i))
    syms = [syms[i] for i in order]
    labels = ["X" + "".join(_SUB[int(c)] for c in str(i + 1)) for i in range(len(syms))]
    return {"syms": syms, "labels": labels}


def red_result(obj, blocks, trafo, coords, fixed, settings):
    """The reduction result: blocks, trafo, removed and remaining directions and the zero compatibility."""
    status = [b["status"] for b in blocks]
    van = []
    for b in blocks:
        for r in (b.get("zero_compatibility") or []):
            van.append(dict(r, block=", ".join(b["labels"])))
    family = [{"labels": b["labels"], "admissible": b.get("admissible"), "matroid": b.get("matroid"),
               "gauge_note": b.get("gauge_note")}
              for b in blocks if b["status"] == "reduced" and
              (b.get("admissible") is not None or b.get("matroid") is not None or
               b.get("gauge_note") is not None)]
    removed = [l for b, s in zip(blocks, status) if s in ("fixed", "reduced") for l in b["labels"]]
    remaining = [l for b, s in zip(blocks, status) if s not in ("fixed", "reduced") for l in b["labels"]]
    # a listed direction that no block contains depends on the others
    dependent = [l for l in ordered(obj)["labels"] if l not in set(removed) | set(remaining)]
    return {"method": obj.get("method"), "coordinates": coords, "fixed": fixed, "blocks": blocks,
            "trafo": trafo, "family": family, "removed": removed, "remaining": remaining,
            "dependent": dependent, "zero_compatibility": van or None, "settings": settings}


def _tidy_root(x):
    """A trafo entry with square roots of perfect squares taken, the coordinates positive."""
    if "sqrt" not in str(x):
        return x
    import sympy as spy

    from ..sym.parse import _make_local_parse
    try:
        _, parse = _make_local_parse([str(x)])
        e = parse(str(x).replace("^", "**"))
        pos = {s: spy.Symbol(str(s), positive=True) for s in e.free_symbols}
        y = e.xreplace(pos).replace(lambda a: a.is_Pow and a.exp == spy.Rational(1, 2),
                                    lambda a: spy.sqrt(spy.factor(a.base)))
        y = y.xreplace({v: k for k, v in pos.items()})
    except Exception:
        return x
    out = str(y).replace("**", "^")
    return out if len(out) < len(str(x)) else x


def _rational_chart(res):
    return not any(re.search(r"sqrt|\^\(1/|\^\([0-9]+/", str(v)) for v in (res.get("trafo") or {}).values())


def _chart_valid(res, dim):
    """Whether the trafo leaves exactly dim minus the removed directions free parameters, so
    that no pin was overwritten by a section."""
    if not res.get("trafo") or dim is None:
        return True
    names = {t for v in res["trafo"].values() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(v))}
    names -= {"sqrt", "exp", "log"}
    return len(names) == dim - len(res.get("removed") or [])


def _chart_rank(res):
    """Sort key of a reduction: more directions removed, then a rational chart; ties keep
    the order in which the charts were built."""
    return (-len(res.get("removed") or []), not _rational_chart(res))


def symmetry_reduction(obj, *args, alternatives=False, **kw):
    """Reparametrisation removing the non-identifiable directions of a detection result.

    The first chart stands unless it holds a root that does not simplify, or `alternatives`
    asks for the others: then the charts of the other admissible pins are built too, the
    best one is returned and the rest are listed under `alternatives`."""
    def run(**opts):
        try:
            with state.options(**opts):
                res = _symmetry_reduction(obj, *args, **kw)
        except Exception:
            return None
        if isinstance(res, dict) and res.get("trafo"):
            res["trafo"] = {k: _tidy_root(v) for k, v in res["trafo"].items()}
        return res
    res = run()
    if not isinstance(res, dict) or not res.get("blocks"):
        return res
    if _rational_chart(res) and not alternatives:
        res["alternatives"] = []
        return res
    found = [res]
    for inside in (False, True):
        for skip in range(3):
            if not inside and not skip:
                continue
            alt = run(reduce_pin_inside=inside or None, transversal_skip=skip or None)
            if isinstance(alt, dict) and alt.get("trafo") and _chart_valid(alt, obj.get("dim")) and \
                    all(alt["trafo"] != f.get("trafo") for f in found):
                found.append(alt)
    found.sort(key=_chart_rank)
    best = found[0]
    best["alternatives"] = [{"trafo": f["trafo"], "removed": f["removed"], "remaining": f["remaining"],
                             "rational": _rational_chart(f),
                             "pins": {k: v for b in f["blocks"] for k, v in (b.get("pins") or {}).items()}}
                            for f in found[1:]]
    return best


def _symmetry_reduction(obj, fixed=None, positive=True, d_poly=3, d_darboux=2, d_exp=2, separable_=True,
                        zero_compatibility=False, timeout=600, verbose=False):
    settings = {"d_poly": int(d_poly), "d_darboux": int(d_darboux), "d_exp": int(d_exp),
                "separable": bool(separable_), "primes": list(PRIMES)}
    coords = red_coordinates(obj)
    if fixed is None:
        fixed = obj.get("gauge") or []
    fixed = list(dict.fromkeys(str(f) for f in fixed))
    unknown = [f for f in fixed if f not in set(coords)]
    if unknown:
        warnings.warn(f"no effect: {', '.join(unknown)}: not a coordinate of the analysis.",
                      stacklevel=2)
    pos_set = positive_set_of(positive, coords)
    settings["positive"] = True if pos_set is None else (False if not pos_set else pos_set)
    old_pos = set_positive(pos_set)
    old_dl = DOMAIN["deadline"]
    try:
        if pos_set is not None and zero_compatibility:
            warnings.warn("zero limits are the faces of the positive orthant; "
                          "for a coordinate that is not declared positive, zero is an interior point "
                          "and the verdict does not apply.", stacklevel=2)
        if obj.get("identifiable") or not obj.get("symmetries"):
            return red_result(obj, [], None, coords, fixed, settings)
        lc = log_chart(obj, coords)
        if lc is not None:
            if isinstance(positive, (list, tuple, str)):
                pos2 = ([positive] if isinstance(positive, str) else list(positive)) + lc["X"]
            elif positive is False:
                pos2 = lc["X"]
            else:
                pos2 = positive
            old_nr = DOMAIN["no_roots"]
            DOMAIN["no_roots"] = True
            try:
                res = symmetry_reduction(lc["object"], fixed=lc["ren"](fixed), positive=pos2,
                                         d_poly=d_poly, d_darboux=d_darboux, d_exp=d_exp,
                                         separable_=separable_,
                                         zero_compatibility=zero_compatibility,
                                         timeout=timeout, verbose=verbose)
            finally:
                DOMAIN["no_roots"] = old_nr
            return log_back(res, lc)
        ec = exp_chart(obj, coords, positive)
        if ec is not None:
            if positive is True:
                pos2 = [c for c in ec["object"]["info"]["coordinates"] if c not in set(ec["L"])]
            else:
                pp = [] if positive is False else ([positive] if isinstance(positive, str) else list(positive))
                pos2 = [c for c in ec["ren"](pp) if c not in set(ec["L"])]
            old_nr = DOMAIN["no_roots"]
            DOMAIN["no_roots"] = True
            try:
                res = symmetry_reduction(ec["object"], fixed=ec["ren"](fixed),
                                         positive=pos2 if pos2 else False, d_poly=d_poly,
                                         d_darboux=d_darboux, d_exp=d_exp, separable_=separable_,
                                         zero_compatibility=zero_compatibility,
                                         timeout=timeout, verbose=verbose)
            finally:
                DOMAIN["no_roots"] = old_nr
            return exp_back(res, ec)
        o = ordered(obj)
        wr = weight_rows(o["syms"])
        W, Wc = wr["W"], wr["cols"]
        # a scaling dependent on the earlier ones needs no reduction of its own
        indep, dep_rows = [], []
        for r in range(W.shape[0]):
            if rank_modp(W[indep + [r]].T) > len(indep):
                indep.append(r)
            else:
                dep_rows.append(wr["rows"][r])
        if dep_rows:
            W = W[indep]
            wr = dict(wr, W=W, rows=[wr["rows"][r] for r in indep])
        curved_idx0 = [i for i in range(len(o["syms"])) if i not in set(wr["rows"]) | set(dep_rows)]
        curved_supp0 = list(dict.fromkeys(c for i in curved_idx0 for c in coords_of(o["syms"][i])))

        def demote(outside_ok):
            curved_supp = list(curved_supp0)
            scal_rows = list(range(W.shape[0]))
            demoted, kept = [], []
            if W.shape[0] and curved_supp:
                while True:
                    overlaps = [r for r in scal_rows if any(Wc[j] in curved_supp for j in np.nonzero(W[r])[0])]
                    if not overlaps:
                        break
                    keep = []
                    if outside_ok:
                        out_c = [j for j, c in enumerate(Wc) if c not in set(curved_supp)]
                        acc = np.zeros((0, len(out_c)), dtype=np.int64)
                        for r in overlaps:
                            aug = np.vstack([acc, W[r, out_c]])
                            if out_c and rank_modp(aug.T) > acc.shape[0]:
                                acc = aug
                                keep.append(r)
                    down = [r for r in overlaps if r not in keep]
                    if not down:
                        kept = list(dict.fromkeys(kept + keep))
                        break
                    demoted += down
                    curved_supp = list(dict.fromkeys(curved_supp + [Wc[j] for r in down for j in np.nonzero(W[r])[0]]))
                    scal_rows = [r for r in scal_rows if r not in down]
            allowed = [c for c in Wc if c not in set(curved_supp)] if curved_supp else None
            return {"scal_rows": scal_rows, "demoted": demoted, "curved_supp": curved_supp,
                    "allowed": allowed, "kept": kept}

        def scaling_blocks(dm):
            blocks = []
            if dm["scal_rows"]:
                Wk = W[dm["scal_rows"]]
                for cp in components(Wk):
                    rows = [dm["scal_rows"][c] for c in cp]
                    nzc = [j for j in range(W.shape[1]) if np.any(W[rows][:, j] != 0)]
                    Wb = W[rows][:, nzc]
                    blocks.append(scaling_block(Wb, [Wc[j] for j in nzc],
                                                [o["labels"][wr["rows"][r]] for r in rows], fixed,
                                                dm["allowed"]))
            return blocks
        if state.get("reduce_pin_inside"):
            # an alternative chart: the scalings may pin coordinates of the curved directions
            dm = {"scal_rows": list(range(W.shape[0])), "demoted": [], "curved_supp": curved_supp0,
                  "allowed": None, "kept": []}
            blocks = scaling_blocks(dm)
        else:
            dm = demote(True)
            blocks = scaling_blocks(dm)
            if dm["kept"] and any(b["status"] not in ("reduced", "fixed") for b in blocks):
                dm = demote(False)
                blocks = scaling_blocks(dm)
        curved_idx = sorted(curved_idx0 + [wr["rows"][r] for r in dm["demoted"]])
        scal_pins = {}
        for b in blocks:
            if b["type"] == "scaling" and b.get("pins"):
                scal_pins.update(b["pins"])
        if curved_idx:
            blocks += curved(o["syms"], curved_idx, o["labels"], fixed, d_poly, d_darboux, d_exp,
                             verbose, separable_, scal_pins)
        blocks = verify_blocks(blocks)
        inv_start = 0
        dens = None
        for b in blocks:
            if b["type"] != "curved" or not b.get("invariants"):
                continue
            if b.get("target") is not None and len(b["invariants"]) < b["target"]:
                continue
            if dens is None:
                dens = model_denominators(obj["info"].get("model_exprs") or [])
            DOMAIN["deadline"] = time.monotonic() + timeout if math.isfinite(timeout) else None
            try:
                sol = face_section(b, scal_pins, dens)
            except Exception:
                sol = {"solved": False}
            if sol.get("solved"):
                sol = face_carriers(sol, coords, inv_start)
            split = sol.get("split")
            split_pair = None
            if split is not None:
                split_pair = [re.sub(r" = 0$", "", x) for x in split["positive"] + split["negative"]]
            if not sol.get("solved"):
                sol = solve_invariants(b, scal_pins, coords, inv_start, split_pair=split_pair)
            if not sol.get("solved") and split is not None:
                sol["reason"] = (f"the orbits leave the {domain_name()} through "
                                 f"{', '.join(split['positive'])} where {split['invariant']} > 0 and through "
                                 f"{', '.join(split['negative'])} where it is < 0; no face covers both, and "
                                 f"no section was certified. An assumption on the sign of {split['invariant']} "
                                 "selects the face")
            rooted = sol.get("solved") and sol.get("root_note") is not None
            pos = positive_set()
            if (not sol.get("solved") or rooted) and pos is not None and set(b["support"]) - set(pos):
                sol_r = solve_invariants(b, scal_pins, coords, inv_start, prefer_real=True, split_pair=split_pair)
                if sol_r.get("solved") and (not rooted or sol_r.get("root_note") is None):
                    sol = sol_r
            if sol.get("solved"):
                nums = []
                for q in sol.get("inv_names") or []:
                    try:
                        nums.append(int(q[2:]))
                    except ValueError:
                        pass
                inv_start = max([inv_start + len(sol.get("inv_names") or [])] + nums)
                b["pins"] = sol["pins"]
                b["transversal"] = sol.get("gauge")
                b["section"] = sol.get("section")
                b["face"] = bool(sol.get("face"))
                gauge_val = list(dict.fromkeys(sol["pins"][g] for g in sol.get("gauge") or []))
                b["coverage"] = sol.get("coverage")
                b["carrier_domain"] = sol.get("carrier_domain")
                notes = [(", ".join(sol["section"])) if sol.get("section") is not None
                         else f"{', '.join(sol.get('gauge') or [])} = {', '.join(gauge_val)}"]
                notes.append(f"entries certified positive for positive carrier values only; a carrier that "
                             f"takes both signs leaves part of the {domain_name()} outside the chart"
                             if sol.get("coverage") == "partial" else
                             "entries certified positive for every value of the invariants")
                if sol.get("root_note"):
                    notes.append(sol["root_note"])
                b["gauge_note"] = "; ".join(notes)
                b["survivor_meaning"] = sol.get("meaning")
                b["status"] = "reduced"
                certs = ["solved exactly: each invariant becomes a fresh q_<k> parameter"]
                if sol.get("face"):
                    certs.append(f"face section certified: every orbit reaches {', '.join(sol['section'])} "
                                 f"with each other coordinate positive (the face point, solved from the "
                                 f"invariants, is positive on the whole {domain_name()})")
                elif sol.get("section") is not None:
                    certs.append("section pre-certified: balance ratio strictly monotone along every orbit")
                if sol.get("coverage") == "partial":
                    certs.append(f"chart certified for POSITIVE carrier values only: a carrier that takes "
                                 f"both signs leaves part of the {domain_name()} uncovered")
                else:
                    certs.append("chart certified: every solved entry positive on the carrier domains (" +
                                 ", ".join(f"{k} {v}" for k, v in (sol.get("carrier_domain") or {}).items()) + ")")
                if sol.get("shifted"):
                    certs.append(f"carrier offset(s) certified for {', '.join(sol['shifted'])}: the shifted "
                                 f"invariant exceeds its offset on the whole {domain_name()}")
                b["certificates"] = b["certificates"] + certs
            else:
                b["reason"] = sol.get("reason") or b.get("reason")
                b["split"] = split
        if zero_compatibility:
            for b in blocks:
                try:
                    b["zero_compatibility"] = zero_compat(b, fixed, verbose)
                except Exception:
                    b["zero_compatibility"] = None
        trafo = assemble_trafo(blocks, coords)
        for b in blocks:
            for k in ("preps", "Wres", "inv_exps", "face_invariants", "face_back", "Wcols"):
                b.pop(k, None)
        return red_result(obj, blocks, trafo, coords, fixed, settings)
    finally:
        set_positive(old_pos)
        DOMAIN["deadline"] = old_dl
