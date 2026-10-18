"""The reduction of a detection result: scalings, curved blocks, trafos and the result."""

import math
import re
import time
import warnings

import numpy as np

from .. import session, state
from ..errors import SYMPY_FAILS, swallow, sympy_fail
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
from .sections import (
    face_carriers,
    face_reach,
    face_regular,
    face_section,
    face_state_limit,
    model_denominators,
    rest_checked,
)
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
    """The result in the coordinates X = b^theta, with the name maps."""
    gens = [None if d.get("generator") is None else
            {k: str(v).replace("^", "**") for k, v in d["generator"].items()} for d in obj["symmetries"]]
    if not gens or all(g is None for g in gens):
        return None
    try:
        ch = log_chart_generators(gens, list(coords))
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
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
    o["info"]["recast_coordinates"] = True
    return {"object": o, "theta": th, "X": X, "base": bs, "ren": ren}


def exp_chart(obj, coords, positive):
    """The detection result in the coordinates L = log(v) with the name maps, or None without them."""
    gens = [None if d.get("generator") is None else
            {k: str(v).replace("^", "**") for k, v in d["generator"].items()} for d in obj["symmetries"]]
    if not gens or all(g is None for g in gens):
        return None
    pos = True if positive is True else ([] if positive is False else list(positive))
    try:
        ch = exp_chart_generators(gens, list(coords), pos)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
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
    o["info"]["recast_coordinates"] = True
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
        if b.get("new_parameters"):
            b["new_parameters"] = {unren([k])[0]: back(v) for k, v in b["new_parameters"].items()}
        blocks.append(b)
    res["blocks"] = blocks
    return res


def exp_back(res, ec):
    """The reduction result `res` mapped back from the exp coordinates `ec`."""
    def back(x, solve_for=None):
        return str(exp_chart_back(str(x).replace("^", "**"), ec["v"], ec["L"], solve_for)).replace("**", "^")
    return _chart_back(res, ec["L"], ec["v"], back)


def log_back(res, lc):
    """The reduction result `res` mapped back from the log coordinates `lc`."""
    def back(x, solve_for=None):
        return str(log_chart_back(str(x).replace("^", "**"), lc["X"], lc["theta"], lc["base"],
                                   solve_for)).replace("**", "^")
    return _chart_back(res, lc["X"], lc["theta"], back)


_SUB = "₀₁₂₃₄₅₆₇₈₉"


# candidate sections the search of one block may try
WORK_CAP = 4000


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
    family = [{"labels": b["labels"], "choices": b.get("choices"), "matroid": b.get("matroid"),
               "gauge_note": b.get("gauge_note")}
              for b in blocks if b["status"] == "reduced" and
              (b.get("choices") is not None or b.get("matroid") is not None or
               b.get("gauge_note") is not None)]
    removed = [l for b, s in zip(blocks, status) if s in ("fixed", "reduced") for l in b["labels"]]
    remaining = [l for b, s in zip(blocks, status) if s not in ("fixed", "reduced") for l in b["labels"]]
    # a listed direction that no block contains depends on the others
    o = ordered(obj)
    dependent = [l for l in o["labels"] if l not in set(removed) | set(remaining)]
    of = {l: d.get("level") or "exact" for l, d in zip(o["labels"], o["syms"])}
    for b in blocks:
        b["unproven"] = unproven_of(b)
        b["generator_level"] = lowest_level(of.get(l, "exact") for l in b["labels"])
        b["level"] = lowest_level([b["generator_level"]] + (["open"] if b["unproven"] else []) +
                                  (["checked"] if b.get("face") and b.get("face_level") == "checked" else []))
    done = [b for b in blocks if b["status"] in ("reduced", "fixed")]
    return {"method": obj.get("method"), "coordinates": coords, "fixed": fixed, "blocks": blocks,
            "trafo": trafo, "family": family, "removed": removed, "remaining": remaining,
            "dependent": dependent, "zero_compatibility": van or None, "settings": settings,
            "level": lowest_level(b["level"] for b in done)}


LEVELS = ("exact", "verified", "checked", "open")


def unproven_of(b):
    """The claims of a reduced block that are not proven, named."""
    if b.get("status") != "reduced" or b.get("type") == "scaling":
        return []
    out = [] if b.get("reach") else ["every level set of the invariants is one orbit"]
    sec = b.get("section") or []
    pins = b.get("pins") or {}
    if b.get("face"):
        if b.get("face_why") is not None or not b.get("reach"):
            out.append(f"the section {', '.join(sec)} lies on a face and gives the output of every orbit "
                       f"in the limit" + (f" ({b['face_why']})" if b.get("face_why") else ""))
    elif b.get("injective"):
        pass
    elif sec:
        if len(b.get("transversal") or []) > 1 or not b.get("section_monotone"):
            out.append(f"each orbit meets the section {', '.join(sec)} at most once")
    else:
        gauge = b.get("transversal") or [k for k, v in pins.items() if re.fullmatch(r"[0-9.]+", str(v).strip())]
        for k in gauge:
            if k in pins and k not in (b.get("monotone") or []):
                out.append(f"each orbit meets {k} = {pins[k]} at most once")
    if b.get("count") == "upper bound" and not b.get("reach"):
        out.append("the number of invariants, an upper bound")
    return out

def finish_reduction(obj, red):
    """The assumptions of the reduction `red` of `obj`, and a warning for a section claim left unproven."""
    if not isinstance(red, dict):
        return
    ax = dict((obj.get("info") or {}).get("assumptions") or {})
    ax["domain"] = (red.get("settings") or {}).get("positive", True)
    ax.setdefault("starts", None)
    ax.setdefault("error_bound", None)
    red["assumptions"] = ax
    if any(b.get("unproven") and b.get("status") in ("reduced", "fixed") for b in red.get("blocks") or []):
        warnings.warn("a claim of a section is not proven; Reduction.blocks names it under 'unproven'",
                      stacklevel=3)


def check_levels(obj, allow_open=False):
    """An error for a direction without closed form unless `allow_open`, a warning for a verified or checked one."""
    levels = [d.get("level") for d in obj.get("symmetries") or []]
    if "open" in levels and not allow_open:
        raise ValueError("a direction has no closed form (level 'open'); detect with reconstruct=True, "
                         "or pass allow_open=True to reduce the others")
    if "checked" in levels:
        warnings.warn("a direction is checked at random points without an error bound; the reduction "
                      "rests on its closed form (Reduction.level)", stacklevel=3)
    elif "verified" in levels:
        warnings.warn("a direction is verified, not proven; the reduction rests on its closed form "
                      "(Reduction.level)", stacklevel=3)


def monotone_pins(b, gens):
    """The constant pins of a one-generator curved block whose generator component has one sign on the domain."""
    if b.get("status") != "reduced" or b.get("type") == "scaling" or len(gens) != 1 or gens[0] is None:
        return []
    from .expr import red_locals, red_sympify
    from .sign import sgn
    gen = gens[0].get("generator") or {}
    out = []
    for k, v in (b.get("pins") or {}).items():
        if not re.fullmatch(r"[0-9.]+", str(v).strip()) or str(v) == "0" or k not in gen:
            continue
        x = str(gen[k]).replace("^", "**")
        try:
            if sgn(red_sympify(x, red_locals([x]))) != 0:
                out.append(k)
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
    return out


def _face_proof(obj, b, pins, sol, gen_of, coords, degrees):
    """Whether every level set is one orbit with the face in its closure, why the face is not regular, and the level."""
    info = obj.get("info") or {}
    if info.get("recast_coordinates"):
        return False, "the model is not checked at the face in log or exponential coordinates", None
    gens = block_generators(b["labels"], gen_of)
    reach = gens is not None and not set(pins) & set(b["support"]) and \
        face_reach(b, pins, sol["gauge"], gens) is not None
    if not info.get("model_exprs"):
        return reach, "the model expressions are not known", None
    eq = (info.get("settings") or {}).get("equilibrate")
    eq = True if eq is None else bool(eq)
    rhs = info.get("model_rhs")
    rest, lift_level = None, "exact"
    if eq and rhs:
        lifted = lifted_generators(b["labels"], gen_of, rhs)
        if lifted is not None:
            rest = face_state_limit(b, pins, sol["gauge"], lifted[0], list(rhs), degrees)
            lift_level = lifted[1]
    why, level = face_regular(sol["gauge"], info["model_exprs"], rhs, eq, coords, b["support"], rest=rest,
                              inputs=info.get("model_inputs") or ())
    if why is None and rest and lift_level == "checked":
        if not rest_checked(rhs, (), info.get("model_inputs") or ()):
            return reach, "the Jacobian of the resting state is not found regular at random points", None
        level = "checked"
    return reach, why, level


def lifted_generators(labels, gen_of, rhs):
    """The generators of `labels` with state components tangent to f = 0 and the level of the lift, or None."""
    # stated components lift when X(f_i) = sum_j f_j d(xi_i)/d(x_j); otherwise xi_x = -J^{-1} f_theta xi_theta,
    # exact when det J has one sign on the domain, else checked
    import sympy as spy

    from .expr import red_locals, red_sympify
    from .sign import sgn
    states = [x for x, f in rhs.items() if str(f).strip() not in ("0", "0.0")]
    loc = red_locals(list(rhs) + [str(v) for v in rhs.values()] +
                     [str(v) for l in labels for v in ((gen_of.get(l) or {}).get("generator") or {}).values()] +
                     [str(v) for l in labels for v in ((gen_of.get(l) or {}).get("state_components") or {}).values()])
    X = [spy.Symbol(x) for x in states]
    level, J = "exact", None
    try:
        F = {x: red_sympify(str(f).replace("^", "**"), loc) for x, f in rhs.items()}
        out = []
        for l in labels:
            d = gen_of.get(l) or {}
            if not d.get("generator"):
                return None
            xi = {spy.Symbol(k): red_sympify(str(v).replace("^", "**"), loc) for k, v in d["generator"].items()}
            if d.get("state_components") is not None:
                xi.update({spy.Symbol(k): red_sympify(str(v).replace("^", "**"), loc)
                           for k, v in d["state_components"].items()})
                for x in rhs:
                    lhs = sum(c * spy.diff(F[x], k) for k, c in xi.items())
                    rhs_ = sum(F[y] * spy.diff(xi.get(spy.Symbol(x), 0), spy.Symbol(y)) for y in rhs)
                    if spy.cancel(lhs - rhs_) != 0:
                        return None
            else:
                if J is None:
                    J = spy.Matrix([F[x] for x in states]).jacobian(X)
                    det = spy.cancel(J.det(method="berkowitz"))
                    if det == 0:
                        return None
                    if sgn(det) == 0:
                        level = "checked"
                ft = spy.Matrix([sum(c * spy.diff(F[x], k) for k, c in xi.items()) for x in states])
                xs = J.LUsolve(-ft)
                xi.update({x: spy.cancel(e) for x, e in zip(X, xs)})
            out.append({str(k): str(v) for k, v in xi.items() if v != 0})
        return out, level
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None


def block_generators(labels, gen_of):
    """The generator components of the directions `labels`, from the weights for a scaling; None if one has neither."""
    out = []
    for l in labels:
        d = gen_of.get(l) or {}
        if d.get("generator"):
            out.append({k: str(v) for k, v in d["generator"].items()})
        elif d.get("weights"):
            out.append({k: f"({w})*{k}" for k, w in d["weights"].items()})
        else:
            return None
    return out


def lowest_level(levels):
    """The lowest of the levels exact, verified, checked and open; exact for none."""
    return max(levels, key=LEVELS.index, default="exact")


def _positive_symbols(res):
    """The coordinates declared positive and the new parameters of positive carriers."""
    p = (res.get("settings") or {}).get("positive", True)
    coords = res.get("coordinates") or []
    out = set(coords) if p is True else set() if p is False else set(p)
    for b in res.get("blocks") or []:
        out |= {q for q, d in (b.get("solved_domain") or {}).items() if d != "real"}
        out |= {q for q in (b.get("new_parameters") or {}) if (b.get("solved_domain") or {}).get(q) != "real"}
    return out


def composed_check(res):
    """Over Q: every pin holds under the trafo and every block invariant maps to its q_l under the pins of its block."""
    import sympy as spy

    from ..sym.parse import _make_local_parse
    tr = res.get("trafo") or {}
    out = {"pins": True, "invariants": True, "failed": []}
    pos_names = _positive_symbols(res)

    def zero(e):
        e = e.xreplace({x: spy.Symbol(str(x), positive=True) if str(x) in pos_names else
                        spy.Symbol(str(x), real=True) for x in e.free_symbols})
        return spy.simplify(spy.expand_log(e)) == 0
    try:
        _, parse = _make_local_parse([str(v) for v in tr.values()] + list(tr))
        for b in res.get("blocks") or []:
            if b.get("status") not in ("reduced", "fixed"):
                continue
            for k, v in (b.get("pins") or {}).items():
                gap = parse(str(tr.get(k, k)).replace("^", "**")) - parse(str(v).replace("^", "**"))
                if not zero(gap):
                    out["pins"] = False
                    out["failed"].append(f"pin {k} = {v}")
            own = {spy.Symbol(k): parse(str(v).replace("^", "**")) for k, v in (b.get("pins") or {}).items()}
            for q, inv in (b.get("new_parameters") or {}).items():
                e = parse(str(inv).replace("^", "**")).xreplace(own)
                if not zero(e - spy.Symbol(q)):
                    out["invariants"] = False
                    out["failed"].append(f"{q} = {inv}")
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return {"pins": None, "invariants": None, "failed": [], "why": f"{type(exc).__name__}"}
    return out


def _tidy_root(x, positive=None):
    """A trafo entry with square roots of perfect squares taken where the symbols are positive."""
    if "sqrt" not in str(x):
        return x
    import sympy as spy

    from ..sym.parse import _make_local_parse
    try:
        _, parse = _make_local_parse([str(x)])
        e = parse(str(x).replace("^", "**"))
        pos = {s: spy.Symbol(str(s), positive=True) for s in e.free_symbols
               if positive is None or str(s) in positive}
        y = e.xreplace(pos).replace(lambda a: a.is_Pow and a.exp == spy.Rational(1, 2),
                                    lambda a: spy.sqrt(spy.factor(a.base)))
        y = y.xreplace({v: k for k, v in pos.items()})
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return x
    out = str(y).replace("**", "^")
    return out if len(out) < len(str(x)) else x


def _rational_chart(res):
    return not any(re.search(r"sqrt|\^\(1/|\^\([0-9]+/", str(v)) for v in (res.get("trafo") or {}).values())


def _chart_valid(res, dim):
    """Whether the trafo leaves exactly dim minus the removed directions free, so that no section overwrote a pin."""
    if not res.get("trafo") or dim is None:
        return True
    names = {t for v in res["trafo"].values() for t in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(v))}
    names -= {"sqrt", "exp", "log"}
    return len(names) == dim - len(res.get("removed") or [])


def _chart_rank(res):
    """Sort key of a reduction: more directions removed, then a rational trafo; ties keep the build order."""
    return (-len(res.get("removed") or []), not _rational_chart(res))


def symmetry_reduction(obj, *args, alternatives=False, **kw):
    """The reduction of `_reduction_charts`; the outermost call reports the errors of alternative trafos."""
    outer = state.get("session") is None
    with session.opened({}) as ses:
        res = _reduction_charts(obj, *args, alternatives=alternatives, **kw)
    if outer and isinstance(res, dict):
        res["errors"] = list(ses.notes["errors"])
        res["limits"] = list(ses.notes["limits"])
    return res


def _reduction_charts(obj, *args, alternatives=False, **kw):
    """Reparametrization removing the non-identifiable directions, the best trafo first when `alternatives` asks."""
    def run(**opts):
        with state.options(**opts):
            res = _symmetry_reduction(obj, *args, **kw)
        if isinstance(res, dict) and res.get("trafo"):
            pos = _positive_symbols(res)
            res["trafo"] = {k: _tidy_root(v, pos) for k, v in res["trafo"].items()}
            res["checks"] = composed_check(res)
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
            try:
                alt = run(reduce_pin_inside=inside or None, transversal_skip=skip or None)
            except Exception as exc:
                swallow(exc, "alternative chart")
                alt = None
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
                        zero_compatibility=False, timeout=math.inf, work_cap=WORK_CAP):
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
                                         timeout=timeout, work_cap=work_cap)
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
                                         timeout=timeout, work_cap=work_cap)
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
            # an alternative trafo: the scalings may pin coordinates of the curved directions
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
                             separable_, scal_pins)
        blocks = verify_blocks(blocks)
        for b in blocks:
            short = b.get("target") is not None and len(b.get("invariants") or []) < b["target"]
            for c in (b.get("certificates") or []) if b["type"] == "curved" and short else []:
                if re.search(r"budget|\bcap\b|capped", c):
                    session.note("limits", f"invariants of {', '.join(b['labels'])}: {c}")
        inv_start = 0
        dens = None
        gen_of = dict(zip(o["labels"], o["syms"]))
        for b in blocks:
            if b["type"] != "curved" or not b.get("invariants"):
                continue
            if b.get("target") is not None and len(b["invariants"]) < b["target"]:
                continue
            if dens is None:
                dens = model_denominators(obj["info"].get("model_exprs") or [])
            DOMAIN["deadline"] = time.monotonic() + timeout if math.isfinite(timeout) else None
            DOMAIN["work"], DOMAIN["work_cap"] = 0, work_cap
            try:
                sol = face_section(b, scal_pins, dens)
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                sol = {"solved": False}
            except Exception as exc:
                swallow(exc, "face section")
                sol = {"solved": False}
            if work_cap is not None and DOMAIN["work"] > work_cap and not sol.get("solved"):
                session.note("limits", f"face section search of {', '.join(b['labels'])}: "
                                       f"reduce(work_cap={work_cap}) reached")
            # the gauge section search has a budget of its own
            DOMAIN["work"] = 0
            if sol.get("solved"):
                sol = face_carriers(sol, coords, inv_start)
            split = sol.get("split")
            split_pair = None
            if split is not None:
                split_pair = [re.sub(r" = 0$", "", x) for x in split["positive"] + split["negative"]]
            if not sol.get("solved"):
                sol = solve_invariants(b, scal_pins, coords, inv_start, split_pair=split_pair,
                                       gens=block_generators(b["labels"], gen_of))
            if not sol.get("solved") and split is not None:
                sol["reason"] = (f"the orbits leave the {domain_name()} through "
                                 f"{', '.join(split['positive'])} where {split['invariant']} > 0 and through "
                                 f"{', '.join(split['negative'])} where it is < 0; no face covers both, and "
                                 f"no section was certified. An assumption on the sign of {split['invariant']} "
                                 "selects the face")
            rooted = sol.get("solved") and sol.get("root_note") is not None
            pos = positive_set()
            if (not sol.get("solved") or rooted) and pos is not None and set(b["support"]) - set(pos):
                sol_r = solve_invariants(b, scal_pins, coords, inv_start, prefer_real=True, split_pair=split_pair,
                                         gens=block_generators(b["labels"], gen_of))
                if sol_r.get("solved") and (not rooted or sol_r.get("root_note") is None):
                    sol = sol_r
            if work_cap is not None and DOMAIN["work"] > work_cap:
                session.note("limits", f"section search of {', '.join(b['labels'])}: reduce(work_cap={work_cap}) "
                                       "reached")
            elif DOMAIN["deadline"] is not None and time.monotonic() > DOMAIN["deadline"]:
                session.note("limits", f"section search of {', '.join(b['labels'])}: reduce(timeout={timeout}) "
                                       "reached")
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
                b["injective"] = bool(sol.get("injective")) and not b["face"]
                b["section_monotone"] = bool(sol.get("monotone")) and not b["face"]
                b["reach"] = bool(sol.get("reach")) and not b["face"]
                if b["face"]:
                    b["reach"], b["face_why"], b["face_level"] = _face_proof(obj, b, scal_pins, sol, gen_of,
                                                                             coords, (d_poly, d_darboux, d_exp))
                gauge_val = list(dict.fromkeys(sol["pins"][g] for g in sol.get("gauge") or []))
                b["coverage"] = sol.get("coverage")
                b["solved_domain"] = sol.get("solved_domain")
                notes = [(", ".join(sol["section"])) if sol.get("section") is not None
                         else f"{', '.join(sol.get('gauge') or [])} = {', '.join(gauge_val)}"]
                notes.append(f"entries certified positive for positive values of the new parameters only; one that "
                             f"takes both signs leaves part of the {domain_name()} outside the trafo"
                             if sol.get("coverage") == "partial" else
                             "entries certified positive for every value of the invariants")
                if sol.get("root_note"):
                    notes.append(sol["root_note"])
                b["gauge_note"] = "; ".join(notes)
                b["new_parameters"] = sol.get("meaning")
                b["status"] = "reduced"
                certs = ["solved exactly: each invariant becomes a fresh q_<k> parameter"]
                if sol.get("face"):
                    certs.append(f"face section certified: every orbit reaches {', '.join(sol['section'])} "
                                 f"with each other coordinate positive (the face point, solved from the "
                                 f"invariants, is positive on the whole {domain_name()})")
                elif b["section_monotone"]:
                    certs.append("section: each balance ratio has a derivative of one sign along each generator")
                if b["injective"]:
                    certs.append("section met at most once: the solved entries give back every point of "
                                 "the section from its invariants, so two points of one orbit on it coincide")
                if b["face"] and b["reach"] and b.get("face_why") is None:
                    certs.append("face reached in the limit: every orbit closes on the face point, where "
                                 "the model is continuous, so the face point gives the output of the orbit" +
                                 ("; the resting state there is checked at random points of the face, without "
                                  "an error bound" if b.get("face_level") == "checked" else ""))
                if b["reach"]:
                    certs.append("every level set of the invariants is one orbit: it is the graph of the "
                                 "solved coordinates over a connected set of gauge values, and the generators are "
                                 "independent on the gauge everywhere")
                if sol.get("coverage") == "partial":
                    certs.append(f"trafo certified for positive values of the new parameters only: one that takes "
                                 f"both signs leaves part of the {domain_name()} uncovered")
                else:
                    certs.append("trafo certified: every solved entry positive on the domains of the new parameters (" +
                                 ", ".join(f"{k} {v}" for k, v in (sol.get("solved_domain") or {}).items()) + ")")
                if sol.get("shifted"):
                    certs.append(f"offsets of the new parameters certified for {', '.join(sol['shifted'])}: the "
                                 f"shifted invariant exceeds its offset on the whole {domain_name()}")
                b["certificates"] = b["certificates"] + certs
            else:
                b["reason"] = sol.get("reason") or b.get("reason")
                b["split"] = split
        if zero_compatibility:
            for b in blocks:
                try:
                    b["zero_compatibility"] = zero_compat(b, fixed)
                except Exception as exc:
                    swallow(exc, "zero compatibility")
                    b["zero_compatibility"] = None
        trafo = assemble_trafo(blocks, coords)
        for b in blocks:
            b["monotone"] = monotone_pins(b, [gen_of.get(l) for l in b["labels"]])
        for b in blocks:
            for k in ("preps", "originals", "Wres", "inv_exps", "face_invariants", "face_back", "Wcols"):
                b.pop(k, None)
        return red_result(obj, blocks, trafo, coords, fixed, settings)
    finally:
        set_positive(old_pos)
        DOMAIN["deadline"] = old_dl
