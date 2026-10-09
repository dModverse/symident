"""The public result: directions with generators, display forms and complete generators."""

import re
import warnings

import sympy as spy

from .. import state
from ..rexpr import get_symbols, replace_symbols, sym_sort
from ..sym.charts import log_arg_back_vec, scaling_weights
from .conditions import scaling_component

_ID = re.compile(r"(?<![0-9.A-Za-z_])[A-Za-z_][A-Za-z0-9_.]*(?![A-Za-z0-9_.])(?!\s*\()")


def red_locals(strings):
    """SymPy symbols for the identifiers in `strings`, as locals for sympify."""
    if isinstance(strings, str):
        strings = [strings]
    ids = {m for s in strings for m in _ID.findall(str(s))}
    return {x: spy.Symbol(x) for x in ids if x}


def public_symmetry(d):
    """One raw direction as a public direction with generator and, for a scaling, weights."""
    gen = d.get("vector") if d.get("vector") is not None else d.get("infinitesimals")
    weights = None
    if d.get("type") == "scaling" and d.get("vector") is not None and d.get("infinitesimals") is None:
        weights = dict(d["vector"])
        gen = {k: scaling_component(v, k) for k, v in d["vector"].items()}
    if gen is not None:
        gen = {k: str(v).replace("**", "^") for k, v in gen.items()}
    return {"type": re.sub(r"^Type: ", "", str(d.get("type"))), "generator": gen,
            "weights": weights, "degree": d.get("degree"),
            "support": d.get("support") if d.get("support") is not None else list(gen or []),
            "explicit": gen is not None, "reason": d.get("reason"), "route": d.get("route"),
            "certified": bool(d.get("certified"))}


def display_form(syms):
    """Factored display form of the components where it is shorter."""
    out = []
    for d in syms:
        if d["generator"] is None or d["type"] == "scaling":
            out.append(d)
            continue
        disp = {}
        for k, x in d["generator"].items():
            x = str(x)
            if len(x) > 4000:
                disp[k] = x.replace("**", "^")
                continue
            try:
                y = str(spy.factor(spy.sympify(x.replace("^", "**"))))
            except Exception:
                y = x
            if len(y) >= len(x):
                y = x
            disp[k] = y.replace("**", "^")
        d = dict(d, display=disp)
        out.append(d)
    return out


def denominator(xi):
    """Common denominator of the components, "1" when polynomial or undecided."""
    try:
        comp = [str(x).replace("^", "**") for x in xi.values()]
        loc = red_locals(comp)
        acc = spy.Integer(1)
        for x in comp:
            acc = spy.lcm(acc, spy.fraction(spy.cancel(spy.together(spy.sympify(x, locals=loc))))[1])
        return str(acc).replace("**", "^")
    except Exception:
        return "1"


def tidy(x):
    """`x` factored or over a common denominator by SymPy if that is shorter, else `x`."""
    n_sym = len(set(re.findall(r"[A-Za-z_.][A-Za-z0-9_.]*", x)))
    wide = n_sym > 10 or len(x) > 600
    try:
        e = spy.sympify(x.replace("^", "**"), locals=red_locals(x))
        y = str(spy.together(e) if wide else spy.factor(spy.cancel(e))).replace("**", "^")
    except Exception:
        y = x
    return y if len(y) < len(x) else x


_COMPLETE_MAX_CHARS = 20000


def complete(syms, positive=True, coordinates=None):
    """Complete form f X of a general generator; a scaling is complete already."""
    if isinstance(positive, (list, tuple)) and coordinates:
        unknown = [p for p in positive if p not in set(coordinates)]
        if unknown:
            warnings.warn("`positive` names no coordinate: "
                          f"{', '.join(unknown)}; ignored.", stacklevel=2)
    out = []
    for d in syms:
        if d["generator"] is None:
            out.append(d)
            continue
        d = dict(d, complete_generator=dict(d["generator"]), factor="1")
        if d["type"] == "scaling":
            out.append(d)
            continue
        xi = d["generator"]
        if sum(len(str(x)) for x in xi.values()) > _COMPLETE_MAX_CHARS:
            out.append(dict(d, complete_generator=None, factor=None))
            continue
        if positive is True:
            pos = set(xi)
        elif positive is False:
            pos = set()
        else:
            pos = set(xi) & set(positive)
        D2 = f"({denominator(xi)})^2"
        Q = [f"{D2}*({x})/{k}" if k in pos else f"{D2}*({x})" for k, x in xi.items()]
        f = tidy(f"{D2}/(1 + " + " + ".join(f"({q})^2" for q in Q) + ")")
        d["complete_generator"] = {k: f"({f})*({x})" for k, x in xi.items()}
        d["factor"] = f
        out.append(d)
    return out


def finalize(raw, method, settings, call=None, elapsed=None, coordinates=None):
    """The public result: verdict at top level, computation in info."""
    is_obs = method == "observability"
    raw_syms = raw.get("non_identifiable") if isinstance(raw, dict) else raw
    raw_syms = raw_syms or []
    syms = [public_symmetry(d) for d in raw_syms]
    if not state.get("inner"):
        syms = complete(display_form(syms), settings.get("positive", True), coordinates)
    rank = int(raw["rank"]) if raw.get("rank") is not None else None
    dim = int(raw["dim"]) if raw.get("dim") is not None else None
    identifiable = (rank == dim) if is_obs else None
    engine = "integer-kernel" if method == "scaling" else \
        ("symbolic" if raw.get("engine") == "symbolic" else "modular")
    keep = bool(state.get("keep_kernel"))
    info = {"engine": engine, "lie_order_used": raw.get("lie_order_used"),
            "lie_order_driver": raw.get("lie_order_driver"), "lie_plateau": raw.get("lie_plateau"),
            "lie_certified": raw.get("lie_certified"), "lie_uncertified": raw.get("lie_uncertified"),
            "lie_block_orders": raw.get("lie_block_orders"),
            "kernel": raw.get("kernel") if keep else None,
            "sat_score": raw.get("sat_score") if keep else None,
            "stacked": raw.get("stacked") if keep else None,
            "rank_proven": raw.get("rank_proven"), "rank_proof": raw.get("rank_proof"),
            "gap_order_used": raw.get("gap_order_used"), "conditions": raw.get("conditions"),
            "segments": raw.get("segments"),
            "coordinates": list(raw["coordinates"]) if raw.get("coordinates") is not None else coordinates,
            "settings": settings, "elapsed": elapsed, "verification": raw.get("verification")}
    return {"method": method, "identifiable": identifiable, "rank": rank, "dim": dim,
            "symmetries": syms, "info": info, "call": call}


def log_arg_back(raw, la):
    """Directions analysed in log_v = log(v) mapped back to v."""
    if la is None or raw is None or not isinstance(raw, dict):
        return raw

    def ren(x):
        if not x:
            return x
        return replace_symbols(la["L"], la["v"], list(x))

    def back(vec):
        if not vec:
            return vec
        out = log_arg_back_vec({k: str(x).replace("^", "**") for k, x in vec.items()},
                               la["map"])
        return {k: str(x).replace("**", "^") for k, x in out.items()}
    ab_syms = set(get_symbols([x for m in la["map"] for x in (m["a"], m["b"])]))
    Lset = set(la["L"])
    pat = re.compile(r"\b(" + "|".join(re.escape(l) for l in la["L"]) + r")\b") if la["L"] else None

    def fix_dir(d):
        if not isinstance(d, dict):
            return d
        names = list((d.get("vector") or {}).keys()) + list((d.get("infinitesimals") or {}).keys()) + \
            list(d.get("support") or [])
        vals = [str(v) for v in list((d.get("vector") or {}).values()) +
                list((d.get("infinitesimals") or {}).values())]
        hit = any(n in Lset | ab_syms for n in names) or (pat is not None and any(pat.search(v) for v in vals))
        if not hit:
            return d
        d = dict(d)
        if d.get("vector") is not None:
            vec = d["vector"]
            if d.get("type") == "scaling":
                vec = {k: scaling_component(v, k) for k, v in vec.items()}
            vec = back(vec)
            try:
                w = scaling_weights({k: x.replace("^", "**") for k, x in vec.items()})
            except Exception:
                w = None
            if w is not None:
                d["vector"] = {k: str(x) for k, x in w.items()}
                d["type"] = "scaling"
                d["degree"] = 1
            else:
                d["vector"] = vec
                d["type"] = "general"
                if d.get("degree") is not None:
                    d["degree"] = -1
            d["support"] = sym_sort(list(d["vector"]))
        elif d.get("support") is not None:
            sup = ren(d["support"])
            while True:
                add = [m["v"] for m in la["map"] if m["v"] not in sup and
                       any(s in sup for s in get_symbols([m["a"], m["b"]]))]
                if not add:
                    break
                sup = sup + add
            d["support"] = sym_sort(sup)
        if d.get("infinitesimals") is not None:
            d["infinitesimals"] = back(d["infinitesimals"])
        return d
    raw = dict(raw)
    if raw.get("coordinates") is not None:
        raw["coordinates"] = ren(raw["coordinates"])
    if raw.get("non_identifiable") is not None:
        raw["non_identifiable"] = [fix_dir(d) for d in raw["non_identifiable"]]
    return raw
