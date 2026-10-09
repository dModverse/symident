"""JSON specs of reference runs: keyword arguments for symmetry_detection and comparison."""

import itertools
import json
import math
import os
import re
import warnings

import sympy as spy

from symident.options import reconst_control


def snake(name):
    """A camelCase name of the R reference as the snake_case name of symident."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower() if name[:1].islower() or "_" in s else name


R_ARGS = {"separable": "separable_", "report_zero_compatibility": "zero_compatibility"}


def kwargs_of(spec):
    """Keyword arguments of symmetry_detection from an R reference spec."""
    s = {snake(k): v for k, v in spec.items()}
    kw = {}
    for k in ("f", "g", "trafo", "conditions", "events", "reactions", "totals", "parameters",
              "fixed", "forcings", "free_initial", "equilibrate", "reduce_cq", "reconstruct",
              "verify", "positive", "sym_engine", "cores"):
        if k in s and s[k] is not None:
            kw[k] = s[k]
    if "gauge_preference" in s:
        gp = s["gauge_preference"]
        kw["gauge_preference"] = None if gp == "AUTO" else gp
    if s.get("method") == "scaling":
        kw["scalings_only"] = True
    if s.get("control"):
        c = {snake(k): (math.inf if v == "Inf" else v) for k, v in s["control"].items()
             if k not in ("certifyPoly", "certifyPolyDeg")}
        kw["control"] = reconst_control(**c)
    return kw


def env_of(g):
    """Environment switches of an R reference run under the SYMIDENT_ prefix."""
    return {"SYMIDENT_" + k.split("_SYM_", 1)[1]: str(v) for k, v in (g.get("env") or {}).items()
            if str(v) and "_SYM_" in k}


def options_of(g):
    """state.options() of an R reference run."""
    return {snake(k.rsplit(".", 1)[-1]): v for k, v in (g.get("options") or {}).items()
            if v is not None}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _expr_equal(a, b):
    if str(a).strip() == str(b).strip():
        return True
    try:
        ea = spy.sympify(str(a).replace("^", "**"))
        eb = spy.sympify(str(b).replace("^", "**"))
        return spy.simplify(ea - eb) == 0
    except Exception:
        return False


def compare(py, ref):
    """Differences between a symident result and the reference, as a list of strings."""
    diffs = []
    if "error" in ref:
        if not isinstance(py, Exception):
            diffs.append(f"reference errored ({ref['error'][:80]}) but symident returned")
        return diffs
    if isinstance(py, Exception):
        return [f"symident error: {py!r}"[:300]]
    for k in ("rank", "dim", "identifiable"):
        rv = None if ref.get(k) == "NA" else ref.get(k)
        if py.get(k) != rv:
            diffs.append(f"{k}: {py.get(k)} != {ref.get(k)}")
    rs = ref.get("symmetries") or []
    ps = py.get("symmetries") or []
    if len(rs) != len(ps):
        diffs.append(f"number of directions: {len(ps)} != {len(rs)}")

    def key(d):
        return (d["type"], tuple(sorted(d.get("support") or [])))
    for t in ("scaling", "general"):
        a = sum(d["type"] == t for d in ps)
        b = sum(d["type"] == t for d in rs)
        if a != b:
            diffs.append(f"{t} directions: {a} != {b}")
    a = sum(bool(d.get("explicit")) for d in ps)
    b = sum(bool(d.get("explicit")) for d in rs)
    if a != b:
        diffs.append(f"explicit: {a} != {b}")
    unmatched = list(ps)
    for d in rs:
        if not d.get("explicit"):
            continue
        hit = None
        for e in unmatched:
            if not e.get("explicit") or e["type"] != d["type"]:
                continue
            ge, gd = e["generator"], d["generator"]
            if d["type"] == "scaling" and d.get("weights"):
                # the R reference wrote a compound weight unbracketed: w*z, not (w)*z
                gd = {k: (f"({w})*{k}" if v == f"{w}*{k}" and re.search(r"[+-]", str(w).lstrip("-"))
                          else v) for k, v, w in ((k, v, d["weights"].get(k)) for k, v in gd.items())}
            if set(ge) != set(gd):
                continue
            if all(_expr_equal(ge[k], gd[k]) for k in gd):
                hit = e
                break
        if hit is None:
            diffs.append(f"no matching generator for {d['type']} on {sorted(d['generator'])}")
        else:
            unmatched.remove(hit)
    py_proven = py.get("info", {}).get("rank_proven")
    if ref.get("rankProven") is None and py_proven is not None and not diffs:
        return ["BETTER: rank proven by symident only"]
    if (ref.get("rankProven") is None) != (py_proven is None):
        diffs.append(f"rank_proven: {py_proven} != {ref.get('rankProven')}")
    return diffs


def obj_of(o):
    syms = []
    for d in o["symmetries"]:
        syms.append({"type": d["type"], "generator": d.get("generator"), "weights": d.get("weights"),
                     "support": d.get("support") or list((d.get("generator") or {}).keys()),
                     "degree": d.get("degree")})
    return {"method": o.get("method"), "identifiable": o.get("identifiable"), "gauge": o.get("gauge"),
            "info": {"coordinates": o["info"].get("coordinates"),
                     "model_exprs": o["info"].get("modelExprs")}, "symmetries": syms}


def kw_of(a):
    kw = {}
    for k, v in (a or {}).items():
        k = R_ARGS.get(snake(k), snake(k))
        if isinstance(v, list) and len(v) == 1 and k in ("d_poly", "d_darboux", "d_exp", "timeout"):
            v = v[0]
        if k == "timeout" and v == "Inf":
            v = math.inf
        if k in ("fixed", "positive") and isinstance(v, list) and v and isinstance(v[0], bool):
            v = v[0]
        kw[k] = v
    return kw


def compare_reduction(py, ref):
    if "error" in ref:
        return [] if isinstance(py, Exception) else [f"reference errored: {ref['error'][:60]}"]
    if isinstance(py, Exception):
        return [f"symident error: {py!r}"[:300]]
    d = []
    if sorted(py["removed"]) != sorted(ref.get("removed") or []):
        d.append(f"removed {sorted(py['removed'])} != {sorted(ref.get('removed') or [])}")
    if sorted(py["remaining"]) != sorted(ref.get("remaining") or []):
        d.append(f"remaining {sorted(py['remaining'])} != {sorted(ref.get('remaining') or [])}")
    st_p = sorted(b["status"] for b in py["blocks"])
    st_r = sorted(snake(b["status"]) for b in ref.get("blocks") or [])
    if st_p != st_r:
        d.append(f"statuses {st_p} != {st_r}")
    tr, tp = ref.get("trafo") or {}, py.get("trafo") or {}
    if set(tr) != set(tp):
        d.append(f"trafo keys differ: {sorted(set(tr) ^ set(tp))[:6]}")
    else:
        bad = min((_trafo_diff(_relabel(tp, perm), tr) for perm in _q_perms(tp)), key=len)
        if bad:
            d.append("trafo entries differ: " + "; ".join(f"{k}: {tp[k]} vs {tr[k]}" for k in bad[:3]))
    return d


def _trafo_diff(tp, tr):
    return [k for k in tr if not _expr_equal(tp[k], tr[k])]


def _q_perms(trafo):
    """Relabellings of the new parameters q_<k>, whose numbering is arbitrary."""
    qs = sorted({q for v in trafo.values() for q in re.findall(r"\bq_[0-9]+\b", str(v))})
    if len(qs) > 5:
        return [dict(zip(qs, qs))]
    return [dict(zip(qs, p)) for p in itertools.permutations(qs)]


def _relabel(trafo, perm):
    pat = re.compile(r"\bq_[0-9]+\b")
    return {k: pat.sub(lambda m: perm[m.group(0)], str(v)) for k, v in trafo.items()}


def run_detection(path):
    """Run one reference detection spec; the differences to the reference."""
    from symident import state
    from symident.detection import symmetry_detection
    g = load(path)
    env = env_of(g)
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    opts = options_of(g)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with state.options(**opts):
                res = symmetry_detection(**kwargs_of(g["spec"]))
    except Exception as e:
        res = e
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return [d for d in compare(res, g["result"]) if not d.startswith("BETTER")]


def run_reduction(path):
    from symident.reduction import symmetry_reduction
    g = load(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = symmetry_reduction(obj_of(g["object"]), **kw_of(g.get("args")))
    except Exception as e:
        res = e
    return compare_reduction(res, g["result"])
