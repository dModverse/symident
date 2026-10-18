"""JSON specs of test models: keyword arguments of symmetry_detection and symmetry_reduction."""

import json
import math
import re

from symident.options import reconst_control


def snake(name):
    """A camelCase name of a spec as the snake_case name of symident."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower() if name[:1].islower() or "_" in s else name


ARGS = {"separable": "separable_", "report_zero_compatibility": "zero_compatibility"}


def kwargs_of(spec):
    """Keyword arguments of symmetry_detection from a spec."""
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
    """Environment switches of a spec under the SYMIDENT_ prefix."""
    return {"SYMIDENT_" + k.split("_SYM_", 1)[1]: str(v) for k, v in (g.get("env") or {}).items()
            if str(v) and "_SYM_" in k}


def options_of(g):
    """state.options() of a spec."""
    return {snake(k.rsplit(".", 1)[-1]): v for k, v in (g.get("options") or {}).items()
            if v is not None}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


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
        k = ARGS.get(snake(k), snake(k))
        # the specs pass arguments the reduction no longer has
        if k == "verbose":
            continue
        if isinstance(v, list) and len(v) == 1 and k in ("d_poly", "d_darboux", "d_exp", "timeout"):
            v = v[0]
        if k == "timeout" and v == "Inf":
            v = math.inf
        # a time limit this short stops the section search; zero stops it on any machine
        if k == "timeout" and isinstance(v, (int, float)) and v < 1:
            v = 0
        if k in ("fixed", "positive") and isinstance(v, list) and v and isinstance(v[0], bool):
            v = v[0]
        kw[k] = v
    return kw
