"""JSON interface for host languages: specification in, result out."""

import contextlib
import json
import math
import os
import warnings

import numpy as np

from . import state
from .detection import symmetry_detection
from .options import reconst_control
from .reduction import symmetry_reduction
from .reduction.core import check_levels, detection_positive, finish_reduction


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def _clean(x):
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items() if not callable(v)}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


def _dumps(x):
    return json.dumps(_clean(x), default=_default)


DETECT_ARGS = ("f", "g", "trafo", "conditions", "events", "reactions", "totals", "parameters",
               "fixed", "forcings", "free_initial", "equilibrate", "reduce_cq", "reconstruct",
               "verify", "positive", "sym_engine", "cores", "scalings_only", "regimes", "unknown_inputs",
               "known_inputs")


def kwargs_of(spec):
    """symmetry_detection() arguments of a spec; ``"AUTO"`` gauge_preference and ``"Inf"`` in control as such."""
    kw = {k: spec[k] for k in DETECT_ARGS if spec.get(k) is not None}
    if "gauge_preference" in spec:
        gp = spec["gauge_preference"]
        kw["gauge_preference"] = None if gp == "AUTO" else gp
    if spec.get("control"):
        kw["control"] = reconst_control(**{k: (math.inf if v == "Inf" else v)
                                           for k, v in spec["control"].items() if v is not None})
    return kw


@contextlib.contextmanager
def _context(env, options):
    env = json.loads(env) if env else {}
    opts = json.loads(options) if options else {}
    opts = {k: v for k, v in (opts or {}).items() if v is not None and not isinstance(v, (dict, list))}
    saved = {k: os.environ.get(k) for k in env}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        with state.options(**opts):
            yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _messages(rec):
    return [str(w.message) for w in rec]


def detect_json(spec, env=None, options=None):
    """Detection from a JSON specification; the result as JSON with the warnings and the spec."""
    with _context(env, options), warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        res = symmetry_detection(**kwargs_of(json.loads(spec)))
    if res.get("info") and res["info"].get("kernel") is not None:
        res["info"]["kernel"] = None
    res["warnings"] = list(dict.fromkeys(_messages(rec)))
    # the closed loop of reduce_json runs the detection again from it
    res["spec"] = json.loads(spec)
    return _dumps(res)


def reduce_json(obj, args, env=None, options=None):
    """Reduction of a detection result given as JSON; the reduction as JSON."""
    o = json.loads(obj)
    a = json.loads(args) if args else {}
    kw = {}
    for k, v in a.items():
        if k == "timeout" and v == "Inf":
            v = math.inf
        kw["separable_" if k == "separable" else k] = v
    if kw.get("positive") is None:
        kw["positive"] = detection_positive(o)
    allow_open = bool(kw.pop("allow_open", False))
    allow_checked = bool(kw.pop("allow_checked", False))
    check = bool(kw.pop("check", True))
    with _context(env, options), warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        check_levels(o, allow_open, allow_checked)
        res = symmetry_reduction(o, **kw)
        finish_reduction(o, res)
    if check and isinstance(o.get("spec"), dict) and res.get("trafo") and o.get("method") == "observability":
        with _context(env, options), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res.setdefault("checks", {})["loop"] = _loop(o["spec"], res, o.get("gauge"))
    res["warnings"] = list(dict.fromkeys(_messages(rec)))
    return _dumps(res)


def _loop(spec, red, gauge):
    """The detection of `spec` with the trafo of the reduction applied: its directions against the remaining ones."""
    from .api import loop_outcome, reduced_model
    want = len(red.get("remaining") or [])
    rm = reduced_model(dict(spec.get("f") or {}), spec.get("g"), spec.get("trafo"), spec.get("conditions"),
                       spec.get("events"), red["trafo"], bool(spec.get("equilibrate")))
    if rm["why"] is not None:
        return {"ok": None, "directions": None, "expected": want, "reason": "not run, " + rm["why"]}
    s = dict(spec, reconstruct=False, trafo=rm["trafo"], conditions=rm["conditions"],
             fixed=list(dict.fromkeys(list(spec.get("fixed") or []) + list(gauge or []) +
                                      list(red.get("fixed") or []) + rm["fixed"])))
    s.pop("gauge_preference", None)
    return loop_outcome(lambda: symmetry_detection(**kwargs_of(s)), want)
