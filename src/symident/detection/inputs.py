"""Form checks of the model arguments, before any analysis: an error names the argument."""

import keyword
import re
from tokenize import TokenError

import sympy as spy

from ..errors import InputError
from ..sym.parse import model_sympify
from .conditions import Grid
from .cq import check_conserved

_METHODS = ("replace", "add", "multiply")


def _dicts(x):
    """The dict or list of dicts `x` as a list of dicts (None as an empty list)."""
    return [d for d in (x if isinstance(x, (list, tuple)) else [x]) if d is not None]


# texts that passed _expression during one check_arguments call: conditions repeat the same entries
_PASSED = set()
# a plain name or decimal number always parses
_PLAIN = re.compile(r"[A-Za-z_][A-Za-z_0-9]*|[0-9]+(\.[0-9]*)?([eE][-+]?[0-9]+)?")


def _expression(text, where):
    """InputError for a missing, empty or unparsable expression; numbers pass."""
    if isinstance(text, str) and (text in _PASSED or (_PLAIN.fullmatch(text) and not keyword.iskeyword(text))):
        return
    if text is None or isinstance(text, bool):
        raise InputError(f"{where} has no expression")
    if isinstance(text, (int, float)):
        return
    if not isinstance(text, str):
        raise InputError(f"{where} must be a string or a number, not {type(text).__name__}")
    if not text.strip():
        raise InputError(f"{where} is an empty expression")
    try:
        e = model_sympify(text)
    except (SyntaxError, TokenError) as exc:
        kws = [w for w in re.findall(r"[A-Za-z_][A-Za-z_0-9]*", text) if keyword.iskeyword(w)]
        if kws:
            raise InputError(f"{where} cannot be parsed: {text!r}; {kws[0]} is a Python keyword and cannot "
                             f"name a state, parameter or input, rename it") from exc
        raise InputError(f"{where} cannot be parsed: {text!r}") from exc
    except Exception:  # an expression the engine handles or refuses by name later
        return
    if e.has(spy.zoo, spy.nan, spy.oo):
        raise InputError(f"{where} divides by zero or is no number: {text!r}")
    _PASSED.add(text)


def _names(kw):
    """InputError for a state, parameter or input named like a Python keyword."""
    names = list(dict(kw.get("f") or {}))
    for key in ("parameters", "fixed", "forcings", "unknown_inputs", "known_inputs"):
        names += [str(x) for x in kw.get(key) or []]
    for t in _dicts(kw.get("trafo")):
        names += list(t)
    bad = [n for n in names if keyword.iskeyword(str(n))]
    if bad:
        raise InputError(f"{bad[0]} is a Python keyword and cannot name a state, parameter or input")


def _events(kw, n_cond):
    for i, e in enumerate(kw.get("events") or [], start=1):
        where = f"event {i}"
        if not isinstance(e, dict):
            raise InputError(f"{where} must be a dict with `var`, `time` and `value`")
        for key in ("var", "value"):
            if e.get(key) is None:
                raise InputError(f"{where} has no `{key}`")
        if e.get("time") is None and e.get("root") is None:
            raise InputError(f"{where} has neither `time` nor `root`")
        if str(e.get("method", "replace")) not in _METHODS:
            raise InputError(f"{where}: `method` is {e['method']!r}, one of {', '.join(_METHODS)} is expected")
        for key in ("time", "value", "root"):
            if e.get(key) is not None and str(e[key]) not in ("NA", "Inf", "-Inf"):
                _expression(e[key], f"{where}: `{key}`")
        for c in e.get("cond") or []:
            if not 0 <= int(c) < n_cond:
                raise InputError(f"{where}: `cond` lists condition {c}, but the conditions are 0 to {n_cond - 1}")


def _n_conditions(kw, grid):
    sizes = [len(x) for x in (kw.get("trafo"), kw.get("g")) if isinstance(x, (list, tuple))]
    return max([1, grid.nrow if grid is not None else 1] + sizes)


def check_arguments(kw):
    """InputError for a malformed model argument of the detection `kw`."""
    _PASSED.clear()
    try:
        _check_arguments(kw)
    finally:
        _PASSED.clear()


def _check_arguments(kw):
    f = kw.get("f")
    if f is not None:
        if not isinstance(f, dict):
            raise InputError(f"`odes` must be a dict of state to right-hand side, not {type(f).__name__}")
        if not f:
            raise InputError("`odes` holds no state")
        for k, v in f.items():
            _expression(v, f"the right-hand side of {k}")
    for g in _dicts(kw.get("g")):
        for k, v in dict(g).items():
            _expression(v, f"the observable {k}")
    for t in _dicts(kw.get("trafo")):
        for k, v in dict(t).items():
            _expression(v, f"the `trafo` entry {k}")
    for k, v in dict(kw.get("totals") or {}).items():
        _expression(v, f"the total {k}")
    grid = Grid.from_any(kw.get("conditions"))
    if kw.get("totals") and isinstance(f, dict):
        check_conserved(dict(kw["totals"]), {str(k): str(v) for k, v in f.items()})
    _names(kw)
    _events(kw, _n_conditions(kw, grid))
    states = set(dict(f or {}))
    clash = [str(p) for p in kw.get("parameters") or [] if str(p) in states]
    if clash:
        raise InputError(f"`parameters` names {clash[0]}, which is a state of `odes`")
