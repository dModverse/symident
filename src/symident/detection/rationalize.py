"""The parts of the prolonged model in rational form, with the same fiber: log coordinates, injective observables and
exponentials in time."""

import sympy as spy

from ..sym.charts import (
    _detect_log_params,
    _log_param_denominators,
    _log_param_sub,
    _log_params_only_in_atoms,
)
from ..sym.parse import _is_rational_expr, _RPrinter, model_sympify

TIME = spy.Symbol("time")


class NotRationalError(ValueError):
    """A term keeps the model from being rational."""


def _str(e):
    return _RPrinter().doprint(e).replace("**", "^")


def _blocking(e):
    """The first subexpression of `e` that keeps it from being rational."""
    for a in spy.preorder_traversal(e):
        if a.is_Function or (a.is_Pow and not a.exp.is_Integer):
            return a
    return None


def _strings(parts):
    """(container, key) of every expression in `parts`, once: the segments of a condition share their dicts."""
    seen = set()
    for d, k in _all_strings(parts):
        if (id(d), k) not in seen:
            seen.add((id(d), k))
            yield d, k


def _all_strings(parts):
    for part in parts:
        for k in range(len(part.get("surfaces") or [])):
            yield part["surfaces"], k
        for seg in part["segments"]:
            for d in ("f", "g", "ic"):
                for k, v in (seg.get(d) or {}).items():
                    if v is not None:
                        yield seg[d], k
            for e in seg.get("events") or []:
                yield e, "value"
            for k in ("on", "t"):
                if seg.get(k) is not None:
                    yield seg, k


def _time_rate(at, states):
    """a of an atom exp(a*time) with a free of time and of the states, else None."""
    if not isinstance(at, spy.exp):
        return None
    a = spy.cancel(at.args[0] / TIME)
    if a.free_symbols & ({TIME} | states) or not at.args[0].has(TIME):
        return None
    return a


def _split_time(e, states):
    """`e` with every exp(a*time + b) split into exp(a*time)*exp(b)."""
    def split(at):
        arg = spy.expand(at.args[0])
        tpart = sum((t for t in spy.Add.make_args(arg) if t.has(TIME)), spy.Integer(0))
        if tpart == 0 or tpart == arg or _time_rate(spy.exp(tpart), states) is None:
            return at
        return spy.exp(tpart) * spy.exp(arg - tpart)
    return e.replace(lambda a: isinstance(a, spy.exp), split)


def _injective_forms(e):
    """Expressions w(e) with w injective where e is real: exp(c*e) for the logarithms of base e, 10 and 2, the base
    of a power with odd numerator, the argument of exp."""
    out = []
    for c in (spy.Integer(1), spy.log(10), spy.log(2)):
        out.append((spy.expand_power_exp(spy.exp(spy.expand(e * c))), f"exp({_str(c)}*y)" if c != 1 else "exp(y)",
                    "exp is injective on the reals"))
    if isinstance(e, spy.exp):
        out.append((e.args[0], "log(y)", "exp is injective on the reals"))
    if e.is_Pow and e.exp.is_Rational and e.exp.p % 2:
        out.append((e.base, f"y^({_str(1 / e.exp)})", "an odd power is injective where it is real"))
    return out


def rationalize(parts, coords, states):
    """(parts, rewrites, logs): `parts` (as in discrete.Series) in rational form with the same fiber, the rewrites
    applied and {coordinate: (base, L)} for a coordinate theta read as b^(theta/L). A ValueError names the term
    that keeps the model from being rational."""
    states = {spy.Symbol(x) for x in states} | {spy.Symbol(x) for part in parts for seg in part["segments"]
                                                  for x in seg["f"]}
    cs = {spy.Symbol(c) for c in coords} - states
    rewrites = []
    orig = {}
    for d, k in _strings(parts):
        orig[(id(d), k)] = str(d[k])
        d[k] = model_sympify(str(d[k]))
    exprs = [d[k] for d, k in _strings(parts)]
    # log coordinates: theta enters only through b^(c*theta), as X = b^(theta/L) with integer powers
    base_of = {t: b for t, b in _detect_log_params(exprs, states).items() if t in cs}
    base_of = _log_params_only_in_atoms(base_of, exprs)
    logs = {}
    if base_of:
        den = _log_param_denominators(base_of, exprs)
        sub = _log_param_sub({t: (t, b, den[t]) for t, b in base_of.items()})
        for d, k in _strings(parts):
            d[k] = sub(d[k])
        for t, b in sorted(base_of.items(), key=lambda kv: str(kv[0])):
            logs[str(t)] = (b, den[t])
            ex = f"{_str(b)}^({t}/{den[t]})" if den[t] != 1 else (f"exp({t})" if b == spy.E else f"{_str(b)}^{t}")
            rewrites.append({"rewrite": "coordinate", "from": ex, "to": str(t),
                             "why": f"{ex} is a bijection of the real line onto the positive reals: the fiber in "
                                    f"{t} is the image of the fiber in {ex}"})
    # observables under an injective function: the function goes
    seen = {}
    for part in parts:
        for seg in part["segments"]:
            for y, e in list(seg["g"].items()):
                if _is_rational_expr(_split_time(e, states).replace(
                        lambda a: _time_rate(a, states) is not None, lambda a: spy.Integer(1))):
                    continue
                if (y, e) in seen:
                    seg["g"][y] = seen[(y, e)]
                    continue
                for w, how, why in _injective_forms(e):
                    w = _split_time(w, states)
                    if _is_rational_expr(w.replace(lambda a: _time_rate(a, states) is not None,
                                                   lambda a: spy.Integer(1))):
                        seen[(y, e)] = seg["g"][y] = w
                        rewrites.append({"rewrite": "observable", "observable": y, "from": _str(e), "to": _str(w),
                                         "why": f"y = {_str(e)} as {how}: {why}, so two values of the coordinates "
                                                "with the same output keep it after the rewrite and back"})
                        break
    # exp(a*time) as a state E with E' = a*E, E(0) = 1
    rates = {}
    for d, k in _strings(parts):
        d[k] = _split_time(d[k], states)
        for at in d[k].atoms(spy.exp):
            a = _time_rate(at, states)
            if a is not None:
                rates.setdefault(a, None)
    if rates:
        taken = {str(s) for e in exprs for s in e.free_symbols} | {str(s) for s in states}
        for j, a in enumerate(rates, start=1):
            nm = f"si_exp{j}"
            while nm in taken:
                nm += "_"
            rates[a] = spy.Symbol(nm)
        for part in parts:
            for j, seg in enumerate(part["segments"]):
                if j == 0 and seg.get("ic") is not None and spy.sympify(seg["t"]) != 0 and \
                        any(v is not None for v in seg["ic"].values()):
                    raise ValueError("discrete() rewrites exp(a*time) as a state that starts at 1 at time 0; the "
                                     f"condition {seg.get('label') or ''} starts at {seg['t']}")
        for d, k in _strings(parts):
            d[k] = d[k].replace(lambda a: _time_rate(a, states) in rates if isinstance(a, spy.exp) else False,
                                lambda a: rates[_time_rate(a, states)])
        for part in parts:
            for j, seg in enumerate(part["segments"]):
                for a, E in rates.items():
                    seg["f"][str(E)] = a * E
                    if j == 0 and seg.get("ic") is not None:
                        free = any(v is None for v in seg["ic"].values())
                        seg["ic"][str(E)] = None if free else spy.Integer(1)
        for a, E in rates.items():
            ex = _str(spy.exp(a * TIME))
            rewrites.append({"rewrite": "time", "from": ex, "to": str(E),
                             "why": f"{E}' = {_str(a * E)}, {E}(0) = 1 has the unique solution {ex}"})
    for d, k in _strings(parts):
        bad = _blocking(d[k])
        if bad is not None:
            raise NotRationalError(f"discrete() needs a model rational in the states and coordinates; {_str(bad)} "
                                   "is not")
        # an entry the rewrites leave keeps the text it had
        o = orig.get((id(d), k))
        d[k] = o if o is not None and d[k] == model_sympify(o) else _str(d[k])
    return parts, rewrites, logs


def to_given(e, logs):
    """An expression in the coordinates of the fiber, X = b^(theta/L) for the log coordinates, in theta."""
    sub = {spy.Symbol(t): b ** (spy.Symbol(t) / L) for t, (b, L) in logs.items()}
    return spy.powsimp(spy.sympify(e).xreplace(sub))


def map_to_given(sig, logs):
    """{coordinate: image} of a map sigma of the fiber in the given coordinates: theta' = L log_b(sigma(X))."""
    out = {}
    for s, e in sig.items():
        img = to_given(e, logs)
        if str(s) in logs:
            b, L = logs[str(s)]
            pos = {x: spy.Symbol(str(x), real=True) for x in img.free_symbols}
            img = spy.expand_log(spy.log(img.xreplace(pos)), force=True) * L / spy.log(b)
            img = spy.expand(spy.simplify(img)).xreplace({v: k for k, v in pos.items()})
        out[s] = img
    return out


def relation_to_given(r, logs):
    """A factor r = 0 or a relation "lhs < rhs" of the fiber in the given coordinates; a binomial M1 - M2 of
    monomials in the log coordinates becomes the linear form log(M1) - log(M2)."""
    if "<" in r or ">" in r:
        op = "<" if "<" in r else ">"
        lhs, rhs = r.split(op, 1)
        if lhs.strip() in logs:
            b, L = logs[lhs.strip()]
            return f"{lhs.strip()} {op} " + _str(map_to_given({spy.Symbol(lhs.strip()): model_sympify(rhs)},
                                                              logs)[spy.Symbol(lhs.strip())])
        return f"{_str(to_given(model_sympify(lhs), logs))} {op} {_str(to_given(model_sympify(rhs), logs))}"
    e = model_sympify(r)
    terms = spy.Add.make_args(spy.expand(e))
    if len(terms) == 2 and e.free_symbols and all(str(s) in logs for s in e.free_symbols):
        a, b = sorted(terms, key=lambda t: t.could_extract_minus_sign())
        b = -b
        if spy.Mul.make_args(a)[0].is_negative is not True and spy.Mul.make_args(b)[0].is_negative is not True:
            real = {x: spy.Symbol(str(x), real=True) for x in e.free_symbols}
            lin = [spy.expand_log(spy.log(to_given(t, logs).xreplace(real)), force=True) for t in (a, b)]
            return _str(spy.expand(lin[0] - lin[1]).xreplace({v: k for k, v in real.items()}))
    return _str(to_given(e, logs))


def report_given(out, logs):
    """The result `out` of discrete.analyze, computed in X = b^(theta/L), in place in the given coordinates theta."""
    for s in out["symmetries"]:
        sig = {spy.Symbol(k): model_sympify(v) for k, v in s["map"].items()}
        s["map"] = {str(k): _str(v) for k, v in map_to_given(sig, logs).items()}
        s["fixed_set"] = [relation_to_given(r, logs) for r in s.get("fixed_set") or []]
        s["domain_conditions"] = [relation_to_given(r, logs) for r in s.get("domain_conditions") or []]
    info = out["info"]
    info["fundamental_domain"] = [relation_to_given(r, logs) for r in info.get("fundamental_domain") or []]
    info["fiber_coordinates"] = {t: (f"exp({t})" if b == spy.E else f"{_str(b)}^{t}") if L == 1 else
                                 f"{_str(b)}^({t}/{L})" for t, (b, L) in logs.items()}
    return out
