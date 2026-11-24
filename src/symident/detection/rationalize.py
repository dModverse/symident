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


def _transcendental(e):
    return [a for a in spy.preorder_traversal(e) if isinstance(a, (spy.exp, spy.log)) or
            (a.is_Pow and not a.exp.is_Integer)]


def _constant_states(parts, coords):
    """Substitutes, per segment, the value of a state without dynamics that a transcendental term reads and that
    starts at a number and is set to numbers by the events: within a segment the state is that number."""
    out = []
    for part in parts:
        segs = part["segments"]
        f0 = segs[0]["f"]
        ic = segs[0].get("ic") or {}
        inside = set()
        for seg in segs:
            for e in list(seg["f"].values()) + list(seg["g"].values()):
                for a in _transcendental(e):
                    inside |= {str(z) for z in a.free_symbols}
        for x in [x for x in f0 if x in inside and x not in coords]:
            if any(seg["f"][x] != 0 for seg in segs) or ic.get(x) is None or not ic[x].is_number:
                continue
            vals, cur = [], ic[x]
            for j, seg in enumerate(segs):
                for ev in seg.get("events") or [] if j else []:
                    if ev["var"] != x:
                        continue
                    v = ev["value"]
                    cur = v if ev["method"] == "replace" else cur + v if ev["method"] == "add" else cur * v
                vals.append(cur)
            if not all(v.is_number and v.is_real for v in vals):
                continue
            X = spy.Symbol(x)
            for seg, v in zip(segs, vals):
                seg["f"] = {z: (e.xreplace({X: v}) if z != x else e) for z, e in seg["f"].items()}
                seg["g"] = {y: e.xreplace({X: v}) for y, e in seg["g"].items()}
            out.append(x)
    return out


def _reciprocal_log_params(exprs, cs):
    """{theta: g} of the coordinates that enter only through exp(c/theta) with rational c, g the positive rational
    gcd of these c: X = exp(g/theta) then enters with integer powers."""
    cand, bad = {}, set()
    for e in exprs:
        for at in e.atoms(spy.exp):
            terms = spy.expand(at.args[0]).as_coefficients_dict()
            for t, c in terms.items():
                th = 1 / t if t.is_Pow and t.exp == -1 and t.base in cs else None
                if th is None or not c.is_Rational or len(terms) != 1:
                    bad |= at.args[0].free_symbols
                    continue
                cand.setdefault(th, []).append(abs(c))
    out = {}
    for th, cs_ in cand.items():
        if th in bad:
            continue
        num = spy.igcd(*[c.p for c in cs_]) if len(cs_) > 1 else cs_[0].p
        out[th] = spy.Rational(num, spy.ilcm(*[c.q for c in cs_]) if len(cs_) > 1 else cs_[0].q)
    # a coordinate read outside these terms stays
    while out:
        sub = _reciprocal_sub({th: (spy.Dummy(), g) for th, g in out.items()})
        left = set().union(*[sub(e).free_symbols for e in exprs])
        drop = [th for th in out if th in left]
        if not drop:
            break
        for th in drop:
            del out[th]
    return out


def _reciprocal_sub(rp):
    """Rewriter of exp(c/theta) to X^(c/g), rp: theta -> (X, g)."""
    def repl(at):
        terms = spy.expand(at.args[0]).as_coefficients_dict()
        if len(terms) != 1:
            return at
        (t, c), = terms.items()
        th = 1 / t if t.is_Pow and t.exp == -1 else None
        if th not in rp:
            return at
        X, g = rp[th]
        return X ** (c / g) if (c / g).is_Integer else at
    return lambda e: e.replace(lambda a: isinstance(a, spy.exp), repl)


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
    constant = _constant_states(parts, set(coords))
    for x in dict.fromkeys(constant):
        rewrites.append({"rewrite": "constant", "state": x,
                         "why": f"{x} has no dynamics and takes numbers at the start and at the events: within a "
                                "segment it is that number"})
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
    # coordinates read only as exp(c/theta), as X = exp(g/theta) with integer powers
    recip = _reciprocal_log_params([d[k] for d, k in _strings(parts)], cs - set(base_of))
    if recip:
        sub = _reciprocal_sub({t: (t, g) for t, g in recip.items()})
        for d, k in _strings(parts):
            d[k] = sub(d[k])
        for t, g in sorted(recip.items(), key=lambda kv: str(kv[0])):
            logs[str(t)] = (spy.E, g, "reciprocal")
            ex = f"exp({_str(g)}/{t})"
            rewrites.append({"rewrite": "coordinate", "from": ex, "to": str(t),
                             "why": f"{ex} is a bijection of the positive reals onto the reals above 1 and of the "
                                    f"negative reals onto (0, 1): the fiber in {t} is the image of the fiber in {ex}"})
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
                        # the jets of an unknown input start free in a condition with a given start
                        free = any(v is None for k, v in seg["ic"].items() if k not in set(seg.get("inputs") or ()))
                        seg["ic"][str(E)] = None if free else spy.Integer(1)
        for a, E in rates.items():
            ex = _str(spy.exp(a * TIME))
            rewrites.append({"rewrite": "time", "from": ex, "to": str(E),
                             "why": f"{E}' = {_str(a * E)}, {E}(0) = 1 has the unique solution {ex}"})
    # a free power of a state at 0 through the normalized power (x/c_x)^alpha is a state of the part
    from .startpow import normalize
    normalize(parts, cs, states)
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
    sub = {spy.Symbol(t): _fiber_coordinate(t, spec) for t, spec in logs.items()}
    return spy.powsimp(spy.sympify(e).xreplace(sub))


def _fiber_coordinate(t, spec):
    """X in the given coordinate t: b^(t/L), or exp(g/t) for a reciprocal one."""
    t = spy.Symbol(t)
    return spy.exp(spec[1] / t) if len(spec) > 2 else spec[0] ** (t / spec[1])


def map_to_given(sig, logs):
    """{coordinate: image} of a map sigma of the fiber in the given coordinates: theta' = L log_b(sigma(X))."""
    out = {}
    for s, e in sig.items():
        img = to_given(e, logs)
        if str(s) in logs:
            b, L = logs[str(s)][:2]
            pos = {x: spy.Symbol(str(x), real=True) for x in img.free_symbols}
            img = spy.expand_log(spy.log(img.xreplace(pos)), force=True) * L / spy.log(b)
            if len(logs[str(s)]) > 2:
                img = L ** 2 / img
            img = spy.expand(spy.simplify(img)).xreplace({v: k for k, v in pos.items()})
        out[s] = img
    return out


def relation_to_given(r, logs):
    """A factor r = 0 or a relation "lhs < rhs" of the fiber in the given coordinates; a binomial M1 - M2 of
    monomials in the log coordinates becomes the linear form log(M1) - log(M2)."""
    if "<" in r or ">" in r:
        op = "<" if "<" in r else ">"
        lhs, rhs = r.split(op, 1)
        if lhs.strip() in logs and len(logs[lhs.strip()]) == 2:
            return f"{lhs.strip()} {op} " + _str(map_to_given({spy.Symbol(lhs.strip()): model_sympify(rhs)},
                                                              logs)[spy.Symbol(lhs.strip())])
        return f"{_str(to_given(model_sympify(lhs), logs))} {op} {_str(to_given(model_sympify(rhs), logs))}"
    e = model_sympify(r)
    terms = spy.Add.make_args(spy.expand(e))
    if len(terms) == 2 and e.free_symbols and all(len(logs.get(str(s), ())) == 2 for s in e.free_symbols):
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
    info["fiber_coordinates"] = {t: _str(_fiber_coordinate(t, spec)) for t, spec in logs.items()}
    return out
