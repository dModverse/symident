"""Powers b^n of a positive rational b under equilibrate: b joins a base K^n or is held as a constant."""

import re

import sympy as spy

from ..errors import SYMPY_FAILS, sympy_fail
from ..sym.parse import model_sympify

# the start of the refusal of an exponential in the resting model, see sym/tape.py
EXP_REFUSAL = "equilibrate does not support exp() or b^x of a state, or of a parameter that also enters"


def _bases(e, states):
    """Positive rational bases b != 1 of the powers b^x of `e` whose exponent x holds no state."""
    out = set()
    for pw in e.atoms(spy.Pow):
        if pw.base.is_Rational and pw.base > 0 and pw.base != 1 and not pw.exp.is_number \
                and not (pw.exp.free_symbols & states):
            out.add(pw.base)
    return out


def _parsed(x):
    if not isinstance(x, str):
        return None
    try:
        e = model_sympify(x)
        # (2*K)^n as 2^n*K^n
        return e.replace(lambda p: p.is_Pow and p.base.is_Mul and not p.exp.is_number
                         and p.base.as_coeff_Mul()[0] not in (1, -1),
                         lambda p: spy.expand_power_base(p, force=True))
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None


def _leaves(kw):
    """The model expressions of `kw` as (container, key) places."""
    for key in ("f", "totals"):
        d = kw.get(key)
        if isinstance(d, dict):
            yield from ((d, k) for k in d)
    for key in ("g", "trafo"):
        v = kw.get(key)
        for d in [v] if isinstance(v, dict) else v if isinstance(v, (list, tuple)) else []:
            if isinstance(d, dict):
                yield from ((d, k) for k in d)
    for ev in kw.get("events") or []:
        if isinstance(ev, dict):
            yield from ((ev, k) for k in ("value", "time", "root") if ev.get(k) is not None)


def _tag(b):
    return f"nb{b.p}" if b.q == 1 else f"nb{b.p}_{b.q}"


def _symbol_bases(e, states):
    """{exponent: symbols K} of the powers K^x of `e` whose base is a symbol and whose exponent holds no state."""
    out = {}
    for pw in e.atoms(spy.Pow):
        if pw.base.is_Symbol and pw.base not in states and not pw.exp.is_number \
                and not (pw.exp.free_symbols & states):
            out.setdefault(pw.exp, set()).add(pw.base)
    return out


def _numeric_exps(e, states):
    out = {}
    for pw in e.atoms(spy.Pow):
        if pw.base.is_Rational and pw.base > 0 and pw.base != 1 and not pw.exp.is_number \
                and not (pw.exp.free_symbols & states):
            out.setdefault(pw.exp, set()).add(pw.base)
    return out


def _copy(kw):
    kw2 = dict(kw)
    for key in ("f", "totals"):
        if isinstance(kw.get(key), dict):
            kw2[key] = dict(kw[key])
    for key in ("g", "trafo"):
        v = kw.get(key)
        kw2[key] = dict(v) if isinstance(v, dict) else [dict(d) if isinstance(d, dict) else d for d in v] \
            if isinstance(v, (list, tuple)) else v
    kw2["events"] = [dict(ev) if isinstance(ev, dict) else ev for ev in kw.get("events") or []] or kw.get("events")
    return kw2


def _pinned(kw, name):
    """True if `name` is set outside the model expressions: a `trafo` key or a condition column."""
    keys = {str(k) for d in ([kw["trafo"]] if isinstance(kw.get("trafo"), dict) else kw.get("trafo") or [])
            if isinstance(d, dict) for k in d}
    cond = kw.get("conditions")
    cols = set(getattr(cond, "columns", None) or (cond if isinstance(cond, dict) else {}) or [])
    return name in keys or name in {str(c) for c in cols}


def _absorb(kw, states):
    """(kw2, {K: c}) with K^x written for (c*K)^x, K standing for c*K in the whole model; None if no power pairs."""
    scaled, kw2 = {}, kw
    for _ in range(4):
        places = [(d, k, _parsed(d[k])) for d, k in _leaves(kw2)]
        pair = None
        for _, _, e in places:
            if e is None:
                continue
            for x, bs in sorted(_numeric_exps(e, states).items(), key=lambda t: str(t[0])):
                ks = sorted((k for _, _, e2 in places if e2 is not None
                             for k in _symbol_bases(e2, states).get(x, ())), key=str)
                ks = [k for k in ks if not _pinned(kw2, str(k))]
                if ks:
                    pair = (min(bs), ks[0])
                    break
            if pair:
                break
        if pair is None:
            return (kw2, scaled) if scaled else None
        b, K = pair
        kw2 = _copy(kw2)
        for d, k in _leaves(kw2):
            e = _parsed(d[k])
            if e is not None and K in e.free_symbols:
                d[k] = str(spy.expand_power_base(e.subs(K, K / b), force=True)).replace("**", "^")
        scaled[str(K)] = scaled.get(str(K), 1) * b
    return None


def numeric_bases(kw):
    """(kw2, fixed, scaled): the model with every b^x of a positive rational b joined into a base K^x, or
    held as a fixed parameter; fixed is {name: b}, scaled is {K: c} for K standing for c*K; None if none."""
    states = {spy.Symbol(str(k)) for k in dict(kw.get("f") or {})}
    kw2, scaled = _copy(kw), {}
    ab = _absorb(kw2, states)
    if ab is not None:
        kw2, scaled = ab
    places = [(d, k, _parsed(d[k])) for d, k in _leaves(kw2)]
    bases = set()
    for _, _, e in places:
        if e is not None:
            bases |= _bases(e, states)
    if not bases and not scaled:
        return None
    taken = {str(s) for _, _, e in places if e is not None for s in e.free_symbols} | {str(s) for s in states}
    taken |= {str(p) for p in kw.get("parameters") or []} | {str(p) for p in kw.get("fixed") or []}
    names = {}
    for b in sorted(bases):
        name = _tag(b)
        while name in taken:
            name += "_"
        names[b], _ = name, taken.add(name)
    sym = {b: spy.Symbol(n) for b, n in names.items()}
    for d, k in _leaves(kw2):
        e = _parsed(d[k])
        if e is None or not _bases(e, states):
            continue
        e2 = e.replace(lambda x: x.is_Pow and x.base in sym and not x.exp.is_number
                       and not (x.exp.free_symbols & states), lambda x: sym[x.base] ** x.exp)
        d[k] = str(e2).replace("**", "^")
    kw2["fixed"] = list(kw.get("fixed") or []) + list(names.values())
    return kw2, {n: b for b, n in names.items()}, scaled


def _number(b):
    return str(b.p) if b.q == 1 else f"({b.p}/{b.q})"


_VECTORS = ("generator", "display", "complete_generator")


def _vector(d, scaled):
    """The components `d` of a direction in the scaled coordinate K = c*K_old as components in K_old."""
    subs = {spy.Symbol(k): c * spy.Symbol(k) for k, c in scaled.items()}
    out = {}
    for k, v in d.items():
        try:
            e = model_sympify(v).subs(subs, simultaneous=True)
            if k in scaled:
                e = e / scaled[k]
            out[k] = str(e).replace("**", "^")
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            out[k] = v
    return out


def restore(obj, fixed, scaled=None):
    """`obj` of the rewritten model in the names of the model: `fixed` names by their number, the
    coordinates `scaled` as c*K, and the fixed names dropped from keys and lists."""
    scaled = scaled or {}
    num = {n: _number(b) for n, b in fixed.items()}
    both = {**num, **{k: f"({_number(c)}*{k})" for k, c in scaled.items()}}

    def sub(text, toks):
        if not toks:
            return text
        return re.sub(r"\b(%s)\b" % "|".join(map(re.escape, toks)), lambda m: toks[m.group(1)], text)

    def walk(o, key=None):
        if isinstance(o, str):
            return sub(o, both)
        if isinstance(o, dict):
            if key in _VECTORS and scaled:
                comp = {k: sub(v, num) for k, v in o.items() if isinstance(v, str)}
                o = {**{k: v for k, v in o.items() if not isinstance(v, str)}, **_vector(comp, scaled)}
                return {k: v for k, v in o.items() if k not in fixed}
            return {k: walk(v, k) for k, v in o.items() if k not in fixed}
        if isinstance(o, list):
            return [walk(v) for v in o if not (isinstance(v, str) and v in fixed)]
        if isinstance(o, tuple):
            return tuple(walk(v) for v in o if not (isinstance(v, str) and v in fixed))
        return o
    return walk(obj)
