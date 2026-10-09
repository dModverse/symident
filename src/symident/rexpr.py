"""Expressions as strings in R syntax: tokens, symbols, substitution, numeric evaluation.

Model equations are kept as strings with `^` for powers, the form the engine parses.
Substitution works on tokens and inserts the replacement text as it is, without
parentheses, as `cOde::replaceSymbols()` does.
"""

import math
import re

_TOKEN = re.compile(r"""
    (?P<ws>\s+)
  | (?P<num>0[xX][0-9a-fA-F]+L?|(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?L?)
  | (?P<name>[A-Za-z.][A-Za-z0-9._]*|`[^`]*`)
  | (?P<str>"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')
  | (?P<op>\*\*|<=|>=|==|!=|&&|\|\||<-|->|%[^%]*%|[-+*/^(),<>!&|=\[\]{}:;~?$@])
""", re.VERBOSE)

_CONSTS = {"TRUE", "FALSE", "NULL", "NA", "Inf", "NaN", "NA_integer_", "NA_real_",
           "NA_character_"}
_KEYWORDS = {"function", "if", "else", "for", "while", "repeat", "in", "next", "break"}


def tokens(expr):
    """Tokens of one expression as (kind, text); kind is num, sym, fun, str or op."""
    out = []
    pos = 0
    s = str(expr)
    n = len(s)
    while pos < n:
        m = _TOKEN.match(s, pos)
        if m is None:
            raise ValueError(f"cannot tokenize {s!r} at {pos}")
        pos = m.end()
        kind = m.lastgroup
        text = m.group(kind)
        if kind == "ws":
            continue
        if kind == "name":
            if text.startswith("."):
                if len(text) > 1 and text[1].isdigit():
                    kind = "num"
            if kind == "name":
                if text in _CONSTS:
                    kind = "num"
                elif text in _KEYWORDS:
                    kind = "op"
                else:
                    j = pos
                    while j < n and s[j].isspace():
                        j += 1
                    kind = "fun" if j < n and s[j] == "(" else "sym"
        if kind == "op" and text == "**":
            text = "^"
        out.append((kind, text))
    return out


def _as_list(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    if isinstance(x, dict):
        return list(x.values())
    return list(x)


def get_symbols(exprs, exclude=None):
    """Symbols of the expressions in order of first appearance (cOde::getSymbols)."""
    seen = {}
    for e in _as_list(exprs):
        if e is None or str(e) == "0":
            continue
        for kind, text in tokens(e):
            if kind == "sym" and text not in seen:
                seen[text] = True
    out = list(seen)
    if exclude:
        ex = set(exclude)
        out = [s for s in out if s not in ex]
    return out


def _replace_one(expr, table):
    if str(expr) == "0":
        return expr
    return "".join(table.get(t, t) if k in ("sym", "fun", "num") else t
                   for k, t in tokens(expr))


def replace_symbols(what, by, x):
    """Replace tokens named in `what` by `by` (cOde::replaceSymbols).

    `x` is a string, a list of strings or a dict name -> string; the result has the
    same shape. `by` is recycled to the length of `what`.
    """
    what = [str(w) for w in _as_list(what)]
    by = [str(b) for b in _as_list(by)]
    if not what:
        return x
    if len(by) != len(what):
        by = [by[i % len(by)] for i in range(len(what))]
    table = dict(zip(what, by))
    if isinstance(x, str):
        return _replace_one(x, table)
    if isinstance(x, dict):
        return {k: _replace_one(v, table) for k, v in x.items()}
    return [_replace_one(v, table) for v in x]


def subst(keys, vals):
    """Substitution keys -> vals as a function on a string, list or dict (None passes)."""
    keys = [str(k) for k in _as_list(keys)]
    vals = [str(v) for v in _as_list(vals)]

    def apply(e):
        if e is None or not keys:
            return e
        if isinstance(e, (dict, list, str)) and len(e) == 0:
            return e
        return replace_symbols(keys, vals, e)
    return apply


def eqn_lines(e):
    """`name = rhs` lines from a dict, the form the engine entry points read."""
    if not e:
        return None
    return [f"{k} = {v}" for k, v in e.items()]


_FREE_EXP = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\s*\^\s*(?![0-9])")


def free_exponent_bases(exprs, states):
    """States that are the base of a free exponent (`C3` in `C3^nhill`)."""
    exprs = _as_list(exprs)
    if not exprs or not states:
        return []
    bases = []
    for e in exprs:
        for m in _FREE_EXP.finditer(str(e).replace("**", "^")):
            b = re.sub(r"\s*\^.*$", "", m.group(0))
            if b not in bases:
                bases.append(b)
    st = set(states)
    return [b for b in bases if b in st]


_FUNS = {
    "exp": math.exp, "log": lambda x, b=None: math.log(x) if b is None else math.log(x, b),
    "log10": math.log10, "log2": math.log2, "sqrt": math.sqrt, "abs": abs,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "sinh": math.sinh,
    "cosh": math.cosh, "tanh": math.tanh, "exp10": lambda x: 10.0 ** x,
    "max": max, "min": min, "pmax": max, "pmin": min,
    "ifelse": lambda c, a, b: a if c else b,
}


def to_python(expr):
    """The expression as Python source: `^` becomes `**`, R constants map to Python."""
    parts = []
    for kind, text in tokens(expr):
        if kind == "op" and text == "^":
            parts.append("**")
        elif kind == "num":
            t = text.rstrip("L") if text[-1:] == "L" and not text.startswith("0x") else text
            parts.append({"TRUE": "True", "FALSE": "False", "Inf": "inf", "NaN": "nan",
                          "NA": "nan", "NULL": "None"}.get(t, t))
        elif kind == "op" and text in ("&&", "&"):
            parts.append(" and ")
        elif kind == "op" and text in ("||", "|"):
            parts.append(" or ")
        elif kind == "op" and text == "!":
            parts.append(" not ")
        else:
            parts.append(text)
    return "".join(parts)


def eval_numeric(expr, env=None):
    """Numeric value of an R expression with the symbols in `env`; None on failure."""
    names = {"inf": math.inf, "nan": math.nan, "pi": math.pi}
    names.update(_FUNS)
    if env:
        names.update(env)
    try:
        v = eval(compile(to_python(expr), "<expr>", "eval"), {"__builtins__": {}}, names)
        return float(v)
    except Exception:
        return None


def is_number(x):
    """Whether the string parses as an R number (as.numeric() is not NA)."""
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return not math.isnan(x)
    try:
        s = str(x).strip()
        if s in ("Inf", "-Inf"):
            return True
        float(s)
        return s.lower() not in ("nan", "inf", "-inf", "infinity", "-infinity")
    except (TypeError, ValueError):
        return False


def as_number(x):
    """The R number of a string or number, None when it is not one."""
    if not is_number(x):
        return None
    s = str(x).strip()
    if s == "Inf":
        return math.inf
    if s == "-Inf":
        return -math.inf
    return float(s)


def sym_sort(x):
    """Sort in byte order (R radix sort in the C locale)."""
    return sorted(x, key=lambda s: s.encode())
