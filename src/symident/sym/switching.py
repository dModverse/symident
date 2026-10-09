"""State-dependent switches: piecewise(), comparisons, logic, Heaviside, sign, abs, max, min.

A switch whose condition the declared signs decide is resolved. Any other switch makes the
model hybrid: each undecided condition is a switching surface g_j = 0 with an indicator
H_j (1 where g_j > 0), and a regime fixes every indicator.
"""

import re

import sympy as spy

from ..rexpr import tokens

_SWITCHY = re.compile(r"\b(piecewise|ifelse|Heaviside|max|min|pmax|pmin)\s*\(|[<>!&|=]")
_SIGNY = re.compile(r"\b(abs|sign)\s*\(")
_BIN = {"||": 1, "|": 1, "&&": 2, "&": 2, "<": 4, ">": 4, "<=": 4, ">=": 4, "==": 4, "!=": 4,
        "+": 5, "-": 5, "*": 6, "/": 6, "^": 8}
_MATH = {"exp", "log", "sqrt", "sin", "cos", "tan", "sinh", "cosh", "tanh", "asin", "acos",
         "atan", "asinh", "acosh", "atanh"}
TIME = spy.Symbol("time")


class _Bool:
    """A truth value as an indicator expression, 1 where true and 0 where false."""

    def __init__(self, e):
        self.e = e


def _val(x):
    return x.e if isinstance(x, _Bool) else x


def _number(t):
    if t in ("TRUE", "T"):
        return spy.Integer(1)
    if t in ("FALSE", "F"):
        return spy.Integer(0)
    t = t[:-1] if t.endswith("L") and not t.startswith("0x") else t
    if t.startswith(("0x", "0X")):
        return spy.Integer(int(t, 16))
    try:
        return spy.Rational(t)
    except (TypeError, ValueError):
        raise ValueError(f"the number {t} has no finite value") from None


def to_str(e):
    """A sympy expression in the R syntax of the model strings."""
    return str(e).replace("**", "^")


class Switches:
    """Switching functions g_j with indicator symbols H_j = 1 where g_j > 0.

    `positive` is True (every coordinate positive), False, or the positive names; `namer`
    gives the name of the next indicator."""

    def __init__(self, positive, namer=None):
        self.positive = positive
        self.namer = namer or (lambda k: f"H{k}")
        self.funcs = []
        self.ind = []

    def is_pos(self, name):
        """Whether the coordinate `name` is declared positive."""
        return self.positive is True or (self.positive not in (False, None) and
                                         name in self.positive)

    def _assume(self, e):
        rep = {}
        for s in e.free_symbols:
            if s in self.ind:
                continue
            rep[s] = spy.Dummy(s.name, positive=True) if self.is_pos(s.name) else \
                spy.Dummy(s.name, real=True)
        return e.xreplace(rep)

    def sign_of(self, e):
        """+1, -1 or 0 when the declared signs decide the sign of e, else None."""
        e = self._assume(spy.sympify(e))
        if e.is_zero:
            return 0
        if e.is_positive:
            return 1
        if e.is_negative:
            return -1
        return None

    def match(self, p):
        """(j, +1) when p is the switching function g_j, (j, -1) for -g_j, else None."""
        for j, q in enumerate(self.funcs):
            if spy.expand(q - p) == 0:
                return j, 1
            if spy.expand(q + p) == 0:
                return j, -1
        return None

    def _register(self, p):
        flip = p.could_extract_minus_sign()
        p = -p if flip else p
        hit = self.match(p)
        if hit is not None:
            j, s = hit
            h = self.ind[j] if s > 0 else 1 - self.ind[j]
            return 1 - h if flip else h
        h = spy.Symbol(self.namer(len(self.funcs) + 1))
        self.funcs.append(p)
        self.ind.append(h)
        return 1 - h if flip else h

    def indicator(self, e):
        """The indicator of e > 0 in the H_j; the boundary e = 0 has measure zero."""
        e = spy.sympify(e)
        if e.free_symbols & set(self.ind):
            raise ValueError(f"a condition on a switched expression ({to_str(e)}) is not "
                             "supported")
        s = self.sign_of(e)
        if s is not None:
            return spy.Integer(1 if s > 0 else 0)
        num, den = spy.fraction(spy.together(e))
        sgn, parts = 1, []
        for q in (num, den):
            try:
                c, facs = spy.factor_list(q)
            except (spy.PolynomialError, NotImplementedError, TypeError):
                c, facs = 1, [(q, 1)]
            sgn *= 1 if spy.sympify(c) > 0 else -1
            for fac, m in facs:
                if m % 2 == 0:
                    continue
                d = self.sign_of(fac)
                if d is None:
                    parts.append(fac)
                elif d < 0:
                    sgn = -sgn
        # positive exactly when an even number of undecided factors is negative
        prod = spy.Integer(1)
        for p in parts:
            prod *= 2 * self._register(p) - 1
        return spy.expand((1 + sgn * prod) / 2)

    def truth(self, x):
        if isinstance(x, _Bool):
            return x.e
        s = self.sign_of(x)
        if s == 0:
            return spy.Integer(0)
        if s is not None:
            return spy.Integer(1)
        raise ValueError(f"the truth value of {to_str(x)} is not a comparison")

    def binary(self, op, a, b):
        if op in ("&&", "&"):
            return _Bool(self.truth(a) * self.truth(b))
        if op in ("||", "|"):
            ta, tb = self.truth(a), self.truth(b)
            return _Bool(ta + tb - ta * tb)
        a, b = _val(a), _val(b)
        if op in ("<", "<="):
            return _Bool(self.indicator(b - a))
        if op in (">", ">="):
            return _Bool(self.indicator(a - b))
        if op in ("==", "!="):
            s = self.sign_of(a - b)
            if s is None:
                raise ValueError(f"{to_str(a)} {op} {to_str(b)} changes only on a surface; "
                                 "write it with < or >")
            eq = spy.Integer(1 if s == 0 else 0)
            return _Bool(eq if op == "==" else 1 - eq)
        return {"+": lambda: a + b, "-": lambda: a - b, "*": lambda: a * b,
                "/": lambda: a / b, "^": lambda: a ** b}[op]()

    def _extreme(self, args, sign):
        args = [_val(a) for a in args]
        out = spy.Integer(0)
        for i, a in enumerate(args):
            w = spy.Integer(1)
            for k, b in enumerate(args):
                if k < i:
                    w *= self.indicator(sign * (a - b))
                elif k > i:
                    w *= 1 - self.indicator(sign * (b - a))
            out += w * a
        return out

    def call(self, name, args):
        if name == "piecewise":
            n = len(args)
            out = _val(args[-1]) if n % 2 else spy.Integer(0)
            for k in range(n // 2 - 1, -1, -1):
                t = self.truth(args[2 * k + 1])
                out = t * _val(args[2 * k]) + (1 - t) * out
            return out
        if name == "ifelse":
            if len(args) != 3:
                raise ValueError("ifelse() needs three arguments")
            t = self.truth(args[0])
            return t * _val(args[1]) + (1 - t) * _val(args[2])
        args = [_val(a) for a in args]
        if name == "Heaviside":
            return self.indicator(args[0])
        if name == "sign":
            return 2 * self.indicator(args[0]) - 1
        if name == "abs":
            return (2 * self.indicator(args[0]) - 1) * args[0]
        if name in ("max", "pmax"):
            return self._extreme(args, 1)
        if name in ("min", "pmin"):
            return self._extreme(args, -1)
        if name == "log" and len(args) == 2:
            return spy.log(args[0]) / spy.log(args[1])
        if name == "log10":
            return spy.log(args[0]) / spy.log(10)
        if name == "log2":
            return spy.log(args[0]) / spy.log(2)
        if name == "exp10":
            return spy.Integer(10) ** args[0]
        if name in _MATH:
            return getattr(spy, name)(*args)
        return spy.Function(name)(*args)

    def parse(self, text):
        """The expression with its switches written in the indicators."""
        return _val(_Reader(str(text), self).expr())


class _Reader:
    """Precedence parser over the R tokens of one expression."""

    def __init__(self, text, sw):
        self.toks = tokens(text)
        self.i = 0
        self.sw = sw
        self.text = text

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else (None, None)

    def take(self, want=None):
        tok = self.peek()
        if tok[0] is None or (want is not None and tok[1] != want):
            raise ValueError(f"cannot parse {self.text!r}")
        self.i += 1
        return tok

    def expr(self, rbp=0):
        left = self.prefix()
        while True:
            kind, t = self.peek()
            if kind != "op" or t not in _BIN or _BIN[t] <= rbp:
                break
            self.take()
            right = self.expr(_BIN[t] - 1 if t == "^" else _BIN[t])
            left = self.sw.binary(t, left, right)
        if rbp == 0 and self.i < len(self.toks) and self.peek()[1] not in (",", ")"):
            raise ValueError(f"cannot parse {self.text!r}")
        return left

    def prefix(self):
        kind, t = self.take()
        if kind == "num":
            return _number(t)
        if kind == "sym":
            return spy.Symbol(t.strip("`"))
        if kind == "fun":
            self.take("(")
            args = []
            while self.peek()[1] != ")":
                args.append(self.expr())
                if self.peek()[1] == ",":
                    self.take()
            self.take(")")
            return self.sw.call(t, args)
        if t == "(":
            e = self.expr()
            self.take(")")
            return e
        if t == "-":
            return -_val(self.expr(7))
        if t == "+":
            return _val(self.expr(7))
        if t == "!":
            return _Bool(1 - self.sw.truth(self.expr(3)))
        raise ValueError(f"cannot parse {self.text!r}")


def _plain(text):
    """A side of a comparison as a sympy expression; it may hold no switch."""
    sw = Switches(False)
    e = sw.parse(text)
    if sw.funcs:
        raise ValueError(f"{text!r} holds a switch")
    return e


def comparison(text):
    """The comparison `a < b` or `a > b` as (a - b, side): side +1 for a - b > 0."""
    parts = re.split(r"(<=|>=|<|>)", str(text))
    if len(parts) != 3:
        raise ValueError(f"{text!r} is not one comparison a < b or a > b")
    a, op, b = parts
    return spy.expand(_plain(a) - _plain(b)), (1 if op in (">", ">=") else -1)


class Hybrid:
    """A model whose switches are not all decided: right-hand sides and observables in the
    indicators, the switching functions and their kinds, and the root events."""

    def __init__(self, sw, f, g, g_list, states, inputs, roots):
        self.sw = sw
        self.f = f
        self.g = g
        self.g_list = g_list
        self.states = states
        self.inputs = inputs
        self.roots = roots
        dyn = {spy.Symbol(x) for x in states if x not in inputs}
        inp = {spy.Symbol(u) for u in inputs}
        self.kind = []
        for q in sw.funcs:
            fs = q.free_symbols
            if fs & dyn:
                self.kind.append("state")
            elif TIME in fs:
                self.kind.append("time")
            elif fs & inp:
                self.kind.append("input")
            else:
                self.kind.append("param")

    @property
    def names(self):
        """The indicator names in the order of the surfaces."""
        return [str(h) for h in self.sw.ind]

    def text(self, j, side):
        """The side of surface j as a comparison."""
        return f"{to_str(self.sw.funcs[j])} {'>' if side > 0 else '<'} 0"

    def surface(self, j):
        """Surface j as an equation."""
        return f"{to_str(self.sw.funcs[j])} = 0"


def hybrid_model(f, g, positive, events=None, inputs=(), taken=()):
    """The model with its switches resolved, or as a Hybrid.

    Returns None when the model has no switch and no root event, {'hybrid': False, 'f',
    'g'} when the declared signs decide every switch, else a Hybrid. `taken` holds names
    the indicators must avoid."""
    obs_list = isinstance(g, (list, tuple))
    gl = [dict(x or {}) for x in g] if obs_list else [dict(g or {})]
    events = list(events or [])
    roots = [e for e in events if e.get("root") is not None]
    exprs = list(f.values()) + [v for x in gl for v in x.values()]
    if not roots and not any(_SWITCHY.search(str(e)) or _SIGNY.search(str(e)) for e in exprs):
        return None
    pos = positive if positive in (True, False) else set(positive or [])
    used = set(taken) | set(f) | {k for x in gl for k in x}
    for e in exprs:
        used |= {t for kind, t in tokens(e) if kind == "sym"}

    def namer(k):
        nm = f"si_sw{k}"
        while nm in used:
            nm += "_"
        used.add(nm)
        return nm
    sw = Switches(pos, namer)
    F = {k: sw.parse(v) for k, v in f.items()}
    Gs = [{k: sw.parse(v) for k, v in x.items()} for x in gl]
    root_of = []
    for e in roots:
        h = sw.indicator(sw.parse(str(e["root"])))
        hit = [j for j, s in enumerate(sw.ind) if s in h.free_symbols]
        single = len(hit) == 1 and (spy.expand(h - sw.ind[hit[0]]) == 0 or
                                    spy.expand(h - 1 + sw.ind[hit[0]]) == 0)
        if not single:
            raise ValueError(f"the root {e['root']} of an event is not one switching surface; "
                             "give the event a root that changes sign on a single surface")
        root_of.append({"event": e, "surface": hit[0]})
    if not sw.funcs:
        if any(_SWITCHY.search(str(e)) for e in exprs):
            return {"hybrid": False, "f": {k: to_str(v) for k, v in F.items()},
                    "g": [{k: to_str(v) for k, v in x.items()} or None for x in Gs]
                    if obs_list else ({k: to_str(v) for k, v in Gs[0].items()} or None)}
        return None
    states = list(F)
    return Hybrid(sw, F, Gs, obs_list, states, [u for u in inputs if u in F], root_of)


def surface_ratios(surfaces):
    """Ratios of the terms of each surface function: a scaling maps g = 0 to itself
    exactly when it scales every term of g alike, which leaves these ratios invariant."""
    out = []
    for q in surfaces:
        terms = spy.Add.make_args(spy.expand(spy.fraction(spy.together(q))[0]))
        out += [to_str(tm / terms[0]) for tm in terms[1:]]
    return out
