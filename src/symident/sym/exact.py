"""Exact certificates of a field Y = eta(theta) d/dtheta + xi(x, theta) d/dx: every identity
that makes Y a symmetry is cancelled to zero over Q. Each identity holds the symbols of one
equation, so the cost grows with the number of equations, not with their product."""

import functools
import random

import sympy as spy

from ..errors import SYMPY_FAILS, sympy_fail
from .parse import model_sympify

# operations of one identity beyond which it is left undecided rather than cancelled
SIZE_CAP = 20000


@functools.lru_cache(maxsize=4096)
def _expr(text):
    return model_sympify(text)


def _zero(e):
    """True or False when `e` cancels to zero or not over Q; None past SIZE_CAP."""
    if spy.count_ops(e) > SIZE_CAP:
        return None
    try:
        return spy.cancel(spy.together(e)) == 0
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None


class _Field:
    """A field as SymPy expressions with its derivation."""

    def __init__(self, field):
        self.c = {spy.Symbol(str(k)): _expr(str(v)) for k, v in dict(field).items()}

    def __call__(self, e):
        """Y(e): the derivative of `e` along the field."""
        return sum((v * spy.diff(e, k) for k, v in self.c.items() if k in e.free_symbols), spy.Integer(0))

    def at(self, s):
        """The component on the symbol `s`."""
        return self.c.get(s, spy.Integer(0))


def _rhs(lines):
    out = {}
    for line in lines:
        lhs, rhs = str(line).split("=", 1)
        out[spy.Symbol(lhs.strip())] = _expr(rhs)
    return out


def regime_identities(regime, field):
    """(label, expression) of [Y, F] per state equation and Y(h) per observable."""
    Y = _Field(field)
    F = _rhs(regime["f"])
    out = []
    for x, fx in F.items():
        xi = Y.at(x)
        out.append((f"[Y, F] on {x}", Y(fx) - sum((F[s] * spy.diff(xi, s) for s in xi.free_symbols if s in F),
                                               spy.Integer(0))))
    for line in regime["g"]:
        h = _expr(str(line).split("=", 1)[1]) if "=" in str(line) else _expr(str(line))
        out.append((f"Y({h})", Y(h)))
    return out


def event_identities(events, field_from, field_to, states):
    """(label, expression) of xi_to(E(x)) = Y_from(E(x)) per state, E the event group
    (list of {var, method, value}) applied together."""
    Yf, Yt = _Field(field_from), _Field(field_to)
    E = {}
    for ev in events:
        s, v = spy.Symbol(str(ev["var"])), _expr(str(ev["value"]))
        E[s] = {"add": s + v, "replace": v, "multiply": s * v}[ev["method"]]
    out = []
    for name in states:
        s = spy.Symbol(str(name))
        Es = E.get(s, s)
        out.append((f"event on {s}", Yt.at(s).xreplace(E) - Yf(Es)))
    for ev in events:
        if "time" in ev and ev["time"] is not None:
            out.append((f"event time {ev['time']}", Yf(_expr(str(ev["time"])))))
    return out


def initial_identities(ic_lines, field):
    """(label, expression) of xi_X(x0) = Y(x0_X) for the initial values "X = x0_X"."""
    Y = _Field(field)
    x0 = _rhs(ic_lines)
    return [(f"initial value of {x}", Y.at(x).xreplace(x0) - Y(e)) for x, e in x0.items()]


def _lie(F, e):
    return sum((fx * spy.diff(e, x) for x, fx in F.items() if x in e.free_symbols), spy.Integer(0))


# Lie orders and operations of one derivative up to which a polynomial output is sought
_OUTPUT_ORDERS = 3
_OUTPUT_OPS = 2000


def _vanishes(e, rng):
    """Whether `e` cancels to zero, False already at a nonzero value at a random rational point; None past the cap."""
    if spy.count_ops(e) > _OUTPUT_OPS:
        return None
    pt = {s: spy.Rational(rng.randint(2, 997), rng.randint(2, 997)) for s in e.free_symbols}
    try:
        v = e.xreplace(pt)
        if v.is_Rational and v != 0:
            return False
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        pass
    return _zero(e)


def output_identities(regime, ic_lines, vector):
    """(label, v(L^k h(x0))) per output h, k = 0 at rest, else below its first vanishing Lie derivative; or None."""
    F = _rhs(regime["f"])
    x0 = _rhs(ic_lines)
    V = _Field(vector)
    rng = random.Random(1)
    obs = [_expr(str(l).split("=", 1)[1]) if "=" in str(l) else _expr(str(l)) for l in regime["g"]]
    at_rest = all(_vanishes(fx.xreplace(x0), rng) is True for fx in F.values())
    out = []
    for h in obs:
        series = [h]
        while not at_rest:
            if len(series) > _OUTPUT_ORDERS:
                return None
            nxt = _lie(F, series[-1])
            z = _vanishes(nxt, rng)
            if z is None:
                return None
            if z:
                break
            series.append(nxt)
        out += [(f"order {k} of {h}", V(e.xreplace(x0))) for k, e in enumerate(series)]
    return out


# equations up to which a parametrization of the resting variety is sought
_REST_EQS = 12


def _triangular(nums, rest):
    """(substitution, numerators left, symbols left), the rest symbols solved one at a time from linear numerators."""
    sub, todo, left = {}, list(nums), list(rest)

    def degree(n, r):
        try:
            return spy.Poly(n, r).degree()
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            return None
    while left:
        todo = [spy.numer(spy.together(n.xreplace(sub))) for n in todo]
        todo = [n for n in todo if n != 0]
        best = None
        for i, n in enumerate(todo):
            for r in left:
                if r in n.free_symbols and degree(n, r) == 1:
                    key = (sum(r in m.free_symbols for m in todo), left.index(r))
                    if best is None or key < best[0]:
                        best = (key, i, r)
        if best is None:
            break
        _, i, r = best
        c1, c0 = spy.Poly(todo[i], r).all_coeffs()
        val = spy.cancel(-c0 / c1)
        sub = {k: spy.cancel(v.xreplace({r: val})) for k, v in sub.items()}
        sub[r] = val
        left.remove(r)
        del todo[i]
    rem = [spy.numer(spy.together(n.xreplace(sub))) for n in todo]
    return sub, [n for n in rem if n != 0], left


def rest_parametrization(eqs, rest, params):
    """{"sub", "params", "eqs"} parametrizing eqs = 0 by `rest`, then by `params` entering linearly; or None."""
    if not eqs or len(eqs) > _REST_EQS:
        return None
    try:
        nums = [spy.numer(spy.together(e)) for e in eqs]
        sub, rem, left = _triangular(nums, rest)
        if not rem:
            return {"sub": sub, "params": [], "eqs": []}
        rng = random.Random(len(eqs))
        free = sorted(set().union(*[n.free_symbols for n in rem]), key=str)
        pt = {x: spy.Rational(rng.randint(2, 997), rng.randint(2, 997)) for x in free}
        chosen = []
        for p in params:
            trial = chosen + [p]
            if not any(p in n.free_symbols for n in rem) or \
                    any(spy.Poly(n, *trial).total_degree() > 1 for n in rem):
                continue
            A = spy.Matrix([[spy.diff(n, q).xreplace(pt) for q in trial] for n in rem])
            if A.rank() == len(trial):
                chosen = trial
            if len(chosen) == len(rem):
                break
        if len(chosen) < len(rem):
            return None
        sol = spy.solve(rem, chosen, dict=True)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    if len(sol) != 1 or set(sol[0]) != set(chosen):
        return None
    ps = sol[0]
    out = {k: spy.cancel(v.xreplace(ps)) for k, v in sub.items()}
    out.update(ps)
    return {"sub": out, "params": chosen, "eqs": rem}


def rest_identities(conds, vector):
    """(label, v(h) + h_x xi) with J_x xi = -J_theta v, per condition (F, states, observables) at rest."""
    out = []
    for k, (F, xs, obs) in enumerate(conds, start=1):
        vec = {spy.Symbol(str(a)): _expr(str(b)) for a, b in vector.items()}
        Jx = spy.Matrix([[spy.diff(F[x], y) for y in xs] for x in xs])
        b = spy.Matrix([-sum((spy.diff(F[x], p) * c for p, c in vec.items()), spy.Integer(0)) for x in xs])
        xi = Jx.LUsolve(b) if xs else spy.Matrix([])
        for h in obs:
            e = sum((spy.diff(h, p) * c for p, c in vec.items()), spy.Integer(0)) + \
                sum((spy.diff(h, y) * xi[j] for j, y in enumerate(xs)), spy.Integer(0))
            out.append((f"resting output {h} of condition {k}", e))
    return out


def certify(identities):
    """{"ok": True when every identity cancels, False with the first that does not, None
    when one is past the size cap; "why"; "identities": the count}."""
    undecided = None
    for label, e in identities:
        z = _zero(e)
        if z is False:
            return {"ok": False, "why": f"{label} does not vanish", "identities": len(identities)}
        if z is None and undecided is None:
            undecided = label
    if undecided is not None:
        return {"ok": None, "why": f"{undecided} is too large to cancel", "identities": len(identities)}
    return {"ok": True, "why": "", "identities": len(identities)}
