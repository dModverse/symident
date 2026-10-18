"""The Lie order of the symbolic engine, certified exactly."""

import numpy as np
import sympy as spy
from sympy.polys.matrices import DomainMatrix

from .. import _core
from ..errors import SYMPY_FAILS, InapplicableError, sympy_fail
from ..gfp import PRIMES, random_point
from .taylor import _eval

# size of a Lie derivative (count_ops) past which the certificate is not attempted
_OPS = 20000


def _exp_atom(a):
    """(w, base, c) for exp(c*w) or base**(c*w) with c an integer and w a symbol; else None."""
    if isinstance(a, spy.exp):
        base, arg = spy.E, a.args[0]
    elif isinstance(a, spy.Pow) and a.base.is_Rational and a.base > 0:
        base, arg = a.base, a.exp
    else:
        return None
    c, w = arg.as_coeff_Mul()
    if not w.is_Symbol or not c.is_Integer:
        return None
    return w, base, int(c)


class Atoms:
    """Symbols for exponentials of single symbols and one logarithm of a positive rational, collected over calls."""

    def __init__(self):
        self.sig, self.base_of, self.const = {}, {}, None

    def add(self, exprs):
        """None, or why an atom of `exprs` has no symbol: another function or a non-integer power."""
        for e in exprs:
            for a in e.atoms(spy.Function, spy.Pow):
                if a in self.sig or isinstance(a, spy.Pow) and a.exp.is_Integer:
                    continue
                if isinstance(a, spy.log) and a.args[0].is_Rational and a.args[0] > 0:
                    if self.const not in (None, a):
                        return "two logarithms of numbers"
                    self.const = a
                    self.sig[a] = spy.Symbol("_log_const")
                    continue
                ea = _exp_atom(a)
                if ea is None:
                    return f"{a} is not rational"
                w, base, c = ea
                if self.base_of.setdefault(w, base) != base:
                    return f"two exponentials of {w}"
                self.sig[a] = spy.Symbol(f"_exp_{w}") ** c
        return None

    def symbols(self):
        return [spy.Symbol(f"_exp_{w}") for w in self.base_of] + [spy.Symbol("_log_const")]

    def sub(self, rows):
        """`rows` with the atoms replaced; ValueError where one has no symbol."""
        why = self.add([e for r in rows for e in r])
        if why is not None:
            raise InapplicableError(why)
        return [[e.xreplace(self.sig) for e in r] for r in rows]


def _rank_mod(rows, p):
    if not rows:
        return 0
    return int(_core.sym_rref_mod(np.asarray(rows, dtype=np.float64), p)["rank"])


def _values(rows, env, p, memo):
    return [[_eval(e, env, 1, p, memo)[0] for e in r] for r in rows]


class _Condition:
    """Rows of one condition over its states and parameters, order by order, resting states at their initial value."""

    def __init__(self, f, g, ic, S, fixed):
        ic = {X: spy.sympify(ic[X]) for X in S}
        rest = {X for X in S if not ic[X].free_symbols & set(S)}
        while True:
            sub = {X: ic[X] for X in rest}
            moving = {X for X in rest if spy.cancel(f[X].xreplace(sub)) != 0}
            if not moving:
                break
            rest -= moving
        sub = {X: ic[X] for X in rest}
        self.S = [X for X in S if X not in rest]
        self.f = {X: f[X].xreplace(sub) for X in self.S}
        self.ic = {X: ic[X] for X in self.S}
        self.jets = [spy.sympify(h).xreplace(sub) for h in g]
        syms = set().union(set(), *[e.free_symbols for e in list(self.f.values()) + self.jets])
        params = sorted(syms - set(self.S) - set(fixed), key=str)
        self.W = self.S + params
        self.orders = []

    def rows(self, k):
        """The rows up to order k."""
        while len(self.orders) <= k:
            if any(spy.count_ops(h) > _OPS for h in self.jets):
                raise OverflowError("a Lie derivative grows past the size cap")
            self.orders.append([[spy.diff(h, w) for w in self.W] for h in self.jets])
            self.jets = [spy.expand(sum((spy.diff(h, x) * self.f[x] for x in self.S), spy.Integer(0)))
                         for h in self.jets]
        return [r for o in self.orders[:k + 1] for r in o]


def _certify(cond, cap, p, seed):
    """The order from which the restricted rows of one condition add nothing, or why there is none."""
    n = len(cond.W)
    if n == 0:
        return 0
    atoms = Atoms()
    sic = atoms.sub([list(cond.ic.values())])[0]
    base = set().union(set(), *[e.free_symbols for e in list(cond.f.values()) + cond.jets])
    atoms.add(list(cond.f.values()) + cond.jets)
    syms = sorted(base | set(atoms.symbols()) | set().union(set(), *[e.free_symbols for e in sic]), key=str)
    env = dict(zip(syms, (int(x) for x in random_point(len(syms), seed))))
    ranks, k0, memo = [], None, {}
    for k in range(n + 2):
        ranks.append(_rank_mod(_values(atoms.sub(cond.rows(k)), env, p, memo), p))
        if k and ranks[k] == ranks[k - 1]:
            k0 = k - 1
            break
    if k0 is None:
        return "the generic rank does not settle"
    r = ranks[k0]
    dm = DomainMatrix.from_Matrix(spy.Matrix(atoms.sub(cond.rows(k0 + 1))))
    if dm.domain.is_EX:
        return "the rows have no exact domain"
    if dm.rank() != r:
        return "the generic rank at a point is below the exact rank"
    # the initial values: each state at its initial value at the same point
    at, mem = dict(env), {}
    for X, e in zip(cond.S, sic):
        at[X] = _eval(e, env, 1, p, mem)[0]
    mem = {}
    for k in range(k0, max(cap, k0) + n + 1):
        if _rank_mod(_values(atoms.sub(cond.rows(k)), at, p, mem), p) == r:
            return k
    return "the rows at the initial values stay below the generic rank"


def lie_certificate(per_cond, S, fixed=(), order=0):
    """{"order": N} past which no condition's restricted rows add, or {"order": None, "why": reason}."""
    p = PRIMES[0]
    out = 0
    for ci, (f, g, ic) in enumerate(per_cond):
        try:
            cond = _Condition(dict(zip(S, [spy.sympify(e) for e in f])), g, ic, S, {spy.Symbol(str(s)) for s in fixed})
            got = _certify(cond, order, p, 7919 * (ci + 1))
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            got = str(exc) or type(exc).__name__
        if isinstance(got, str):
            return {"order": None, "why": got + (f" (condition {ci + 1})" if len(per_cond) > 1 else "")}
        out = max(out, got)
    return {"order": out, "why": None}


def kernel_exact(rows, vectors, z):
    """Whether the vectors, aligned with z, annihilate the rows identically and span their kernel at a point."""
    p = PRIMES[0]
    try:
        atoms = Atoms()
        srows = atoms.sub(rows)
        svecs = atoms.sub([[spy.sympify(c) for c in v] for v in vectors])
        for v in svecs:
            for r in srows:
                s = sum((a * b for a, b in zip(r, v)), spy.Integer(0))
                if spy.cancel(spy.together(s)) != 0:
                    return False
        syms = sorted(set().union(set(), *[e.free_symbols for r in srows + svecs for e in r]), key=str)
        env = dict(zip(syms, (int(x) for x in random_point(len(syms), 104723))))
        memo = {}
        rk = _rank_mod(_values(srows, env, p, memo), p)
        rv = _rank_mod(_values(svecs, env, p, memo), p)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return False
    return rv == len(vectors) and rk == len(z) - rv
