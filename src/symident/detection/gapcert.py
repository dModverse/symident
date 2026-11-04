"""An upper bound of the rank over the gap series; a truncated rank that reaches it proves the gap order."""

import random

import numpy as np
import sympy as spy

from .. import _core
from ..errors import SYMPY_FAILS, sympy_fail
from ..sym.parse import model_sympify
from .bound import field_degrees, lie_row_degree, rows_degree

# size of a Lie derivative (count_ops) past which no bound is attempted
_OPS = 20000


def _lines(lines):
    out = {}
    for ln in lines or []:
        lhs, rhs = str(ln).split("=", 1)
        out[lhs.strip()] = model_sympify(rhs.strip())
    return out


def _lie(e, F):
    return spy.expand(sum((spy.diff(e, x) * fx for x, fx in F.items()), spy.Integer(0)))


def _values(rows, env, p):
    # an exponential left after the substitution takes an independent random value; a bound stays an upper bound
    rng = random.Random(p)
    gen = {}
    out = []
    for r in rows:
        row = []
        for e in r:
            v = spy.sympify(e).xreplace(env)
            if not v.is_Rational and v.has(spy.exp):
                for a in sorted(v.atoms(spy.exp), key=str):
                    gen.setdefault(a, spy.Integer(rng.randrange(2, p)))
                v = v.xreplace(gen)
            if not v.is_Rational:
                return None
            row.append(int(v.p) * pow(int(v.q), p - 2, p) % p)
        out.append(row)
    return out


def _rank(rows, p):
    if not rows:
        return 0
    return int(_core.sym_rref_mod(np.asarray(rows, dtype=np.float64), p)["rank"])


def _event_map(tape, S):
    E = {x: x for x in S}
    for ev in tape.get("ver_ev") or []:
        x, v = spy.Symbol(ev["var"]), model_sympify(str(ev["value"]))
        if x not in E:
            raise LookupError(f"an event on {x}, which is no state of the segment")
        E[x] = {"replace": v, "add": E[x] + v, "multiply": E[x] * v}[ev.get("method", "replace")]
    return E


def _plateau_term(F, fam, W, top):
    """Schwartz-Zippel numerator of a plateau of the ranks of d_W L_F^k fam at a random point, None without degrees."""
    md = field_degrees(list(F.values()), list(fam), W)
    if md is None:
        return None
    n = len(W)
    # a maximal minor at each order up to the one at which a strictly rising generic rank must stop
    return sum(min((k + 1) * max(1, len(fam)), n) * lie_row_degree(md, k) for k in range(min(n + 1, top) + 1))


def _stationary(F, fam, W, p, rng, top):
    """Order past which the rows d_W L_F^k of `fam` add nothing generically, the functions up to it, _plateau_term()."""
    term = _plateau_term(F, fam, W, top)
    gen = {w: spy.Integer(rng.randrange(2, p)) for w in W}
    jets, out, rows, prev = list(fam), [], [], -1
    for k in range(top + 1):
        if any(spy.count_ops(g) > _OPS for g in jets):
            raise OverflowError("a Lie derivative grows past the size cap")
        vals = _values([[spy.diff(g, w) for w in W] for g in jets], gen, p)
        if vals is None:
            return None
        rows += vals
        r = _rank(rows, p)
        if r == prev:
            return k - 1, out, term
        prev = r
        out += jets
        jets = [_lie(g, F) for g in jets]
    return None


class _Chain:
    """The coefficient rows of one chain over the states and parameters of its first segment."""

    def __init__(self, tapes, chain, n_lie, p, rng):
        segs = [tapes[i] for i in chain]
        Fs = [{spy.Symbol(x): v for x, v in _lines(t.get("model_lines")).items()} for t in segs]
        hs = [list(_lines(t.get("obs_lines")).values()) for t in segs]
        self.S = list(Fs[0])
        fam, self.term = [], 0
        for h in hs[-1]:
            g = h
            for _ in range(n_lie + 1):
                fam.append(spy.expand(g))
                g = _lie(g, Fs[-1])
                if spy.count_ops(g) > _OPS:
                    raise OverflowError("a Lie derivative grows past the size cap")
        for s in range(len(segs) - 2, -1, -1):
            E = _event_map(segs[s + 1], list(Fs[s]))
            base = list(hs[s]) + [spy.expand(f.xreplace(E)) for f in fam]
            if s == 0:
                fam = base
                break
            W = self._vars(Fs[s], base)
            st = _stationary(Fs[s], base, W, p, rng, 2 * len(W) + 4)
            if st is None:
                raise OverflowError("no stationary order within the cap")
            fam = st[1]
            self.term = None if self.term is None or st[2] is None else self.term + st[2]
        if len(segs) == 1:
            fam = list(hs[0])
        self.F = Fs[0]
        self.ic = {spy.Symbol(x): v for x, v in _lines(segs[0].get("ver_ic")).items()}
        self.jets, self.orders = fam, []
        self.W = self._vars(self.F, fam)
        self.params = [w for w in self.W if w not in set(self.S)]

    def _vars(self, F, fam):
        free = set().union(*[e.free_symbols for e in list(F.values()) + list(fam)])
        S = list(F)
        return S + sorted(free - set(S), key=str)

    def rows(self, k):
        """The rows d_W L_{F1}^i g up to order k."""
        while len(self.orders) <= k:
            if any(spy.count_ops(g) > _OPS for g in self.jets):
                raise OverflowError("a Lie derivative grows past the size cap")
            self.orders.append([[spy.diff(g, w) for w in self.W] for g in self.jets])
            self.jets = [_lie(g, self.F) for g in self.jets]
        return [r for o in self.orders[:k + 1] for r in o]


def gap_bound(tapes, chains, znames, n_lie, p, rng, cap=None):
    """{"bound": U} with U at least the rank over the gap series, or {"bound": None, "why"}."""
    try:
        return _gap_bound(tapes, chains, znames, n_lie, p, rng, cap)
    except OverflowError as exc:
        return {"bound": None, "why": f"no symbolic family ({exc})"}


def _gap_bound(tapes, chains, znames, n_lie, p, rng, cap):
    # a moving boundary shifts the series and the state at the event, which the family leaves out
    for c in chains:
        if any(str(tapes[i].get("ver_time") or "").strip() and model_sympify(str(tapes[i]["ver_time"])).free_symbols
               for i in c):
            return {"bound": None, "why": "an event time with a free symbol"}
    try:
        cs = [_Chain(tapes, c, n_lie, p, rng) for c in chains]
    except (OverflowError, LookupError, *SYMPY_FAILS) as exc:
        if not isinstance(exc, (OverflowError, LookupError)):
            sympy_fail(exc)
        return {"bound": None, "why": str(exc) if isinstance(exc, LookupError) else
                f"no symbolic family ({type(exc).__name__})"}
    zs = [spy.Symbol(z) for z in znames]
    zval = {z: spy.Integer(rng.randrange(2, p)) for z in zs}
    total, term, tdeg = [], 0, 0
    for ch in cs:
        if any(str(w) in ("t", "time") for w in ch.W):
            return {"bound": None, "why": "a field that depends on time"}
        top = cap or 2 * len(ch.W) + 4
        t = _plateau_term(ch.F, ch.jets, ch.W, top)
        term = None if term is None or ch.term is None or t is None else term + ch.term + t
        # the generic rank, at a random point of states and parameters
        gen = {w: spy.Integer(rng.randrange(2, p)) for w in ch.W}
        prev, rho, kg = -1, None, None
        for k in range(top + 1):
            vals = _values(ch.rows(k), gen, p)
            if vals is None:
                return {"bound": None, "why": "a row that is not rational"}
            r = _rank(vals, p)
            if r == prev:
                rho, kg = r, k - 1
                break
            prev = r
        if rho is None:
            return {"bound": None, "why": "no stationary order within the cap"}
        # the rows at the start, the initial values a function of the coordinates; the
        # coordinate named like a state is its initial value
        ren = {z: spy.Symbol(f"{z}__z") for z in zs}
        x1 = {x: ch.ic.get(x, x).xreplace(ren) for x in ch.S}
        at = {**{x: x1[x] for x in ch.S}, **{w: ren.get(w, w) for w in ch.params}}
        zv = {ren[z]: zval[z] for z in zs}
        J = [[spy.diff(at[w], ren[z]) for z in zs] for w in ch.W]
        reach = None
        for k in range(kg, top + 1):
            rows = [[e.xreplace(at) for e in r] for r in ch.rows(k)]
            vals = _values(rows, zv, p)
            if vals is None:
                return {"bound": None, "why": "a row at the start that is not rational"}
            if _rank(vals, p) >= rho:
                reach = (rows, vals)
                break
        if reach is None:
            return {"bound": None, "why": "the rows at the start do not reach the generic rank"}
        Jv = _values(J, zv, p)
        if Jv is None:
            return {"bound": None, "why": "initial values that are not rational"}
        A = np.asarray(reach[1], dtype=object)
        B = np.asarray(Jv, dtype=object)
        total += [[int(sum(int(a) * int(b) for a, b in zip(A[i], B[:, j])) % p) for j in range(len(zs))]
                  for i in range(A.shape[0])]
        dr, dj = rows_degree(reach[0]), rows_degree([[e for r in J for e in r]])
        tdeg = None if tdeg is None or dr is None or dj is None else max(tdeg, dr + dj)
    U = _rank(total, p)
    # a rank of the rows at the start below the generic one at zval lowers the bound: a maximal minor
    term = None if term is None or tdeg is None else term + min(len(total), len(zs)) * tdeg
    return {"bound": U, "why": None, "terms": None if term is None else float(term) / (p - 2)}
