"""Hybrid models: the rank where stacked regimes and exact symmetries of every regime give equal bounds."""

import itertools
import re
import time

import numpy as np
import sympy as spy

from .. import session, state
from ..gfp import PRIMES
from ..rexpr import sym_sort
from ..sym.scaling import scaling_symmetries_multi
from ..sym.switching import BACK, CONSTS, TIME, comparison, surface_ratios, to_str
from ..sym.taylor import NoSeriesError, output_series
from .conditions import Grid, as_events, num_str
from .finalize import finalize, public_symmetry

_MAX_REGIMES = 4096
_VISIBLE, _INVISIBLE, _UNDECIDED = "visible", "invisible", "undecided"


def _sym(x):
    return spy.Symbol(str(x))


def _parse(text):
    """A model string (R syntax) as a sympy expression."""
    from ..sym.switching import _plain
    return _plain(str(text)).xreplace(CONSTS)


def _rational(t):
    """A numeric time as a rational number."""
    return spy.Rational(num_str(float(t)))


def _rational_point(syms, *key):
    rng = session.current().rng("hybrid.point", *key)
    return {s: spy.Rational(rng.randint(2, 89), rng.randint(2, 89)) for s in sorted(syms, key=str)}


def _value(e, pt):
    """e at the point as a rational number, None for a pole or a value that is not rational."""
    try:
        v = spy.sympify(e).xreplace(pt)
    except (ZeroDivisionError, TypeError, ValueError):
        return None
    return v if v.is_Rational else None


def is_zero(e):
    """Whether `e` vanishes identically, refuted at rational points and proven by simplification."""
    e = spy.sympify(e)
    if e == 0:
        return True
    for i in range(2):
        v = _value(e, _rational_point(e.free_symbols, str(e), i))
        if v is not None and v != 0:
            return False
    try:
        r = spy.cancel(spy.together(e))
    except (spy.PolynomialError, TypeError):
        r = e
    if r == 0:
        return True
    return spy.simplify(r) == 0


def solve_linear(g, prefer):
    """(s, expr) solving g = 0 for the first symbol of `prefer` in which the numerator of g is linear, else None."""
    num = spy.fraction(spy.together(spy.sympify(g)))[0]
    for s in prefer:
        if s not in num.free_symbols:
            continue
        try:
            p = spy.Poly(num, s)
        except spy.PolynomialError:
            continue
        if p.degree() != 1:
            continue
        a1, a0 = p.all_coeffs()
        if is_zero(a1):
            continue
        return s, spy.cancel(-a0 / a1)
    return None


class HybridError(ValueError):
    """A hybrid model the analysis does not support or cannot certify."""


class _Cond:
    """One condition in the coordinates of the model: substitutions, initial values, observables, events."""

    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.ic = {}
        self.obs = {}
        self.events = []
        self.equil = False


class _Setup:
    """The hybrid model with conditions, regimes and surfaces."""

    def __init__(self, hy, kw, regimes=None):
        self.hy = hy
        self.kw = kw
        self.sw = hy.sw
        if kw.get("gauge_preference", False) is not False:
            raise HybridError("a model with switches does not support `gauge`")
        for bad, why in (("reduce_cq", "`reduce_cq`"), ("free_initial", "`free_initial`")):
            if kw.get(bad) not in (None, False, [], ()):
                raise HybridError(f"a model with switches does not support {why}")
        if kw.get("sym_engine", "modular") != "modular":
            raise HybridError('a model with switches needs engine="modular"')
        self.positive = kw.get("positive", True)
        self.fixed = [str(x) for x in (kw.get("fixed") or [])]
        self.inputs = [str(u) for u in (kw.get("forcings") or [])]
        self.unknown = [str(w) for w in (kw.get("unknown_inputs") or [])]
        self.equilibrate = bool(kw.get("equilibrate"))
        self.f = dict(hy.f)
        if "time" in self.f:
            raise HybridError("a state named `time` clashes with the clock")
        self.root_events = [r["event"] for r in hy.roots]
        self.time_events = as_events([e for e in (kw.get("events") or []) if e.get("root") is None])
        self.dyn = [x for x in self.f if x not in self.inputs]
        self._conditions()
        if self.unknown:
            self._check_unknown()
        # the clock where a condition's model reads time
        self.clock = any(TIME in cd.apply(e).free_symbols for cd in self.conds for e in self.f.values()) or \
            any(TIME in e.free_symbols for cd in self.conds for e in cd.obs.values())
        exprs = list(self.f.values()) + [e for x in hy.g for e in x.values()] + list(self.sw.funcs)
        for cd in self.conds:
            exprs += [e for e in cd.ic.values() if e is not None] + list(cd.obs.values())
            exprs += [e[k] for e in cd.events for k in ("time", "value")]
            exprs += list(cd.sub_t.values()) + list(cd.sub_g.values())
        used = set().union(*[spy.sympify(e).free_symbols for e in exprs])
        self.fixed += [str(s) for s in CONSTS.values() if s in used and str(s) not in self.fixed]
        self.kw = dict(kw, fixed=list(self.fixed))
        self.regimes, self.excluded = self._regimes()
        self.roots = {r["surface"] for r in hy.roots}
        decl = self._declarations(regimes)
        for cd in self.conds:
            self._select(cd, decl[cd.index])
            cd.vis = {j: self.visibility(cd, j) for j in cd.surfaces}

    def _check_unknown(self):
        """HybridError where an unknown input meets a part of the model that the rule does not cover."""
        us = {_sym(w) for w in self.unknown}
        for j, q in enumerate(self.sw.funcs):
            hit = sorted(str(s) for s in q.free_symbols & us)
            if hit:
                raise HybridError(f"a switch on the unknown input {hit[0]} ({self.hy.surface(j)}) is not supported")
        if self.root_events or any(not e["t0"] for cd in self.conds for e in cd.events):
            raise HybridError("unknown inputs together with switches and events after the start are not "
                              "supported yet")
        seen = {}
        for cd in self.conds:
            vals = [v for v in cd.ic.values() if v is not None] + [e["value"] for e in cd.events]
            hit = sorted(str(s) for v in vals for s in v.free_symbols & us)
            if hit:
                raise HybridError(f"an initial value or event reads the unknown input {hit[0]}")
            exprs = [cd.apply(e) for e in self.f.values()] + list(cd.obs.values())
            for u in set().union(*[e.free_symbols for e in exprs]) & us:
                seen.setdefault(str(u), []).append(cd.name)
        shared = sorted(u for u, c in seen.items() if len(c) > 1)
        if shared:
            raise HybridError(f"the unknown input {shared[0]} enters several conditions; with switches each "
                              "condition needs an input of its own, renamed in `conditions`")

    def _conditions(self):
        kw, hy = self.kw, self.hy
        grid = Grid.from_any(kw.get("conditions"))
        trafo = kw.get("trafo")
        tlist = isinstance(trafo, (list, tuple))
        K = max(1, grid.nrow if grid is not None else 0, len(trafo) if tlist else 0,
                len(hy.g) if hy.g_list else 0)
        names = list(grid.rows) if grid is not None else [str(i + 1) for i in range(K)]
        self.grid = grid
        states = set(self.f)
        single = {} if tlist or trafo is None else {str(k): str(v) for k, v in dict(trafo).items()}
        t_sub = {_sym(k): _parse(v) for k, v in single.items() if k not in states}
        conds = []
        for c in range(K):
            cd = _Cond(c, names[c])
            per = {} if not tlist else {str(k): str(v) for k, v in dict(trafo[c] or {}).items()}
            gp = {}
            if grid is not None:
                for col in grid.colnames():
                    if col in states:
                        continue
                    gp[_sym(col)] = _parse(grid.cell(c, col))
            gp.update({_sym(k): _parse(v) for k, v in per.items() if k not in states})
            cd.sub_t, cd.sub_g = t_sub, gp

            def app(e, t_sub=t_sub, gp=gp):
                return spy.sympify(e).xreplace(t_sub).xreplace(gp)
            cd.apply = app
            ic = {}
            for x in self.dyn:
                v = per.get(x, single.get(x))
                if v is None and grid is not None and x in grid.colnames():
                    v = grid.cell(c, x)
                ic[x] = None if v is None else app(_parse(v))
            for u in self.inputs:
                ic[u] = spy.Integer(0)
            cd.ic = ic
            g = hy.g[c] if hy.g_list else hy.g[0]
            cd.obs = {k: app(v) for k, v in g.items()}
            cd.events = []
            for i, e in enumerate(self.time_events):
                tv = e["time"]
                vv = e["value"]
                if grid is not None and tv in grid.colnames():
                    tv = grid.cell(c, tv)
                if grid is not None and vv in grid.colnames():
                    vv = grid.cell(c, vv)
                T = app(_parse(tv))
                # an event value reads the clock at the event time
                cd.events.append({"var": e["var"], "time": T, "value": app(_parse(vv)).xreplace({TIME: T}),
                                  "method": e["method"], "index": i})
            cd.equil = self.equilibrate
            conds.append(cd)
        # the start: 0, or the earliest event time that is a number in a condition
        self.t0e = min([_rational(e["time"]) for cd in conds for e in cd.events if e["time"].is_number] +
                       [spy.Integer(0)])
        self.t0 = float(self.t0e)
        for cd in conds:
            for e in cd.events:
                e["t0"] = bool(e["time"].is_number and _rational(e["time"]) <= self.t0e)
            cd.ic = {x: None if v is None else v.xreplace({TIME: self.t0e}) for x, v in cd.ic.items()}
        self.conds = conds

    def model(self, cd, sides):
        """Right-hand sides and observables of condition `cd` in the regime `sides`."""
        rep = {h: spy.Integer(1 if sides[j] > 0 else 0) for j, h in enumerate(self.sw.ind)}
        f = {k: cd.apply(v.xreplace(rep)) for k, v in self.f.items()}
        if self.clock:
            f["time"] = spy.Integer(1)
        h = {k: v.xreplace(rep) for k, v in cd.obs.items()}
        return f, h

    def gfun(self, cd, j):
        """Switching function j in the coordinates of condition `cd`."""
        return cd.apply(self.sw.funcs[j])

    def _regimes(self):
        """(kept, dropped) sign patterns of the switches; dropped ones contradict a declared sign in every condition."""
        m = len(self.sw.funcs)
        if 2 ** m > _MAX_REGIMES:
            raise HybridError(f"the switches give {2 ** m} sign patterns, more than {_MAX_REGIMES}")
        per = []
        for cd in self.conds:
            rules = set()
            for a, b in itertools.combinations(range(m), 2):
                ga, gb = self.gfun(cd, a), self.gfun(cd, b)
                d = self.sw.sign_of(ga - gb)
                if d is not None and d != 0:
                    # g_a - g_b > 0 rules out g_a < 0 < g_b, and the reverse
                    rules.add((a, -d, b, d))
                d = self.sw.sign_of(ga + gb)
                if d is not None and d != 0:
                    rules.add((a, -d, b, -d))
            per.append(rules)
        # a regime is dropped only where every condition excludes it
        rules = set.intersection(*per) if per else set()
        keep, drop = [], []
        for pat in itertools.product((1, -1), repeat=m):
            bad = any(pat[a] == sa and pat[b] == sb for a, sa, b, sb in rules)
            (drop if bad else keep).append(list(pat))
        return keep, drop

    def _declarations(self, regimes):
        names = [cd.name for cd in self.conds]
        if regimes is None:
            return [None] * len(names)
        if isinstance(regimes, dict):
            bad = [str(k) for k in regimes if str(k) not in names]
            if bad:
                raise HybridError(f"`regimes` names unknown conditions: {', '.join(bad)}")
            by = {str(k): v for k, v in regimes.items()}
            return [by.get(n) for n in names]
        if not isinstance(regimes, (list, tuple)):
            raise HybridError("`regimes` is a list of regimes or a dict of them per condition")
        return [list(regimes)] * len(names)

    def _sides_of(self, entry):
        """{surface: side} of one entry of `regimes`."""
        out = {}
        for txt in ([entry] if isinstance(entry, str) else list(entry)):
            d, side = comparison(txt)
            hit = self.sw.match(d)
            if hit is None:
                # the surface as the model registered it: the one factor of undecided sign
                sgn, parts = self.sw.factored(d)
                hit = self.sw.match(parts[0]) if len(parts) == 1 else None
                hit = None if hit is None else (hit[0], hit[1] * sgn)
            if hit is None:
                raise HybridError(f"`regimes`: {txt!r} names no switching surface of the model")
            j, sgn = hit
            if out.get(j, side * sgn) != side * sgn:
                raise HybridError(f"`regimes`: an entry names both sides of {self.hy.surface(j)}")
            out[j] = side * sgn
        return out

    def _select(self, cd, decl):
        """Set the regimes, surfaces and declared start sides of condition `cd`."""
        cd.declared = []
        if not decl:
            cd.regimes = [list(r) for r in self.regimes]
        else:
            picks = [self._sides_of(e) for e in decl]
            cd.regimes = []
            for pick in picks:
                hit = [r for r in self.regimes if all(r[j] == v for j, v in pick.items())]
                if not hit:
                    raise HybridError(f"condition {cd.name}: `regimes` names no feasible regime "
                                      f"with {', '.join(self.hy.text(j, v) for j, v in pick.items())}")
                cd.regimes += [r for r in hit if r not in cd.regimes]
            self._check_start(cd, picks[0])
        cd.surfaces = [j for j in range(len(self.sw.funcs))
                       if {r[j] for r in cd.regimes} == {1, -1} or j in self.roots]
        # no trajectory crosses a surface in the parameters alone
        moving = {_sym(x) for x in self.f} | {TIME}
        for j in cd.surfaces:
            if not self.gfun(cd, j).free_symbols & moving:
                raise HybridError(f"condition {cd.name}: the switch on {self.hy.surface(j)} depends on "
                                  f"parameters only, so no trajectory crosses it; name the side with "
                                  f"`regimes`, for instance regimes=[{self.hy.text(j, 1)!r}]")

    def _check_start(self, cd, pick):
        """Check the declared start against the initial values and record the sides they leave undecided."""
        x0 = {TIME: self.t0e}
        x0.update({_sym(u): v for u, v in _start_inputs(self, cd).items()})
        if not cd.equil:
            for x in self.dyn:
                if cd.ic.get(x) is not None:
                    x0[_sym(x)] = cd.ic[x]
            for e in cd.events:
                if e["t0"] and e["var"] in self.dyn and _sym(e["var"]) in x0:
                    x0[_sym(e["var"])] = _compose(x0[_sym(e["var"])], e["method"], e["value"])
        dyn = {_sym(x) for x in self.dyn}
        for j, side in pick.items():
            g = self.gfun(cd, j).xreplace(x0)
            d = None if g.free_symbols & dyn else self.sw.sign_of(g)
            if d is not None and d != side:
                raise HybridError(f"condition {cd.name}: `regimes` starts in "
                                  f"{self.hy.text(j, side)}, but the initial values give "
                                  f"{self.hy.text(j, d) if d else self.hy.surface(j)}")
            if d is None:
                cd.declared.append(f"{self.hy.text(j, side)} at the start")

    def _reach(self, cd, pair):
        """States that reach an observable in either regime of `pair`."""
        out = set()
        models = [self.model(cd, r) for r in pair]
        for _, h in models:
            for v in h.values():
                out |= {str(s) for s in v.free_symbols if str(s) in self.f}
        while True:
            more = set(out)
            for f, _ in models:
                for x in out:
                    more |= {str(s) for s in f[x].free_symbols if str(s) in self.f}
            if more == out:
                return out
            out = more

    def visibility(self, cd, j):
        """Whether the switch on surface j is visible, invisible or undecided in the observables of `cd`."""
        if j in self.roots:
            return _VISIBLE
        pairs = []
        for a in cd.regimes:
            b = list(a)
            b[j] = -a[j]
            if a[j] > 0 and b in cd.regimes:
                pairs.append((a, b))
        sol = self.on_surface(cd, j)
        verdicts = []
        for a, b in pairs:
            fa, ha = self.model(cd, a)
            fb, hb = self.model(cd, b)
            reach = self._reach(cd, (a, b))
            if all(is_zero(ha[y] - hb[y]) for y in ha) and \
                    all(is_zero(fa[x] - fb[x]) for x in reach):
                verdicts.append(_INVISIBLE)
                continue
            seen = sol is not None and self._jets_differ(fa, fb, ha, hb, sol)
            verdicts.append(_VISIBLE if seen else _UNDECIDED)
        if _VISIBLE in verdicts:
            return _VISIBLE
        return _INVISIBLE if verdicts and all(v == _INVISIBLE for v in verdicts) else _UNDECIDED

    def _jets_differ(self, fa, fb, ha, hb, sol):
        """Whether the observable jets differ between the regimes at a seeded point of the surface, modulo a prime."""
        syms = set().union(*[e.free_symbols for m in (fa, fb, ha, hb) for e in m.values()])
        # a surface symbol that no expression contains does not enter the key
        key = tuple(tuple(sorted((str(k), str(v)) for k, v in m.items())) for m in (fa, fb, ha, hb)) + \
            ((str(sol[0]), str(sol[1])) if sol[0] in syms else ())
        memo = self.__dict__.setdefault("_jet_memo", {})
        if key in memo:
            return memo[key]
        ys = sorted(ha)
        # a derivative of an unknown input may jump at the switch, so only the orders below it count
        caps = [self._input_order(fa, fb, ha[y], hb[y]) for y in ys]
        syms |= {_sym(x) for x in fa} | {sol[0]} | spy.sympify(sol[1]).free_symbols
        rng = session.current().rng("hybrid.jets", *key)
        p = PRIMES[0]
        seen = False
        for _ in range(3):
            pt = {x: rng.randrange(2, p) for x in sorted(syms, key=str)}
            try:
                pt[sol[0]] = output_series({}, [sol[1]], pt, 1, p)[0][0]
                va = output_series({_sym(k): v for k, v in fa.items()}, [ha[y] for y in ys], pt,
                                   len(self.dyn) + 2, p)
                vb = output_series({_sym(k): v for k, v in fb.items()}, [hb[y] for y in ys], pt,
                                   len(self.dyn) + 2, p)
            except NoSeriesError:
                continue
            seen = any(x != y for A, B, c in zip(va, vb, caps)
                       for x, y in list(zip(A, B))[:None if c is None else c + 1])
            break
        memo[key] = seen
        return seen

    def _input_order(self, fa, fb, *hs):
        """The lowest Lie order at which an unknown input can enter the observables `hs` in either regime, or None."""
        us = {_sym(w) for w in self.unknown}
        seen = set().union(*[h.free_symbols for h in hs])
        k = 0
        while us and not seen & us:
            more = seen.union(*[fa[str(s)].free_symbols | fb[str(s)].free_symbols for s in seen if str(s) in fa])
            if more == seen:
                return None
            seen, k = more, k + 1
        return k if us else None

    def text(self, sides):
        """A regime as the list of the sides of the surfaces."""
        return [self.hy.text(j, s) for j, s in enumerate(sides)]

    def on_surface(self, cd, j):
        """Surface j solved for a symbol in which it is linear, states first: (symbol, value)."""
        g = self.gfun(cd, j)
        prefer = [_sym(x) for x in self.dyn] + ([TIME] if TIME in g.free_symbols else []) + \
            sorted(g.free_symbols - {_sym(x) for x in self.f} - {TIME}, key=str)
        return solve_linear(g, prefer)


class _Names:
    """Fresh names that avoid every symbol of the model."""

    def __init__(self, taken):
        self.taken = set(taken)

    def __call__(self, base):
        nm, k = base, 1
        while nm in self.taken:
            k += 1
            nm = f"{base}_{k}"
        self.taken.add(nm)
        return nm


def _model_symbols(st):
    out = set(st.f) | set(st.fixed)
    for e in st.f.values():
        out |= {str(s) for s in e.free_symbols}
    for x in st.hy.g:
        for e in x.values():
            out |= {str(s) for s in e.free_symbols}
    for q in st.sw.funcs:
        out |= {str(s) for s in q.free_symbols}
    for cd in st.conds:
        for v in cd.ic.values():
            if v is not None:
                out |= {str(s) for s in v.free_symbols}
        for e in cd.events:
            out |= {str(s) for s in e["time"].free_symbols | e["value"].free_symbols}
    return out


def _start_inputs(st, cd):
    """The values of the inputs after the events at the start."""
    vals = {u: spy.Integer(0) for u in st.inputs}
    for e in cd.events:
        if e["t0"] and e["var"] in vals:
            vals[e["var"]] = _compose(vals[e["var"]], e["method"], e["value"])
    return vals


def _copies(st, names=None):
    """(copies, fresh coordinates): the copies whose observability rows every direction of the rule satisfies."""
    names = _Names(_model_symbols(st)) if names is None else names
    copies, fresh = [], []
    for cd in st.conds:
        first = len(copies)
        for sides in cd.regimes:
            # a resting start is each closed-form resting state of the regime
            starts = [dict(cd.ic)] if not cd.equil else \
                [{x: s[_sym(x)] for x in st.dyn} for s in _steady_states(st, cd, sides)]
            for ic in starts:
                copies += _switched({"cd": cd, "sides": sides, "ic": ic, "events": True,
                                     "obs": {}, "time": st.t0e}, st)
        points = []
        for j in cd.surfaces:
            g = st.gfun(cd, j)
            if cd.vis[j] != _VISIBLE:
                continue
            if st.hy.kind[j] == "time":
                tau = solve_linear(g, [TIME])
                if tau is not None and tau[1].free_symbols:
                    copies[first]["obs"][names("si_tau")] = tau[1]
                continue
            sol = st.on_surface(cd, j)
            if st.hy.kind[j] == "state" and sol is not None and str(sol[0]) in st.dyn:
                points.append((sol, j))
        if st.unknown or any(v != _VISIBLE for v in cd.vis.values()):
            points.append((None, None))
        resets = []
        for r in st.hy.roots:
            j = r["surface"]
            sol = st.on_surface(cd, j) if j in cd.surfaces and st.hy.kind[j] == "state" else None
            resets.append((sol if sol is not None and str(sol[0]) in st.dyn else None, r["event"]))
        # a copy from a point of a surface names that surface under "surface"
        for sol, ev, on in [(s, None, j) for s, j in points] + [(s, e, None) for s, e in resets]:
            pt = {}
            for x in st.dyn:
                if sol is None or _sym(x) != sol[0]:
                    pt[_sym(x)] = _sym(names(f"{x}_si"))
                    fresh.append(str(pt[_sym(x)]))
            tsym = None
            if st.clock:
                tsym = _sym(names("si_time"))
                fresh.append(str(tsym))
                pt[TIME] = tsym
            pt.update({_sym(u): v for u, v in _start_inputs(st, cd).items()})
            if sol is not None:
                pt[sol[0]] = sol[1].xreplace(pt)
            if ev is not None:
                # the state just after a root event fired at this point of the event surface
                v = _sym(ev["var"])
                if str(v) not in st.dyn:
                    continue
                pt[v] = _compose(pt[v], ev.get("method", "replace"), cd.apply(_parse(ev["value"])).xreplace(pt))
            ic = {x: pt[_sym(x)] for x in st.dyn}
            for sides in cd.regimes:
                copies += _switched({"cd": cd, "sides": sides, "ic": ic, "events": False, "obs": {},
                                     "time": tsym if tsym is not None else st.t0e,
                                     "inputs": {u: pt[_sym(u)] for u in st.inputs}, "surface": on}, st)
    return copies, fresh


def _jet_chains(st, order, names):
    """{input: [input, jets of the input]} of the unknown inputs the model reads, `order` names each."""
    # in a condition that renames an input the chain stands for the new name
    read = set().union(*[e.free_symbols for e in st.f.values()],
                       *[e.free_symbols for g in st.hy.g for e in g.values()])
    return {w: [w] + [names(f"{w}_d{k}") for k in range(1, order)] for w in st.unknown if _sym(w) in read}


def _jet_values(copies, jets, names):
    """Fresh values of the jets per copy, set in the copies, as {name: jet}: one start per start point and regime."""
    fresh, starts, shared = {}, {}, {}

    def new(x):
        nm = names(f"{x}_si")
        fresh[nm] = x
        return _sym(nm)
    for cp in copies:
        own = (id(cp["ic"]), tuple(cp["sides"]))
        if own not in starts:
            j, vals = cp.get("surface"), {}
            for w, chain in jets.items():
                v = cp["cd"].apply(_sym(w))
                if v.is_number:
                    vals.update({w: v, **{x: spy.Integer(0) for x in chain[1:]}})
                    continue
                if j is None:
                    vals[w] = new(w)
                else:
                    # the input itself is shared by the two sides of the surface the copy starts on
                    other = list(cp["sides"])
                    other[j] = -other[j]
                    key = (id(cp["ic"]), w, frozenset([tuple(cp["sides"]), tuple(other)]))
                    if key not in shared:
                        shared[key] = new(w)
                    vals[w] = shared[key]
                vals.update({x: new(x) for x in chain[1:]})
            starts[own] = vals
        cp["jets"] = starts[own]
    return fresh


def _uncovered(st, copies):
    """(condition, regime, "start" or "switched") of each allowed regime no start copy, or no switched copy, holds."""
    out = []
    for cd in st.conds:
        own = [c for c in copies if c["cd"] is cd]
        starts = [c["sides"] for c in own if c.get("switch") is None and c.get("events", True)]
        into = [c["switch"][1] for c in own if c.get("switch") is not None] + \
            [c["sides"] for c in own if st.unknown and not c.get("events", True)]
        for r in cd.regimes:
            if r not in starts:
                out.append((cd.name, list(r), "start"))
            if len(cd.regimes) > 1 and r not in into:
                out.append((cd.name, list(r), "switched"))
    return out


def _switched(cp, st):
    """The copy and the versions of it switched at a fixed time into every other regime of the condition."""
    # with unknown inputs the input restarts at a switch off the surfaces, and a copy from a free point stands in
    if st.unknown:
        return [cp]
    taken = {_rational(e["time"]) for cd in st.conds for e in cd.events if e["time"].is_number}
    ts = st.t0e + 1
    while ts in taken:
        ts += 1
    out = [cp]
    for other in cp["cd"].regimes:
        if other != cp["sides"]:
            out.append(dict(cp, switch=(ts, other)))
    return out


def _core_text(e):
    """`e` as a model string of the core, which takes pi and Euler's number as the numbers."""
    return to_str(spy.sympify(e).xreplace(BACK))


def _rows_model(st, copies, fresh, jets=None, jet_fresh=None, order=0):
    """Keyword arguments of _symmetry_detection for the copies, the unknown-input jets as chains of nuisance values."""
    kw = dict(st.kw)
    kw["fixed"] = [s for s in kw.get("fixed") or [] if s not in {str(c) for c in CONSTS.values()}]
    f = {x: _core_text(v) for x, v in st.f.items()}
    for chain in (jets or {}).values():
        f.update({x: chain[k + 1] if k + 1 < len(chain) else "0" for k, x in enumerate(chain)})
    for h in st.hy.names:
        f[h] = "0"
    if st.clock:
        f["time"] = "1"
    used = set()
    for e in st.f.values():
        used |= e.free_symbols
    for x in st.hy.g:
        for e in x.values():
            used |= e.free_symbols
    rows, cols, trafos, obs, events = [], {}, [], [], []
    grid = st.grid
    for i, cp in enumerate(copies):
        cd = cp["cd"]
        rows.append(f"{cd.name}#{i + 1}")
        if grid is not None:
            # the jets of a copy hold the input the condition of the copy names
            for col in grid.colnames():
                if col not in st.unknown:
                    cols.setdefault(col, []).append(grid.cols[col][cd.index])
        for j, h in enumerate(st.hy.names):
            cols.setdefault(h, []).append(1 if cp["sides"][j] > 0 else 0)
        tr = {}
        evs = cd.events if cp["events"] else []
        here = set(used)
        for e in evs:
            here |= e["time"].free_symbols | e["value"].free_symbols
        for v in cp["ic"].values():
            if v is not None:
                here |= spy.sympify(v).free_symbols
        for k, v in list(cd.sub_t.items()) + list(cd.sub_g.items()):
            if k in here and (grid is None or str(k) not in grid.colnames()):
                tr[str(k)] = _core_text(v)
        for x in st.dyn:
            v = cp["ic"].get(x)
            if v is not None:
                tr[x] = _core_text(v)
        tr.update({x: _core_text(v) for x, v in (cp.get("jets") or {}).items()})
        for u, v in (cp.get("inputs") or {}).items():
            # an input starts at zero; an event at the start sets it
            events.append({"var": u, "time": _core_text(st.t0e), "value": _core_text(v), "method": "replace",
                           "cond": [i]})
        if st.clock:
            tr["time"] = _core_text(cp["time"])
        trafos.append(tr)
        o = {k: _core_text(v) for k, v in cd.obs.items()}
        o.update({k: _core_text(v) for k, v in cp["obs"].items()})
        obs.append(o)
        for e in evs:
            events.append({"var": e["var"], "time": _core_text(e["time"]), "value": _core_text(e["value"]),
                           "method": e["method"], "cond": [i]})
        if cp.get("switch") is not None:
            ts, other = cp["switch"]
            # an event at the start keeps the clock origin at t0 for the switch time
            events.append({"var": st.hy.names[0], "time": _core_text(st.t0e),
                           "value": "1" if cp["sides"][0] > 0 else "0", "method": "replace",
                           "cond": [i]})
            for j, h in enumerate(st.hy.names):
                if other[j] != cp["sides"][j]:
                    events.append({"var": h, "time": _core_text(ts),
                                   "value": "1" if other[j] > 0 else "0", "method": "replace",
                                   "cond": [i]})
    kw.update(f=f, g=obs, trafo=trafos, conditions={"rows": rows, "cols": cols},
              events=events or None, reconstruct=True, reactions=None, totals=None,
              equilibrate=False, unknown_inputs=None)
    if jets:
        kw["nuisance"] = {"states": list(jet_fresh), "kept": [], "tops": [], "exact_order": int(order) - 1,
                          "prolonged": False, "jet_states": [], "starts": dict(jet_fresh)}
    pos = st.positive
    if pos not in (True, False):
        pos = list(pos or []) + [n for n in fresh if n.startswith("si_time") or
                                 any(n.startswith(f"{x}_si") and x in set(pos or []) for x in st.f)]
        kw["positive"] = pos
    return kw


def _compose(cur, method, value):
    return {"replace": value, "add": cur + value, "multiply": cur * value}[method]


def _steady_states(st, cd, sides):
    """The resting states of a regime with the inputs at zero, each in closed form."""
    f, _ = st.model(cd, sides)
    # at rest the model reads the clock at the start
    rhs = [f[x].xreplace({_sym(u): 0 for u in st.inputs + st.unknown}).xreplace({TIME: st.t0e}) for x in st.dyn]
    try:
        sol = spy.solve(rhs, [_sym(x) for x in st.dyn], dict=True)
    except NotImplementedError:
        sol = []
    sol = [s for s in sol if set(s) == {_sym(x) for x in st.dyn}]
    if not sol:
        raise HybridError(f"condition {cd.name}: a resting state of a regime has no closed form, "
                          "so a symmetry cannot be certified on it")
    return sol


def _start_values(st, cd):
    """The states of `cd` after the start events, at rest zero where all allowed regimes rest at zero; None if free."""
    if cd.equil:
        try:
            rests = [r for sides in cd.regimes for r in _steady_states(st, cd, sides)]
        except HybridError:
            return {}
        out = {x: spy.Integer(0) if rests and all(is_zero(r[_sym(x)]) for r in rests) else None for x in st.dyn}
    else:
        out = {x: cd.ic.get(x) for x in st.dyn}
    for e in cd.events:
        if e["t0"] and out.get(e["var"]) is not None:
            out[e["var"]] = _compose(out[e["var"]], e["method"], e["value"])
    return out


_INFINITE = (spy.zoo, spy.nan, spy.oo, -spy.oo)


def _regular_zero(e, sub):
    """Whether `e` vanishes where the states of `sub` are zero, with finite derivatives in them there."""
    e = spy.sympify(e)
    v = e.xreplace(sub)
    if v.has(*_INFINITE) or not is_zero(v):
        return False
    return not any(spy.diff(e, a).xreplace(sub).has(*_INFINITE) for a in e.free_symbols & sub.keys())


def _zero_set(st, cd):
    """The states that start at zero in `cd` and that every allowed regime and every later event keep at zero."""
    zs = {x for x, v in _start_values(st, cd).items() if v is not None and is_zero(v)}
    fields = [st.model(cd, r)[0] for r in cd.regimes]
    later = [(e["var"], e["method"], e["value"]) for e in cd.events if not e["t0"]] + \
        [(str(e["var"]), e.get("method", "replace"), cd.apply(_parse(e["value"]))) for e in st.root_events]
    # a state is checked again only when a state named in the right-hand sides or later events
    # of the checked state has left the set
    uses = {x: {str(s) for e in [f[x] for f in fields] + [v for var, _, v in later if var == x]
                for s in spy.sympify(e).free_symbols} for x in zs}
    check = set(zs)
    while check:
        sub = {_sym(x): spy.Integer(0) for x in zs}
        drop = {x for x in check if not (all(_regular_zero(f[x], sub) for f in fields) and
                all(is_zero(_compose(spy.Integer(0), m, spy.sympify(v).xreplace(sub))) for var, m, v in later
                    if var == x))}
        zs -= drop
        check = {x for x in zs if uses[x] & drop}
    return zs


def zero_restriction(hy, kw, regimes=None):
    """(kw, states, notes, setup): the model without the states every condition, regime and event keeps at zero."""
    st = _Setup(hy, kw, regimes)
    per = {cd.name: _zero_set(st, cd) for cd in st.conds}
    shared = set.intersection(*per.values()) if per and regimes is None else set()
    kw2 = _without_states(kw, sorted(shared, key=str)) if shared else None
    common = shared if kw2 is not None else set()

    def why(z):
        if regimes is not None:
            return "`regimes` is declared"
        return "the reactions, totals or conditions keep them in the model" if z & shared else \
            "not in every condition"
    notes = [f"condition {n}: {', '.join(sorted(z - common))} start at zero and stay at zero there, but "
             + why(z) + "; the switched copies of that condition start at any point of the surface"
             for n, z in per.items() if z - common]
    return kw2, sorted(common), notes, st


def _zeroed(text, names):
    pat = re.compile(r"(?<![A-Za-z0-9_.])(" + "|".join(map(re.escape, names)) + r")(?![A-Za-z0-9_.])")
    return pat.sub("0", str(text))


def _without_states(kw, names):
    """`kw` with the states `names` at zero and dropped, None where a reaction network or another grid form has them."""
    drop = set(names)
    if kw.get("reactions") is not None or kw.get("totals"):
        return None
    cond = kw.get("conditions")
    if cond is not None:
        grid = Grid.from_any(cond)
        if grid is not None and set(grid.colnames()) & drop:
            if not (isinstance(cond, dict) and isinstance(cond.get("cols"), dict)):
                return None
            cond = dict(cond, cols={c: v for c, v in cond["cols"].items() if c not in drop})

    def z(t):
        return t if t is None else _zeroed(t, names)

    def m(d):
        return None if d is None else {str(k): z(v) for k, v in dict(d).items() if str(k) not in drop}
    g, tr = kw.get("g"), kw.get("trafo")
    events = [{k: z(v) if k in ("time", "value", "root") else v for k, v in e.items()}
              for e in as_events(kw.get("events")) or [] if str(e["var"]) not in drop]
    pos = kw.get("positive", True)
    return dict(kw, f=m(kw["f"]), g=[m(x) for x in g] if isinstance(g, (list, tuple)) else m(g),
                trafo=[m(x) for x in tr] if isinstance(tr, (list, tuple)) else m(tr), conditions=cond,
                events=events or None,
                positive=pos if pos in (True, False) else [x for x in pos or [] if str(x) not in drop])


def _coords(st):
    """The coordinates of the model: free initial values, then parameters."""
    pars = set()
    for cd in st.conds:
        for sides in st.regimes:
            f, h = st.model(cd, sides)
            for e in list(f.values()) + list(h.values()):
                pars |= {str(s) for s in e.free_symbols}
        for j in range(len(st.sw.funcs)):
            pars |= {str(s) for s in st.gfun(cd, j).free_symbols}
        for v in cd.ic.values():
            if v is not None:
                pars |= {str(s) for s in v.free_symbols}
        for e in cd.events:
            pars |= {str(s) for s in e["time"].free_symbols | e["value"].free_symbols}
        for e in st.root_events:
            pars |= {str(s) for s in cd.apply(_parse(e["value"])).free_symbols}
    pars -= set(st.f) | {"time"} | set(st.fixed) | set(st.inputs) | set(st.unknown)
    free = []
    for cd in st.conds:
        if cd.equil:
            continue
        for x in st.dyn:
            if cd.ic.get(x) is None and x not in free:
                free.append(x)
    return free, sorted(pars - set(free), key=str)


class _Field:
    """A candidate symmetry: components on the parameters and, per condition, on the states."""

    def __init__(self, theta, states, kind, weights=None, origin=""):
        self.theta = theta
        self.states = states
        self.kind = kind
        self.weights = weights
        self.origin = origin

    def on(self, cd):
        out = {_sym(k): v for k, v in self.theta.items()}
        for x, v in self.states.get(cd.index, {}).items():
            out[_sym(x)] = v
        return out


def _apply(X, e):
    """X(e) for a field X given as {symbol: component}."""
    e = spy.sympify(e)
    return sum((spy.diff(e, s) * v for s, v in X.items() if s in e.free_symbols), spy.Integer(0))


def _span(vectors, b, keys):
    """Coefficients c with sum c_i vectors_i = b over `keys`, proven exactly, or None."""
    bb = spy.Matrix([spy.sympify(b.get(k, 0)) for k in keys])
    if not vectors:
        return [] if all(is_zero(x) for x in bb) else None
    A = spy.Matrix([[spy.sympify(v.get(k, 0)) for v in vectors] for k in keys])
    try:
        sol, params = A.gauss_jordan_solve(bb)
    except ValueError:
        return None
    sol = sol.xreplace({q: 0 for q in params})
    if all(is_zero(r) for r in A * sol - bb):
        return list(sol)
    return None


def _bracket(st, X, f):
    """[X, F] on the states for a field X and right-hand sides f."""
    sx = [_sym(x) for x in f]
    out = {}
    for k, comp in X.items():
        if str(k) not in st.f:
            # a parameter component that varies with the states turns under the flow
            out[k] = -sum((spy.diff(comp, s) * f[str(s)] for s in sx), spy.Integer(0))
    for x in st.f:
        comp = X.get(_sym(x), spy.Integer(0))
        out[_sym(x)] = _apply(X, f[x]) - sum((spy.diff(comp, s) * f[str(s)] for s in sx),
                                             spy.Integer(0))
    return out


def _events_of(st, cd):
    return [e for e in cd.events if not e["t0"]] + \
        [{"var": e["var"], "time": None, "value": cd.apply(_parse(e["value"])),
          "method": e.get("method", "replace")} for e in st.root_events]


def _starts(st, cd):
    """The start states of `cd`, given or resting, after the events at the start."""
    if cd.equil:
        raw = [s for sides in cd.regimes for s in _steady_states(st, cd, sides)]
    else:
        raw = [{_sym(x): (cd.ic[x] if cd.ic.get(x) is not None else _sym(x)) for x in st.dyn}]
    out = []
    for x0 in raw:
        x0 = dict(x0)
        x0.update({_sym(u): spy.Integer(0) for u in st.inputs})
        for e in cd.events:
            if e["t0"]:
                v = _sym(e["var"])
                x0[v] = _compose(x0.get(v, v), e["method"], e["value"])
        if st.clock:
            x0[TIME] = st.t0e
        out.append(x0)
    return out


def certify_fields(st, cands, why=None):
    """Indices of the candidates proven symbolically to be directions of the rule in every condition."""
    why = [] if why is None else why
    alive = []
    for i, F in enumerate(cands):
        moved = [nm for nm in st.fixed if not is_zero(F.theta.get(nm, 0))]
        if moved:
            why.append(f"{F.origin}: it moves the fixed {moved[0]}")
        else:
            alive.append(i)
    changed = True
    while changed and alive:
        changed = False
        for cd in st.conds:
            out = _family(st, cd, cands, alive, why)
            if out != alive:
                alive, changed = out, True
    return alive


def _family(st, cd, cands, alive, why):
    keys = sorted({_sym(z) for i in alive for z in cands[i].theta} |
                  {_sym(x) for x in st.f}, key=str)
    fam = []
    for i in alive:
        # an input change follows from the brackets below
        X = {k: cd.apply(v) for k, v in cands[i].on(cd).items() if str(k) not in st.unknown}
        if any(not is_zero(X.get(_sym(u), 0)) for u in st.inputs) or not is_zero(X.get(TIME, 0)):
            why.append(f"{cands[i].origin}: it moves an input or time")
            continue
        fam.append((i, X))
    fam += [(None, {_sym(x): spy.Integer(1)}) for x in st.dyn]
    models = [st.model(cd, r) for r in cd.regimes]
    surf = {j: (st.gfun(cd, j), st.on_surface(cd, j)) for j in cd.surfaces}
    evs = _events_of(st, cd)
    drives = [_drives(st, f, h) for f, h in models]

    def local(X):
        for _f, h in models:
            if any(not is_zero(_apply(X, v)) for v in h.values()):
                return "it changes an observable"
        for j in cd.surfaces:
            g, sol = surf[j]
            if cd.vis[j] == _VISIBLE:
                if sol is None:
                    return f"the surface {st.hy.surface(j)} is linear in no symbol"
                if not is_zero(_apply(X, g).xreplace({sol[0]: sol[1]})):
                    return f"it moves the visible surface {st.hy.surface(j)}"
        for e in evs:
            if e["time"] is not None and not is_zero(_apply(X, e["time"])):
                return "it moves an event time"
        return None
    while True:
        kept = []
        for i, X in fam:
            bad = local(X)
            if bad is not None:
                if i is not None:
                    why.append(f"{cands[i].origin}: condition {cd.name}: {bad}")
                continue
            kept.append((i, X))
        fam = kept
        vecs = [X for _, X in fam]
        drop, changes = None, {}
        for k, (_i, X) in enumerate(fam):
            for r, (f, _) in enumerate(models):
                # modulo the directions of the unknown inputs, whose coefficients are the input change
                sol = _span(vecs + list(drives[r].values()), _bracket(st, X, f), keys)
                if sol is None:
                    drop = (k, "the regimes do not leave the span invariant")
                    break
                changes[k, r] = dict(zip(drives[r], sol[len(vecs):]))
            if drop is None:
                for e in evs:
                    post = _post(e)
                    img = _event_image(st, X, e, post)
                    at = [{kk: spy.sympify(v).xreplace(post) for kk, v in Y.items()} for Y in vecs]
                    if _span(at, img, keys) is None:
                        drop = (k, f"the event on {e['var']} leaves the span")
                        break
            if drop is not None:
                break
        if drop is None:
            drop = _hidden_jumps(st, cd, fam, keys, surf)
        if drop is None and st.unknown:
            drop = _input_jumps(st, cd, fam, keys, surf, models, drives, changes)
        if drop is None:
            break
        k, reason = drop
        if fam[k][0] is not None:
            why.append(f"{cands[fam[k][0]].origin}: condition {cd.name}: {reason}")
        fam.pop(k)
    vecs = [X for _, X in fam]
    out = []
    for i, X in fam:
        if i is None:
            continue
        ok = True
        for x0 in _starts(st, cd):
            at = [{k: spy.sympify(v).xreplace(x0) for k, v in Y.items()} for Y in vecs]
            v0 = {k: spy.sympify(v).xreplace(x0) for k, v in X.items() if str(k) not in st.f}
            for x in st.dyn:
                v0[_sym(x)] = _apply(X, x0[_sym(x)]) if x0[_sym(x)] != _sym(x) else \
                    spy.sympify(X.get(_sym(x), 0)).xreplace(x0)
            if _span(at, v0, keys) is None:
                ok = False
                break
        if ok:
            out.append(i)
        else:
            why.append(f"{cands[i].origin}: condition {cd.name}: the start vector leaves the span")
    return [i for i in alive if i in out]


def _drives(st, f, h):
    """{input: direction of the input on the states} of the unknown inputs that no observable of the regime reads."""
    read = set().union(*[v.free_symbols for v in h.values()])
    return {u: {_sym(x): spy.diff(f[x], _sym(u)) for x in st.f} for u in st.unknown if _sym(u) not in read}


def _input_jumps(st, cd, fam, keys, surf, models, drives, changes):
    """(index, reason) of a member whose input change jumps where a trajectory crosses a surface, else None."""
    vecs = [X for _, X in fam]
    # a regime where the direction of an input lies in the span leaves the change of that input free
    free = [{u for u, g in d.items() if _span(vecs, g, keys) is not None} for d in drives]
    index = {tuple(r): i for i, r in enumerate(cd.regimes)}
    for j in cd.surfaces:
        g, sol = surf[j]
        rep = {} if sol is None else {sol[0]: sol[1]}
        moved = [_apply(X, g).xreplace(rep) for _, X in fam]
        for a in cd.regimes:
            b = list(a)
            b[j] = -a[j]
            if tuple(b) not in index:
                continue
            ia, ib = index[tuple(a)], index[tuple(b)]
            us = [u for u in st.unknown if u not in free[ia] | free[ib]]
            if not us:
                continue
            shift = None
            if any(not is_zero(m) for m in moved):
                # a member that moves the surface shifts the crossing, where the state jumps by the difference of
                # the two fields
                fa, fb = models[ia][0], models[ib][0]
                speed = _apply({_sym(x): v for x, v in fa.items()}, g).xreplace(rep)
                at = [{k: spy.sympify(v).xreplace(rep) for k, v in Y.items()} for Y in vecs]
                shift = _span(at, {_sym(x): (fa[x] - fb[x]).xreplace(rep) for x in st.f}, keys)
                if shift is None or is_zero(speed):
                    k = next(k for k, m in enumerate(moved) if not is_zero(m))
                    return k, f"it moves the surface {st.hy.surface(j)} of a switch an unknown input passes"
            for k in range(len(fam)):
                for u in us:
                    jump = changes[k, ib].get(u, 0) - changes[k, ia].get(u, 0)
                    if shift is not None:
                        jump -= moved[k] / speed * sum(c * changes[n, ib].get(u, 0) for n, c in enumerate(shift))
                    if not is_zero(spy.sympify(jump).xreplace(rep)):
                        return k, f"the change of the unknown input {u} jumps at {st.hy.surface(j)}"
    return None


def _post(e):
    v = _sym(e["var"])
    return {v: _compose(v, e["method"], e["value"])}


def _event_image(st, X, e, post):
    """D E applied to X, as a vector at the state after the event."""
    v = _sym(e["var"])
    img = {k: spy.sympify(c) for k, c in X.items() if str(k) not in st.f}
    for x in st.f:
        s = _sym(x)
        img[s] = _apply(X, post[v]) if s == v else spy.sympify(X.get(s, 0))
    return img


def _hidden_jumps(st, cd, fam, keys, surf):
    """(index, reason) of a member that moves a surface that is not visible and would see the switch of that surface,
    else None."""
    vecs = [X for _, X in fam]
    for j in cd.surfaces:
        if cd.vis[j] == _VISIBLE:
            continue
        g, sol = surf[j]
        moving = [k for k, (_, X) in enumerate(fam)
                  if sol is None or not is_zero(_apply(X, g).xreplace({sol[0]: sol[1]}))]
        if not moving:
            continue
        rep = {} if sol is None else {sol[0]: sol[1]}
        for a in cd.regimes:
            b = list(a)
            b[j] = -a[j]
            if a[j] < 0 or b not in cd.regimes:
                continue
            fa, ha = st.model(cd, a)
            fb, hb = st.model(cd, b)
            jump = {_sym(x): (fa[x] - fb[x]).xreplace(rep) for x in st.f}
            at = [{k: spy.sympify(v).xreplace(rep) for k, v in Y.items()} for Y in vecs]
            if any(not is_zero((ha[y] - hb[y]).xreplace(rep)) for y in ha) or \
                    _span(at, jump, keys) is None:
                cand = [k for k in moving if fam[k][0] is not None]
                k = cand[0] if cand else moving[0]
                return k, f"it moves the surface {st.hy.surface(j)}, whose switch it would see"
    return None


def _components(st, F, coords):
    """The components of F on the coordinates, as sympy expressions."""
    out = []
    for z in coords:
        if z in F.theta:
            out.append(spy.sympify(F.theta[z]))
            continue
        comp = None
        for cd in st.conds:
            if cd.ic.get(z) is None and z in F.states.get(cd.index, {}):
                comp = F.states[cd.index][z]
                break
        out.append(spy.Integer(0) if comp is None else spy.sympify(comp))
    return out


def _rank_at_point(rows):
    """The rank of the component rows at a rational point; None where a value is not rational."""
    syms = set()
    for r in rows:
        for e in r:
            syms |= spy.sympify(e).free_symbols
    for i in range(4):
        pt = _rational_point(syms, str(rows), i)
        vals = [[_value(e, pt) for e in r] for r in rows]
        if any(v is None for r in vals for v in r):
            continue
        return spy.Matrix(vals).rank()
    return None




def _scaling_candidates(st):
    """Scalings the integer kernel proposes over every regime, constrained by surfaces, initial values, events."""
    models, obs, xm, xo = [], [], [], []
    for cd in st.conds:
        cons_m, cons_o = [], []
        for x in st.dyn:
            v = cd.ic.get(x)
            if v is not None and not cd.equil:
                cons_m.append(f"{x} = {to_str(v)}")
        resets = [{"var": r["var"], "time": None, "method": r.get("method", "replace"),
                   "value": cd.apply(_parse(r["value"]))} for r in st.root_events]
        for e in list(cd.events) + resets:
            if e["var"] in st.inputs or e["method"] == "multiply":
                cons_o.append(f"_ = {to_str(e['value'])}")
            else:
                cons_m.append(f"{e['var']} = {to_str(e['value'])}")
            if e["time"] is not None and e["time"].free_symbols:
                cons_o.append(f"_ = {to_str(e['time'])}")
        cons_o += [f"_ = {r}" for r in surface_ratios([st.gfun(cd, j) for j in cd.surfaces
                                                        if cd.vis[j] == _VISIBLE])]
        for sides in cd.regimes:
            f, h = st.model(cd, sides)
            models.append([f"{x} = {to_str(v)}" for x, v in f.items()])
            obs.append([f"{y} = {to_str(v)}" for y, v in h.items()])
            xm.append(cons_m)
            xo.append(cons_o)
    r = scaling_symmetries_multi(per_cond_model=models, per_cond_obs=obs, inputs=st.inputs or None,
                                 fixed=list(st.fixed) + ["time"], logs=True, extra_model=xm,
                                 extra_obs=xo)
    out = []
    for d in r["non_identifiable"]:
        # the weight of an unknown input is the input change
        w = {k: int(spy.Rational(v)) for k, v in d["vector"].items() if k not in st.unknown}
        theta = {k: v * _sym(k) for k, v in w.items() if k not in st.f}
        states = {cd.index: {k: v * _sym(k) for k, v in w.items() if k in st.f} for cd in st.conds}
        out.append(_Field(theta, states, "scaling", weights=w, origin="exact scaling"))
    return out


def _kernel_candidates(st, res, fresh):
    """Closed forms of the stacked analysis that hold no fresh coordinate, lifted to the states."""
    out = []
    for d in res.get("symmetries") or []:
        gen = d.get("generator")
        if not gen:
            continue
        try:
            comp = {k: spy.sympify(str(v).replace("^", "**"),
                                   locals={n: spy.Symbol(n) for n in _names_in(v)})
                    for k, v in gen.items()}
        except (spy.SympifyError, TypeError):
            continue
        syms = set(comp) | {str(s) for v in comp.values() for s in v.free_symbols}
        if syms & set(fresh):
            continue
        theta = {k: v for k, v in comp.items() if k not in st.f}
        lift = _parsed(d.get("state_components") or {})
        states = {}
        for cd in st.conds:
            part = {}
            for x in st.dyn:
                v = cd.ic.get(x)
                if v is None and x in comp:
                    part[x] = comp[x]
                elif v is not None and v.is_Symbol and str(v) in comp:
                    part[x] = comp[str(v)].xreplace({v: _sym(x)})
                elif x in lift:
                    part[x] = lift[x]
            states[cd.index] = part
        out.append(_Field(theta, states, d.get("type", "general"), origin="stacked analysis"))
    return out


def _parsed(comp):
    """String components as SymPy expressions; empty where one does not parse."""
    try:
        return {k: spy.sympify(str(v).replace("^", "**"), locals={n: spy.Symbol(n) for n in _names_in(v)})
                for k, v in comp.items()}
    except (spy.SympifyError, TypeError):
        return {}


def _unit_candidates(st, coords):
    """The unit field of every coordinate, moving that coordinate alone."""
    out = []
    for z in coords:
        if z in st.f:
            states = {cd.index: {z: spy.Integer(1)} for cd in st.conds if cd.ic.get(z) is None}
            out.append(_Field({}, states, "general", origin=f"unit field of {z}"))
        else:
            out.append(_Field({z: spy.Integer(1)}, {}, "general", origin=f"unit field of {z}"))
    return out


def _names_in(v):
    return set(re.findall(r"[A-Za-z_.][A-Za-z0-9_.]*", str(v)))


def _certify(st, cands, coords, need):
    """Up to `need` proven directions among the candidates, independent on the coordinates."""
    pre, rows = [], []
    for F in cands:
        comps = _components(st, F, coords)
        if all(is_zero(c) for c in comps):
            continue
        rk = _rank_at_point(rows + [comps])
        if rk is None or rk <= len(rows):
            continue
        pre.append((F, comps))
        rows.append(comps)
    why = []
    alive = certify_fields(st, [F for F, _ in pre], why)
    chosen, rows = [], []
    for i in alive:
        F, comps = pre[i]
        rk = _rank_at_point(rows + [comps])
        if rk is not None and rk > len(rows) and len(chosen) < need:
            chosen.append((F, comps))
            rows.append(comps)
    return chosen, why


def _entry(F, comps, coords, reconstruct):
    support = sym_sort([z for z, c in zip(coords, comps) if not is_zero(c)])
    if F.kind == "scaling" and F.weights is not None:
        w = {z: str(F.weights.get(z, 0)) for z in support}
        return {"support": support, "vector": w, "type": "scaling", "closed_form": True,
                "certified": True, "exact": True, "route": "hybrid symmetry"}
    w = {z: spy.cancel(c / _sym(z)) for z, c in zip(coords, comps) if z in support}
    if w and all(r.is_Integer for r in w.values()):
        return {"support": support, "vector": {z: str(r) for z, r in w.items()}, "type": "scaling",
                "closed_form": True, "certified": True, "exact": True, "route": "hybrid symmetry"}
    vec = {z: to_str(spy.factor(c).xreplace(BACK)) for z, c in zip(coords, comps) if z in support}
    if not reconstruct:
        return {"support": support, "type": "general", "closed_form": False, "certified": True}
    return {"support": support, "vector": vec, "type": "general", "closed_form": True,
            "certified": True, "exact": True, "route": "hybrid symmetry"}




def _info(st, extra=None):
    out = {"surfaces": [st.hy.surface(j) for j in range(len(st.sw.funcs))],
           "kinds": list(st.hy.kind),
           "regimes": [st.text(r) for r in st.regimes],
           "excluded": [st.text(r) for r in st.excluded],
           "conditions": [{"name": cd.name, "regimes": [st.text(r) for r in cd.regimes],
                           "visibility": {st.hy.surface(j): cd.vis[j] for j in cd.surfaces},
                           "declared": list(cd.declared)} for cd in st.conds]}
    out.update(extra or {})
    return out


def _settings(st):
    kw = st.kw
    return {"positive": st.positive, "reduce_cq": False, "equilibrate": st.equilibrate,
            "reconstruct": bool(kw.get("reconstruct")), "verify": bool(kw.get("verify", True)),
            "sym_engine": "modular", "degree_cap": None, "scalings_only": bool(kw.get("scalings_only"))}


def _entering(st, copies, fresh):
    """The fresh coordinates that can enter a row: entry values of states that reach an observable, and the clock."""
    fr, out = set(fresh), set()
    for cp in copies:
        pair = (cp["sides"], cp["switch"][1]) if cp.get("switch") is not None else (cp["sides"], cp["sides"])
        for x in st._reach(cp["cd"], pair):
            v = cp["ic"].get(x)
            if v is not None:
                out |= {str(a) for a in spy.sympify(v).free_symbols} & fr
        if st.clock:
            out |= {str(a) for a in spy.sympify(cp["time"]).free_symbols} & fr
    return out


def _relaxed_dim(res, coords, fresh, entering):
    """(dimension, proven): the dimension of the stacked kernel projected onto the coordinates of the model."""
    need, sub = _projected(res, coords)
    if sub is None or sub.shape[1] == 0:
        return need, True
    from .. import _core
    ker = res["info"]["kernel"]
    within = sub.shape[1] - int(_core.sym_rref_mod(sub.T.astype(np.float64), int(ker["p"]))["rank"])
    idle = [z for z in ker["coords"] if z in set(fresh) and z not in entering]
    # proven where every direction within the fresh coordinates alone is the unit direction of one that enters no row
    return need, within == len(idle)


_OF_FRESH = "the fresh coordinates of the switch surfaces"


def _at_random(rows, res, coords, fresh, entering, need):
    """(projection, dimension) with the projection checked at a point uniform modulo the verification prime."""
    ker = res["info"]["kernel"]
    rk = ker.get("random")
    if rk is None:
        return {"status": "unchecked", "of": _OF_FRESH}, need
    at = {"info": {"kernel": {"coords": ker["coords"], "N": rk["N"], "p": rk["p"]},
                   "coordinates": res["info"]["coordinates"]}}
    need_r, proven = _relaxed_dim(at, coords, fresh, entering)
    if proven:
        return {"status": "proven"}, need_r
    if need_r != need:
        return {"status": "differs", "of": _OF_FRESH, "base": len(coords) - need,
                "random": len(coords) - need_r}, need
    cols = len(ker["coords"]) + len([z for z in ker["coords"] if z in set(fresh)])
    return {"status": "verified", "of": _OF_FRESH, "probability": _minor_bound(rows, res, cols)}, need


def _minor_bound(rows, res, cols):
    """Schwartz-Zippel bound on a maximal minor of the stacked rows or their fresh columns vanishing, or None."""
    from .core import _model_bound
    order = int(res["info"].get("lie_order_used") or 0)
    evs = list(rows.get("events") or [])
    deg, q = 0, None
    for i, (tr, g) in enumerate(zip(rows["trafo"], rows["g"])):
        own = [{k: v for k, v in e.items() if k != "cond"} for e in evs if i in e.get("cond", [i])]
        b = _model_bound(dict(rows, trafo=tr, g=g, conditions=None, events=own or None), res, [{"_": "1"}], 1, order)
        if not isinstance(b, dict) or b.get("probability") is None:
            return None
        deg, q = max(deg, int(b["degree"])), int(b["prime"])
    return None if q is None else float(min(1.0, cols * deg / q))


def _projected(res, coords):
    """The dimension of the stacked kernel projected onto the coordinates of the model, and that projection."""
    from .. import _core
    ker = res["info"].get("kernel")
    if ker is None and res.get("identifiable"):
        return len([z for z in coords if z not in res["info"]["coordinates"]]), None
    if ker is None:
        raise HybridError("internal: the stacked analysis returned no kernel")
    kc = list(ker["coords"])
    N = np.asarray(ker["N"], dtype=np.int64).reshape(len(kc), -1) if np.size(ker["N"]) else \
        np.zeros((len(kc), 0), dtype=np.int64)
    missing = [z for z in coords if z not in kc]
    pub = [z for z in res["info"]["coordinates"] if not re.search(r"\|c[0-9]+$", z)]
    unknown = [z for z in missing if z in pub]
    if unknown:
        raise HybridError("internal: the kernel of the stacked analysis is in other coordinates "
                          f"({', '.join(kc)}) than the model ({', '.join(coords)})")
    rows = [kc.index(z) for z in coords if z in kc]
    if not rows or N.shape[1] == 0:
        return len(missing), None
    sub = N[rows, :] % int(ker["p"])
    return len(missing) + int(_core.sym_rref_mod(sub.T.astype(np.float64), int(ker["p"]))["rank"]), sub


def hybrid_detection(hy, kw, regimes=None, setup=None):
    """The detection result of a model with switches, `kw` the arguments of _symmetry_detection."""
    t_start = time.monotonic()
    st = setup if setup is not None else _Setup(hy, kw, regimes)
    free, pars = _coords(st)
    coords = free + pars
    if kw.get("scalings_only"):
        return _scalings_only(st, coords, t_start)
    from .core import _jet_order, detection_with_primes
    given = int(dict(kw.get("control") or {}).get("input_jet_order") or 0)
    order = given or (_jet_order(kw) if st.unknown else 0)
    while True:
        names = _Names(_model_symbols(st))
        jets = _jet_chains(st, order, names)
        copies, fresh = _copies(st, names)
        miss = _uncovered(st, copies)
        if miss:
            raise HybridError("internal: a regime enters no copy of the condition: " +
                              "; ".join(f"{c}: {', '.join(st.text(r))} ({k})" for c, r, k in miss))
        jet_fresh = _jet_values(copies, jets, names)
        rows = _rows_model(st, copies, fresh, jets, jet_fresh, order)
        with state.options(keep_kernel=True, random_kernel=True):
            res = detection_with_primes(rows)
        short = bool((res["info"].get("nuisance") or {}).get("short"))
        # the chains are lengthened until exact at every Lie order taken
        if not short or given:
            break
        order *= 2
    entering = _entering(st, copies, fresh)
    need, proven = _relaxed_dim(res, coords, fresh, entering)
    projection = None
    if not proven:
        projection, need = _at_random(rows, res, coords, fresh, entering, need)
    if jets:
        projection = _with_inputs_projection(rows, res, projection)
    chosen, why = [], []
    off = fresh + list(jet_fresh) + [x for chain in jets.values() for x in chain]
    if need > 0:
        cands = _scaling_candidates(st) + _kernel_candidates(st, res, off) + \
            _unit_candidates(st, coords)
        chosen, why = _certify(st, cands, coords, need)
    uncertified, candidates = None, None
    syms = [_entry(F, c, coords, bool(kw.get("reconstruct"))) for F, c in chosen]
    if len(chosen) < need:
        uncertified = (f"the stacked regimes leave {need} direction(s) open, and {len(chosen)} were proven "
                       "directions of the rule" + ("; rejected candidates: " + "; ".join(dict.fromkeys(why))
                                                    if why else ""))
        candidates = [_candidate(e) for e in _stacked_entries(res, off, syms, need - len(chosen))]
    dim = len(coords)
    info = res["info"]
    proven = need if need and uncertified is None else None
    raw = {"rank": dim - need, "dim": dim, "coordinates": coords, "non_identifiable": syms,
           "rank_proven": proven,
           "rank_proof": "stacked regimes and exact symmetries of every regime" if proven else None,
           "lie_order_used": info.get("lie_order_used"), "lie_certified": info.get("lie_certified"),
           "lie_block_orders": info.get("lie_block_orders"),
           "gap_order_used": info.get("gap_order_used"), "gap_certified": info.get("gap_certified"),
           "gap_certificate": info.get("gap_certificate"), "gap_bound": info.get("gap_bound"),
           "order_bound": info.get("order_bound"), "order_bound_why": info.get("order_bound_why"),
           "order_draws": info.get("order_draws"),
           "conditions": len(st.conds),
           "segments": info.get("segments"),
           "projection": projection}
    out = finalize(raw, "observability", _settings(st), elapsed=time.monotonic() - t_start,
                   coordinates=coords)
    out["info"]["model_exprs"] = info.get("model_exprs")
    out["info"]["model_rhs"] = info.get("model_rhs")
    out["info"]["model_inputs"] = info.get("model_inputs")
    out["info"]["switching"] = _info(st, {"copies": len(copies), "uncertified": uncertified,
                                          "candidates": candidates})
    if jets:
        out["info"]["unknown_inputs"] = {"names": list(jets), "jets": jets, "jet_order": int(order),
                                         "exact_order": int(order) - 1, "short": short}
        if short:
            session.note("limits", f"unknown inputs: the {order} jets of each input are exact up to Lie order "
                                   f"{order - 1}, below the orders the analysis took")
    return out


def _with_inputs_projection(rows, res, projection):
    """The projection with the rank modulo the unknown inputs: the weaker status, chances added where both verified."""
    pj = res["info"].get("projection")
    if not pj or pj.get("status") == "proven":
        return projection
    nu = dict(pj, of="the unknown inputs")
    if nu["status"] == "verified":
        ker = res["info"]["kernel"]
        nu["probability"] = _minor_bound(rows, res, len(ker["coords"]))
    if projection is None or projection.get("status") == "proven":
        return nu
    if projection["status"] != "verified":
        return projection
    if nu["status"] != "verified":
        return nu
    both = None if None in (projection.get("probability"), nu.get("probability")) else \
        float(min(1.0, projection["probability"] + nu["probability"]))
    return {"status": "verified", "of": f"{projection['of']} and the unknown inputs", "probability": both}


def _candidate(e):
    """A direction of the stacked kernel that the rule does not prove: support and closed form."""
    pub = public_symmetry(e)
    return {"type": pub["type"], "support": pub["support"], "generator": pub["generator"]}


def _stacked_entries(res, fresh, have, n):
    """Up to `n` directions of the stacked analysis without fresh coordinates, not among `have`."""
    sup = {tuple(sorted(d["support"])) for d in have}
    out = []
    for d in res.get("symmetries") or []:
        if len(out) >= n:
            break
        support = [z for z in d.get("support") or [] if z not in set(fresh)]
        if not support or tuple(sorted(support)) in sup:
            continue
        sup.add(tuple(sorted(support)))
        closed = d.get("generator") is not None and len(support) == len(d["support"])
        e = {"support": support, "type": d.get("type", "general"), "closed_form": closed}
        if closed:
            scal = d.get("type") == "scaling" and d.get("weights")
            e["vector"] = dict(d["weights"]) if scal else dict(d["generator"])
        out.append(e)
    return out


def _scalings_only(st, coords, t_start):
    out_syms = []
    cands = _scaling_candidates(st)
    why = []
    alive = certify_fields(st, cands, why)
    if len(alive) < len(cands):
        raise HybridError("the scaling kernel proposes a scaling that is not proven a direction "
                          f"of the rule ({why[0] if why else ''})")
    for F in cands:
        out_syms.append({"support": sym_sort([k for k, v in F.weights.items() if v]),
                         "vector": {k: str(v) for k, v in F.weights.items() if v},
                         "type": "scaling", "certified": True, "exact": True})
    raw = {"non_identifiable": out_syms, "coordinates": coords}
    out = finalize(raw, "scaling", _settings(st), elapsed=time.monotonic() - t_start,
                   coordinates=coords)
    out["info"]["switching"] = _info(st)
    return out
