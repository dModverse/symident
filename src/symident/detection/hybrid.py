"""Hybrid models: every regime that the declared signs do not exclude, analysed together.

A direction is non-identifiable when it is an exact symmetry of every such regime, maps
every switching surface to itself and is compatible with the events and the initial
values. Necessary conditions on such a direction (the observability rows of every regime
from the start of every condition and from generic points of every surface) bound the
rank from below; exact symmetries verified symbolically bound it from above. The result
is returned only when the two bounds meet."""

import itertools
import random
import re
import time

import numpy as np
import sympy as spy

from .. import state
from ..rexpr import is_number, sym_sort
from ..sym.scaling import scaling_symmetries_multi
from ..sym.switching import TIME, comparison, surface_ratios, to_str
from .conditions import Grid, as_events, num_str
from .finalize import finalize

_RNG = random.Random(20261008)
_MAX_REGIMES = 4096
_VISIBLE, _INVISIBLE, _UNDECIDED = "visible", "invisible", "undecided"


def _sym(x):
    return spy.Symbol(str(x))


def _parse(text):
    """A model string (R syntax) as a sympy expression."""
    from ..sym.switching import _plain
    return _plain(str(text))


def _rational_point(syms):
    return {s: spy.Rational(_RNG.randint(2, 89), _RNG.randint(2, 89)) for s in syms}


def _value(e, pt):
    """e at the point as a rational number, None for a pole or a value that is not rational."""
    try:
        v = spy.sympify(e).xreplace(pt)
    except (ZeroDivisionError, TypeError, ValueError):
        return None
    return v if v.is_Rational else None


def is_zero(e):
    """Whether `e` vanishes identically: a nonzero exact value at a rational point refutes
    it, a symbolic simplification to zero proves it."""
    e = spy.sympify(e)
    if e == 0:
        return True
    for _ in range(2):
        v = _value(e, _rational_point(e.free_symbols))
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
    """(s, expr) with g = 0 solved for the first symbol of `prefer` in which its numerator
    is linear with a coefficient free of it, else None."""
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
    """One condition in the coordinates of the model: substitutions, initial values,
    observables and events."""

    def __init__(self, index, name):
        self.index = index
        self.name = name
        self.ic = {}
        self.obs = {}
        self.events = []
        self.equil = False


class _Setup:
    """The hybrid model with its conditions, regimes and surfaces."""

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
        self.equilibrate = bool(kw.get("equilibrate"))
        self.f = dict(hy.f)
        if "time" in self.f:
            raise HybridError("a state named `time` clashes with the clock")
        self.root_events = [r["event"] for r in hy.roots]
        self.time_events = as_events([e for e in (kw.get("events") or []) if e.get("root") is None])
        self.clock = any(TIME in e.free_symbols for e in self.f.values()) or \
            any(TIME in e.free_symbols for x in hy.g for e in x.values())
        nums = [float(e["time"]) for e in self.time_events if is_number(e["time"])]
        self.t0 = min(nums) if nums else 0.0
        self.t0e = spy.Rational(num_str(self.t0))
        self.dyn = [x for x in self.f if x not in self.inputs]
        self._conditions()
        self.regimes, self.excluded = self._regimes()
        self.roots = {r["surface"] for r in hy.roots}
        decl = self._declarations(regimes)
        for cd in self.conds:
            self._select(cd, decl[cd.index])
            cd.vis = {j: self.visibility(cd, j) for j in cd.surfaces}

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
                cd.events.append({"var": e["var"], "time": T, "value": app(_parse(vv)),
                                  "method": e["method"], "index": i,
                                  "t0": is_number(e["time"]) and float(e["time"]) <= self.t0})
            cd.equil = self.equilibrate
            conds.append(cd)
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
        """Every sign pattern of the switching functions, less those the declared signs
        prove infeasible: a decided sign of g_a - g_b or g_a + g_b excludes the patterns
        it contradicts."""
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
                raise HybridError(f"`regimes`: {txt!r} names no switching surface of the model")
            j, sgn = hit
            if out.get(j, side * sgn) != side * sgn:
                raise HybridError(f"`regimes`: an entry names both sides of {self.hy.surface(j)}")
            out[j] = side * sgn
        return out

    def _select(self, cd, decl):
        """The regimes of condition `cd`, its surfaces (those with regimes on both sides) and
        the assumptions of a declared start."""
        cd.assumptions = []
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

    def _check_start(self, cd, pick):
        """The declared start against the initial values: a sign of g at the start that the
        declared signs decide must agree; an undecided one is an assumption."""
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
                cd.assumptions.append(f"{self.hy.text(j, side)} at the start")

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
        """Whether the switch on surface j shows in the observables of condition `cd`.
        Visible: a jet of the two neighbouring regimes differs at a rational point of the
        surface. Invisible: the observables and every state that reaches them have the
        same right-hand sides in both regimes, so every jet agrees. Else undecided."""
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
            seen = False
            if sol is not None:
                rep = {sol[0]: sol[1]}
                for y in ha:
                    ea, eb = ha[y], hb[y]
                    for _ in range(len(self.dyn) + 2):
                        v = _value((ea - eb).xreplace(rep), _rational_point(
                            (ea - eb).xreplace(rep).free_symbols))
                        if v is not None and v != 0:
                            seen = True
                            break
                        ea, eb = self.lie(fa, ea), self.lie(fb, eb)
                    if seen:
                        break
            verdicts.append(_VISIBLE if seen else _UNDECIDED)
        if _VISIBLE in verdicts:
            return _VISIBLE
        return _INVISIBLE if verdicts and all(v == _INVISIBLE for v in verdicts) else _UNDECIDED

    def lie(self, f, e):
        """The Lie derivative of e along the right-hand sides f."""
        return sum((spy.diff(e, _sym(x)) * v for x, v in f.items()), spy.Integer(0))

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


def _copies(st):
    """The copies whose observability rows every direction of the rule satisfies: each
    regime from the start of each condition, from a generic point of each visible surface
    that a state parametrises and, where a switch is not visible, from a generic point;
    each switched into every other regime; the times of the visible time switches as
    observables. Returns the copies and the fresh coordinates."""
    names = _Names(_model_symbols(st))
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
                points.append(sol)
        if any(v != _VISIBLE for v in cd.vis.values()):
            points.append(None)
        for sol in points:
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
            ic = {x: pt[_sym(x)] for x in st.dyn}
            for sides in cd.regimes:
                copies += _switched({"cd": cd, "sides": sides, "ic": ic, "events": False, "obs": {},
                                     "time": tsym if tsym is not None else st.t0e,
                                     "inputs": {u: pt[_sym(u)] for u in st.inputs}}, st)
    return copies, fresh


def _switched(cp, st):
    """The copy and, for every other regime of its condition, the copy switched into it at
    a fixed time: its rows hold Lie derivatives along words of two regimes, which every
    direction of the rule keeps."""
    taken = {spy.Rational(num_str(float(e["time"]))) for e in st.time_events if is_number(e["time"])}
    ts = st.t0e + 1
    while ts in taken:
        ts += 1
    out = [cp]
    for other in cp["cd"].regimes:
        if other != cp["sides"]:
            out.append(dict(cp, switch=(ts, other)))
    return out


def _rows_model(st, copies, fresh):
    """Keyword arguments of _symmetry_detection for the copies."""
    kw = dict(st.kw)
    f = {x: to_str(v) for x, v in st.f.items()}
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
            for col in grid.colnames():
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
                tr[str(k)] = to_str(v)
        for x in st.dyn:
            v = cp["ic"].get(x)
            if v is not None:
                tr[x] = to_str(v)
        for u, v in (cp.get("inputs") or {}).items():
            tr[u] = to_str(v)
        if st.clock:
            tr["time"] = to_str(cp["time"])
        trafos.append(tr)
        o = {k: to_str(v) for k, v in cd.obs.items()}
        o.update({k: to_str(v) for k, v in cp["obs"].items()})
        obs.append(o)
        for e in evs:
            events.append({"var": e["var"], "time": to_str(e["time"]), "value": to_str(e["value"]),
                           "method": e["method"], "cond": [i]})
        if cp.get("switch") is not None:
            ts, other = cp["switch"]
            # an event at the start keeps the clock origin at t0 for the switch time
            events.append({"var": st.hy.names[0], "time": to_str(st.t0e),
                           "value": "1" if cp["sides"][0] > 0 else "0", "method": "replace",
                           "cond": [i]})
            for j, h in enumerate(st.hy.names):
                if other[j] != cp["sides"][j]:
                    events.append({"var": h, "time": to_str(ts),
                                   "value": "1" if other[j] > 0 else "0", "method": "replace",
                                   "cond": [i]})
    kw.update(f=f, g=obs, trafo=trafos, conditions={"rows": rows, "cols": cols},
              events=events or None, reconstruct=True, reactions=None, totals=None,
              equilibrate=False)
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
    rhs = [f[x].xreplace({_sym(u): 0 for u in st.inputs}) for x in st.dyn]
    try:
        sol = spy.solve(rhs, [_sym(x) for x in st.dyn], dict=True)
    except NotImplementedError:
        sol = []
    sol = [s for s in sol if set(s) == {_sym(x) for x in st.dyn}]
    if not sol:
        raise HybridError(f"condition {cd.name}: a resting state of a regime has no closed form, "
                          "so a symmetry cannot be certified on it")
    return sol


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
    pars -= set(st.f) | {"time"} | set(st.fixed) | set(st.inputs)
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
    """The start states of condition `cd`: the given ones, or the resting states of its
    regimes, after the events at the start."""
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
    """The candidates that are proven directions of the rule. Per condition the
    candidates and the state unit fields form a family; members are dropped until the
    family spans a distribution that every regime leaves invariant ([Y, F] in the span),
    that kills every observable, is tangent to every visible surface, maps into itself
    under every event, and, at a surface that is not visible and not kept, holds the jump
    of the dynamics with equal observables. A candidate survives when its start vector
    lies in that distribution. Every identity is proven symbolically."""
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
        X = {k: cd.apply(v) for k, v in cands[i].on(cd).items()}
        if any(not is_zero(X.get(_sym(u), 0)) for u in st.inputs) or not is_zero(X.get(TIME, 0)):
            why.append(f"{cands[i].origin}: it moves an input or time")
            continue
        fam.append((i, X))
    fam += [(None, {_sym(x): spy.Integer(1)}) for x in st.dyn]
    models = [st.model(cd, r) for r in cd.regimes]
    surf = {j: (st.gfun(cd, j), st.on_surface(cd, j)) for j in cd.surfaces}
    evs = _events_of(st, cd)

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
        drop = None
        for k, (_i, X) in enumerate(fam):
            for f, _ in models:
                if _span(vecs, _bracket(st, X, f), keys) is None:
                    drop = (k, "the regimes do not leave its span invariant")
                    break
            if drop is None:
                for e in evs:
                    post = _post(st, e)
                    img = _event_image(st, X, e, post)
                    at = [{kk: spy.sympify(v).xreplace(post) for kk, v in Y.items()} for Y in vecs]
                    if _span(at, img, keys) is None:
                        drop = (k, f"the event on {e['var']} leaves the span")
                        break
            if drop is not None:
                break
        if drop is None:
            drop = _hidden_jumps(st, cd, fam, keys, surf)
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
            why.append(f"{cands[i].origin}: condition {cd.name}: its start vector leaves the span")
    return [i for i in alive if i in out]


def _post(st, e):
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
    """At a surface that is not visible, a member that moves it must leave the output
    unchanged: the jump of the dynamics lies in the span and the observables agree there."""
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
    """The rank of the component rows at a rational point; None where a value is not
    rational."""
    syms = set()
    for r in rows:
        for e in r:
            syms |= spy.sympify(e).free_symbols
    for _ in range(4):
        pt = _rational_point(syms)
        vals = [[_value(e, pt) for e in r] for r in rows]
        if any(v is None for r in vals for v in r):
            continue
        return spy.Matrix(vals).rank()
    return None




def _scaling_candidates(st):
    """Scalings proposed by the integer kernel over the regimes of every condition, with the
    visible surfaces, initial values and events as constraints; the certificate decides."""
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
        w = {k: int(spy.Rational(v)) for k, v in d["vector"].items()}
        theta = {k: v * _sym(k) for k, v in w.items() if k not in st.f}
        states = {cd.index: {k: v * _sym(k) for k, v in w.items() if k in st.f} for cd in st.conds}
        out.append(_Field(theta, states, "scaling", weights=w, origin="exact scaling"))
    return out


def _kernel_candidates(st, res, fresh):
    """Closed forms of the stacked analysis that hold no fresh coordinate; a state whose
    start is a coordinate moves as that coordinate does."""
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
        states = {}
        for cd in st.conds:
            part = {}
            for x in st.dyn:
                v = cd.ic.get(x)
                if v is None and x in comp:
                    part[x] = comp[x]
                elif v is not None and v.is_Symbol and str(v) in comp:
                    part[x] = comp[str(v)].xreplace({v: _sym(x)})
            states[cd.index] = part
        out.append(_Field(theta, states, d.get("type", "general"), origin="stacked analysis"))
    return out


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
                "certified": True, "route": "hybrid symmetry"}
    vec = {z: to_str(spy.factor(c)) for z, c in zip(coords, comps) if z in support}
    if not reconstruct:
        return {"support": support, "type": "general", "closed_form": False, "certified": True}
    return {"support": support, "vector": vec, "type": "general", "closed_form": True,
            "certified": True, "route": "hybrid symmetry"}




def _info(st, extra=None):
    out = {"surfaces": [st.hy.surface(j) for j in range(len(st.sw.funcs))],
           "kinds": list(st.hy.kind),
           "regimes": [st.text(r) for r in st.regimes],
           "excluded": [st.text(r) for r in st.excluded],
           "conditions": [{"name": cd.name, "regimes": [st.text(r) for r in cd.regimes],
                           "visibility": {st.hy.surface(j): cd.vis[j] for j in cd.surfaces},
                           "assumptions": list(cd.assumptions)} for cd in st.conds]}
    out.update(extra or {})
    return out


def _settings(st):
    kw = st.kw
    return {"positive": st.positive, "reduce_cq": False, "equilibrate": st.equilibrate,
            "reconstruct": bool(kw.get("reconstruct")), "verify": bool(kw.get("verify", True)),
            "sym_engine": "modular", "degree_cap": None, "scalings_only": bool(kw.get("scalings_only"))}


def _relaxed_dim(res, coords):
    """The dimension of the stacked kernel projected onto the coordinates of the model."""
    from .. import _core
    ker = res["info"].get("kernel")
    if ker is None and res.get("identifiable"):
        return len([z for z in coords if z not in res["info"]["coordinates"]])
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
        return len(missing)
    sub = N[rows, :] % int(ker["p"])
    return len(missing) + int(_core.sym_rref_mod(sub.T.astype(np.float64), int(ker["p"]))["rank"])


def hybrid_detection(hy, kw, regimes=None):
    """The detection of a model with switches; `kw` holds the arguments of
    _symmetry_detection, `regimes` the optional regimes of the experiment."""
    t_start = time.monotonic()
    st = _Setup(hy, kw, regimes)
    free, pars = _coords(st)
    coords = free + pars
    if kw.get("scalings_only"):
        return _scalings_only(st, coords, t_start)
    from .core import detection_with_primes
    copies, fresh = _copies(st)
    with state.options(keep_kernel=True):
        res = detection_with_primes(_rows_model(st, copies, fresh))
    need = _relaxed_dim(res, coords)
    chosen, why = [], []
    if need > 0:
        cands = _scaling_candidates(st) + _kernel_candidates(st, res, fresh) + \
            _unit_candidates(st, coords)
        chosen, why = _certify(st, cands, coords, need)
    if len(chosen) < need:
        msg = (f"the analysis of the switches is not certified: the stacked regimes leave "
               f"{need} direction(s) open, and {len(chosen)} were proven directions of the rule")
        if why:
            msg += "; rejected candidates: " + "; ".join(dict.fromkeys(why))
        raise HybridError(msg)
    dim = len(coords)
    syms = [_entry(F, c, coords, bool(kw.get("reconstruct"))) for F, c in chosen]
    info = res["info"]
    raw = {"rank": dim - need, "dim": dim, "coordinates": coords, "non_identifiable": syms,
           "rank_proven": need if need else None,
           "rank_proof": "stacked regimes and exact symmetries of every regime" if need else None,
           "lie_order_used": info.get("lie_order_used"), "lie_certified": info.get("lie_certified"),
           "lie_block_orders": info.get("lie_block_orders"),
           "gap_order_used": info.get("gap_order_used"), "conditions": len(st.conds),
           "segments": info.get("segments")}
    out = finalize(raw, "observability", _settings(st), elapsed=time.monotonic() - t_start,
                   coordinates=coords)
    out["info"]["model_exprs"] = info.get("model_exprs")
    out["info"]["switching"] = _info(st, {"copies": len(copies)})
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
                         "type": "scaling"})
    raw = {"non_identifiable": out_syms, "coordinates": coords}
    out = finalize(raw, "scaling", _settings(st), elapsed=time.monotonic() - t_start,
                   coordinates=coords)
    out["info"]["switching"] = _info(st)
    return out
