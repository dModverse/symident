"""Invariant distributions: the field certificate of the rank and static subsystems."""

import functools
import random
import sys
from fractions import Fraction

import numpy as np
import sympy as spy

from .. import state
from .parse import _as_list, _clean, _make_local_parse

_LIFT_CACHE = {}


class _DistCheck:
    """Rational expressions over the coordinates, evaluated exactly mod p."""

    def __init__(self, lines):
        self.local, self.parse = _make_local_parse(list(lines))
        self.fns = {}

    def expr(self, text):
        """`text` parsed into a SymPy expression."""
        return spy.sympify(self.parse(str(text)))

    def value(self, e, syms, vals, p, skey=None):
        """e at the point vals (Fractions aligned with syms) mod p; None on a pole or a
        non-rational value. `skey` names the symbol list in the cache key, which saves
        hashing a long list of symbols on every call."""
        key = (e, tuple(syms) if skey is None else skey)
        fn = self.fns.get(key)
        if fn is None:
            # decimals as exact rationals, rationals as Fraction: no float enters
            ex = e.xreplace({x: spy.Rational(str(x)) for x in e.atoms(spy.Float)})
            frac = spy.Function('Fraction')
            ex = ex.xreplace({x: frac(x.p, x.q) for x in ex.atoms(spy.Rational)
                              if not x.is_Integer})
            fn = spy.lambdify(syms, ex, modules=[{'Fraction': Fraction}])
            self.fns[key] = fn
        try:
            v = fn(*vals)
        except ZeroDivisionError:
            return None
        if isinstance(v, int):
            return v % p
        if isinstance(v, Fraction):
            if v.denominator % p == 0:
                return None
            return v.numerator % p * pow(v.denominator % p, p - 2, p) % p
        return None


def _span_coeffs(B, v, p):
    """Coefficients of v over the vectors B (lists of residues) mod p, earlier vectors
    first and dependent ones at zero, or None if v is not in their span. An incremental
    echelon that stores each row's combination of B; p < 2^31 keeps products in int64."""
    n = len(B)
    rows = []
    for k, b in enumerate(B):
        x = np.asarray(b, dtype=np.int64) % p
        comb = np.zeros(n, dtype=np.int64)
        comb[k] = 1
        for piv, r, cm in rows:
            f = int(x[piv])
            if f:
                x = (x - f * r) % p
                comb = (comb - f * cm) % p
        nz = np.flatnonzero(x)
        if nz.size == 0:
            continue
        piv = int(nz[0])
        inv = pow(int(x[piv]), p - 2, p)
        rows.append((piv, x * inv % p, comb * inv % p))
    y = np.asarray(v, dtype=np.int64) % p
    coef = np.zeros(n, dtype=np.int64)
    for piv, r, cm in rows:
        f = int(y[piv])
        if f:
            y = (y - f * r) % p
            coef = (coef + f * cm) % p
    if y.any():
        return None
    return [int(c) for c in coef]


class _DualMod:
    """a + b*eps over GF(p) with eps^2 = 0, for directional derivatives of lambdified
    rational expressions; constants are ints or Fractions."""
    __slots__ = ('a', 'b', 'p')

    def __init__(self, a, b, p):
        self.a, self.b, self.p = a % p, b % p, p

    def _c(self, o):
        if isinstance(o, _DualMod):
            return o
        if isinstance(o, Fraction):
            if o.denominator % self.p == 0:
                raise ZeroDivisionError
            return _DualMod(o.numerator % self.p * pow(o.denominator % self.p, self.p - 2,
                                                       self.p), 0, self.p)
        if isinstance(o, int):
            return _DualMod(o, 0, self.p)
        raise TypeError('unsupported operand')

    def __add__(self, o):
        o = self._c(o)
        return _DualMod(self.a + o.a, self.b + o.b, self.p)

    __radd__ = __add__

    def __neg__(self):
        return _DualMod(-self.a, -self.b, self.p)

    def __sub__(self, o):
        o = self._c(o)
        return _DualMod(self.a - o.a, self.b - o.b, self.p)

    def __rsub__(self, o):
        return self._c(o) - self

    def __mul__(self, o):
        o = self._c(o)
        return _DualMod(self.a * o.a, self.a * o.b + self.b * o.a, self.p)

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._c(o)
        if o.a % self.p == 0:
            raise ZeroDivisionError
        inv = pow(o.a, self.p - 2, self.p)
        q = self.a * inv % self.p
        return _DualMod(q, (self.b - q * o.b) * inv, self.p)

    def __rtruediv__(self, o):
        return self._c(o) / self

    def __pow__(self, n):
        if isinstance(n, Fraction) and n.denominator == 1:
            n = n.numerator
        if not isinstance(n, int):
            raise TypeError('non-integer power')
        if n < 0:
            return _DualMod(1, 0, self.p) / (self ** (-n))
        if n == 0:
            return _DualMod(1, 0, self.p)
        an = pow(self.a, n - 1, self.p)
        return _DualMod(an * self.a, n * an * self.b, self.p)

    def __pos__(self):
        return self


def verify_distribution(regimes, gens, events, lifts, p, npts=2, seed=1):
    """Certificate that every direction of `lifts` lies in the kernel of all orders.

    `regimes`: list of {'f': ["X = rhs"], 'g': ["y = h"]}; `gens[r]`: fields of regime r,
    each a {coordinate: expression}; `lifts`: one per chain, {'regime': r, 'point':
    {coordinate: int}, 'vectors': [{coordinate: int}], 'times': [expression]}, the start
    state and the kernel directions lifted onto it; `events`: the joins between
    consecutive segments in chain order, {'from', 'to', 'ev': [{var, method, value}],
    'lift'}. Per chain the fields that the directions reach are closed under the
    brackets with each regime's field and propagated through the joins; (A) and (B) are
    checked on them. Returns {'ok': bool, 'why': str}."""
    p, npts = int(p), int(npts)
    lines = []
    for rg in regimes:
        lines += list(_as_list(rg['f'])) + list(_as_list(rg['g']))
    for gl in gens:
        for gg in gl:
            lines += ['_ = %s' % v for v in dict(gg).values()]
    for ev in events:
        lines += ['_ = %s' % e['value'] for e in _as_list(ev['ev'])]
    ck = _DistCheck(lines)
    rng = random.Random(seed)
    states = []
    for rg in regimes:
        for l in _as_list(rg['f']):
            nm = _clean(l).split('=', 1)[0].strip()
            if nm not in states:
                states.append(nm)
    F, H = [], []
    for rg in regimes:
        rhs = {}
        for l in _as_list(rg['f']):
            lhs, e = _clean(l).split('=', 1)
            rhs[lhs.strip()] = ck.expr(e)
        F.append(rhs)
        H.append([ck.expr(_clean(l).split('=', 1)[1]) for l in _as_list(rg['g'])])
    Xm = [[{str(k): ck.expr(v) for k, v in dict(gg).items()} for gg in gl] for gl in gens]
    syms = set()
    for rhs in F:
        for e in rhs.values():
            syms |= e.free_symbols
    for hs in H:
        for e in hs:
            syms |= e.free_symbols
    for gl in Xm:
        for gg in gl:
            for e in gg.values():
                syms |= e.free_symbols
    for ev in events:
        for e in _as_list(ev['ev']):
            syms |= ck.expr(e['value']).free_symbols
    for lf in lifts:
        syms |= {spy.Symbol(str(k)) for k in dict(lf['point'])}
        for v in _as_list(lf['vectors']):
            syms |= {spy.Symbol(str(k)) for k in dict(v)}
    coords = states + sorted({str(s) for s in syms} - set(states))
    in_model = set()
    for rhs in F:
        for e in rhs.values():
            in_model |= {str(x) for x in e.free_symbols}
    for hs in H:
        for e in hs:
            in_model |= {str(x) for x in e.free_symbols}
    free = [nm for nm in coords if nm not in states and nm not in in_model]
    sym_list = [ck.local.get(nm, spy.Symbol(nm)) for nm in coords]
    zero = spy.Integer(0)

    def comp(field, nm):
        return field.get(nm, zero)

    # regimes share most right-hand sides, so the partials are memoised per expression
    dcache = {}

    def dpart(e, s):
        k = (e, s)
        d = dcache.get(k)
        if d is None:
            d = dcache[k] = spy.diff(e, s)
        return d

    def deriv_dir(e, field):
        return sum((dpart(e, s) * comp(field, str(s)) for s in e.free_symbols
                    if str(s) in field), zero)

    def vec(exprs, vals):
        out = []
        for e in exprs:
            v = ck.value(e, sym_list, vals, p, skey='vd')
            if v is None:
                return None
            out.append(v)
        return out

    pos_of = {nm: k for k, nm in enumerate(coords)}

    def field_at(x, vals):
        out = [0] * len(coords)
        for nm, e in x.items():
            if nm not in pos_of or e == zero:
                continue
            v = ck.value(e, sym_list, vals, p, skey='vd')
            if v is None:
                return None
            out[pos_of[nm]] = v
        return out

    def rand_point():
        return [Fraction(rng.randrange(2, p - 1)) for _ in coords]

    bracket_cache = {}

    def bracket(r, j):
        key = (r, j)
        if key not in bracket_cache:
            x, Fr = Xm[r][j], {nm: F[r].get(nm, zero) for nm in states}
            bracket_cache[key] = [deriv_dir(comp(Fr, nm), x) - deriv_dir(comp(x, nm), Fr)
                                  if nm in Fr else -deriv_dir(comp(x, nm), Fr)
                                  for nm in coords]
        return bracket_cache[key]

    def fields_at(r, vals):
        """Values of the model fields of regime r and of the unit fields at a point."""
        out = [field_at(x, vals) for x in Xm[r]]
        if any(o is None for o in out):
            return None
        return out + [[1 if nm == u else 0 for nm in coords] for u in free]

    def support(r, vals, v):
        """Model fields of regime r in the decomposition of v at the point, or None."""
        B = fields_at(r, vals)
        if B is None:
            return None
        c = _span_coeffs(B, v, p)
        if c is None:
            return None
        nx = len(Xm[r])
        return {k for k in range(nx) if c[k]}

    # lambdified vector functions over the coordinates, evaluated on duals: the
    # derivative of an expression list along a direction without symbolic brackets
    lam_cache = {}

    def lam(key, exprs):
        fn = lam_cache.get(key)
        if fn is None:
            ex = []
            frac = spy.Function('Fraction')
            for e in exprs:
                e = spy.sympify(e)
                e = e.xreplace({x: spy.Rational(str(x)) for x in e.atoms(spy.Float)})
                e = e.xreplace({x: frac(x.p, x.q) for x in e.atoms(spy.Rational)
                                if not x.is_Integer})
                ex.append(e)
            fn = lam_cache[key] = spy.lambdify(sym_list, ex, modules=[{'Fraction': Fraction}])
        return fn

    def along(key, exprs, vals, dirv):
        """Values and derivatives of exprs at vals along dirv (residues), or None."""
        args = [_DualMod(int(v.numerator) % p * pow(int(v.denominator) % p, p - 2, p),
                         d, p) for v, d in zip(vals, dirv)]
        try:
            out = lam(key, exprs)(*args)
        except Exception:
            return None
        res = []
        for o in out:
            if isinstance(o, _DualMod):
                res.append((o.a, o.b))
            elif isinstance(o, Fraction):
                if o.denominator % p == 0:
                    return None
                res.append((o.numerator % p * pow(o.denominator % p, p - 2, p), 0))
            elif isinstance(o, int):
                res.append((o % p, 0))
            else:
                return None
        return res

    # the rational right-hand sides of a regime go through duals; one with a function
    # (exp) is differentiated symbolically, where the terms along a field cancel
    rat_of = {}

    def rat_idx(r):
        if r not in rat_of:
            rat_of[r] = [k for k, nm in enumerate(states)
                         if not F[r].get(nm, zero).atoms(spy.Function)]
        return rat_of[r]

    def bracket_at(r, j, vals):
        """[X_j, F^(r)] at vals, or None on a pole: DF.X - DX.F over the coordinates."""
        X = Xm[r][j]
        x = field_at(X, vals)
        if x is None:
            return None
        n_s = len(states)
        ri = rat_idx(r)
        df = [0] * n_s
        fx = along(('F', r), [F[r].get(states[k], zero) for k in ri], vals, x)
        if fx is None:
            return None
        for k, (_, d) in zip(ri, fx):
            df[k] = d
        supp = set(X)
        for k in set(range(n_s)) - set(ri):
            e = F[r].get(states[k], zero)
            if not supp & {str(t) for t in e.free_symbols}:
                continue
            v = ck.value(deriv_dir(e, X), sym_list, vals, p, skey='vd')
            if v is None:
                return None
            df[k] = v
        Fr = {nm: F[r].get(nm, zero) for nm in states}
        out = []
        for k, nm in enumerate(coords):
            cx = comp(X, nm)
            dx = 0
            if cx != zero:
                dx = ck.value(deriv_dir(cx, Fr), sym_list, vals, p, skey='vd')
                if dx is None:
                    return None
            out.append(((df[k] if k < n_s else 0) - dx) % p)
        return out

    def close(r, S):
        """Close the field set S of regime r under brackets with F^(r); check (A), (B)."""
        S, todo = set(S), list(S)
        while todo:
            j = todo.pop()
            for _ in range(npts):
                vals = rand_point()
                x = field_at(Xm[r][j], vals)
                hr = [h for h in H[r] if not h.atoms(spy.Function)]
                hs = [h for h in H[r] if h.atoms(spy.Function)]
                ha = along(('H', r), hr, vals, x) if x is not None else None
                hb = vec([deriv_dir(h, Xm[r][j]) for h in hs], vals) if hs else []
                if ha is None or hb is None or any(d for _, d in ha) or any(hb):
                    return None, 'regime %d: a field changes the output' % (r + 1)
                bv = bracket_at(r, j, vals)
                if bv is None:
                    return None, 'regime %d: a bracket has a pole' % (r + 1)
                sup = support(r, vals, bv)
                if sup is None:
                    return None, ('regime %d: the fields are not invariant under the '
                                  'dynamics' % (r + 1))
                for k in sup - S:
                    S.add(k)
                    todo.append(k)
        return S, ''

    def image(ev_list, vals, x):
        """D E applied to the field or vector x at the point, and E of the point."""
        val_of = dict(zip(coords, vals))
        img = dict(val_of)
        xd = dict(zip(coords, x))
        for e in ev_list:
            ve = ck.expr(e['value'])
            v = ck.value(ve, sym_list, vals, p, skey='vd')
            if v is None:
                return None, None
            dv = 0
            for s_ in ve.free_symbols:
                d = ck.value(spy.diff(ve, s_), sym_list, vals, p, skey='vd')
                if d is None:
                    return None, None
                dv = (dv + d * x[coords.index(str(s_))]) % p if str(s_) in coords else dv
            s0 = int(val_of[e['var']]) % p
            img[e['var']] = Fraction({'replace': v, 'add': (s0 + v) % p,
                                      'multiply': s0 * v % p}[e['method']])
            xd[e['var']] = {'replace': dv, 'add': (xd[e['var']] + dv) % p,
                            'multiply': (xd[e['var']] * v + s0 * dv) % p}[e['method']]
        return [xd[nm] for nm in coords], [img[nm] for nm in coords]

    joins_of = {}
    for ev in events:
        if ev.get('lift') is not None:
            joins_of.setdefault(int(ev['lift']), []).append(ev)

    for li, lf in enumerate(lifts):
        r = int(lf['regime'])
        pt = {str(k): int(v) for k, v in dict(lf['point']).items()}
        vals0 = [Fraction(pt.get(nm, 0)) for nm in coords]
        taus = [ck.expr(t) for t in _as_list(lf.get('times', []))]
        vecs = [{str(k): int(x) % p for k, x in dict(v).items()}
                for v in _as_list(lf['vectors'])]
        # (D) the lifted directions in the span at the start, and the fields they reach
        S = set()
        for vd in vecs:
            sup = support(r, vals0, [vd.get(nm, 0) for nm in coords])
            if sup is None:
                return {'ok': False, 'why': 'a kernel direction is not in the span of the fields'}
            S |= sup
            for tau in taus:
                dv = [ck.value(spy.diff(tau, s_), sym_list, vals0, p, skey='vd') for s_ in tau.free_symbols]
                if any(x is None for x in dv) or sum(
                        x * vd.get(str(sy), 0) for x, sy in zip(dv, tau.free_symbols)) % p:
                    return {'ok': False, 'why': 'a kernel direction moves an event time'}
        S, why = close(r, S)
        if S is None:
            return {'ok': False, 'why': why}
        us = [[vd.get(nm, 0) if nm in free else 0 for nm in coords] for vd in vecs]
        # (C) the joins of the chain in order
        for ev in joins_of.get(li, []):
            r2 = int(ev['to'])
            evs = _as_list(ev['ev'])
            S2 = set()
            for _ in range(npts):
                vals = rand_point()
                items = [field_at(Xm[r][j], vals) for j in S] + us
                for x in items:
                    if x is None:
                        return {'ok': False, 'why': 'a field has a pole before an event'}
                    xi, zi = image(evs, vals, x)
                    if xi is None:
                        return {'ok': False, 'why': 'an event value has a pole'}
                    sup = support(r2, zi, xi)
                    if sup is None:
                        return {'ok': False, 'why': 'an event leaves the span of the fields'}
                    S2 |= sup
            S, why = close(r2, S2)
            if S is None:
                return {'ok': False, 'why': why}
            r = r2
    return {'ok': True, 'why': ''}


def static_fields_symbolic(conds, param_names, forcings=None):
    """Symmetries from the static subsystems of resting conditions.
    `conds` lists per condition {'rest': resting model lines, 'segs': [{'f', 'g', 'ev'}],
    't0': events at the start}; lines are "X = expr", events {'var', 'value', 'method'}.
    A state is static if no event changes it and its right-hand side in every segment
    depends on static states and parameters only and equals the resting one. Returns
    {'fields': [{param: expr}], 'absorbed': [...], 'static': [[...] per condition]} or
    {'fields': [], 'why': ...}."""
    forcings = set(_as_list(forcings))
    pnames = [str(x) for x in _as_list(param_names)]
    pset = set(pnames)

    def eqs(lines):
        out = {}
        for ln in _as_list(lines):
            k, v = str(ln).split('=', 1)
            out[k.strip()] = v.strip()
        return out

    def evkey(evs):
        return tuple((str(e['var']), str(e['value']), str(e['method'])) for e in _as_list(evs))
    lines = []
    for c in conds:
        lines += list(eqs(c['rest']).values())
        for sg in c['segs']:
            lines += list(eqs(sg['f']).values()) + list(eqs(sg['g']).values())
            lines += [str(e['value']) for e in _as_list(sg.get('ev'))]
        lines += [str(e['value']) for e in _as_list(c.get('t0'))]
    local, parse0 = _make_local_parse(lines + pnames)
    parse = functools.lru_cache(maxsize=None)(parse0)

    # pass 1, per distinct condition: static states, the parameters seen outside their
    # resting equations, and the static states seen
    info = {}
    static_of = []
    reduced = set()
    for c in conds:
        key = (tuple(_as_list(c['rest'])),
               tuple((tuple(_as_list(sg['f'])), tuple(_as_list(sg['g'])), evkey(sg.get('ev')))
                     for sg in c['segs']), evkey(c.get('t0')))
        if key in info:
            static_of.append(info[key]['static'])
            continue
        rest = {k: parse(v) for k, v in eqs(c['rest']).items()}
        segs = [({k: parse(v) for k, v in eqs(sg['f']).items()},
                 {k: parse(v) for k, v in eqs(sg['g']).items()},
                 _as_list(sg.get('ev'))) for sg in c['segs']]
        states = set(rest) | {k for f, _, _ in segs for k in f}
        # forcings held and the condition's zero states (as the resting model has them)
        held = {spy.Symbol(x): 0 for x in forcings}
        held.update({spy.Symbol(str(z)): 0 for z in _as_list(c.get('zero'))})
        rest = {k: v.xreplace(held) for k, v in rest.items()}
        moved = set()
        for ev in list(_as_list(c.get('t0'))) + [e for _, _, evs in segs for e in evs]:
            v = parse(str(ev['value']))
            m = str(ev['method'])
            var = str(ev['var'])
            if (m == 'add' and v != 0) or (m == 'multiply' and v != 1) or \
               (m == 'replace' and not (var in forcings and v == 0)):
                moved.add(var)
        dyn = set(moved)
        while True:
            n0 = len(dyn)
            for s_ in states - dyn - forcings:
                r0 = rest.get(s_)
                for f, _, _ in segs:
                    e = f.get(s_)
                    if e is None:
                        continue
                    if {str(x) for x in e.free_symbols} & dyn or r0 is None or \
                       (e.xreplace(held) != r0 and spy.expand(e.xreplace(held) - r0) != 0):
                        dyn.add(s_)
                        break
            if len(dyn) == n0:
                break
        static = sorted(states - dyn - forcings)
        seen = set()
        for x in dyn - forcings:
            if x in rest:
                seen |= rest[x].free_symbols
            for f, _, _ in segs:
                if x in f:
                    seen |= f[x].free_symbols
        for _, g, evs in segs:
            for e in g.values():
                seen |= e.free_symbols
            for ev in evs:
                seen |= parse(str(ev['value'])).free_symbols
        for ev in _as_list(c.get('t0')):
            seen |= parse(str(ev['value'])).free_symbols
        reduced |= {str(x) for x in seen if str(x) in pset}
        info[key] = {'static': static, 'rest': rest, 'seen': seen,
                     'zero': [str(z) for z in _as_list(c.get('zero'))]}
        static_of.append(static)
    # a parameter outside every static resting equation (one that enters only through
    # a recast atom, say) is not absorbed
    in_static = set()
    for inf in info.values():
        for x in inf['static']:
            if x in inf['rest']:
                in_static |= {str(y) for y in inf['rest'][x].free_symbols}
    absorbed = [x for x in pnames if x not in reduced and x in in_static]
    if not absorbed:
        return {'fields': [], 'absorbed': [], 'static': static_of}
    aset = {parse(x) for x in absorbed}

    # pass 2: fields that leave every static rest in place (delta rho = 0). Each resting
    # equation of a static state with an absorbed parameter, solved for that state, gives
    # a row d e / d theta_A; zero states (forced or sink clusters) are taken out first
    rows = []
    for inf in info.values():
        rest, static = inf['rest'], inf['static']
        zero = {spy.Symbol(x): 0 for x in inf.get('zero', [])}
        for x in static:
            if x not in rest or spy.Symbol(x) in zero:
                continue
            e = spy.expand(rest[x].xreplace(zero))
            if not (e.free_symbols & aset):
                continue
            xs = spy.Symbol(x)
            num, den = spy.fraction(spy.together(e))
            try:
                linear = xs not in den.free_symbols and spy.Poly(num, xs).degree() == 1
            except spy.PolynomialError:
                linear = False
            if not linear:
                return {'fields': [], 'why': 'a static resting equation not linear in its state',
                        'static': static_of}
            try:
                own = spy.solve(num, xs)
            except NotImplementedError:
                own = []
            if len(own) != 1:
                return {'fields': [], 'why': 'a static resting equation not solved for its state',
                        'static': static_of}
            rows.append([spy.cancel(spy.diff(e, a).xreplace({xs: own[0]})) for a in
                         [parse(y) for y in absorbed]])
    if state.env("STATICDIAG"):
        sys.stderr.write("[staticdiag] %d row(s) over %d absorbed parameter(s)\n"
                         % (len(rows), len(absorbed)))
    asy = [parse(x) for x in absorbed]
    if rows:
        basis = spy.Matrix(rows).nullspace(simplify=True)
    else:
        basis = [spy.Matrix([1 if j == i else 0 for j in range(len(asy))])
                 for i in range(len(asy))]
    state_syms = {spy.Symbol(x) for inf in info.values() for x in inf['rest']}
    fields = []
    for v in basis:
        v = spy.Matrix([spy.cancel(spy.together(x)) for x in v])
        if any(x.free_symbols & state_syms for x in v):
            continue
        den = spy.lcm([spy.fraction(x)[1] for x in v])
        v = spy.Matrix([spy.factor(spy.cancel(x * den)) for x in v])
        fields.append({absorbed[i]: str(v[i]) for i in range(len(absorbed)) if v[i] != 0})
    return {'fields': fields, 'absorbed': absorbed, 'static': static_of}


def extend_fields(fields, ic_lines, event_groups, states):
    """Components of each field on the parameters that are the value of an initial value
    or an event, from compatibility with that map. `event_groups` holds the events of one
    time each, applied together as E. x0 = X gives eta_X = eta_x(x0); an event x -> x + v,
    v, x v on a state s gives eta_v = eta_s(E x) - eta_s(x), eta_s(E x),
    (eta_s(E x) - v eta_s(x)) / x_s. A component that depends on the states means no
    extension, and the field is dropped. Returns the extended fields."""
    ic_lines = _as_list(ic_lines)
    groups = [_as_list(g) for g in _as_list(event_groups)]
    lines = list(ic_lines) + ['_ = %s' % e['value'] for g in groups for e in g]
    for fld in _as_list(fields):
        lines += ['_ = %s' % v for v in dict(fld).values()]
    local, parse = _make_local_parse(lines + ['_ = %s' % st for st in _as_list(states)])
    state_set = set(_as_list(states))
    def sym(nm):
        return local.get(nm, spy.Symbol(nm))
    state_syms = {sym(nm) for nm in state_set}
    out = []
    for fld in _as_list(fields):
        eta = {str(k): spy.sympify(parse(str(v))) for k, v in dict(fld).items()}
        ext = {}
        for l in ic_lines:
            lhs, rhs = _clean(l).split('=', 1)
            x, e = lhs.strip(), spy.sympify(parse(rhs))
            if e.is_Symbol and str(e) not in state_set:
                ext[str(e)] = eta.get(x, spy.Integer(0)).subs(sym(x), e)
        for g in groups:
            vals = {str(ev['var']): spy.sympify(parse(str(ev['value']))) for ev in g}
            E = {}
            for ev in g:
                s_, v = str(ev['var']), vals[str(ev['var'])]
                xs = sym(s_)
                E[xs] = {'add': xs + v, 'replace': v, 'multiply': xs * v}[ev['method']]
            for ev in g:
                s_, v = str(ev['var']), vals[str(ev['var'])]
                if not (v.is_Symbol and str(v) not in state_set):
                    continue
                xs = sym(s_)
                es = eta.get(s_, spy.Integer(0))
                e_e = es.xreplace(E) if E else es
                ext[str(v)] = {'add': e_e - es, 'replace': e_e,
                               'multiply': (e_e - v * es) / xs}[ev['method']]
        ok = True
        for k, val in ext.items():
            val = spy.simplify(val)
            if val.free_symbols & state_syms:
                ok = False
                break
            if val != 0:
                eta[k] = val
        if ok:
            out.append({k: str(v) for k, v in eta.items()})
    return out


def lift_start(ic_lines, t0events, point, vector, p, joint=False):
    """Start state of a chain and a direction lifted onto it: the initial values `ic_lines`
    ("X = e", unless `joint`, where the resting state and its direction are in `point`
    and `vector`), then the t0 events. Returns {'point', 'vector'} over states and
    parameters, or None on a pole."""
    p = int(p)
    ic_lines = _as_list(ic_lines)
    evs = _as_list(t0events)
    pt = {str(k): int(v) % p for k, v in dict(point).items()}
    vv = {str(k): int(v) % p for k, v in dict(vector).items()}
    names = sorted(pt)
    # parsed lines, lambdified values and derivatives are reused across points and primes
    key = (tuple(ic_lines), tuple((e['var'], str(e['value']), e['method']) for e in evs),
           tuple(names))
    ck = _LIFT_CACHE.get(key)
    if ck is None:
        if len(_LIFT_CACHE) > 256:
            _LIFT_CACHE.clear()
        ck = _DistCheck(ic_lines + ['_ = %s' % e['value'] for e in evs] +
                        ['_ = %s' % k for k in names])
        ck.parsed, ck.derivs = {}, {}
        _LIFT_CACHE[key] = ck
    syms = [ck.local.get(n, spy.Symbol(n)) for n in names]
    vals = [Fraction(pt[n]) for n in names]

    def parsed(text):
        e = ck.parsed.get(text)
        if e is None:
            e = ck.parsed[text] = ck.expr(text)
        return e

    skey = tuple(names)

    def val_and_dir(e):
        v = ck.value(e, syms, vals, p, skey=skey)
        ds = ck.derivs.get(e)
        if ds is None:
            ds = ck.derivs[e] = [(str(s), spy.diff(e, s)) for s in e.free_symbols]
        dv = 0
        for s, de in ds:
            d = ck.value(de, syms, vals, p, skey=skey)
            if d is None:
                return None, None
            dv = (dv + d * vv.get(s, 0)) % p
        return v, dv

    if not joint:
        for l in ic_lines:
            lhs, e = _clean(l).split('=', 1)
            x, dx = val_and_dir(parsed(e))
            if x is None:
                return None
            pt[lhs.strip()], vv[lhs.strip()] = x, dx
    for e in evs:
        v, dv = val_and_dir(parsed(str(e['value'])))
        if v is None:
            return None
        s0, d0 = pt.get(e['var'], 0), vv.get(e['var'], 0)
        if e['method'] == 'replace':
            pt[e['var']], vv[e['var']] = v, dv
        elif e['method'] == 'add':
            pt[e['var']], vv[e['var']] = (s0 + v) % p, (d0 + dv) % p
        else:
            pt[e['var']], vv[e['var']] = s0 * v % p, (d0 * v + s0 * dv) % p
    return {'point': pt, 'vector': vv}
