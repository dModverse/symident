"""Exponentials and trigonometric functions of states as auxiliary coordinates."""

import math

import numpy as np
import sympy as spy

from .charts import _exp_atom, _has_exp
from .linalg import _rank_mod
from .modp import _modp_rational
from .parse import _as_list, _is_rational_expr, _make_local_parse


def _canon_exp_base(b):
    """(r, k) with b = r^k, r = E or a positive rational that is not a perfect power;
    None for any other base."""
    if b == spy.E:
        return spy.E, 1
    b = spy.nsimplify(b, rational=True)
    if not (b.is_Rational and b.is_positive) or b == 1:
        return None
    sign = 1
    if b < 1:
        b, sign = 1 / b, -1
    fp, fq = spy.factorint(b.p), spy.factorint(b.q)
    g = 0
    for e in list(fp.values()) + list(fq.values()):
        g = math.gcd(g, int(e))
    num = den = 1
    for pr, e in fp.items():
        num *= pr ** (e // g)
    for pr, e in fq.items():
        den *= pr ** (e // g)
    return spy.Rational(num, den), sign * g


def _exp_terms(b, u):
    """b^u = r^(c0 + sum q*t): (r, c0, {t: q}) with q rational, or None."""
    rk = _canon_exp_base(b)
    if rk is None:
        return None
    r, k = rk
    u = spy.expand(spy.nsimplify(k * u, rational=True))
    terms = dict(u.as_coefficients_dict())
    c0 = terms.pop(spy.Integer(1), spy.Integer(0))
    if not (c0.is_Rational and all(q.is_Rational for q in terms.values())):
        return None
    return r, c0, terms


def _log_basis(r):
    """log(r) as {1: 1} for r = E, else {prime: exponent}."""
    if r == spy.E:
        return {1: 1}
    d = dict(spy.factorint(r.p))
    for pr, e in spy.factorint(r.q).items():
        d[pr] = d.get(pr, 0) - e
    return d


def _terms_independent(atoms):
    """True if 1 and the exponents tau*log(r) of the atoms [(r, tau)] are linearly
    independent over Q, with 1 and the logs of the primes independent. Checked at
    random integer points."""
    if not atoms:
        return True
    comps = [_log_basis(r) for r, _ in atoms]
    basis = sorted({b for d in comps for b in d}, key=str)
    taus = [t for _, t in atoms]
    syms = sorted(set().union(*[t.free_symbols for t in taus]), key=str)
    m, nb = len(atoms), len(basis)
    rng = np.random.default_rng(7)
    rows, npts = [], 0
    for _ in range(4 * (m + 3)):
        pt = {s: spy.Integer(int(v)) for s, v in zip(syms, rng.integers(2, 997, len(syms)))}
        vals = [t.xreplace(pt) for t in taus]
        vals = [v if v.is_Rational else spy.nsimplify(v) for v in vals]
        if not all(v.is_Rational for v in vals):
            continue
        for bi, b in enumerate(basis):
            rows.append([comps[k].get(b, 0) * vals[k] for k in range(m)] +
                        [1 if j == bi else 0 for j in range(nb)])
        npts += 1
        if npts >= m + 3:
            break
    if npts < m + 1:
        return False
    # full rank over GF(p) implies full rank over Q
    for p in (2147483629, 2147483587):
        if any(spy.Rational(v).q % p == 0 for r in rows for v in r):
            continue
        mod_rows = [[_modp_rational(v, p) for v in r] for r in rows]
        if _rank_mod(mod_rows, p) == m + nb:
            return True
    return spy.Matrix(rows).rank() == m + nb


class _ExpCtx:
    """Fresh symbols and constant leaves shared by one exponential rewrite: e^(1/L0)
    as `ec` (a power of it until L0 is known), log(p) as one leaf per prime p."""
    def __init__(self, taken):
        self.taken = set(taken)
        self.ec = self.fresh('_e_', positive=True)
        self.ln_sym = {}

    def fresh(self, stem, **kw):
        """A new symbol from `stem`, with underscores appended until unused, marked as taken."""
        nm = stem
        while nm in self.taken:
            nm += '_'
        self.taken.add(nm)
        return spy.Symbol(nm, **kw)

    def ln_of(self, r):
        """log(r) through one leaf per prime."""
        out = spy.Integer(0)
        for pr, e in _log_basis(r).items():
            if pr == 1:
                out += e
                continue
            if pr not in self.ln_sym:
                self.ln_sym[pr] = self.fresh('_ln%d_' % pr, positive=True)
            out += e * self.ln_sym[pr]
        return out

    @staticmethod
    def rpow(r, e):
        """`r` to the power `e`, as exp(e) for the base E."""
        return spy.exp(e) if r == spy.E else r ** e

    def const(self, r, c0):
        """r^c0, None when it is irrational and not a power of e."""
        if c0 == 0:
            return spy.Integer(1)
        if r == spy.E:
            return self.ec ** c0
        return r ** c0 if c0.is_Integer else None

    def numbers(self, e):
        """E as the leaf ec, log(b) of a number as a multiple of a log leaf."""
        e = spy.sympify(e).xreplace({spy.E: self.ec})
        rep = {}
        for L in e.atoms(spy.log):
            if L.args[0].is_number:
                rk = _canon_exp_base(L.args[0])
                if rk is not None:
                    rep[L] = rk[1] * self.ln_of(rk[0])
        return e.xreplace(rep) if rep else e


def _exp_leaf_canon(exprs, ctx):
    """Every exponential in `exprs`, innermost first, as a product of generic leaves
    W = r^(tau/L), one per term tau, and powers of ctx.ec. Returns {'exprs', 'atoms'}
    with atoms [{'W', 'r', 'tau', 'L'}], after ec -> ec^L0; or {'why': ...}."""
    exprs = [ctx.numbers(e) for e in exprs]
    atoms = []
    while True:
        found = {}
        for e in exprs:
            for at in e.atoms(spy.exp, spy.Pow):
                if at in found or _exp_atom(at) is None:
                    continue
                if any(_exp_atom(a) is not None
                       for a in _exp_atom(at)[1].atoms(spy.exp, spy.Pow)):
                    continue
                dec = _exp_terms(*_exp_atom(at))
                if dec is None or ctx.const(dec[0], dec[1]) is None:
                    return {'why': 'the exponential %s is not supported by '
                            'engine="modular"; try engine="symbolic"' % at}
                found[at] = dec
        if not found:
            break
        den = {}
        for r, _c0, terms in found.values():
            for t, q in terms.items():
                den[(r, t)] = spy.ilcm(den.get((r, t), 1), q.q)
        key_w = {}
        for (r, t), L in sorted(den.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            W = ctx.fresh('_ew%d_' % (len(atoms) + 1), positive=True)
            key_w[(r, t)] = W
            atoms.append({'W': W, 'r': r, 'tau': t, 'L': L})
        repl = {}
        for at, (r, c0, terms) in found.items():
            v = ctx.const(r, c0)
            for t, q in terms.items():
                v = v * key_w[(r, t)] ** int(q * den[(r, t)])
            repl[at] = v
        exprs = [e.xreplace(repl) for e in exprs]
    # e^c as integer powers of the leaf e^(1/L0)
    L0 = 1
    for e in exprs + [w['tau'] for w in atoms]:
        for pw in e.atoms(spy.Pow):
            if pw.base == ctx.ec:
                L0 = spy.ilcm(L0, spy.Rational(pw.exp).q)
    ec_sub = {ctx.ec: ctx.ec ** L0}
    exprs = [e.xreplace(ec_sub) for e in exprs]
    for w in atoms:
        w['tau'] = w['tau'].xreplace(ec_sub)
    ctx.L0 = L0
    if not _terms_independent([(w['r'], w['tau']) for w in atoms]):
        return {'why': 'exponents that are linearly dependent over the rationals '
                'are not supported'}
    return {'exprs': exprs, 'atoms': atoms}


def _exp_back(atoms, ctx, used):
    """Back-substitution map {name: value string} for the leaves and constants."""
    back = {str(w['W']): str(ctx.rpow(w['r'], w['tau'] / w['L'])) for w in atoms}
    if ctx.ec in used:
        back[str(ctx.ec)] = str(spy.exp(spy.Rational(1, ctx.L0)))
    for r, s in ctx.ln_sym.items():
        back[str(s)] = 'log(%s)' % r
    return back


def _apply_exp_recast(S, per_cond, ev_per, tm_per, taken):
    """Replace every exponential in the per-condition model by auxiliary states and
    generic exponential leaves (see above). `ev_per` holds each condition's later
    events as {'var', 'value', 'method'} with sympy values, `tm_per` its segment start
    time or None. Returns a dict with the extended S, per_cond, ev_per and tm_per, the
    relation list [(W, z, expr)], the fixed constant leaves, the back-substitution map
    and the codimension added; or {'why': ...} for an unsupported form."""
    ctx = _ExpCtx(taken)
    rpow, const = ctx.rpow, ctx.const
    S = list(S)
    K = len(per_cond)
    f = [[ctx.numbers(e) for e in p[0]] for p in per_cond]
    g = [[ctx.numbers(e) for e in p[1]] for p in per_cond]
    ic = [{k: ctx.numbers(v) for k, v in p[2].items()} for p in per_cond]
    fss = [list(p[3]) for p in per_cond]
    ev = [[dict(e, value=ctx.numbers(e['value'])) for e in evs] for evs in ev_per]
    tm = [None if t is None else ctx.numbers(t) for t in tm_per]

    # ---- dynamics: auxiliary states, innermost exponential first
    aux = []                         # {'X', 'r', 'phi'}
    n_real = len(S)
    while True:
        states = set(S)

        def state_atom(a):
            be = _exp_atom(a)
            return be is not None and bool(be[1].free_symbols & states)
        found = {}
        for c in range(K):
            for e in f[c] + g[c]:
                for at in e.atoms(spy.exp, spy.Pow):
                    if at in found or not state_atom(at):
                        continue
                    if any(state_atom(a) for a in _exp_atom(at)[1].atoms(spy.exp, spy.Pow)):
                        continue
                    dec = _exp_terms(*_exp_atom(at))
                    if dec is None or const(dec[0], dec[1]) is None:
                        return {'why': 'the exponential %s is not supported by '
                            'engine="modular"; try engine="symbolic"' % at}
                    found[at] = dec
        if not found:
            break
        den = {}
        for r, _c0, terms in found.values():
            for t, q in terms.items():
                if t.free_symbols & states:
                    den[(r, t)] = spy.ilcm(den.get((r, t), 1), q.q)
        key_x = {}
        for (r, t), D in sorted(den.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
            X = ctx.fresh('_ex%d_' % (len(aux) + 1))
            key_x[(r, t)] = X
            aux.append({'X': X, 'r': r, 'phi': t / D})
        repl = {}
        for at, (r, c0, terms) in found.items():
            v = const(r, c0)
            for t, q in terms.items():
                if (r, t) in key_x:
                    v = v * key_x[(r, t)] ** int(q * den[(r, t)])
                else:
                    v = v * rpow(r, q * t)
            repl[at] = v
        f = [[e.xreplace(repl) for e in fc] for fc in f]
        g = [[e.xreplace(repl) for e in gc] for gc in g]
        new_aux = aux[len(aux) - len(key_x):]
        for c in range(K):
            rhs_of = dict(zip(S, f[c]))
            for a in new_aux:
                phi = a['phi']
                dphi = sum(spy.diff(phi, s) * rhs_of[s] for s in phi.free_symbols & states)
                f[c].append(spy.expand(a['X'] * ctx.ln_of(a['r']) * dphi))
                ic_sub = {s: ic[c][str(s)] for s in phi.free_symbols & states}
                ic[c][str(a['X'])] = rpow(a['r'], phi.xreplace(ic_sub))
                fss[c].append(spy.Integer(0))
        S += [a['X'] for a in new_aux]

    # an auxiliary state no observable depends on stays at 1 in that condition
    idx = {X: i for i, X in enumerate(S)}
    for c in range(K):
        used = set()
        front = set()
        for e in f[c][:n_real] + g[c]:
            front |= e.free_symbols
        while front:
            nxt = set()
            for a in aux:
                if a['X'] in front and a['X'] not in used:
                    used.add(a['X'])
                    nxt |= f[c][idx[a['X']]].free_symbols
            front = nxt - used
        for a in aux:
            if a['X'] not in used:
                f[c][idx[a['X']]] = spy.Integer(0)
                ic[c][str(a['X'])] = spy.Integer(1)

    # later events on a state inside an exponent move the auxiliary states with it
    Sset = set(S)
    for c in range(K):
        out = []
        for e in ev[c]:
            out.append(e)
            queue = [e]
            while queue:
                cur = queue.pop(0)
                for a in aux:
                    phi = a['phi']
                    vs = [s for s in phi.free_symbols if str(s) == str(cur['var'])]
                    if not vs or f[c][idx[a['X']]] == 0:
                        continue
                    vsym = vs[0]
                    others = (phi.free_symbols & Sset) - {vsym}
                    d1 = spy.diff(phi, vsym)
                    if cur['method'] == 'replace' and not others:
                        new = {'var': str(a['X']), 'method': 'replace',
                               'value': rpow(a['r'], phi.xreplace({vsym: cur['value']}))}
                    elif (cur['method'] == 'add' and spy.diff(d1, vsym) == 0 and
                          not (d1.free_symbols & Sset)):
                        new = {'var': str(a['X']), 'method': 'multiply',
                               'value': rpow(a['r'], cur['value'] * d1)}
                    else:
                        return {'why': 'an event (%s) on %s, which enters an exponent, '
                                'is not supported' % (cur['method'], cur['var'])}
                    out.append(new)
                    queue.append(new)
        ev[c] = out

    # ---- leaf level: generic exponential leaves W
    refs = ([('f', c, i) for c in range(K) for i in range(len(f[c]))] +
            [('g', c, i) for c in range(K) for i in range(len(g[c]))] +
            [('ic', c, k) for c in range(K) for k in ic[c]] +
            [('ev', c, i) for c in range(K) for i in range(len(ev[c]))] +
            [('tm', c, None) for c in range(K) if tm[c] is not None])
    store = {'f': f, 'g': g, 'ic': ic}
    vals = []
    for kind, c, i in refs:
        vals.append(ev[c][i]['value'] if kind == 'ev' else tm[c] if kind == 'tm'
                    else store[kind][c][i])
    lc = _exp_leaf_canon(vals, ctx)
    if 'why' in lc:
        return lc
    for (kind, c, i), v in zip(refs, lc['exprs']):
        if kind == 'ev':
            ev[c][i]['value'] = v
        elif kind == 'tm':
            tm[c] = v
        else:
            store[kind][c][i] = v
    w_atoms = lc['atoms']

    const_syms = set(ctx.ln_sym.values())
    used = set()
    for v in lc['exprs'] + [w['tau'] for w in w_atoms]:
        used |= v.free_symbols
    if ctx.ec in used:
        const_syms.add(ctx.ec)
    rel = []
    for w in w_atoms:
        phi = w['tau'] / w['L']
        for z in sorted(phi.free_symbols - const_syms, key=str):
            rel.append((str(w['W']), str(z),
                        spy.together(w['W'] * ctx.ln_of(w['r']) * spy.diff(phi, z))))
    per_cond = [(f[c], g[c], ic[c], fss[c]) for c in range(K)]
    return {'S': S, 'per_cond': per_cond, 'ev_per': ev, 'tm_per': tm, 'rel': rel,
            'consts': sorted(str(s) for s in const_syms),
            'atoms': [str(w['W']) for w in w_atoms], 'back': _exp_back(w_atoms, ctx, used),
            'n_aux': len(aux), 'codim': len(aux) + len(w_atoms)}


_TRIG = (spy.sin, spy.cos, spy.tan)


_TRIG_RECIPROCAL = {spy.cot: lambda a: spy.cos(a) / spy.sin(a),
                    spy.sec: lambda a: 1 / spy.cos(a),
                    spy.csc: lambda a: 1 / spy.sin(a)}


def _has_trig(e):
    e = spy.sympify(e)
    return e.has(*_TRIG) or e.has(*_TRIG_RECIPROCAL)


def _trig_basic(e, pi_sym):
    """cot, sec and csc through sin and cos, a model symbol pi as the number."""
    e = spy.sympify(e)
    if pi_sym is not None:
        e = e.xreplace({pi_sym: spy.pi})
    return e.replace(lambda x: type(x) in _TRIG_RECIPROCAL,
                     lambda x: _TRIG_RECIPROCAL[type(x)](x.args[0]))


def _trig_terms(fn, u):
    """fn(u) with u = c0 + cp*pi + sum q*t as (c0, cp, {t: q}), all rational; tan in
    the doubled argument, tan(u) = sin(2u)/(1 + cos(2u)). None for another form."""
    if fn is spy.tan:
        u = 2 * u
    u = spy.expand(spy.nsimplify(u, rational=True))
    terms = dict(u.as_coefficients_dict())
    c0 = terms.pop(spy.Integer(1), spy.Integer(0))
    cp = terms.pop(spy.pi, spy.Integer(0))
    if not (c0.is_Rational and cp.is_Rational and (2 * cp).is_Integer and
            all(q.is_Rational for q in terms.values())):
        return None
    # an inner trig atom left in a term is one the later leaf pass resolves
    if any(_has_exp(t) or not _is_rational_expr(
            t.replace(lambda x: isinstance(x, _TRIG), lambda x: spy.Dummy())
             .xreplace({spy.pi: 1})) for t in terms):
        return None
    return c0, cp, terms


def _trig_value(fn, factors, k, rest):
    """fn(u) for e^(i*v) = i^k e^(i*rest) prod ((1 + i*T)/(1 - i*T))^m over factors
    [(T, m)], v = u or 2u for tan; cos and sin of rest stay as they are."""
    I = spy.I
    reps, A, n = {}, spy.Integer(1), spy.Integer(1)
    for T, m in factors:
        d = spy.Dummy(real=True)
        reps[d] = T
        A *= (1 + I * d) ** m if m > 0 else (1 - I * d) ** (-m)
        n *= (1 + T ** 2) ** abs(m)
    Z = I ** (int(k) % 4) * spy.expand(A ** 2)
    if rest != 0:
        C, Sn = spy.Dummy(real=True), spy.Dummy(real=True)
        reps[C], reps[Sn] = spy.cos(rest), spy.sin(rest)
        Z = Z * (C + I * Sn)
    re, im = spy.expand(Z).as_real_imag()
    re, im = re.xreplace(reps), im.xreplace(reps)
    if fn is spy.tan:
        return spy.cancel(im / (n + re))
    return spy.cancel((im if fn is spy.sin else re) / n)


def _trig_find(exprs, in_scope, why):
    """Innermost trig atoms of `exprs` whose argument `in_scope` accepts, with their
    decomposition; {'why': ...} for an unsupported argument."""
    found = {}
    for e in exprs:
        for at in e.atoms(*_TRIG):
            if at in found or not in_scope(at):
                continue
            if any(in_scope(a) for a in at.args[0].atoms(*_TRIG)):
                continue
            dec = _trig_terms(type(at), at.args[0])
            if dec is None:
                return {'why': why % at}
            found[at] = dec
    return found


def _trig_den(found, keep):
    """Per term t that `keep` takes the least D with q*D integer over all atoms."""
    den = {}
    for _c0, _cp, terms in found.values():
        for t, q in terms.items():
            if keep(t):
                den[t] = spy.ilcm(den.get(t, 1), q.q)
    return den


def _apply_trig_recast(S, per_cond, ev_per, tm_per, taken):
    """Replace every sin, cos and tan (and cot, sec, csc) in the per-condition model by
    auxiliary half-angle states and generic half-angle leaves (see above). Arguments
    and return as _apply_exp_recast, or {'why': ...} for an unsupported form."""
    ctx = _ExpCtx(taken)
    pis = ctx.fresh('_pi_', positive=True)
    pi_sym = next((s for p in per_cond
                   for e in list(p[0]) + list(p[1]) + list(p[2].values())
                   for s in spy.sympify(e).free_symbols if str(s) == 'pi'), None)
    if pi_sym is not None and pi_sym in set(S):
        pi_sym = None
    def tb(e):
        return _trig_basic(e, pi_sym)
    S = list(S)
    K = len(per_cond)
    f = [[tb(e) for e in p[0]] for p in per_cond]
    g = [[tb(e) for e in p[1]] for p in per_cond]
    ic = [{k: tb(v) for k, v in p[2].items()} for p in per_cond]
    fss = [[tb(e).xreplace({spy.pi: pis}) for e in p[3]] for p in per_cond]
    ev = [[dict(e, value=tb(e['value'])) for e in evs] for evs in ev_per]
    tm = [None if t is None else tb(t) for t in tm_per]
    unsupported = ('the argument of %s is not supported by engine="modular"; '
                   'try engine="symbolic"')

    # ---- dynamics: auxiliary half-angle states, innermost first
    aux = []                         # {'X', 'phi'}
    while True:
        states = set(S)
        def state_atom(a):
            return bool(a.args[0].free_symbols & states)
        found = _trig_find([e for c in range(K) for e in f[c] + g[c]], state_atom,
                           unsupported)
        if 'why' in found:
            return found
        if not found:
            break
        den = _trig_den(found, lambda t: bool(t.free_symbols & states))
        key_x = {}
        for t, D in sorted(den.items(), key=lambda kv: str(kv[0])):
            X = ctx.fresh('_tx%d_' % (len(aux) + 1))
            key_x[t] = X
            aux.append({'X': X, 'phi': t / D})
        repl = {}
        for at, (c0, cp, terms) in found.items():
            factors = [(key_x[t], int(q * den[t]))
                       for t, q in terms.items() if t in key_x]
            rest = c0 + sum((q * t for t, q in terms.items() if t not in key_x),
                            spy.Integer(0))
            repl[at] = _trig_value(type(at), factors, 2 * cp, rest)
        f = [[e.xreplace(repl) for e in fc] for fc in f]
        g = [[e.xreplace(repl) for e in gc] for gc in g]
        new_aux = aux[len(aux) - len(key_x):]
        for c in range(K):
            rhs_of = dict(zip(S, f[c]))
            for a in new_aux:
                phi = a['phi']
                dphi = sum(spy.diff(phi, s) * rhs_of[s]
                           for s in phi.free_symbols & states)
                f[c].append(spy.expand((1 + a['X'] ** 2) * dphi / 2))
                ic_sub = {s: ic[c][str(s)] for s in phi.free_symbols & states}
                ic[c][str(a['X'])] = spy.tan(phi.xreplace(ic_sub) / 2)
                fss[c].append(spy.Integer(0))
        S += [a['X'] for a in new_aux]

    # an auxiliary state no observable depends on stays at 0 in that condition
    n_real = len(S) - len(aux)
    idx = {X: i for i, X in enumerate(S)}
    for c in range(K):
        used, front = set(), set()
        for e in f[c][:n_real] + g[c]:
            front |= e.free_symbols
        while front:
            nxt = set()
            for a in aux:
                if a['X'] in front and a['X'] not in used:
                    used.add(a['X'])
                    nxt |= f[c][idx[a['X']]].free_symbols
            front = nxt - used
        for a in aux:
            if a['X'] not in used:
                f[c][idx[a['X']]] = spy.Integer(0)
                ic[c][str(a['X'])] = spy.Integer(0)

    # a later replace event on a state inside an argument resets the auxiliary state
    Sset = set(S)
    for c in range(K):
        out = []
        for e in ev[c]:
            out.append(e)
            for a in aux:
                phi = a['phi']
                vs = [s for s in phi.free_symbols if str(s) == str(e['var'])]
                if not vs or f[c][idx[a['X']]] == 0:
                    continue
                if e['method'] != 'replace' or (phi.free_symbols & Sset) - {vs[0]}:
                    return {'why': 'an event (%s) on %s, which enters the argument of '
                            'a trigonometric function, is not supported'
                            % (e['method'], e['var'])}
                out.append({'var': str(a['X']), 'method': 'replace',
                            'value': spy.tan(phi.xreplace({vs[0]: e['value']}) / 2)})
        ev[c] = out

    # ---- leaf level: generic half-angle leaves V and the constant leaves
    refs = ([('f', c, i) for c in range(K) for i in range(len(f[c]))] +
            [('g', c, i) for c in range(K) for i in range(len(g[c]))] +
            [('ic', c, k) for c in range(K) for k in ic[c]] +
            [('ev', c, i) for c in range(K) for i in range(len(ev[c]))] +
            [('tm', c, None) for c in range(K) if tm[c] is not None])
    store = {'f': f, 'g': g, 'ic': ic}
    vals = [ev[c][i]['value'] if kind == 'ev' else tm[c] if kind == 'tm'
            else store[kind][c][i] for kind, c, i in refs]
    # rational offsets c0 as powers of e^(i/M0), M0 over all arguments
    atoms, M0 = [], 1
    for e in vals:
        for at in e.atoms(*_TRIG):
            dec = _trig_terms(type(at), at.args[0])
            if dec is not None:
                M0 = spy.ilcm(M0, dec[0].q)
    tc = ctx.fresh('_tc_', positive=True)
    while True:
        found = _trig_find(vals, lambda a: True, unsupported)
        if 'why' in found:
            return found
        if not found:
            break
        den = _trig_den(found, lambda t: True)
        key_v = {}
        for t, D in sorted(den.items(), key=lambda kv: str(kv[0])):
            V = ctx.fresh('_tw%d_' % (len(atoms) + 1))
            key_v[t] = V
            atoms.append({'V': V, 'phi': t / D})
        repl = {}
        for at, (c0, cp, terms) in found.items():
            factors = [(key_v[t], int(q * den[t])) for t, q in terms.items()]
            if c0 != 0:
                if not (c0 * M0).is_Integer:
                    return {'why': unsupported % at}
                factors.append((tc, int(c0 * M0)))
            repl[at] = _trig_value(type(at), factors, 2 * cp, spy.Integer(0))
        vals = [e.xreplace(repl) for e in vals]
    if not _terms_independent([(spy.E, a['phi'].xreplace({spy.pi: spy.Symbol('_pi_')}))
                               for a in atoms]):
        return {'why': 'arguments of trigonometric functions that are linearly '
                'dependent over the rationals are not supported'}

    vals = [e.xreplace({spy.pi: pis}) for e in vals]
    for a in atoms:
        a['phi'] = a['phi'].xreplace({spy.pi: pis})
    for (kind, c, i), v in zip(refs, vals):
        if kind == 'ev':
            ev[c][i]['value'] = v
        elif kind == 'tm':
            tm[c] = v
        else:
            store[kind][c][i] = v

    used = set()
    for v in vals + [a['phi'] for a in atoms]:
        used |= v.free_symbols
    consts = {pis, tc} & used
    rel = []
    for a in atoms:
        for z in sorted(a['phi'].free_symbols - consts, key=str):
            rel.append((str(a['V']), str(z),
                        spy.together((1 + a['V'] ** 2) * spy.diff(a['phi'], z) / 2)))
    back = {str(a['V']): str(spy.tan(a['phi'].xreplace({pis: spy.pi}) / 2))
            for a in atoms}
    if pis in consts:
        back[str(pis)] = 'pi'
    if tc in consts:
        back[str(tc)] = str(spy.tan(spy.Rational(1, 2 * M0)))
    per_cond = [(f[c], g[c], ic[c], fss[c]) for c in range(K)]
    return {'S': S, 'per_cond': per_cond, 'ev_per': ev, 'tm_per': tm, 'rel': rel,
            'consts': sorted(str(s) for s in consts),
            'atoms': [str(a['V']) for a in atoms], 'back': back,
            'n_aux': len(aux), 'codim': len(aux) + len(atoms)}


def _merge_recast(a, b):
    """Two recast results as one: b applied after a."""
    if a is None or b is None:
        return b if a is None else a
    back = dict(a['back'])
    back.update(b['back'])
    return dict(b, rel=a['rel'] + b['rel'], atoms=a['atoms'] + b['atoms'], back=back,
                consts=sorted(set(a['consts']) | set(b['consts'])),
                n_aux=a['n_aux'] + b['n_aux'], codim=a['codim'] + b['codim'])


def _tidy_logs(e):
    """Common factors pulled out and each numeric sum of logs as one log."""
    e = spy.factor_terms(spy.cancel(e))
    return e.replace(lambda x: x.is_Add and x.is_number and x.has(spy.log),
                     lambda x: spy.logcombine(x, force=True))


def drop_monomial_content(vector, skip=None):
    """A direction {coordinate: component} with denominators cleared, divided by the
    power product of symbols common to every component and signed so that the first
    coordinate has a positive component: the same direction. Components in `skip`
    are scaled along but do not set the factor."""
    skip = set(_as_list(skip)) if skip is not None else set()
    items = [(str(k), str(v)) for k, v in dict(vector).items()]
    local, parse = _make_local_parse([v for _, v in items])
    es = [spy.cancel(parse(v)) for _, v in items]
    L = spy.Integer(1)
    for (k, _), e in zip(items, es):
        if k not in skip:
            L = spy.lcm(L, spy.fraction(e)[1])
    es = [spy.factor_terms(spy.cancel(e * L)) for e in es]
    first = [e for (k, _), e in sorted(zip(items, es)) if e != 0 and k not in skip]
    if first and first[0].could_extract_minus_sign():
        es = [-e for e in es]
    common = None
    for (k, _), e in zip(items, es):
        if k in skip:
            continue
        pw = {}
        num, den = spy.fraction(e)
        for sgn, part in ((1, num), (-1, den)):
            for f in spy.Mul.make_args(part):
                b, x = f.as_base_exp()
                if b.is_Symbol and x.is_Integer:
                    pw[b] = pw.get(b, 0) + sgn * int(x)
        common = pw if common is None else {
            b: (min(x, common[b]) if x > 0 else max(x, common[b]))
            for b, x in pw.items() if b in common and (x > 0) == (common[b] > 0)}
    m = spy.Integer(1)
    for b, x in (common or {}).items():
        m *= b ** x
    return {k: str(spy.cancel(e / m)) for (k, _), e in zip(items, es)}


def exp_backsub_expr(expr, names, values):
    """A reported component in the exponential leaves back in the model symbols:
    each name is replaced by its value until none is left."""
    def as_list(v):
        return list(v) if isinstance(v, (list, tuple)) else [v]
    names, values = as_list(names), as_list(values)
    local, parse = _make_local_parse([str(expr)] + [str(v) for v in values] +
                                     [str(n) for n in names])
    e = parse(str(expr))
    sub = {parse(str(n)): parse(str(v)) for n, v in zip(names, values)}
    for _ in range(len(sub) + 1):
        e2 = e.xreplace(sub)
        if e2 == e:
            break
        e = e2
    return str(_tidy_logs(spy.powsimp(spy.cancel(e))))


def trig_full_angle(vector, skip=None):
    """A direction {coordinate: component} over half-angle leaves V = tan(u/2) in
    sin(u) and cos(u): denominators cleared and common factors divided out in V, then
    divided by (1 + V^2)^k with the least k that makes every component a polynomial
    in sin(u) and cos(u). Components in `skip` are scaled along but do not set the
    factor."""
    skip = set(_as_list(skip)) if skip is not None else set()
    items = [(str(k), str(v)) for k, v in dict(vector).items()]
    local, parse = _make_local_parse([v for _, v in items])
    es = [spy.sympify(parse(v)) for _, v in items]
    halves = sorted(set().union(*[e.atoms(spy.tan) for e in es]), key=str)
    if not halves:
        return dict(items)
    Vs = [spy.Dummy('V') for _ in halves]
    es = [spy.cancel(e.xreplace(dict(zip(halves, Vs)))) for e in es]
    keep = [k not in skip for k, _ in items]
    L = spy.Integer(1)
    for e, kp in zip(es, keep):
        if kp:
            L = spy.lcm(L, spy.fraction(e)[1])
    es = [spy.cancel(e * L) for e in es]
    G = spy.Integer(0)
    for e, kp in zip(es, keep):
        if kp and e != 0:
            G = spy.gcd(G, e)
    if G != 0:
        es = [spy.cancel(e / G) for e in es]
    # sin(u) = 2V/(1 + V^2), cos(u) = (1 - V^2)/(1 + V^2), 1 + V^2 = 2/(1 + cos(u))
    rels, sub = [], {}
    for h, V in zip(halves, Vs):
        deg = max((spy.degree(e, V) for e, kp in zip(es, keep) if kp and e != 0),
                  default=0)
        es = [e / (1 + V ** 2) ** ((deg + 1) // 2) for e in es]
        sn, cs = spy.sin(2 * h.args[0]), spy.cos(2 * h.args[0])
        sub[V] = sn / (1 + cs)
        rels.append((sn, cs))
    out = {}
    for (k, _), e in zip(items, es):
        v = spy.cancel(spy.together(e.xreplace(sub)))
        num, den = spy.fraction(v)
        for sn, cs in rels:
            num = spy.expand(num).subs(sn ** 2, 1 - cs ** 2)
            den = spy.expand(den).subs(sn ** 2, 1 - cs ** 2)
        v = spy.factor(spy.cancel(num / den))
        if spy.count_ops(v) < 200:
            v = spy.factor(spy.trigsimp(v))
        out[k] = str(v)
    return out


# ---- elementary functions of states ----
# log, inverse trigonometric and hyperbolic functions and non-integer powers of sums become
# auxiliary states with d phi(u)/dt = phi'(u)*du/dt; a transcendental function of leaves
# only becomes a generic leaf tied by relation rows, an algebraic one is left to the charts.
_FUNCS = (spy.log, spy.atan, spy.atanh, spy.asin, spy.acos, spy.asinh, spy.acosh)


def _func_pow(a):
    return a.is_Pow and not a.base.is_Symbol and not a.base.is_number and \
        a.exp.is_Rational and not a.exp.is_Integer


def _func_atoms(e):
    e = spy.sympify(e)
    return [a for a in e.atoms(*_FUNCS) | e.atoms(spy.Pow) if not isinstance(a, spy.Pow) or _func_pow(a)]


def _has_func(e):
    return bool(_func_atoms(e))


def _func_derivative(a, u):
    """phi'(u) of the atom a = phi(u); None for the power, whose rhs is c*W*u'/u."""
    t = type(a)
    if t is spy.log:
        return 1 / u
    if t is spy.atan:
        return 1 / (1 + u ** 2)
    if t is spy.atanh:
        return 1 / (1 - u ** 2)
    if t is spy.asin:
        return 1 / spy.sqrt(1 - u ** 2)
    if t is spy.acos:
        return -1 / spy.sqrt(1 - u ** 2)
    if t is spy.asinh:
        return 1 / spy.sqrt(1 + u ** 2)
    if t is spy.acosh:
        return 1 / spy.sqrt(u ** 2 - 1)
    return None


def _engine_holds(constants=(), degrees=()):
    """Whether every engine prime holds the roots (see gfp.holds_roots); raises
    NeedPrimesError otherwise, so that the analysis reruns over primes that do."""
    from ..gfp import PRIMES, NeedPrimesError, holds_roots, verify_prime
    if not all(holds_roots(constants, q, degrees) for q in list(PRIMES) + [verify_prime()]):
        raise NeedPrimesError(constants, degrees)
    return True


def _is_sqrt_rational(e):
    """Whether e is rational in its symbols and rational powers of such expressions; a
    root only where the engine's primes hold it (raises NeedPrimesError otherwise)."""
    e = spy.sympify(e)
    roots = [a for a in e.atoms(spy.Pow) if a.exp.is_Rational and not a.exp.is_Integer]
    for a in roots:
        q, k = int(a.exp.q), 0
        while q % 2 == 0:
            q, k = q // 2, k + 1
        if a.base.is_Rational:
            if a.base <= 0:
                return False
            _engine_holds([(a.base, k)] if k else [], [q] if q > 1 else [])
        else:
            if not _is_sqrt_rational(a.base):
                return False
            if q > 1:
                _engine_holds([], [q])
    return _is_rational_expr(e.xreplace({a: spy.Dummy() for a in roots}))


def _func_arg(a):
    return a.base if isinstance(a, spy.Pow) else a.args[0]


def _apply_func_recast(S, per_cond, ev_per, tm_per, taken):
    """Replace the elementary functions of states (see above) by auxiliary states, and
    the transcendental ones of leaves by generic leaves. Arguments and return as
    _apply_exp_recast, or {'why': ...} for an unsupported form."""
    ctx = _ExpCtx(taken)
    S = list(S)
    K = len(per_cond)
    f = [[spy.sympify(e) for e in p[0]] for p in per_cond]
    g = [[spy.sympify(e) for e in p[1]] for p in per_cond]
    ic = [{k: spy.sympify(v) for k, v in p[2].items()} for p in per_cond]
    fss = [list(p[3]) for p in per_cond]
    ev = [[dict(e, value=spy.sympify(e['value'])) for e in evs] for evs in ev_per]
    tm = [None if t is None else spy.sympify(t) for t in tm_per]

    # ---- dynamics: auxiliary states, innermost function first
    aux = []
    while True:
        states = set(S)
        found = []
        for c in range(K):
            for e in f[c] + g[c]:
                for a in _func_atoms(e):
                    u = _func_arg(a)
                    if a in found or not (u.free_symbols & states):
                        continue
                    if any(_func_atoms(x) and (_func_arg(x).free_symbols & states)
                           for x in _func_atoms(u)):
                        continue
                    found.append(a)
        if not found:
            break
        found.sort(key=spy.default_sort_key)
        key = {a: ctx.fresh('_fx%d_' % (len(aux) + i + 1)) for i, a in enumerate(found)}
        f = [[e.xreplace(key) for e in fc] for fc in f]
        g = [[e.xreplace(key) for e in gc] for gc in g]
        for a in found:
            aux.append({'X': key[a], 'atom': a})
        for c in range(K):
            rhs_of = dict(zip(S, f[c]))
            for a in found:
                u = _func_arg(a).xreplace(key)
                du = sum((spy.diff(u, s) * rhs_of[s] for s in u.free_symbols & states), spy.Integer(0))
                dphi = _func_derivative(a, u)
                W = key[a]
                f[c].append(a.exp * W * du / u if dphi is None else dphi * du)
                ic_sub = {s: ic[c][str(s)] for s in _func_arg(a).free_symbols & states}
                ic[c][str(W)] = a.xreplace(ic_sub)
                fss[c].append(spy.Integer(0))
        for c in range(K):
            for e in ev[c]:
                if any(str(e['var']) in {str(s) for s in _func_arg(a).free_symbols} for a in found):
                    return {'why': 'an event on %s, which enters the argument of %s, is not '
                            'supported' % (e['var'], next(a for a in found if str(e['var']) in
                                                          {str(s) for s in _func_arg(a).free_symbols}))}
        S += [key[a] for a in found]

    # ---- leaf level: transcendental functions of leaves as generic leaves
    refs = ([('f', c, i) for c in range(K) for i in range(len(f[c]))] +
            [('g', c, i) for c in range(K) for i in range(len(g[c]))] +
            [('ic', c, k) for c in range(K) for k in ic[c]] +
            [('ev', c, i) for c in range(K) for i in range(len(ev[c]))] +
            [('tm', c, None) for c in range(K) if tm[c] is not None])
    store = {'f': f, 'g': g, 'ic': ic}
    vals = [ev[c][i]['value'] if kind == 'ev' else tm[c] if kind == 'tm'
            else store[kind][c][i] for kind, c, i in refs]
    atoms, consts, back = [], [], {}
    leaf_of = {}

    def log_leaf(p):
        # one generic leaf per irreducible factor: logs of distinct irreducibles are independent
        if p not in leaf_of:
            V = ctx.fresh('_fl%d_' % (len(atoms) + 1))
            leaf_of[p] = V
            atoms.append({'V': V, 'phi': p, 'kind': 'log', 'atom': spy.log(p), 'dphi': 1 / p})
        return leaf_of[p]

    def const_leaf(a):
        if a not in leaf_of:
            C = ctx.fresh('_fc%d_' % (len(consts) + 1), positive=bool(a.is_positive))
            leaf_of[a] = C
            consts.append(C)
            back[str(C)] = str(a)
        return leaf_of[a]

    def as_logs(u):
        num, den = spy.fraction(spy.together(u))
        out = spy.Integer(0)
        for part, sgn in ((num, 1), (den, -1)):
            c, facs = spy.factor_list(part)
            if c <= 0:
                return None
            if c != 1:
                out += sgn * const_leaf(spy.log(c))
            for p, m in facs:
                out += sgn * m * log_leaf(p)
        return out

    def leaf_value(a):
        u = _func_arg(a)
        if not u.free_symbols:
            v = spy.simplify(a)
            return v if v.is_Rational else const_leaf(a)
        t = type(a)
        if t is spy.log:
            return as_logs(u)
        if t is spy.atanh:
            lo, hi = as_logs(1 + u), as_logs(1 - u)
            return None if lo is None or hi is None else (lo - hi) / 2
        if t in (spy.atan, spy.asin, spy.acos, spy.asinh, spy.acosh):
            if a not in leaf_of:
                V = ctx.fresh('_fa%d_' % (len(atoms) + 1))
                leaf_of[a] = V
                atoms.append({'V': V, 'phi': u, 'kind': 'inv', 'atom': a,
                              'dphi': _func_derivative(a, u)})
            return leaf_of[a]
        return None
    while True:
        found = {}
        for e in vals:
            for a in _func_atoms(e):
                if a in found or any(_func_atoms(x) for x in _func_atoms(_func_arg(a))):
                    continue
                v = leaf_value(a)
                if v is not None:
                    found[a] = v
        if not found:
            break
        vals = [e.xreplace(found) for e in vals]
    # a leaf can cancel, as log(1 + x) in atanh(x/(1 + x)); one in no value is no coordinate
    used = set().union(*(spy.sympify(e).free_symbols for e in vals))
    atoms = [a for a in atoms if a['V'] in used]
    # pi and e left by an evaluated function value are transcendental constants
    vals = [e.xreplace({c: const_leaf(c) for c in e.atoms(spy.NumberSymbol) if c in (spy.pi, spy.E)})
            for e in vals]
    for a in aux:
        for (kind, _c, k), v in zip(refs, vals):
            if kind == 'ic' and k == str(a['X']) and not _is_sqrt_rational(v):
                return {'why': 'the term %s starts at %s, which the finite-field engine '
                        'cannot evaluate exactly; use engine="symbolic"'
                        % (str(a['atom']).replace('**', '^'), str(v).replace('**', '^'))}
    inv_args = [a['phi'] for a in atoms if a['kind'] == 'inv']
    for i, u in enumerate(inv_args):
        if any(u.free_symbols & v.free_symbols for v in inv_args[i + 1:]):
            return {'why': 'inverse trigonometric or hyperbolic functions of leaves that share '
                    'a coordinate (%s) are not supported' % u}
    for (kind, c, i), v in zip(refs, vals):
        if kind == 'ev':
            ev[c][i]['value'] = v
        elif kind == 'tm':
            tm[c] = v
        else:
            store[kind][c][i] = v

    cset = set(consts)
    rel = []
    for a in atoms:
        u = a['phi']
        for z in sorted(u.free_symbols - cset, key=str):
            rel.append((str(a['V']), str(z), spy.together(a['dphi'] * spy.diff(u, z))))
        back[str(a['V'])] = str(a['atom'])
    per_cond = [(f[c], g[c], ic[c], fss[c]) for c in range(K)]
    return {'S': S, 'per_cond': per_cond, 'ev_per': ev, 'tm_per': tm, 'rel': rel,
            'consts': sorted(str(s) for s in consts),
            'atoms': [str(a['V']) for a in atoms], 'back': back,
            'n_aux': len(aux), 'codim': len(aux) + len(atoms)}
