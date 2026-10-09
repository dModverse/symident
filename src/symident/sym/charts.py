"""Coordinate charts: signs, logarithms, exponents and their back-substitution."""

import sympy as spy

from .parse import _make_local_parse, _make_parse

_HYPERBOLIC = (spy.sinh, spy.cosh, spy.tanh, spy.coth, spy.sech, spy.csch)


def _hyp_to_exp(e):
    """Hyperbolic functions written in exponentials."""
    e = spy.sympify(e)
    if not e.has(*_HYPERBOLIC):
        return e
    return e.replace(lambda x: isinstance(x, _HYPERBOLIC), lambda x: x.rewrite(spy.exp))


def _has_exp(e):
    """True if e contains an exponential or the number E."""
    e = spy.sympify(e)
    return e.has(spy.E) or any(_exp_atom(a) is not None
                               for a in e.atoms(spy.exp, spy.Pow))


def _strip_log_obs(g):
    """An observable kappa*(sum a_i log(h_i)) + c, with a_i rational and kappa 1 or
    1/log(b) for a numeric b, is a strictly monotone function of exp(g/kappa) =
    prod h_i^a_i * exp(c/kappa) and holds the same information; that expression is
    returned. Anything else comes back unchanged."""
    g = spy.sympify(g)
    logs = [a for a in g.atoms(spy.log) if not a.args[0].is_number]
    if not logs:
        return g
    consts = [spy.Integer(1)] + [a for a in g.atoms(spy.log) if a.args[0].is_number]
    dums = [spy.Dummy() for _ in logs]
    gd = g.xreplace(dict(zip(logs, dums)))
    for kap in consts:
        g2 = spy.expand(gd * kap)
        coeffs = [g2.coeff(d) for d in dums]
        rest = spy.expand(g2 - sum(c * d for c, d in zip(coeffs, dums)))
        if rest.has(*dums):
            continue
        if not all(c.is_Rational for c in coeffs):
            continue
        if any(not a.args[0].is_number for a in rest.atoms(spy.log)):
            continue
        out = spy.Integer(1)
        for c, L in zip(coeffs, logs):
            out = out * L.args[0] ** c
        for t in spy.Add.make_args(rest):
            nl = [a for a in t.atoms(spy.log) if a.args[0].is_number]
            out = out * (nl[0].args[0] ** spy.cancel(t / nl[0]) if len(nl) == 1
                         else spy.exp(t))
        if not out.atoms(spy.log):
            return out
    return g


def _exp_atom(e):
    """(base, exponent) of an exponential atom with a numeric base, else None."""
    if isinstance(e, spy.exp):
        return spy.E, e.args[0]
    if e.is_Pow and e.base.is_number and e.base.is_positive and not e.exp.is_number:
        return e.base, e.exp
    return None


def _detect_log_params(exprs, states):
    """Parameters that occur only in exponents of numeric-base powers, base^(c*theta +
    ...) with c rational and one base per parameter. Returns {theta: base}. A constant
    term in the exponent must leave a rational factor base^k."""
    base_of, bad = {}, set()
    for e in exprs:
        for at in spy.sympify(e).atoms(spy.exp, spy.Pow):
            be = _exp_atom(at)
            if be is None:
                continue
            base, ex = be
            terms = spy.expand(ex).as_coefficients_dict()
            k = terms.pop(spy.Integer(1), spy.Integer(0))
            ok = k == 0 or (base != spy.E and base.is_Rational and k.is_Integer)
            for t, cf in terms.items():
                if not (t.is_Symbol and cf.is_Rational and t not in states) or not ok:
                    bad |= ex.free_symbols
                    break
                if base_of.get(t, base) != base:
                    bad.add(t)
                base_of[t] = base
    return {t: b for t, b in base_of.items() if t not in bad}


def _log_param_denominators(base_of, exprs):
    """Per parameter the lcm L of the denominators of its exponent coefficients, so that
    X = base^(theta/L) enters with integer powers only."""
    den = {t: 1 for t in base_of}
    for e in exprs:
        for at in spy.sympify(e).atoms(spy.exp, spy.Pow):
            be = _exp_atom(at)
            if be is None:
                continue
            for t, cf in spy.expand(be[1]).as_coefficients_dict().items():
                if t in den and cf.is_Rational:
                    den[t] = int(spy.ilcm(den[t], cf.q))
    return den


def _log_param_sub(lp):
    """Rewrite base^(sum c_i theta_i + k) as base^k * prod X_i^(c_i L_i), X_i = base^(theta_i/L_i);
    `lp` maps theta to (X, base) or (X, base, L)."""
    def repl(at):
        be = _exp_atom(at)
        if be is None:
            return at
        base, ex = be
        terms = spy.expand(ex).as_coefficients_dict()
        k = terms.pop(spy.Integer(1), spy.Integer(0))
        if not terms or any(t not in lp or lp[t][1] != base for t in terms):
            return at
        out = base ** k if k != 0 else spy.Integer(1)
        for t, cf in terms.items():
            L = lp[t][2] if len(lp[t]) > 2 else 1
            out = out * lp[t][0] ** (cf * L)
        return out

    def sub(e):
        e = spy.sympify(e)
        if not lp:
            return e
        return e.replace(lambda x: _exp_atom(x) is not None, repl)
    return sub


def _log_params_only_in_atoms(base_of, exprs):
    """Drop every candidate that still occurs after the rewrite: it enters somewhere
    other than an exponent and cannot be traded for base^theta."""
    base_of = dict(base_of)
    while base_of:
        sub = _log_param_sub({t: (spy.Dummy(), b) for t, b in base_of.items()})
        left = set()
        for e in exprs:
            left |= sub(e).free_symbols
        drop = [t for t in base_of if t in left]
        if not drop:
            break
        for t in drop:
            del base_of[t]
    return base_of


def sign_chart(f, gs, positive):
    """abs() and sign() resolved with the declared signs (abs(v) = v for positive
    v). `f` maps state -> rhs, `gs` is a list of observable dicts. Returns {'f',
    'g'} as strings, or {'why': ...} for max, min, a step (not analytic) or an
    abs() of undecided sign."""
    fl = dict(f)
    gl = [dict(g) for g in gs]
    lines = list(fl.values()) + [v for g in gl for v in g.values()] + [str(k) for k in fl]
    local, parse = _make_local_parse(lines)
    local = dict(local)
    local.update({'abs': spy.Abs, 'sign': spy.sign})
    parse = _make_parse(local)
    pos = None if positive is True else {str(x) for x in (positive or [])}
    names = {str(sym) for sym in local.values() if isinstance(sym, spy.Symbol)}
    to_pos = {spy.Symbol(n): spy.Symbol(n, positive=True)
              for n in names if pos is None or n in pos}
    back = {v: k for k, v in to_pos.items()}

    def one(k, rhs):
        if any(t in rhs for t in ('max(', 'min(', 'pmax(', 'pmin(', 'ifelse(',
                                  'Heaviside(')):
            raise ValueError('%s: max, min and steps are not analytic; analyse each '
                             'regime on its own, e.g. through `conditions`' % k)
        e = parse(rhs).xreplace(to_pos)
        if e.atoms(spy.Abs, spy.sign):
            raise ValueError('%s: the sign inside abs() or sign() is not decided by '
                             'the coordinates declared `positive`' % k)
        return str(e.xreplace(back))
    try:
        Fn = {k: one(k, str(v)) for k, v in fl.items()}
        Gn = [{k: one(k, str(v)) for k, v in g.items()} for g in gl]
    except ValueError as err:
        return {'why': str(err)}
    return {'f': Fn, 'g': Gn}


def _log_numbers(e, consts):
    """Rewrite log of a positive rational as sum e_p*lognum_p over its prime factors,
    collecting the constants lognum_p in `consts`. Exact: log 4 = 2 log 2 is kept, and
    logs of distinct primes are independent (Schanuel)."""
    rep = {}
    for a in e.atoms(spy.log):
        n = a.args[0]
        if not n.is_number:
            continue
        n = spy.nsimplify(n)
        if not (n.is_Rational and n > 0):
            raise ValueError('log(%s) of a number that is not a positive rational' % n)
        out = spy.Integer(0)
        for q, sgn in ((n.p, 1), (n.q, -1)):
            for pr, ex in spy.factorint(q).items():
                sym = spy.Symbol('lognum_%d' % pr)
                consts.add(str(sym))
                out += sgn * ex * sym
        rep[a] = out
    return e.xreplace(rep)


def log_arg_chart(f, gs, positive, taken, ics=None):
    """Log chart for positive arguments of log() and fractional powers: log(a + b*v),
    a and b free of v, a + b*v positive, becomes L with v = (exp(L) - a)/b; log(v) and
    v^(p/q) are a = 0, b = 1. Powers of products are split into powers of their positive
    factors, non-integer powers of sums become exp(exponent*log(sum)). A state v becomes
    L with rhs u'*exp(-L), u = a + b*v, and initial value log(u) at the initial values.
    `f` maps state -> rhs, `gs` and `ics` are lists of dicts, `positive` True or a list
    of names. Returns None when nothing qualifies, else {'f', 'g', 'ic', 'consts',
    'map': [{'v', 'L', 'a', 'b'}]}, or {'why': ...}."""
    fl = dict(f)
    gl = [dict(g) for g in gs]
    il = [dict(ic) for ic in (ics or [])]
    lines = (list(fl.values()) + [v for g in gl for v in g.values()] +
             [v for ic in il for v in ic.values()] +
             [str(k) for k in fl] + [str(t) for t in taken])
    local, parse = _make_local_parse(lines)
    pos = None if positive is True else {str(x) for x in (positive or [])}

    def is_pos(sym):
        return pos is None or str(sym) in pos

    # a power of a product splits into powers of its positive factors, and a non-integer
    # power of a sum becomes exp(exponent*log(sum)), which the chart makes rational
    def norm(e):
        P = {x: spy.Symbol(str(x), positive=True) for x in e.free_symbols if is_pos(x)}
        e = spy.expand_power_base(e.xreplace(P), deep=True)
        e = e.xreplace({v: k for k, v in P.items()})
        return e.replace(lambda x: x.is_Pow and x.base.is_Add and not x.exp.is_number,
                         lambda x: spy.exp(x.exp * spy.log(x.base)))

    def sum_powers(e):
        # u^c of a sum u with c not an integer is exp(c*log(u)): its base is charted like log(u)
        return [pw.base for pw in e.atoms(spy.Pow) if pw.base.is_Add and pw.exp.is_Rational
                and not pw.exp.is_Integer]
    F = {k: norm(parse(str(v))) for k, v in fl.items()}
    G = [{k: norm(parse(str(v))) for k, v in g.items()} for g in gl]
    I = [{k: norm(parse(str(v))) for k, v in ic.items()} for ic in il]
    normed = any(spy.simplify(F[k] - parse(str(v))) != 0 for k, v in fl.items()) or \
        any(spy.simplify(G[i][k] - parse(str(v))) != 0
            for i, g in enumerate(gl) for k, v in g.items()) or \
        any(sum_powers(e) for e in list(F.values()) + [e for g in G for e in g.values()])
    states = {spy.Symbol(k) for k in fl}
    one, zero = spy.Integer(1), spy.Integer(0)

    def positive_on(e):
        # sign certificate on the declared domain: coordinates are real, the declared ones positive
        sub = {x: spy.Symbol(str(x), positive=True) if is_pos(x) else spy.Symbol(str(x), real=True)
               for x in e.free_symbols}
        return bool(e.xreplace(sub).is_positive)

    chart = {}                               # v -> (a, b)

    def claim(v, a, b):
        if v in chart and (spy.simplify(chart[v][0] - a) != 0 or
                           spy.simplify(chart[v][1] - b) != 0):
            raise ValueError('%s enters two different logarithms, which no single '
                             'chart makes rational' % v)
        chart[v] = (a, b)

    # pass 0 claims log(v) and v^(p/q) of symbols, pass 1 the affine arguments, so a
    # plain logarithm keeps its symbol whatever the order of the terms
    def scan(e, ps):
        e = e.xreplace({at: spy.log(spy.factor(spy.together(at.args[0])))
                        for at in e.atoms(spy.log)})
        e = spy.expand_log(e, force=True)
        args = [at.args[0] for at in e.atoms(spy.log)] + [spy.factor(spy.together(u)) for u in sum_powers(e)]
        for arg in sorted(set(args), key=spy.default_sort_key):
            if arg.is_number:
                continue
            if arg.is_Symbol:
                if is_pos(arg) and ps == 0:
                    claim(arg, zero, one)
                continue
            if ps == 0:
                continue
            # affine in one symbol, a state first, then a parameter; a and b may hold
            # other coordinates (log(1 + x + y) in the chart of y)
            if not positive_on(arg):
                continue
            done = False
            for v in sorted(arg.free_symbols, key=lambda x: (x not in states, str(x))):
                b = spy.diff(arg, v)
                if v in b.free_symbols or b == 0:
                    continue
                a = spy.expand(arg - b * v)
                if v in chart and (spy.simplify(chart[v][0] - a) != 0 or
                                   spy.simplify(chart[v][1] - b) != 0):
                    continue
                claim(v, a, b)
                done = True
                break
            if not done:
                continue  # left to the function recast of the tapes
        for pw in e.atoms(spy.Pow):
            if pw.base.is_Symbol and pw.exp.is_Rational and not pw.exp.is_Integer:
                if is_pos(pw.base) and ps == 0:
                    claim(pw.base, zero, one)

    try:
        for ps in (0, 1):
            for e in F.values():
                scan(e, ps)
            for g in G:
                for e in g.values():
                    scan(_strip_log_obs(e), ps)
        # a free-exponent base with a given initial value has no leaf for the recast
        given = {spy.Symbol(k) for ic in I for k in ic}
        for e in list(F.values()) + [e for g in G for e in g.values()]:
            for pw in e.atoms(spy.Pow):
                if pw.base in given and not pw.exp.is_number and is_pos(pw.base):
                    claim(pw.base, zero, one)
        # an initial value of a charted state enters as the logarithm of its argument
        while True:
            n0 = len(chart)
            for ic in I:
                for k, _e in ic.items():
                    ks = spy.Symbol(k)
                    if ks in chart:
                        a, b = chart[ks]
                        at0 = {spy.Symbol(n): x for n, x in ic.items()}
                        for ps in (0, 1):
                            scan(spy.log((a + b * ks).xreplace(at0)), ps)
            if len(chart) == n0:
                break
    except ValueError as err:
        return {'why': str(err)}
    if not chart and not normed:
        return None

    used = {str(t) for t in taken} | {str(k) for k in fl}
    L = {}
    for v in sorted(chart, key=str):
        nm = 'log_%s' % v
        while nm in used:
            nm += '_'
        used.add(nm)
        L[v] = spy.Symbol(nm, real=True)
    sub0 = {v: (spy.exp(L[v]) - chart[v][0]) / chart[v][1] for v in chart}
    # a and b may hold other charted symbols
    sub = dict(sub0)
    for _ in chart:
        sub = {v: e.xreplace(sub) for v, e in sub0.items()}
    consts = set()

    def tidy(e):
        e = e.xreplace(sub)
        e = e.xreplace({at: spy.log(spy.cancel(at.args[0])) for at in e.atoms(spy.log)})
        return _log_numbers(spy.powsimp(spy.expand_log(e, force=True)), consts)
    Fn = {}
    for k, e in F.items():
        ks = spy.Symbol(k)
        if ks in chart:
            # log(u)' = u'/u with u = a + b*v and u' over every state it holds
            a, b = chart[ks]
            u = a + b * ks
            du = sum((spy.diff(u, spy.Symbol(s_)) * F[s_] for s_ in F
                      if spy.Symbol(s_) in u.free_symbols), spy.Integer(0))
            Fn[str(L[ks])] = str(spy.powsimp(spy.expand(tidy(du) * spy.exp(-L[ks]))))
        else:
            Fn[k] = str(tidy(e))
    Gn = [{k: str(tidy(e)) for k, e in g.items()} for g in G]
    In = []
    for ic in I:
        d = {}
        for k, e in ic.items():
            ks = spy.Symbol(k)
            if ks in chart:
                a, b = chart[ks]
                at0 = {spy.Symbol(n): x for n, x in ic.items()}
                le = tidy(spy.log((a + b * ks).xreplace(at0)))
                if any(not at.args[0].is_number for at in le.atoms(spy.log)):
                    return {'why': 'the initial value %s = %s does not split into '
                            'logarithms of positive coordinates' % (k, e)}
                try:
                    d[str(L[ks])] = str(_log_numbers(le, consts))
                except ValueError as err:
                    return {'why': str(err)}
            else:
                d[k] = str(tidy(e))
        In.append(d)
    return {'f': Fn, 'g': Gn, 'ic': In, 'consts': sorted(consts),
            'map': [{'v': str(v), 'L': str(L[v]), 'a': str(chart[v][0]),
                     'b': str(chart[v][1])} for v in sorted(chart, key=str)]}


def log_arg_event(var, value, method, maps):
    """One event in the log chart: v -> (exp(L) - a)/b in its value, and an event
    on a charted state acts on L: a replacement by log(a + b*value), and for
    a = 0 a multiplication as the addition of log(value). Returns {'var',
    'value', 'method', 'consts'} as strings or {'why': ...}."""
    maps = list(maps)
    lines = [str(value)] + [str(m[k]) for m in maps for k in ('v', 'L', 'a', 'b')]
    local, parse = _make_local_parse(lines)
    ent = {str(m['v']): (spy.Symbol(str(m['L']), real=True), parse(str(m['a'])),
                         parse(str(m['b']))) for m in maps}
    sub = {parse(v): (spy.exp(Lv) - a) / b for v, (Lv, a, b) in ent.items()}
    e = parse(str(value)).subs(sub)
    if str(var) not in ent:
        return {'var': str(var), 'value': str(e), 'method': str(method), 'consts': []}
    Lv, a, b = ent[str(var)]
    if method == 'add' and e == 0:
        return {'var': str(Lv), 'value': '0', 'method': 'add', 'consts': []}
    if method == 'replace':
        le = spy.log(a + b * e)
        kind = 'replace'
    elif method == 'multiply' and a == 0:
        le = spy.log(e)
        kind = 'add'
    else:
        return {'why': 'a %s event on %s, which is analysed as %s, is not supported'
                % (method, var, 'log(%s)' % (a + b * parse(str(var))))}
    le = spy.powsimp(spy.expand_log(le, force=True))
    if any(not at.args[0].is_number for at in le.atoms(spy.log)):
        return {'why': 'the event value %s for %s does not split into logarithms of '
                'positive coordinates' % (value, var)}
    consts = set()
    try:
        le = _log_numbers(le, consts)
    except ValueError as err:
        return {'why': str(err)}
    return {'var': str(Lv), 'value': str(le), 'method': kind, 'consts': sorted(consts)}


def log_arg_back_vec(vec, maps):
    """A generator of the log chart back in the user's symbols. L -> log(a + b*v)
    everywhere; for a component eta_v = ((a + b*v)*eta_L - eta(a) - v*eta(b))/b with
    eta(a) = grad(a) . eta."""
    maps = list(maps)
    lines = [str(x) for x in vec.values()] + [str(k) for k in vec] + \
        [str(m[k]) for m in maps for k in ('v', 'L', 'a', 'b')]
    local, parse = _make_local_parse(lines)
    V = {str(k): parse(str(x)) for k, x in vec.items()}
    ent = [(parse(str(m['v'])), str(m['L']), parse(str(m['a'])), parse(str(m['b'])))
           for m in maps]
    Lnames = {Ln for _, Ln, _, _ in ent}
    vp = {v: spy.Symbol(str(v), positive=True) for v, _, _, _ in ent}
    to_l = {parse(Ln): spy.log(a + b * v) for v, Ln, a, b in ent}
    out = {k: e.subs(to_l) for k, e in V.items() if k not in Lnames}
    # charts whose a, b hold another charted symbol come after it
    order = sorted(ent, key=lambda t: len((t[2].free_symbols | t[3].free_symbols) &
                                          set(vp)))
    for v, Ln, a, b in order:
        e_l = V.get(Ln, spy.Integer(0)).subs(to_l)

        def grad(h):
            return sum((spy.diff(h, parse(k)) * e for k, e in out.items()), spy.Integer(0))
        nv = ((a + b * v) * e_l - grad(a) - v * grad(b)) / b
        out[str(v)] = nv
    res = {}
    for k, e in out.items():
        e = spy.powsimp(spy.expand_log(e.xreplace(vp), force=True))
        e = spy.factor_terms(spy.cancel(e)).xreplace({s: v for v, s in vp.items()})
        if e != 0 or k in vec:
            res[k] = str(e)
    return res


def scaling_weights(comps):
    """{name: component} -> {name: weight} when every component is a number times
    its own coordinate, else None."""
    local, parse = _make_local_parse([str(v) for v in comps.values()] +
                                     [str(k) for k in comps])
    out = {}
    for k, v in comps.items():
        w = spy.cancel(parse(str(v)) / parse(str(k)))
        if not w.is_Rational:
            return None
        out[str(k)] = str(w)
    return out


def log_chart_generators(gens, coords):
    """Generators in the chart X = b^theta for every coordinate theta that enters
    them only through b^(c*theta), or whose own component contains 1/log(b): there
    they are rational. `gens` is a list of {name: expression} (None for a support-only
    direction). Returns {'map': [...], 'gens': [...]} or None when there is no such
    coordinate or the chart does not make every component rational."""
    exprs_of = []
    names = [str(c) for c in coords]
    for g in gens:
        if g is None:
            exprs_of.append(None)
            continue
        local, parse = _make_local_parse([str(v) for v in g.values()] + names)
        exprs_of.append({str(k): spy.sympify(parse(str(v))) for k, v in g.items()})
    all_e = [e for g in exprs_of if g for e in g.values()]
    coord_syms = {spy.Symbol(n) for n in names}
    base_of = _detect_log_params(all_e, set())
    base_of = {t: b for t, b in base_of.items() if t in coord_syms}
    # a coordinate whose own component contains 1/log(b) and that enters nowhere else
    for g in exprs_of:
        if not g:
            continue
        for k, v in g.items():
            t = spy.Symbol(k)
            nl = {a.args[0] for a in v.atoms(spy.log) if a.args[0].is_number}
            if t in base_of or len(nl) != 1:
                continue
            if all(t not in e.free_symbols for e in all_e):
                base_of[t] = nl.pop()
    base_of = _log_params_only_in_atoms(base_of, all_e)
    if not base_of:
        return None
    taken = set(names)
    lp = {}
    for t, b in sorted(base_of.items(), key=lambda kv: str(kv[0])):
        nm = 'exp_%s' % t
        while nm in taken:
            nm += '_'
        taken.add(nm)
        lp[t] = (spy.Symbol(nm), b)
    sub = _log_param_sub(lp)
    out = []
    for g in exprs_of:
        if g is None:
            out.append(None)
            continue
        h = {}
        for k, v in g.items():
            t = spy.Symbol(k)
            if t in lp:
                X, b = lp[t]
                h[str(X)] = sub(X * spy.log(b) * v)
            else:
                h[k] = sub(v)
        h = {k: spy.cancel(v) for k, v in h.items()}
        # a translation of theta is X*log(b) d/dX: the constant factor goes
        if any(v.atoms(spy.log) for v in h.values()):
            for b in {b for _, b in lp.values()}:
                h2 = {k: spy.cancel(v / spy.log(b)) for k, v in h.items()}
                if not any(v.atoms(spy.log) for v in h2.values()):
                    h = h2
                    break
        for v in h.values():
            if v.atoms(spy.log, spy.exp) or any(t in v.free_symbols for t in lp):
                return None
        out.append({k: str(v) for k, v in h.items()})
    return {'map': [{'theta': str(t), 'X': str(X), 'base': 'E' if b == spy.E else str(b)}
                    for t, (X, b) in lp.items()],
            'gens': out}


def exp_chart_generators(gens, coords, positive):
    """Generators in the chart L = log(v) for every positive coordinate v that enters
    them through log(v), or that a scaling with symbolic weights moves:
    eta_L = eta_v/v, v = exp(L). `gens` as in log_chart(),
    `positive` True or a list of names. Returns {'map': [{'v', 'L'}], 'gens': [...]}
    or None when no coordinate qualifies or a component stays transcendental."""
    names = [str(c) for c in coords]
    pos = None if positive is True else {str(x) for x in (positive or [])}
    exprs_of = []
    for g in gens:
        if g is None:
            exprs_of.append(None)
            continue
        local, parse = _make_local_parse([str(v) for v in g.values()] + names)
        exprs_of.append({str(k): spy.expand_log(spy.sympify(parse(str(v))), force=True)
                         for k, v in g.items()})
    coord_syms = {spy.Symbol(n) for n in names}
    cand = set()
    for g in exprs_of:
        for e in (g or {}).values():
            for a in e.atoms(spy.log):
                if a.args[0] in coord_syms and (pos is None or str(a.args[0]) in pos):
                    cand.add(a.args[0])
    # a scaling with symbolic weights w translates log(z) at the rate w
    for g in exprs_of:
        if not g:
            continue
        supp = {spy.Symbol(k) for k in g}
        w = {k: spy.cancel(e / spy.Symbol(k)) for k, e in g.items()}
        if any(ww.free_symbols & supp or ww.atoms(spy.log, spy.exp) for ww in w.values()):
            continue
        if all(ww.is_number for ww in w.values()):
            continue
        if all(pos is None or str(z) in pos for z in supp):
            cand |= supp
    if not cand:
        return None
    taken = set(names)
    L = {}
    for v in sorted(cand, key=str):
        nm = 'log_%s' % v
        while nm in taken:
            nm += '_'
        taken.add(nm)
        L[v] = spy.Symbol(nm, real=True)
    sub = {v: spy.exp(L[v]) for v in cand}
    out = []
    for g in exprs_of:
        if g is None:
            out.append(None)
            continue
        h = {}
        for k, e in g.items():
            ks = spy.Symbol(k)
            e2 = spy.expand_log(e.subs(sub), force=True)
            if ks in cand:
                h[str(L[ks])] = spy.cancel(spy.powsimp(e2 * spy.exp(-L[ks])))
            else:
                h[k] = spy.cancel(spy.powsimp(e2))
        for v in h.values():
            if v.atoms(spy.log, spy.exp):
                return None
        out.append({k: str(v) for k, v in h.items()})
    return {'map': [{'v': str(v), 'L': str(L[v])} for v in sorted(cand, key=str)],
            'gens': out}


def exp_chart_back(expr, v_names, l_names, solve_for=None):
    """An expression of the chart L = log(v) back in v; with `solve_for` naming an L
    the expression is its value and exp() of it, the value of v, is returned."""
    def as_list(v):
        return list(v) if isinstance(v, (list, tuple)) else [v]
    v_names, l_names = as_list(v_names), as_list(l_names)
    local, parse = _make_local_parse([str(expr)] + [str(x) for x in v_names + l_names])
    e = parse(str(expr))
    vp = {str(l): spy.Symbol(str(v), positive=True) for v, l in zip(v_names, l_names)}
    e = e.subs({parse(l): spy.log(v) for l, v in vp.items()})
    if solve_for is not None and str(solve_for) in vp:
        e = spy.exp(e)
    e = spy.powsimp(spy.expand_log(e, force=True))
    return str(e.subs({v: spy.Symbol(str(v)) for v in vp.values()}))


def log_chart_back(expr, x_names, thetas, bases, solve_for=None):
    """An expression of the chart X = b^theta back in theta. With `solve_for` naming
    an X, the expression is the value of that X and log_b of it is returned, the
    value of theta."""
    def as_list(v):
        return list(v) if isinstance(v, (list, tuple)) else [v]
    x_names, thetas, bases = as_list(x_names), as_list(thetas), as_list(bases)
    local, parse = _make_local_parse([str(expr)] + [str(x) for x in x_names + thetas])
    e = parse(str(expr))
    subs = {}
    logb = None
    for X, th, b in zip(x_names, thetas, bases):
        bb = spy.E if str(b) == 'E' else spy.sympify(str(b))
        subs[parse(str(X))] = bb ** parse(str(th))
        if solve_for is not None and str(X) == str(solve_for):
            logb = spy.log(bb)
    e = e.subs(subs)
    if logb is not None:
        e = spy.cancel(spy.expand_log(spy.log(e), force=True) / logb)
    return str(e)


def log_param_backsub(expr, comp, x_names, thetas, bases):
    """A reported component in the rational coordinates X = base^theta back in theta:
    X -> base^theta everywhere, and the component of X itself divided by X*log(base)
    when `comp` names an X."""
    def as_list(v):
        return list(v) if isinstance(v, (list, tuple)) else [v]
    x_names, thetas, bases = as_list(x_names), as_list(thetas), as_list(bases)
    local, parse = _make_local_parse([str(expr)] + [str(x) for x in x_names + thetas])
    e = parse(str(expr))
    subs = {}
    for X, th, b in zip(x_names, thetas, bases):
        bb = spy.E if str(b) == 'E' else spy.sympify(str(b))
        Xs = parse(str(X))
        if str(comp) == str(X):
            e = e / (Xs * spy.log(bb))
        subs[Xs] = bb ** parse(str(th))
    return str(spy.powsimp(spy.cancel(e.subs(subs))))


def _detect_power_atoms(per_cond, states=()):
    """Find base^exp terms with a non-numeric exponent. base must be a single
    symbol and exp = c*n (c rational, n a symbol other than a state); returns the
    unique (base, n) pairs, or None if a power is outside this form (then it stays
    non-rational). A numeric base is an exponential, see _apply_exp_recast."""
    pairs = set()
    states = set(states)
    for (f_c, g_c, ic_c, f_ss) in per_cond:
        for e in list(f_c) + list(g_c) + list(ic_c.values()) + list(f_ss):
            for pw in spy.sympify(e).atoms(spy.Pow):
                if pw.exp.is_number or _exp_atom(pw) is not None:
                    continue
                if not pw.base.is_Symbol:
                    return None
                c, rest = pw.exp.as_coeff_Mul()
                if not (rest.is_Symbol and c.is_rational) or rest in states:
                    return None
                pairs.add((pw.base, rest))
    return sorted(pairs, key=lambda t: (str(t[0]), str(t[1])))


def _apply_power_recast(pairs, S, per_cond):
    """Recast each base^(c*n) as E^c with E = base^n a new state, and add a
    companion L = log(base) per base. E' = n*E*base'/base, L' = base'/base when
    base is a state (0 when it is a parameter). E, L are appended to the state
    list; f_ss keeps only the real states (E is held generic in the solve)."""
    recast = {}
    Lof = {}
    for (base, n) in pairs:
        recast[(base, n)] = spy.Symbol('_E_%s_%s' % (base, n))
        if base not in Lof:
            Lof[base] = spy.Symbol('_L_%s' % base)

    def sub(expr):
        def repl(pw):
            if pw.exp.is_number or not pw.base.is_Symbol:
                return pw
            c, rest = pw.exp.as_coeff_Mul()
            E = recast.get((pw.base, rest))
            return E ** c if E is not None else pw
        return spy.sympify(expr).replace(lambda x: x.is_Pow, repl)

    extra = [recast[(b, n)] for (b, n) in pairs] + \
            [Lof[b] for b in sorted(Lof, key=str)]
    new_per_cond = []
    for (f_c, g_c, ic_c, f_ss) in per_cond:
        f_r = [sub(e) for e in f_c]
        g_r = [sub(e) for e in g_c]
        ic_r = {k: sub(v) for k, v in ic_c.items()}
        ss_r = [sub(e) for e in f_ss]
        rhs_of = {str(X): f_r[i] for i, X in enumerate(S)}
        for (b, n) in pairs:
            E = recast[(b, n)]
            brhs = rhs_of.get(str(b), spy.Integer(0))
            ic_r[str(E)] = E
            f_r.append(n * E * brhs / b if brhs != 0 else spy.Integer(0))
        for b in sorted(Lof, key=str):
            L = Lof[b]
            brhs = rhs_of.get(str(b), spy.Integer(0))
            ic_r[str(L)] = L
            f_r.append(brhs / b if brhs != 0 else spy.Integer(0))
        new_per_cond.append((f_r, g_r, ic_r, ss_r))
    meta = [{'E': str(recast[(b, n)]), 'L': str(Lof[b]),
             'base': str(b), 'exp': str(n)} for (b, n) in pairs]
    return list(S) + extra, new_per_cond, meta


def recast_backsub_expr(expr, e_names, l_names, bases, exps):
    """Substitute recast coordinates back, E -> base**exp and L -> log(base), and
    cancel. The name vectors are aligned per recast atom; the shared symbol table
    keeps names like E, I, N as symbols."""
    def as_list(v):
        return list(v) if isinstance(v, (list, tuple)) else [v]
    e_names, l_names, bases, exps = (as_list(e_names), as_list(l_names),
                                     as_list(bases), as_list(exps))
    local, parse = _make_local_parse(
        [str(expr)] + [str(x) for x in e_names + l_names + bases + exps])
    e = parse(str(expr))
    subs = {}
    for E, L, base, exp in zip(e_names, l_names, bases, exps):
        b = parse(str(base))
        subs[parse(str(E))] = b ** parse(str(exp))
        subs[parse(str(L))] = spy.log(b)
    return str(spy.cancel(e.subs(subs)))
