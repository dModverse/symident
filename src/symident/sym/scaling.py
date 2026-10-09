"""Scaling symmetries from the integer kernel of the monomial exponents."""

import math

import sympy as spy

from .. import state
from .charts import _exp_atom, _hyp_to_exp
from .linalg import exact_nullspace
from .parse import _as_list, _clean, _make_local_parse, _read_equations
from .recast import _TRIG, _TRIG_RECIPROCAL


def _poly_monomials(expr, zvars):
    """Numerator and denominator monomial-exponent lists of expr over zvars
    (other symbols are treated as weight-zero coefficients). Exponents may be
    rational, as in sqrt(x). Raises if expr is not a ratio of such sums.
    Read term by term: a dense Poly nests one level per coordinate."""
    idx = {z: i for i, z in enumerate(zvars)}
    zset = set(zvars)
    zero = spy.Integer(0)

    def mons(poly):
        out = {}
        for t in spy.Add.make_args(spy.expand(poly)):
            if t == 0:
                continue
            a = [zero] * len(zvars)
            for k, v in t.as_powers_dict().items():
                if k in idx:
                    a[idx[k]] = spy.Rational(v)
                elif k.free_symbols & zset:
                    raise ValueError('not a monomial in the coordinates: %s' % t)
            out[tuple(a)] = None
        return list(out)
    p, q = spy.fraction(spy.together(spy.sympify(expr)))
    return mons(p), mons(q)


def _exp_split(expr, logs=False):
    """expr with every exponential atom (exp(u), or b^u with a numeric base), every
    trigonometric function of u, and with `logs` every log(u), replaced by a fresh
    symbol, innermost first, and the list of the u. Each is invariant exactly when u
    is: log(lambda^w*u) = log(u) + w*log(lambda)."""
    exps = []

    def rec(e):
        if not e.args:
            return e
        be = _exp_atom(e)
        if be is not None:
            exps.append(rec(be[1]))
            return spy.Dummy('exp')
        if isinstance(e, _TRIG) or type(e) in _TRIG_RECIPROCAL:
            exps.append(rec(e.args[0]))
            return spy.Dummy('trig')
        if logs and isinstance(e, spy.log):
            exps.append(rec(e.args[0]))
            return spy.Dummy('log')
        return e.func(*[rec(a) for a in e.args])
    return rec(_hyp_to_exp(expr)), exps


def _scaling_rows(diff_equations, obs_functions, m, zvars, inter_offset, logs=False,
                  seen=None):
    """Sparse {col: coeff} monomial-exponent rows of one (f, g) system for the scaling
    kernel: weight columns 0..nz-1 over zvars, intermediate columns from inter_offset.
    Returns (rows, ninter, skipped). An exponential has weight zero and its exponent
    is an invariant, like an observable; with `logs` so is the argument of a log(),
    off for the power recast, which keeps log(base) as a coordinate. An (expression,
    target) pair already in `seen` is skipped."""
    nz = len(zvars)
    exprs = []          # (numer monomials, denom monomials, target weight vector)
    skipped = 0
    seen = set() if seen is None else seen

    def fresh(e, t):
        key = (e, t)
        if key in seen:
            return False
        seen.add(key)
        return True

    # pairs seen in an earlier system add no rows and are not split again
    obs_new = [g for g in obs_functions if fresh(g, -1)]
    eq_new = [i for i in range(m) if fresh(diff_equations[i], i)]
    split = [_exp_split(e, logs) for e in obs_new + [diff_equations[i] for i in eq_new]]
    obs_functions = ([s[0] for s in split[:len(obs_new)]] +
                    [u for s in split for u in s[1]])
    eq_split = {i: s[0] for i, s in zip(eq_new, split[len(obs_new):])}

    for g in obs_functions:
        try:
            pmon, qmon = _poly_monomials(g, zvars)
        except Exception:
            skipped += 1
            continue
        exprs.append((pmon, qmon, [0] * nz))
    for i in eq_new:
        try:
            pmon, qmon = _poly_monomials(eq_split[i], zvars)
        except Exception:
            skipped += 1
            continue
        tvec = [0] * nz
        tvec[i] = 1     # state i is column i of zvars; x_i has weight c_i
        exprs.append((pmon, qmon, tvec))

    def diff_row(a, b, t=None):
        # (a - b - t) . c = 0 over the weight columns, times the lcm of the denominators
        d = [x - y for x, y in zip(a, b)]
        if t is not None:
            d = [x - int(y) for x, y in zip(d, t)]
        nzj = [(j, x) for j, x in enumerate(d) if x != 0]
        if not nzj:
            return None
        D = 1
        for _, x in nzj:
            q = int(getattr(x, 'q', 1))
            if q != 1:
                D = D * q // math.gcd(D, q)
        return {j: int(x * D) for j, x in nzj}

    # every numerator monomial has the weight of the first, every denominator monomial
    # that of the first denominator one, and the two differ by the target weight: the
    # rows of the intermediate-column form with the intermediates eliminated
    rows, keys = [], set()
    for pmon, qmon, tvec in exprs:
        cand = [diff_row(a, pmon[0]) for a in pmon[1:]] + \
               [diff_row(b, qmon[0]) for b in qmon[1:]]
        if pmon and qmon:
            cand.append(diff_row(pmon[0], qmon[0], tvec))
        for row in cand:
            if row is None:
                continue
            k = tuple(sorted(row.items()))
            if k not in keys:
                keys.add(k)
                rows.append(row)
    return rows, 0, skipped


def _materialize_rows(rows, ncols):
    """Dense matrix (list of lists) from sparse {col: coeff} rows."""
    out = []
    for r in rows:
        dense = [0] * ncols
        for c, v in r.items():
            dense[c] = v
        out.append(dense)
    return out


def _scaling_gens(rows, ncols, nz):
    """Reduced integer generators (primitive, over the first nz weight columns)
    of the scaling kernel given the determining rows."""
    basis = exact_nullspace(rows, ncols, integer=True) if rows else []
    cvecs = [[v[j] for j in range(nz)] for v in basis
             if any(v[j] != 0 for j in range(nz))]
    gens = []
    if cvecs:
        red, _ = spy.Matrix(cvecs).rref()
        for r in range(red.rows):
            row = [red[r, j] for j in range(nz)]
            if all(x == 0 for x in row):
                continue
            L = 1
            for x in row:
                L = spy.ilcm(L, spy.Rational(x).q)
            ints = [int(spy.Rational(x) * L) for x in row]
            d = 0
            for x in ints:
                d = spy.igcd(d, x)
            if d:
                ints = [x // d for x in ints]
            gens.append(ints)
    return gens


def _weighted_degree(e, w):
    """Weighted degree of `e` with symbol weights `w` (name -> number, default 0), or None
    when `e` is not weighted-homogeneous."""
    if e.is_Number or e.is_NumberSymbol:
        return 0
    if e.is_Symbol:
        return w.get(str(e), 0)
    if e.is_Add:
        ds = [_weighted_degree(a, w) for a in e.args]
        return ds[0] if None not in ds and len(set(ds)) == 1 else None
    if e.is_Mul:
        tot = 0
        for a in e.args:
            d = _weighted_degree(a, w)
            if d is None:
                return None
            tot += d
        return tot
    if e.is_Pow:
        b, x = e.args
        db = _weighted_degree(b, w)
        if x.is_Number:
            return None if db is None else db * x
        return 0 if db == 0 and _weighted_degree(x, w) == 0 else None
    if e.args:
        return 0 if all(_weighted_degree(a, w) == 0 for a in e.args) else None
    return None


def verify_scalings(scalings, tapes):
    """Exact check that each scaling (name -> weight) leaves every tape's model, observables,
    initial values, events and start time invariant: a right-hand side has the weight of
    its state, an observable weight 0, a replaced or added value the weight of its target,
    a factor and a time weight 0. Returns one bool per scaling."""
    lines = []
    for t in tapes:
        lines += _as_list(t.get('model_lines', [])) + _as_list(t.get('obs_lines', []))
        lines += _as_list(t.get('ver_ic', []))
        lines += ['_ = %s' % e['value'] for e in _as_list(t.get('ver_ev', [])) +
                  _as_list(t.get('ver_t0', []))]
        if t.get('ver_time'):
            lines.append('_ = %s' % t['ver_time'])
    local, parse = _make_local_parse(lines)

    def split(line):
        lhs, rhs = _clean(line).split('=', 1)
        return lhs.strip(), spy.sympify(parse(rhs))

    checks = []                       # (expression, target: state name, None for 0)
    for t in tapes:
        for l in _as_list(t.get('model_lines', [])) + _as_list(t.get('ver_ic', [])):
            if '=' in l:
                x, e = split(l)
                checks.append((e, x))
        for l in _as_list(t.get('obs_lines', [])):
            if '=' in l:
                checks.append((split(l)[1], None))
        for ev in _as_list(t.get('ver_ev', [])) + _as_list(t.get('ver_t0', [])):
            e = spy.sympify(parse(str(ev['value'])))
            checks.append((e, None if str(ev['method']) == 'multiply' else str(ev['var'])))
        if t.get('ver_time'):
            checks.append((spy.sympify(parse(t['ver_time'])), None))
    # each distinct check once, with its symbols; one free of the support has degree 0
    checks = list(dict.fromkeys(checks))
    syms = [{str(x) for x in e.free_symbols} for e, _ in checks]
    out = []
    for sc in scalings:
        w = {str(k): spy.Rational(str(v)) for k, v in dict(sc).items()}
        sup = {k for k, v in w.items() if v != 0}
        ok = True
        for (e, target), es in zip(checks, syms):
            if e == 0:                # zero has every weight
                continue
            d = _weighted_degree(e, w) if es & sup else 0
            if d is None or d != (w.get(target, 0) if target else 0):
                ok = False
                if state.env("LIEDIAG"):
                    print('[liediag] scaling fails at %s -> %s: degree %s' %
                          (target, str(e)[:120], d), flush=True)
                break
        out.append(ok)
    return out


def _scaling_nonid(gens, znames, nz):
    """Scaling entries (support, integer vector, type) from generators."""
    non_id = []
    for c in gens:
        comp = {znames[j]: c[j] for j in range(nz) if c[j] != 0}
        non_id.append({'support': sorted(comp.keys()),
                       'vector': {k: str(v) for k, v in comp.items()},
                       'type': 'scaling'})
    return non_id


def _scaling_gens_recast(gens, znames, nz, recast):
    """Impose c_E = exp * c_base on the integer scaling lattice span(gens) and return
    the physical generators, weights possibly symbolic in the exponent. No integer
    weight satisfies c_E = nhill * c_base, so the plain kernel misses Hill scalings;
    intersecting over Q(exp) recovers them exactly (e.g. xi_kinh = -nhill * kinh).
    Holds for inverted and normal recasts alike."""
    if not gens:
        return []
    G = spy.Matrix(gens)                      # rows = basis vectors, cols = coords
    idx = {nm: j for j, nm in enumerate(znames)}
    rel_rows = []
    for rc in recast:
        E, base, exp = str(rc['E']), str(rc['base']), str(rc['exp'])
        if E not in idx or base not in idx:
            continue
        exp_sym = spy.sympify(exp)
        j_e, j_b = idx[E], idx[base]
        # (G[:,E] - exp*G[:,base]) . alpha = 0 over the basis coefficients alpha
        rel_rows.append([G[i, j_e] - exp_sym * G[i, j_b] for i in range(G.rows)])
    if not rel_rows:
        return [[spy.Integer(x) for x in g] for g in gens]
    alphas = spy.Matrix(rel_rows).nullspace()   # over Q(exp)
    out = []
    for a in alphas:
        v = [spy.together(sum(a[i] * G[i, j] for i in range(G.rows)))
             for j in range(nz)]
        # L = log(base) shifts by c_base*log(lam) under the scaling; the kernel above
        # leaves it at 0, the joint nullspace contains the shift
        for rc in recast:
            base, L = str(rc['base']), str(rc['L'])
            if base in idx and L in idx:
                v[idx[L]] = v[idx[base]]
        # clear denominators so the weights are polynomial in the exponents
        dens = [spy.denom(x) for x in v if x != 0]
        Lden = spy.Integer(1)
        for d in dens:
            Lden = spy.lcm(Lden, d)
        v = [spy.expand(x * Lden) for x in v]
        if any(x != 0 for x in v):
            out.append(v)
    return out


def scaling_symmetries(all_variables, diff_equations, obs_functions, m, params,
                      fixed=(), verbose=True):
    """Exact scaling symmetries z_i -> lam^{c_i} z_i from the integer kernel of the
    monomial-exponent conditions (observables invariant, f_i scaling like x_i); no
    expression swell. Symbols in `fixed` get weight zero."""
    zvars = list(all_variables[:m]) + list(params)
    nz = len(zvars)
    znames = [str(s) for s in zvars]
    fixedset = set(str(s) for s in fixed)

    sparse, ninter, skipped = _scaling_rows(diff_equations, obs_functions, m, zvars, nz,
                                            logs=True)
    ncols = nz + ninter
    rows = _materialize_rows(sparse, ncols)
    for j in range(nz):                      # a fixed coordinate does not scale
        if znames[j] in fixedset:
            row = [0] * ncols
            row[j] = 1
            rows.append(row)

    non_id = _scaling_nonid(_scaling_gens(rows, ncols, nz), znames, nz)

    if verbose:
        print('-' * 60)
        print('%d scaling symmetry/ies (exact integer kernel)%s:'
              % (len(non_id), '' if not skipped else
                 ', %d non-polynomial term(s) skipped' % skipped))
        for d in non_id:
            print('  ' + ', '.join('%s^(%s)' % (k, d['vector'][k])
                                   for k in d['support']))

    return {
        'method': 'scaling',
        'count': len(non_id),
        'non_identifiable': non_id,
        'coordinates': znames,
    }


def scaling_symmetries_multi(per_cond_model, per_cond_obs, inputs=None, fixed=None,
                           recast=None, logs=False, extra_model=None, extra_obs=None):
    """Scaling symmetries common to every condition: the integer kernel of all
    conditions' monomial-exponent rows stacked over a shared weight space (own
    intermediate columns each), i.e. the intersection of the per-condition lattices.
    Coordinates are states plus parameters; `inputs` and `fixed` do not scale.
    `extra_model` ("X = e", e of the weight of state X) and `extra_obs` ("_ = e", e of
    weight 0) add constraints per condition, such as initial values and events."""
    per_cond_model = [_as_list(m) for m in per_cond_model]
    per_cond_obs = [_as_list(o) for o in per_cond_obs]
    extra_model = [_as_list(m) for m in (extra_model or [[] for _ in per_cond_model])]
    extra_obs = [_as_list(o) for o in (extra_obs or [[] for _ in per_cond_model])]
    inputset = set(_as_list(inputs))
    fixedset = set(_as_list(fixed))
    K = len(per_cond_model)
    if K == 0:
        return {'method': 'scaling', 'count': 0, 'non_identifiable': [],
                'coordinates': []}

    all_lines = [l for lines in per_cond_model + per_cond_obs + extra_model + extra_obs
                 for l in lines]
    local, parse = _make_local_parse(all_lines)

    state_names = [_clean(l).split('=', 1)[0].strip() for l in per_cond_model[0]]
    state_syms = [local.get(nm, spy.Symbol(nm)) for nm in state_names]

    per_f, per_g, paramset = [], [], set()
    for c in range(K):
        rhs = {}
        for l in per_cond_model[c]:
            lhs, expr = _clean(l).split('=', 1)
            rhs[lhs.strip()] = spy.sympify(parse(expr))
        f = [rhs[nm] for nm in state_names]
        g = [spy.sympify(parse(_clean(l).split('=', 1)[1])) for l in per_cond_obs[c]]
        per_f.append(f)
        per_g.append(g)
        for e in f + g:
            paramset |= set(spy.sympify(e).free_symbols)
    per_xf, per_xg = [], []
    for c in range(K):
        xf = []
        for l in extra_model[c]:
            lhs, expr = _clean(l).split('=', 1)
            if lhs.strip() in state_names:
                e = spy.sympify(parse(expr))
                xf.append((state_names.index(lhs.strip()), e))
                paramset |= set(e.free_symbols)
        xg = [spy.sympify(parse(_clean(l).split('=', 1)[1])) for l in extra_obs[c]]
        for e in xg:
            paramset |= set(e.free_symbols)
        per_xf.append(xf)
        per_xg.append(xg)

    paramset -= set(state_syms)
    paramset -= {local.get(nm, spy.Symbol(nm)) for nm in inputset}
    params = sorted(paramset, key=spy.default_sort_key)
    # re-add free-exponent bases the recast eliminated (Km in C^n/(Km^n+C^n)), so
    # _scaling_gens_recast can impose c_E = exp*c_base on them
    for rc in _as_list(recast):
        b = local.get(str(rc['base']), spy.Symbol(str(rc['base'])))
        if (b not in set(state_syms) and b not in params
                and str(b) not in inputset and str(b) not in fixedset):
            params.append(b)
    params = sorted(params, key=spy.default_sort_key)
    zvars = state_syms + params
    nz = len(zvars)
    znames = [str(s) for s in zvars]
    m = len(state_syms)

    rows, inter_offset, skipped, seen = [], nz, 0, set()
    for c in range(K):
        sparse, ninter, sk = _scaling_rows(per_f[c], per_g[c], m, zvars, inter_offset,
                                           logs, seen)
        rows.extend(sparse)
        inter_offset += ninter
        skipped += sk
        # the extra values in rounds with at most one per state
        pending = list(per_xf[c])
        while pending:
            fx, rest, used = [spy.Integer(0)] * m, [], set()
            for i, e in pending:
                if i in used:
                    rest.append((i, e))
                else:
                    fx[i] = e
                    used.add(i)
            pending = rest
            sparse, ninter, sk = _scaling_rows(fx, [], m, zvars, inter_offset, logs, seen)
            rows.extend(sparse)
            inter_offset += ninter
            skipped += sk
        if per_xg[c]:
            sparse, ninter, sk = _scaling_rows([spy.Integer(0)] * m, per_xg[c], m, zvars,
                                               inter_offset, logs, seen)
            rows.extend(sparse)
            inter_offset += ninter
            skipped += sk
    if skipped and state.env("LIEDIAG"):
        print('[liediag] scaling_symmetries_multi: %d non-polynomial term(s) skipped '
              '(a scaling they would forbid may be over-reported)' % skipped)
    ncols = inter_offset
    dense = _materialize_rows(rows, ncols)
    for j in range(nz):
        if znames[j] in fixedset:
            row = [0] * ncols
            row[j] = 1
            dense.append(row)

    gens = _scaling_gens(dense, ncols, nz)
    # the integer kernel treats each recast E as free; recover the Hill scalings
    recast = _as_list(recast)
    if recast:
        physical = _scaling_gens_recast(gens, znames, nz, recast)
        non_id = _scaling_nonid(physical, znames, nz)
    else:
        non_id = _scaling_nonid(gens, znames, nz)
    return {'method': 'scaling', 'count': len(non_id), 'non_identifiable': non_id,
            'coordinates': znames}


def scaling_symmetries_single(model, observation, inputs=None, fixed=None, parameters=None):
    """Scaling symmetries of one condition from the integer kernel of its exponent rows."""
    model = _as_list(model)
    observation = _as_list(observation)
    input_names = _as_list(inputs)
    fixed_names = _as_list(fixed)
    extra_params = _as_list(parameters)
    local, parse = _make_local_parse(model + observation + input_names + fixed_names + extra_params)
    variables, diff_equations, params = _read_equations(model, parse)
    _, obs_functions, params = _read_observation(observation, variables, params, parse)
    for pn in extra_params:
        s = local.get(pn, spy.Symbol(pn))
        if s not in params and s not in variables:
            params.append(s)
    input_syms = [local.get(nm, spy.Symbol(nm)) for nm in input_names]
    fixed_syms = [local.get(nm, spy.Symbol(nm)) for nm in fixed_names]
    for s in input_syms:
        if s in params:
            params.remove(s)
    # a declared input that is also a state is a constant: its equation is dropped
    keep = [i for i, v in enumerate(variables) if v not in input_syms]
    variables = [variables[i] for i in keep]
    diff_equations = [diff_equations[i] for i in keep]
    all_variables = list(variables) + list(input_syms) + list(params)
    return scaling_symmetries(all_variables, diff_equations, obs_functions, len(variables), params,
                             fixed=fixed_syms)


def _read_observation(observation, variables, parameters, parse):
    obs_vars, obs_functions, obs_parameters = _read_equations(observation, parse)
    for var in variables:
        if var in obs_parameters:
            obs_parameters.remove(var)
    for par in parameters:
        if par in obs_parameters:
            obs_parameters.remove(par)
    return obs_vars, obs_functions, parameters + obs_parameters
