"""Scaling symmetries from the integer kernel of the monomial exponents."""

import functools
import math
from fractions import Fraction

import sympy as spy

from .. import state
from ..errors import SYMPY_FAILS, InapplicableError, sympy_fail
from .charts import _exp_atom, _hyp_to_exp
from .linalg import exact_nullspace
from .parse import _as_list, _clean, _make_local_parse, _read_equations
from .recast import _TRIG, _TRIG_RECIPROCAL


def _ring_elem(x, R, gen):
    """x as an element of the polynomial ring R, None when it is not a polynomial in its symbols."""
    if x.is_Symbol:
        return gen[x]
    if x.is_Rational:
        return R(x)
    if x.is_Add or x.is_Mul:
        out = R(0) if x.is_Add else R(1)
        for a in x.args:
            e = _ring_elem(a, R, gen)
            if e is None:
                return None
            out = out + e if x.is_Add else out * e
        return out
    if x.is_Pow and x.exp.is_Integer and x.exp >= 0:
        b = _ring_elem(x.base, R, gen)
        return None if b is None else b ** int(x.exp)
    return None


def _terms(x):
    """The terms of the expanded polynomial x, each as its powers, by sparse arithmetic where x is a polynomial."""
    syms = sorted(x.free_symbols, key=spy.default_sort_key)
    if not syms:
        return ((),) if x != 0 else ()
    R, *gens = spy.ring(syms, spy.QQ)
    P = _ring_elem(x, R, dict(zip(syms, gens)))
    if P is not None:
        return tuple(tuple((g, spy.Integer(e)) for g, e in zip(syms, m) if e) for m in P.monoms())
    return tuple(tuple(t.as_powers_dict().items()) for t in spy.Add.make_args(spy.expand(x)) if t != 0)


@functools.lru_cache(maxsize=4096)
def _expanded(expr):
    """The terms of the expanded numerator and denominator of expr, each as its powers."""
    return tuple(_terms(x) for x in spy.fraction(spy.together(spy.sympify(expr))))


def _exponent(v):
    r = spy.Rational(v)
    return int(r.p) if r.q == 1 else Fraction(int(r.p), int(r.q))


def _poly_monomials(expr, zvars):
    """Numerator and denominator monomials of expr over zvars as sparse {column: exponent}; raises for other forms."""
    idx = {z: i for i, z in enumerate(zvars)}
    zset = set(zvars)

    def mons(terms):
        out = {}
        for powers in terms:
            a = {}
            for k, v in powers:
                if k in idx:
                    e = _exponent(v)
                    if e:
                        a[idx[k]] = e
                elif k.free_symbols & zset:
                    raise InapplicableError('not a monomial in the coordinates: %s' % spy.Mul(
                        *[b ** e for b, e in powers]))
            out.setdefault(tuple(sorted(a.items())), a)
        return list(out.values())
    p, q = _expanded(expr)
    return mons(p), mons(q)


def _diff_row(a, b, t=None):
    """(a - b - t) . c = 0 over the weight columns, times the lcm of the denominators; None when zero."""
    d = dict(a)
    for j, x in b.items():
        d[j] = d.get(j, 0) - x
    if t is not None:
        d[t] = d.get(t, 0) - 1
    nzj = sorted((j, x) for j, x in d.items() if x != 0)
    if not nzj:
        return None
    D = 1
    for _, x in nzj:
        q = getattr(x, "denominator", 1)
        D = D * q // math.gcd(D, q)
    return {j: int(x * D) for j, x in nzj}


@functools.lru_cache(maxsize=4096)
def _expr_rows(expr, target, zvars):
    """The scaling rows equating the monomial weights of expr per side, the sides apart by the weight of `target`."""
    pmon, qmon = _poly_monomials(expr, zvars)
    cand = [_diff_row(a, pmon[0]) for a in pmon[1:]] + [_diff_row(b, qmon[0]) for b in qmon[1:]]
    if pmon and qmon:
        cand.append(_diff_row(pmon[0], qmon[0], target if target >= 0 else None))
    return tuple(tuple(r.items()) for r in cand if r is not None)


def _exp_split(expr, logs=False):
    """expr with exponentials, trig and, with `logs`, log atoms as fresh symbols, and their arguments."""
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
    """(rows, ninter, skipped): sparse monomial-exponent rows of one (f, g) system for the scaling kernel."""
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

    zkey = tuple(zvars)
    rows, keys = [], set()
    # state i is column i of zvars; x_i has weight c_i
    for e, target in [(g, -1) for g in obs_functions] + [(eq_split[i], i) for i in eq_new]:
        try:
            cand = _expr_rows(e, target, zkey)
        except InapplicableError:
            # not a monomial form: counted as skipped, not as an error
            skipped += 1
            continue
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            skipped += 1
            continue
        for k in cand:
            if k not in keys:
                keys.add(k)
                rows.append(dict(k))
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
    """Primitive integer generators of the scaling kernel over the first nz weight columns."""
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
    """Weighted degree of `e` under the weights `w`, or None when `e` is not weighted-homogeneous."""
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
    """One bool per scaling: whether it leaves every tape's model, observables, values and events invariant."""
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
    """Generators of span(gens) with c_E = exp * c_base imposed, weights possibly symbolic in exp."""
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
    """Exact scaling symmetries z_i -> lam^{c_i} z_i from the integer kernel; `fixed` gets weight zero."""
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
        'skipped': skipped,
    }


def scaling_symmetries_multi(per_cond_model, per_cond_obs, inputs=None, fixed=None,
                           recast=None, logs=False, extra_model=None, extra_obs=None):
    """Scaling symmetries common to every condition; `inputs` and `fixed` do not scale."""
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
    # re-add free-exponent bases the recast eliminated, so _scaling_gens_recast can
    # impose c_E = exp*c_base on them
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
    # the integer kernel treats each recast E as free; recover the symbolic-weight scalings
    recast = _as_list(recast)
    if recast:
        physical = _scaling_gens_recast(gens, znames, nz, recast)
        non_id = _scaling_nonid(physical, znames, nz)
    else:
        non_id = _scaling_nonid(gens, znames, nz)
    return {'method': 'scaling', 'count': len(non_id), 'non_identifiable': non_id,
            'coordinates': znames, 'skipped': skipped}


def scaling_symmetries_single(model, observation, inputs=None, fixed=None, parameters=None,
                              verbose=False):
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
                             fixed=fixed_syms, verbose=verbose)


def _read_observation(observation, variables, parameters, parse):
    obs_vars, obs_functions, obs_parameters = _read_equations(observation, parse)
    for var in variables:
        if var in obs_parameters:
            obs_parameters.remove(var)
    for par in parameters:
        if par in obs_parameters:
            obs_parameters.remove(par)
    return obs_vars, obs_functions, parameters + obs_parameters
