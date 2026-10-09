"""Resting states over GF(p): compiled models, the backward solve and its derivatives."""

import sympy as spy

from .gfpoly import _compile_linear_plan, _dp_subst, _eval_bipoly_dict, _solve_states_reduced
from .modp import _bipoly, _eval_terms, _eval_terms_guarded, _poly_terms, _solve_mod
from .msolve import _solve_states_msolve
from .parse import _as_list, _build_symbol_table, _clean, _function_aliases, _make_local_parse, _make_parse

_ss_modular_cache = {}


# compiled t0 event value/derivative term lists keyed by the event tuple and the
# parameter order
_event_compile_cache = {}


# test seam: force the per-point solve
_SS_FORCE_POINT = False


# force the numeric per-point steady-state seed; a test seam to cross-check it
# against the compiled steady-state IC tape
_FORCE_CONSTRAINT_SEED = False


def _solve_states_fast(dps, solve_states, p):
    """Interior point of f = 0 over GF(p) from the per-point state polynomials in dict
    form. Linear elimination by dict-polynomial substitution mod p, then numeric
    back-substitution; a coupled residual goes to _solve_states_reduced (core of at most
    two states) or _solve_states_msolve. Returns (sol {Symbol: Integer}, None) or
    (None, fail-dict)."""
    nv = len(solve_states)
    names = [str(x) for x in solve_states]
    dps0 = dps
    dps = [dict(d) for d in dps]
    rem_idx = set(range(nv))
    elim = []                          # (var_index, expr dict poly over remaining vars)
    progress = True
    while progress and rem_idx:
        progress = False
        for eqi in range(len(dps)):
            d = dps[eqi]
            if d is None or not d:
                continue
            present = sorted((i for i in rem_idx if any(m[i] for m in d)),
                             key=lambda i: names[i])   # match the symbolic order
            if not present:
                continue
            picked = None
            for i in present:
                if max(m[i] for m in d) != 1:          # degree in var i must be 1
                    continue
                c1 = 0
                ok = True
                for m in d:                            # coeff of var_i^1 must be const
                    if m[i] == 1:
                        if any(m[k] for k in range(nv) if k != i):
                            ok = False
                            break
                        c1 = d[m]
                if not ok or c1 % p == 0:
                    continue
                inv = pow(c1 % p, p - 2, p)
                c0 = {m: c for m, c in d.items() if m[i] == 0}   # var_i^0 part
                expr = {m: (c * (-inv)) % p for m, c in c0.items()}
                expr = {m: c for m, c in expr.items() if c}
                picked = (i, expr)
                break
            if picked is None:
                continue
            i, expr = picked
            elim.append((i, expr))
            dps[eqi] = None
            for j in range(len(dps)):
                if dps[j] and any(m[i] for m in dps[j]):
                    dps[j] = _dp_subst(dps[j], i, expr, nv, p)
            rem_idx.discard(i)
            progress = True

    sol = {}
    coupled_idx = [i for i in range(nv) if i in rem_idx]
    if coupled_idx:
        red = _solve_states_reduced(dps0, solve_states, p)
        if red is not None:
            return red
        return _solve_states_msolve(dps0, solve_states, p)

    val_by_id = {i: int(sol[solve_states[i]]) % p for i in coupled_idx}
    for i, expr in reversed(elim):     # resolve eliminated vars numerically, latest first
        acc = 0
        for m, c in expr.items():
            t = c
            for k in range(nv):
                if m[k]:
                    t = t * pow(val_by_id.get(k, 0), m[k], p) % p
            acc = (acc + t) % p
        val_by_id[i] = acc
        sol[solve_states[i]] = spy.Integer(acc)
    return sol, None


def _compile_t0events(events, param_names):
    """Compile each t0 event's value and its parameter derivatives to
    prime-independent term lists over the parameters, so a point evaluates by
    integer arithmetic. Keyed by the event tuple and parameter order."""
    key = (tuple((str(e['var']), str(e['value']), str(e['method'])) for e in events),
           tuple(param_names))
    compiled = _event_compile_cache.get(key)
    if compiled is not None:
        return compiled
    param_syms = [spy.Symbol(nm) for nm in param_names]
    eloc = _build_symbol_table([str(e['value']) for e in events] + list(param_names))
    eloc.update(_function_aliases())
    eparse = _make_parse(eloc)
    compiled = []
    for evt in events:
        val = spy.sympify(eparse(str(evt['value'])))
        duals = {}
        for nm in param_names:
            d = spy.diff(val, spy.Symbol(nm))
            duals[nm] = None if d == 0 else _poly_terms(d, param_syms)
        compiled.append({'var': str(evt['var']), 'method': str(evt['method']),
                         'val_t': _poly_terms(val, param_syms), 'duals': duals})
    _event_compile_cache[key] = compiled
    return compiled


def _ss_compile(model, state_names, param_names, forcings, held_state_names=()):
    """Prime-independent compile of f = 0 for the modular steady-state solve,
    memoised in _ss_modular_cache. Inputs are the already-normalised model/state/param
    lists and the forcings set. Returns the cached tuple (param_syms, solve_states,
    polys, Jx, Jt, gens, jx_terms, jt_terms, poly_bi, generic_linear, lin_terms, point_plan).

    `held_state_names` are conserved-moiety pivot states with a free resting value
    (without reduce_cq). f = 0 is rank-deficient by one equation per moiety, so these
    states act as parameters of the point solve and their dependent equations are
    dropped, leaving a square system (`point_plan`, None when nothing is held).
    `solve_states` and Jx/Jt still cover all non-forcing states, so the joint
    determining system leaves the pivot directions free."""
    key = (tuple(model), tuple(state_names), tuple(param_names), tuple(sorted(forcings)),
           tuple(sorted(held_state_names)))
    cached = _ss_modular_cache.get(key)
    if cached is not None:
        return cached
    local, parse = _make_local_parse(model + list(param_names))
    rhs_by_name = {}
    for raw in model:
        line = _clean(raw)
        if '=' not in line:
            continue
        lhs, rhs = line.split('=', 1)
        rhs_by_name[lhs.strip()] = parse(rhs)
    state_syms = [spy.Symbol(nm) for nm in state_names]
    param_syms = [spy.Symbol(nm) for nm in param_names]
    solve_states = [s for s in state_syms if str(s) not in forcings]
    forcing_subs = {spy.Symbol(nm): spy.Integer(0) for nm in forcings}
    f_sym = [spy.sympify(rhs_by_name[str(s)]).subs(forcing_subs) for s in solve_states]
    polys = [spy.expand(spy.fraction(spy.together(fi))[0]) for fi in f_sym]
    # sparse Jacobians: each equation is differentiated in its own symbols only
    n_sc = len(solve_states)
    gens = list(param_syms) + list(solve_states)
    fs_sym = [fi.free_symbols for fi in f_sym]
    Jx = spy.zeros(n_sc, n_sc)
    Jt = {str(th): spy.zeros(n_sc, 1) for th in param_syms}
    for i, fi in enumerate(f_sym):
        for j, sj in enumerate(solve_states):
            if sj in fs_sym[i]:
                Jx[i, j] = spy.diff(fi, sj)
        for th in param_syms:
            if th in fs_sym[i]:
                Jt[str(th)][i] = spy.diff(fi, th)
    jx_terms = [[_poly_terms(Jx[i, j], gens) for j in range(n_sc)]
                for i in range(n_sc)]
    jt_terms = {str(th): [_poly_terms(Jt[str(th)][i], gens) for i in range(n_sc)]
                for th in param_syms}
    poly_bi = [_bipoly(pl, list(solve_states), list(param_syms)) for pl in polys]
    generic_linear, lin_terms = _compile_linear_plan(polys, solve_states, param_syms)
    # held states are a moiety pivot set (detection.cq), so their own
    # equations are exactly the dependent rows of f = 0
    held_set = set(held_state_names)
    point_plan = None
    if held_set:
        held_syms = [s for s in solve_states if str(s) in held_set]
        point_states = [s for s in solve_states if str(s) not in held_set]
        point_params = list(param_syms) + held_syms
        keep_polys = [polys[i] for i, s in enumerate(solve_states) if str(s) not in held_set]
        point_poly_bi = [_bipoly(pl, list(point_states), list(point_params)) for pl in keep_polys]
        point_generic_linear, point_lin_terms = _compile_linear_plan(
            keep_polys, point_states, point_params)
        point_plan = (point_states, point_params, held_syms, point_poly_bi,
                      point_generic_linear, point_lin_terms)
    cached = (param_syms, solve_states, polys, Jx, Jt, gens, jx_terms, jt_terms,
              poly_bi, generic_linear, lin_terms, point_plan)
    _ss_modular_cache[key] = cached
    return cached


_NAMES = {}


def _names(cached):
    """{Symbol: name} for the symbols of a compiled steady-state system, kept with it."""
    hit = _NAMES.get(id(cached))
    if hit is None or hit[0] is not cached:
        syms = set()
        for part in list(cached) + list(cached[-1] or ()):
            if isinstance(part, (list, tuple)):
                syms |= {x for x in part if isinstance(x, spy.Symbol)}
        hit = (cached, {x: str(x) for x in syms})
        _NAMES[id(cached)] = hit
    return hit[1]


def solve_steady_state_modular(model, state_names, param_names, param_vals, prime,
                            forcings=None, backend='sympy', t0events=None,
                            recast=None, l_vals=None, joint_mode=False,
                            held_states=None):
    """Numeric point on the interior component of f = 0 over GF(prime) with its
    implicit-function-theorem parameter sensitivities.

    `model` is a list of "X = rhs" lines; `state_names`/`param_names` the state and
    parameter order; `param_vals` a {name: int} map of residues mod `prime`;
    `forcings` are held at 0; `t0events` are dose-style events composed onto the
    point after the solve. Returns {'ok': True, 'xstar': [...], 'dx': {param:
    [...]}} (state-ordered) or {'ok': False, 'why': ...}."""
    model = _as_list(model)
    forcings = set(_as_list(forcings))
    state_names = _as_list(state_names)
    param_names = _as_list(param_names)
    p = int(prime)

    held_states = {str(k): int(v) % p for k, v in (held_states or {}).items()}
    cached = _ss_compile(model, state_names, param_names, forcings,
                         tuple(sorted(held_states.keys())))
    (param_syms, solve_states, polys, Jx, Jt, gens, jx_terms, jt_terms, poly_bi,
     generic_linear, lin_terms, point_plan) = cached
    snm = _names(cached)

    paramvals = [int(param_vals.get(snm[th], 0)) % p for th in param_syms]

    if point_plan is not None:
        # held pivot states act as parameters; solve the square reduced system
        (point_states, point_params, held_syms, point_poly_bi,
         point_generic_linear, point_lin_terms) = point_plan
        ppvals = [(held_states[snm[th]] if snm[th] in held_states
                   else int(param_vals.get(snm[th], 0)) % p) for th in point_params]
        sub = None
        if point_generic_linear and not _SS_FORCE_POINT:
            sub = {}
            for nm, terms in point_lin_terms:
                val = _eval_terms_guarded(terms, ppvals, p)
                if val is None:
                    sub = None
                    break
                sub[spy.Symbol(nm)] = spy.Integer(val)
        if sub is None:
            dps = [_eval_bipoly_dict(bip, ppvals, p) for bip in point_poly_bi]
            sub, fail = _solve_states_fast(dps, list(point_states), p)
            if fail is not None:
                return fail
        sol = {}
        for s in solve_states:
            sol[s] = (spy.Integer(held_states[snm[s]]) if snm[s] in held_states
                      else spy.Integer(int(sub[s]) % p))
    else:
        # fast path: each state is a compiled rational function of the parameters,
        # evaluated in integer arithmetic; a vanishing pivot denominator mod p (None)
        # routes the point to the symbolic solve
        sol = None
        if generic_linear and not _SS_FORCE_POINT:
            sol = {}
            for nm, terms in lin_terms:
                val = _eval_terms_guarded(terms, paramvals, p)
                if val is None:
                    sol = None
                    break
                sol[spy.Symbol(nm)] = spy.Integer(val)
        if sol is None:
            dps = [_eval_bipoly_dict(bip, paramvals, p) for bip in poly_bi]
            sol, fail = _solve_states_fast(dps, list(solve_states), p)
            if fail is not None:
                return fail

    n_s = len(solve_states)
    val_by = {snm[s]: int(sol[s]) % p for s in solve_states}
    s_idx = {snm[s]: i for i, s in enumerate(solve_states)}
    # point in gens order (params then solve-states), already reduced mod p
    ptvals = [int(param_vals.get(snm[th], 0)) % p for th in param_syms] + \
             [val_by[snm[s]] for s in solve_states]
    jt_by = {snm[th]: [_eval_terms(jt_terms[snm[th]][i], ptvals, p) for i in range(n_s)]
             for th in param_syms}
    Jxeff = [[_eval_terms(jx_terms[i][j], ptvals, p) for j in range(n_s)]
             for i in range(n_s)]

    # resting Jacobian df_rest for the joint determining system (tangency
    # df_rest . xi = 0 in uneliminated (x, theta) coordinates), snapshot before the
    # recast folding below mutates Jxeff/jt_by; these rows are stacked by the caller
    if point_plan is not None:
        # reduced tangency: non-pivot rows and state columns; pivot state columns
        # become parameter columns, as the pivot's resting value is free
        held_idx = [i for i, s in enumerate(solve_states) if str(s) in held_states]
        keep_idx = [i for i, s in enumerate(solve_states) if str(s) not in held_states]
        df_jx = [[Jxeff[i][j] for j in keep_idx] for i in keep_idx]
        df_state_cols = [str(solve_states[j]) for j in keep_idx]
        df_jt = {str(th): [jt_by[str(th)][i] for i in keep_idx] for th in param_syms}
        for hj in held_idx:
            df_jt[str(solve_states[hj])] = [Jxeff[i][hj] for i in keep_idx]
        df_param_cols = [str(th) for th in param_syms] + \
                      [str(solve_states[hj]) for hj in held_idx]
    else:
        df_jx = [list(row) for row in Jxeff]
        df_jt = {nm: list(col) for nm, col in jt_by.items()}
        df_state_cols = [str(sst) for sst in solve_states]
        df_param_cols = [str(th) for th in param_syms]

    if joint_mode:
        # joint mode needs only val_by and df_rest; returning here also skips the
        # (occasionally singular) dual solve, which would reject a good point
        return {'ok': True, 'state_names': list(state_names), 'val_by': dict(val_by),
                'df_jx': df_jx, 'df_jt': df_jt, 'df_state_cols': df_state_cols,
                'df_param_cols': df_param_cols}

    # power/Hill recast: a normal entry holds E = base^exp generic, chain rule into base
    # and exponent; an inverted entry solves for E and takes the duals of base and
    # L = log(base) from d_e by the inverse chain rule. base0, E0, L0 are resting values.
    recast_out = []
    if recast:
        l_vals = l_vals or {}
        for rc in recast:
            E, L, base, exp = rc['E'], rc['L'], rc['base'], rc['exp']
            inverted = bool(rc.get('inverted'))
            E0 = val_by[E] if E in val_by else int(param_vals.get(E, 0)) % p
            L0 = int(l_vals.get(L, 0)) % p
            base0 = val_by[base] if base in val_by else int(param_vals.get(base, 0)) % p
            expv = int(param_vals.get(exp, 0)) % p
            if not inverted:
                binv = pow(base0, p - 2, p) if base0 % p else 0
                d_ebase = expv * E0 % p * binv % p
                d_eexp = E0 * L0 % p
                j_e = jt_by.get(E, [0] * n_s)
                if base in s_idx:
                    bc = s_idx[base]
                    for i in range(n_s):
                        Jxeff[i][bc] = (Jxeff[i][bc] + j_e[i] * d_ebase) % p
                elif base in jt_by:
                    jt_by[base] = [(jt_by[base][i] + j_e[i] * d_ebase) % p for i in range(n_s)]
                if exp in jt_by:
                    jt_by[exp] = [(jt_by[exp][i] + j_e[i] * d_eexp) % p for i in range(n_s)]
            recast_out.append({'E': E, 'L': L, 'base': base, 'exp': exp,
                               'inverted': inverted, 'E0': E0, 'L0': L0,
                               'base0': base0, 'expv': expv})

    # solve Jxeff * dx = -Jt for every parameter column at once over GF(p)
    params_l = [str(th) for th in param_syms]
    B = [[(-jt_by[params_l[j]][i]) % p for j in range(len(params_l))]
         for i in range(n_s)]
    X = _solve_mod(Jxeff, B, p)
    if X is None:
        return {'ok': False, 'why': 'singular jacobian mod p'}
    dx_by = {params_l[j]: [X[i][j] % p for i in range(n_s)]
             for j in range(len(params_l))}

    # assemble in the full state order; held states stay at 0 with no duals
    xstar = [val_by.get(nm, 0) for nm in state_names]
    dx = {}
    for nm in param_names:
        col = dx_by.get(nm, [0] * n_s)
        dx[nm] = [col[s_idx[s]] if s in s_idx else 0 for s in state_names]

    # parameter-duals for the recast ic_seed rows. Each entry seeds its generic
    # partner (gen) and L: a normal entry derives them from dx[base], an inverted
    # one from dx[E]. gen is E for a normal entry and base for an inverted one.
    for rc in recast_out:
        E, L, base, exp = rc['E'], rc['L'], rc['base'], rc['exp']
        E0, L0, base0, expv = rc['E0'], rc['L0'], rc['base0'], rc['expv']
        g_d = {}
        l_d = {}
        if not rc['inverted']:
            bi = s_idx.get(base)
            binv = pow(base0, p - 2, p) if base0 % p else 0
            d_ebase = expv * E0 % p * binv % p
            d_eexp = E0 * L0 % p
            for nm in param_names:
                dxb = dx_by[nm][bi] if (bi is not None and nm in dx_by) else 0
                g_d[nm] = (d_ebase * dxb + (d_eexp if nm == exp else 0)) % p
                l_d[nm] = binv * dxb % p
            rc['gen'], rc['gen0'] = E, E0
        else:
            ei = s_idx.get(E)
            xe = pow(expv, p - 2, p) if expv % p else 0
            einv = pow(E0, p - 2, p) if E0 % p else 0
            for nm in param_names:
                d_e = dx_by[nm][ei] if (ei is not None and nm in dx_by) else 0
                g_d[nm] = (base0 * xe % p * einv % p * d_e
                           - (base0 * L0 % p * xe if nm == exp else 0)) % p
                l_d[nm] = (xe * einv % p * d_e - (L0 * xe if nm == exp else 0)) % p
            rc['gen'], rc['gen0'] = base, base0
        rc['gen_dual'] = g_d
        rc['l_dual'] = l_d
        for k in ('E', 'base', 'exp', 'E0', 'base0', 'expv'):
            rc.pop(k, None)

    # compose the t0 events onto the seed (value mod p and its parameter-duals)
    events = list(t0events) if t0events else []
    if events:
        idx_of_state = {nm: i for i, nm in enumerate(state_names)}
        for evt in _compile_t0events(events, list(param_names)):
            X = evt['var']
            if X not in idx_of_state:
                continue
            i = idx_of_state[X]
            v0 = _eval_terms(evt['val_t'], paramvals, p)
            vdu = {nm: (_eval_terms(evt['duals'][nm], paramvals, p)
                        if evt['duals'][nm] is not None else 0)
                   for nm in param_names}
            meth = evt['method']
            if meth == 'replace':
                xstar[i] = v0
                for nm in param_names:
                    dx[nm][i] = vdu[nm]
            elif meth == 'add':
                xstar[i] = (xstar[i] + v0) % p
                for nm in param_names:
                    dx[nm][i] = (dx[nm][i] + vdu[nm]) % p
            elif meth == 'multiply':
                old = xstar[i]
                oldd = {nm: dx[nm][i] for nm in param_names}
                xstar[i] = old * v0 % p
                for nm in param_names:
                    dx[nm][i] = (old * vdu[nm] + v0 * oldd[nm]) % p
    return {'ok': True, 'xstar': xstar, 'dx': dx, 'state_names': list(state_names),
            'recast': recast_out,
            'df_jx': df_jx, 'df_jt': df_jt, 'df_state_cols': df_state_cols,
            'df_param_cols': df_param_cols, 'val_by': dict(val_by)}
