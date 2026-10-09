"""Observability rank with sympy for small models."""

import sympy as spy

from .parse import _make_local_parse, _read_equations
from .tape import _lie_reach


def _lie_deriv(h, states, rhs):
    """Lie derivative L_f h = sum_i (dh/dx_i) f_i; parameters are constant."""
    return sum(spy.diff(h, states[i]) * rhs[i] for i in range(len(states)))


def observability_sympy_multi(model, observation, condition_subs=None, condition_ic0=None,
                            fixed=None, parameters=None, inputs=None, backend='sympy',
                            condition_obs=None, condition_rest_subs=None):
    """Multi-condition pure-symbolic observability, the exact cross-check of the
    modular multi-condition engine (same coordinate and substitution semantics as
    compile_observability_tape_multi()).

    Condition k substitutes `condition_subs[k]` into f and g, starts the jet at
    `condition_ic0[k]` and may replace `observation` by `condition_obs[k]`. A direction
    is non-identifiable iff it lies in the nullspace of the observability matrices
    stacked over one shared coordinate space. Single segment only and no equilibrate;
    give a steady state through the initial values (`trafo`)."""
    def as_l(v):
        return list(v) if isinstance(v, (list, tuple)) else ([] if v is None else [v])
    model = [str(l) for l in as_l(model)]
    observation = [str(l) for l in as_l(observation)]
    condition_subs = [dict(c) for c in as_l(condition_subs)] or [{}]
    condition_ic0 = [dict(c) for c in as_l(condition_ic0)]
    K = len(condition_subs)
    condition_obs = list(condition_obs) if condition_obs else []
    obs_per_cond = [[str(l) for l in as_l(condition_obs[c])]
                    if (c < len(condition_obs) and condition_obs[c] is not None)
                    else observation for c in range(K)]

    all_lines = model + observation + as_l(parameters)
    for o in obs_per_cond:
        all_lines += o
    for c in condition_subs + condition_ic0:
        for k, v in c.items():
            all_lines += [str(k), str(v)]
    local, parse = _make_local_parse(all_lines)

    variables, diff_equations, _ = _read_equations(model, parse)
    obs_fun_per = [_read_equations(o, parse)[1] for o in obs_per_cond]
    fixed_names = set(str(s) for s in
                     [local.get(nm, spy.Symbol(nm)) for nm in as_l(fixed)])

    # constant (u' = 0) states are dropped; their per-condition value is baked
    # into condition_subs, so they enter f/g as substituted constants
    is_const = [spy.sympify(e) == 0 for e in diff_equations]
    S = [variables[i] for i in range(len(variables)) if not is_const[i]]
    Srhs = [spy.sympify(diff_equations[i]) for i in range(len(variables))
            if not is_const[i]]

    def pval(s):
        return spy.sympify(parse(str(s)))

    # per-condition substituted dynamics/observation and initial state
    per_cond = []
    for c in range(K):
        subs_map = {local.get(str(k), spy.Symbol(str(k))): pval(v)
                    for k, v in condition_subs[c].items()}
        ic0 = condition_ic0[c] if c < len(condition_ic0) else {}
        f_c = [e.subs(subs_map) for e in Srhs]
        g_c = [spy.sympify(e).subs(subs_map) for e in obs_fun_per[c]]
        subs_map_ic = dict(subs_map)
        if condition_rest_subs is not None and c < len(condition_rest_subs):
            for k, v in dict(condition_rest_subs[c] or {}).items():
                subs_map_ic[local.get(str(k), spy.Symbol(str(k)))] = pval(v)
        ic_c = {}
        for X in S:
            e = pval(ic0[str(X)]) if str(X) in ic0 else X
            ic_c[X] = spy.sympify(e).subs(subs_map_ic)
        per_cond.append((f_c, g_c, ic_c))

    # a state is a free coordinate iff its symbol survives in its initial value in
    # some condition; parameters are all remaining free symbols; both minus `fixed`
    free_states = [X for X in S
                   if any(X in ic_c[X].free_symbols for (_, _, ic_c) in per_cond)]
    paramset = set()
    for (f_c, g_c, ic_c) in per_cond:
        for e in list(f_c) + list(g_c) + list(ic_c.values()):
            paramset |= set(spy.sympify(e).free_symbols)
    paramset -= set(S)
    params = sorted(paramset, key=spy.default_sort_key)

    z = [X for X in free_states if str(X) not in fixed_names] + \
        [s for s in params if str(s) not in fixed_names]
    znames = [str(s) for s in z]
    nz = len(z)
    if nz == 0:
        return {'ok': True, 'rank': 0, 'dim': 0, 'lie_order': 0,
                'non_identifiable': [], 'identifiable': True, 'coordinates': []}

    # per Lie order, each condition's rows d/dz[L_{f_c}^order g_c at ic_c]; stop at
    # full rank or after two orders without rank gain, counted only past the
    # structural first order (a readout behind a transit chain is flat until it fills)
    reach_per = [_lie_reach(f_c, g_c, S, ic_c, []) for (f_c, g_c, ic_c) in per_cond]
    first_ord = [min([r[nm] for r in reach_per if nm in r] or [-1]) for nm in znames]
    lo0 = max([k for k in first_ord if k >= 0] or [0])
    rows = []
    jets = [list(g_c) for (f_c, g_c, ic_c) in per_cond]     # order 0: the observables
    prev, flat, order = -1, 0, 0
    while True:
        for ci, (_f_c, _g_c, ic_c) in enumerate(per_cond):
            for h in jets[ci]:
                he = spy.sympify(h).subs(ic_c)             # evaluate at x(0) = ic_c
                rows.append([spy.diff(he, zj) for zj in z])
        rank = spy.Matrix(rows).rank()
        if rank == prev:
            flat = flat + 1 if order > lo0 else flat
        else:
            flat = 0
        if rank >= nz or (flat >= 2 and order >= 1) or order >= nz + lo0:
            break
        prev = rank
        order += 1
        for ci, (f_c, _g_c, _ic_c) in enumerate(per_cond):
            jets[ci] = [spy.expand(_lie_deriv(h, S, f_c)) for h in jets[ci]]

    M = spy.Matrix(rows)
    rank = M.rank()
    non_id = []
    if rank < nz:
        for vec in M.nullspace():
            v = [spy.cancel(vec[i]) for i in range(nz)]
            nz_i = [i for i in range(nz) if v[i] != 0]
            non_id.append({'support': sorted(znames[i] for i in nz_i),
                           'vector': {znames[i]: str(v[i]) for i in nz_i},
                           'type': 'general', 'closed_form': True})
    return {'ok': True, 'rank': int(rank), 'dim': int(nz), 'lie_order': int(order),
            'non_identifiable': non_id, 'identifiable': bool(rank >= nz),
            'coordinates': znames}
