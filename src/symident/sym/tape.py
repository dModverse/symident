"""Straight-line tapes of the observability kernel, per condition and segment."""

import sympy as spy

from .charts import (
    _apply_power_recast,
    _detect_log_params,
    _detect_power_atoms,
    _has_exp,
    _hyp_to_exp,
    _log_param_denominators,
    _log_param_sub,
    _log_params_only_in_atoms,
    _strip_log_obs,
)
from .gfpoly import _linear_solution
from .parse import (
    _as_list,
    _is_rational_expr,
    _make_local_parse,
    _NotRationalError,
    _read_equations,
)
from .recast import (
    _apply_exp_recast,
    _apply_func_recast,
    _apply_trig_recast,
    _has_func,
    _has_trig,
    _is_sqrt_rational,
    _merge_recast,
)
from .steady import _FORCE_CONSTRAINT_SEED

# Opcodes for the flat tape consumed by the C++ kernel (cpp/kernel.cpp).
# Integer powers are expanded into multiplications, so the kernel needs only
# these four operations.
_OP_CONST, _OP_ADD, _OP_MUL, _OP_INV, _OP_SQRT, _OP_ROOT = 0, 1, 2, 3, 4, 5


def _emit_tape_shared(fexpr, gexpr, slot_of, base, allow_sqrt=False):
    """Emit a straight-line tape over an explicit slot map of states and leaves;
    instruction i writes slot base + i. States get their own slots, seeded from an
    initial condition (multi-condition compiler). With `allow_sqrt` (initial-value
    tapes), rational powers become roots modulo the prime: square roots for the powers of
    two in the denominator, the unique odd root for the rest."""
    op, a, b, cnum, cden = [], [], [], [], []
    memo = {}

    def emit(opcode, aa, bb, num=0, den=1):
        op.append(int(opcode))
        a.append(int(aa))
        b.append(int(bb))
        cnum.append(str(num))
        cden.append(str(den))
        return base + len(op) - 1

    def build(e):
        s = memo.get(e)
        if s is not None:
            return s
        if e.is_Symbol:
            nm = str(e)
            if nm not in slot_of:
                raise _NotRationalError()
            memo[e] = slot_of[nm]
            return slot_of[nm]
        if e.is_Integer:
            s = emit(_OP_CONST, 0, 0, int(e), 1)
        elif e.is_Rational:
            s = emit(_OP_CONST, 0, 0, int(e.p), int(e.q))
        elif e.is_Float:
            r = spy.nsimplify(e, rational=True)
            if not r.is_Rational:
                r = spy.Rational(e)
            s = emit(_OP_CONST, 0, 0, int(r.p), int(r.q))
        elif e.is_Add:
            args = list(e.args)
            s = build(args[0])
            for t in args[1:]:
                s = emit(_OP_ADD, s, build(t))
        elif e.is_Mul:
            args = list(e.args)
            s = build(args[0])
            for t in args[1:]:
                s = emit(_OP_MUL, s, build(t))
        elif e.is_Pow and e.exp.is_Integer:
            n = int(e.exp)
            base_s = build(e.base)
            if n == 0:
                s = emit(_OP_CONST, 0, 0, 1, 1)
            else:
                s = base_s
                for _ in range(abs(n) - 1):
                    s = emit(_OP_MUL, s, base_s)
                if n < 0:
                    s = emit(_OP_INV, s, 0)
        elif allow_sqrt and e.is_Pow and e.exp.is_Rational and not e.exp.is_Integer:
            q = int(e.exp.q)
            root = build(e.base)
            while q % 2 == 0:
                root = emit(_OP_SQRT, root, 0)
                q //= 2
            if q > 1:
                root = emit(_OP_ROOT, root, q)
            s = root
            for _ in range(abs(int(e.exp.p)) - 1):
                s = emit(_OP_MUL, s, root)
            if e.exp.p < 0:
                s = emit(_OP_INV, s, 0)
        else:
            raise _NotRationalError()
        memo[e] = s
        return s

    repl, red = spy.cse(list(fexpr) + list(gexpr))
    for sym, sub in repl:
        memo[sym] = build(sub)
    outslots = [build(e) for e in red]
    return op, a, b, cnum, cden, outslots


def _compose_t0_events(ic_map, t0evs, pval, subs_map):
    """Apply a segment's t0 events to the initial-state map in place: the jet starts
    at x0 = E(x_ss), the f = 0 constraint stays on the pre-event x_ss. Events on
    non-coordinate states are ignored."""
    for e in t0evs or []:
        Xv = spy.Symbol(str(e['var']))
        if Xv not in ic_map:
            continue
        val = spy.sympify(pval(e['value'])).subs(subs_map)
        meth = str(e['method'])
        if meth == 'replace':
            ic_map[Xv] = val
        elif meth == 'add':
            ic_map[Xv] = ic_map[Xv] + val
        elif meth == 'multiply':
            ic_map[Xv] = ic_map[Xv] * val
    return ic_map


def _lie_reach(f_c, g_c, S, ic_c, evs, f_ss=None):
    """Structural first Lie order at which each symbol can enter the output jets of one
    segment, a lower bound for the order its column first becomes nonzero: 0 for the
    symbols of the observables, else one more than the least order of a state whose
    right-hand side holds it (L_f^{k+1} h = sum_i dL_f^k h/dx_i f_i). An initial value
    or an event value enters with its state; at an implicit steady state (f_ss) every
    parameter of the balance may enter with the earliest state. Returns {name: order}."""
    import collections
    Sstr = [str(X) for X in S]
    rhs = {}
    for i, X in enumerate(Sstr):
        try:
            rhs[X] = {str(s) for s in spy.sympify(f_c[i]).free_symbols}
        except Exception:
            rhs[X] = set()
    dist = {}
    queue = collections.deque()
    for e in g_c:
        for s in spy.sympify(e).free_symbols:
            if str(s) not in dist:
                dist[str(s)] = 0
                queue.append(str(s))
    while queue:
        u = queue.popleft()
        for s in rhs.get(u, ()):
            if s not in dist:
                dist[s] = dist[u] + 1
                queue.append(s)
    out = dict(dist)

    def lower(name, k):
        if k < out.get(name, 10 ** 9):
            out[name] = k

    for X, e in dict(ic_c).items():
        if str(X) in dist:
            for s in spy.sympify(e).free_symbols:
                lower(str(s), dist[str(X)])
    for ev in evs or []:
        X = str(ev['var'])
        if X in dist:
            for s in spy.sympify(ev['value']).free_symbols:
                lower(str(s), dist[X])
    if f_ss is not None:
        reached = [dist[X] for X in Sstr if X in dist]
        if reached:
            k0 = min(reached)
            for e in f_ss:
                for s in spy.sympify(e).free_symbols:
                    lower(str(s), k0)
    return out



def _str_memo():
    """str() with a memo for hashable arguments."""
    memo = {}

    def f(x):
        try:
            v = memo.get(x)
        except TypeError:
            return str(x)
        if v is None:
            v = memo[x] = str(x)
        return v
    return f


def compile_observability_tape_multi(model, observation, condition_subs, condition_ic0,
                                  fixed=None, parameters=None, backend='sympy',
                                  equilibrate=False, forcings=None,
                                  seg_equilibrate=None, condition_events=None,
                                  condition_t0_events=None, joint_steady_state=False,
                                  joint_fixed_states=None, held_state_params=None,
                                  condition_obs=None, condition_times=None,
                                  keep_coords=None, condition_rest_subs=None):
    """Compile one observability tape per experimental condition over a shared
    coordinate space, for the multi-condition observability path.

    `condition_obs` optionally replaces `observation` per condition, so an observable
    measured in some conditions enters only their rows. `condition_subs` holds one
    {symbol: replacement} map per condition: a number bakes the symbol, a symbol renames
    it (e.g. a knockdown rate). `condition_ic0` holds one {state: expression} map per
    condition with the t0+ initial values composed in R (steady state, `initial`, t0
    events); a state without entry starts free.

    Returns the per-condition tapes (each with an IC tape seeding the initial values and
    their duals), the shared leaf/state layout and the coordinates z with dual lanes (free
    initial values + free parameters, minus `fixed`). A non-rational right-hand side,
    observable or initial value returns {'ok': False, 'nonrational': ...}."""
    _s = _str_memo()
    model = _as_list(model)
    observation = _as_list(observation)
    condition_subs = condition_subs or []
    condition_ic0 = condition_ic0 or []
    forcings = set(_as_list(forcings))
    K = len(condition_subs)
    # one flag per (condition, segment): equilibrated first segments are seeded from
    # the steady state, all others from an IC tape (possibly over z coordinates)
    if seg_equilibrate is None:
        seg_eq = [bool(equilibrate)] * K
    else:
        seg_eq = [bool(x) for x in list(seg_equilibrate)]
        seg_eq += [bool(equilibrate)] * (K - len(seg_eq))

    # missing/None entries of `condition_obs` fall back to `observation`
    condition_obs = list(condition_obs) if condition_obs else []
    obs_per_cond = [_as_list(condition_obs[c]) if (c < len(condition_obs) and
                                                   condition_obs[c] is not None)
                    else observation for c in range(K)]

    extra = []
    for d in condition_subs:
        for k, v in dict(d).items():
            extra += [_s(k), _s(v)]
    for d in condition_ic0:
        for k, v in dict(d).items():
            extra += [_s(k), _s(v)]
    all_lines = (model + observation + [l for o in obs_per_cond for l in o] +
                 _as_list(parameters) + extra)
    # symbol collection is order-insensitive and conditions repeat the same
    # trafo/IC strings, so tokenize each distinct line once
    all_lines = list(dict.fromkeys(all_lines))
    local, parse = _make_local_parse(all_lines)

    variables, diff_equations, _ = _read_equations(model, parse)
    obs_read = [_read_equations(o, parse) for o in obs_per_cond]
    obs_vars_per = [r[0] for r in obs_read]
    fixed_names = set(_s(s) for s in
                     [local.get(nm, spy.Symbol(nm)) for nm in _as_list(fixed)])
    # held-variable moieties (equilibrate without reduce_cq): each pivot state's resting
    # value is a shared initial-value parameter (held_map), so in joint mode the moiety
    # freedom lands on a reported parameter, not a per-condition state column
    held_state_params = dict(held_state_params or {})
    held_map = {_s(k): spy.Symbol(_s(v)) for k, v in held_state_params.items()}
    held_param_syms = [spy.Symbol(_s(v)) for v in held_state_params.values()]

    is_const = [spy.sympify(e) == 0 for e in diff_equations]
    S = [variables[i] for i in range(len(variables)) if not is_const[i]]
    Srhs = [spy.sympify(diff_equations[i]) for i in range(len(variables))
            if not is_const[i]]
    n_s = len(S)

    parse_cache = {}

    def pval(s):
        key = _s(s)
        e = parse_cache.get(key)
        if e is None:
            e = spy.sympify(parse(key))
            parse_cache[key] = e
        return e

    # substitution memoised on the transitively relevant subset of the map (conditions
    # repeat the same replacements); exact, since other items can never fire and sympy
    # orders dict items canonically per item
    subs_cache = {}

    def subs_memo(e, smap):
        e = spy.sympify(e)
        rel, frontier = {}, set(e.free_symbols)
        while True:
            add = {k: v for k, v in smap.items() if k not in rel and k in frontier}
            if not add:
                break
            rel.update(add)
            for v in add.values():
                frontier |= v.free_symbols
        if not rel:
            return e
        key = (e, tuple(sorted(rel.items(),
                               key=lambda kv: spy.default_sort_key(kv[0]))))
        r = subs_cache.get(key)
        if r is None:
            r = e.subs(rel)
            subs_cache[key] = r
        return r

    per_cond = []
    subs_per = []            # each condition's substitution map, aligned with per_cond
    for c in range(K):
        subs_map = {}
        for k, v in dict(condition_subs[c]).items():
            subs_map[local.get(_s(k), spy.Symbol(_s(k)))] = pval(v)
        ic0 = dict(condition_ic0[c]) if c < len(condition_ic0) else {}
        f_c = [subs_memo(e, subs_map) for e in Srhs]
        g_c = [subs_memo(e, subs_map) for e in obs_read[c][1]]
        # resting-state model for the equilibrate solve: forcings at 0, no events,
        # non-forcing per-condition substitutions baked in
        forc_zero = {local.get(nm, spy.Symbol(nm)): spy.Integer(0) for nm in forcings}
        subs_map_nf = {k: v for k, v in subs_map.items() if _s(k) not in forcings}
        # constant states at rest and in the initial values hold their value before the
        # t0 events
        subs_map_ic = dict(subs_map)
        if condition_rest_subs is not None and c < len(condition_rest_subs):
            for k, v in dict(condition_rest_subs[c] or {}).items():
                ks = local.get(_s(k), spy.Symbol(_s(k)))
                subs_map_ic[ks] = pval(v)
                if _s(k) not in forcings:
                    subs_map_nf[ks] = pval(v)
        f_ss = [subs_memo(subs_memo(e, subs_map_nf), forc_zero) for e in Srhs]
        ic_c = {}
        for X in S:
            e = pval(ic0[_s(X)]) if _s(X) in ic0 else X
            ic_c[_s(X)] = subs_memo(e, subs_map_ic)
        per_cond.append((f_c, g_c, ic_c, f_ss))
        subs_per.append(subs_map)

    # a logarithmic observable holds the information of its argument; hyperbolic
    # functions are exponentials
    per_cond = [([_hyp_to_exp(e) for e in f_c],
                [_hyp_to_exp(_strip_log_obs(e)) for e in g_c],
                {k: _hyp_to_exp(v) for k, v in ic_c.items()},
                [_hyp_to_exp(e) for e in f_ss])
                for (f_c, g_c, ic_c, f_ss) in per_cond]

    # log-parametrised parameters: one that occurs only as base^(c*theta) is replaced
    # by the rational coordinate X = base^theta, and the report maps X back to theta
    condition_events = condition_events or []
    condition_times = condition_times or []
    ev_exprs = []
    for c in range(K):
        for e in (list(condition_events[c]) if c < len(condition_events) else []):
            ev_exprs.append(subs_memo(pval(e['value']), subs_per[c]))
        if c < len(condition_times) and condition_times[c] is not None:
            ev_exprs.append(subs_memo(pval(condition_times[c]), subs_per[c]))
    lp_exprs = [e for (f_c, g_c, ic_c, f_ss) in per_cond
                for e in list(f_c) + list(g_c) + list(ic_c.values()) + list(f_ss)]
    lp_base = _detect_log_params(lp_exprs + ev_exprs, set(S))
    lp_base = _log_params_only_in_atoms(lp_base, lp_exprs + ev_exprs)
    lp_den = _log_param_denominators(lp_base, lp_exprs + ev_exprs)
    taken = set()
    for e in lp_exprs + ev_exprs:
        taken |= {_s(x) for x in spy.sympify(e).free_symbols}
    lp = {}
    for t, b in lp_base.items():
        nm = 'exp_%s' % t
        while nm in taken:
            nm += '_'
        taken.add(nm)
        lp[t] = (spy.Symbol(nm), b, lp_den[t])
    lp_sub = _log_param_sub(lp)
    if lp:
        per_cond = [([lp_sub(e) for e in f_c], [lp_sub(e) for e in g_c],
                    {k: lp_sub(v) for k, v in ic_c.items()}, [lp_sub(e) for e in f_ss])
                    for (f_c, g_c, ic_c, f_ss) in per_cond]
        for t, (X, _b, _L) in lp.items():
            if _s(t) in fixed_names:
                fixed_names.add(_s(X))
    log_params = [{'X': _s(X), 'theta': _s(t), 'base': 'E' if b ** spy.Rational(1, L) == spy.E
                   else _s(b ** spy.Rational(1, L))}
                  for t, (X, b, L) in sorted(lp.items(), key=lambda kv: _s(kv[0]))]

    # each condition's later events and segment start time, in the leaves
    ev_per = [[{'var': _s(e['var']), 'method': _s(e['method']),
                'value': lp_sub(subs_memo(pval(e['value']), subs_per[c]))}
               for e in (list(condition_events[c]) if c < len(condition_events) else [])]
              for c in range(K)]
    tm_per = [lp_sub(subs_memo(pval(condition_times[c]), subs_per[c]))
              if c < len(condition_times) and condition_times[c] is not None else None
              for c in range(K)]

    # power/Hill recast: base^exp with a parameter exponent becomes a state E with
    # E' = exp*E*base'/base plus L = log(base), so f stays rational; E is held generic
    # in the f = 0 solve, or a free-initial-value leaf when transient.
    n_real = n_s
    power_recast = []
    inv_solve_name = {}
    pairs = _detect_power_atoms(per_cond, S)
    if pairs is None:
        return {'ok': False, 'nonrational':
                ['unsupported power form: base must be a symbol and exponent c*param']}
    if pairs:
        S, per_cond, power_recast = _apply_power_recast(pairs, S, per_cond)
        n_s = len(S)
        if equilibrate:
            # a base with a linear turnover term is solved directly; otherwise its
            # balance is linear in E and it is "inverted": E is solved, base and L
            # stay generic
            real_set = {_s(X) for X in S[:n_real]}
            bare_syms = set()
            for (_f, _g, _ic, ss_r) in per_cond:
                for e in ss_r[:n_real]:
                    bare_syms |= spy.fraction(spy.together(e))[0].free_symbols
            for rc in power_recast:
                rc['inverted'] = (rc['base'] in real_set
                                  and spy.Symbol(rc['base']) not in bare_syms)
            by_base = {}
            for rc in power_recast:
                by_base.setdefault(rc['base'], []).append(rc)
            for base, group in by_base.items():
                if group[0]['inverted'] and len(group) > 1:
                    return {'ok': False, 'nonrational':
                            ['free exponent base %s has multiple independent '
                             'powers with no linear turnover term' % base]}
            inv_solve_name = {rc['base']: rc['E']
                              for rc in power_recast if rc['inverted']}
        else:
            # transient: no steady-state solve, so nothing is inverted
            for rc in power_recast:
                rc['inverted'] = False

    def any_in(pred):
        return (any(pred(e) for (f_c, g_c, ic_c, f_ss) in per_cond
                    for e in list(f_c) + list(g_c) + list(ic_c.values())) or
                any(pred(e['value']) for evs in ev_per for e in evs) or
                any(t is not None and pred(t) for t in tm_per))

    def taken_names():
        taken = {_s(x) for x in S} | set(fixed_names) | {_s(X) for X, _ in lp.values()}
        for (f_c, g_c, ic_c, f_ss) in per_cond:
            for e in list(f_c) + list(g_c) + list(ic_c.values()) + list(f_ss):
                taken |= {_s(x) for x in spy.sympify(e).free_symbols}
        for e in lp_exprs + ev_exprs:
            taken |= {_s(x) for x in spy.sympify(e).free_symbols}
        return taken

    # trigonometric functions: half-angle states and leaves (_apply_trig_recast), before
    # the exponentials so that an exponent may contain them; the remaining elementary
    # functions last (_apply_func_recast), once their arguments are rational
    scal_s, scal_per_cond = list(S), per_cond
    trig_recast = None
    if any_in(_has_trig):
        if equilibrate:
            return {'ok': False, 'why': 'equilibrate does not support '
                    'trigonometric functions; give the steady state through `trafo` '
                    'or start from free initial values'}
        trig_recast = _apply_trig_recast(S, per_cond, ev_per, tm_per, taken_names())
        if 'why' in trig_recast:
            return {'ok': False, 'why': trig_recast['why']}
        S, per_cond = trig_recast['S'], trig_recast['per_cond']
        ev_per, tm_per = trig_recast['ev_per'], trig_recast['tm_per']
        n_s = len(S)
        fixed_names |= set(trig_recast['consts'])

    # exponentials: auxiliary states and generic exponential leaves (_apply_exp_recast)
    exp_recast = None
    if any_in(_has_exp):
        if equilibrate:
            return {'ok': False, 'why': 'equilibrate does not support exp() '
                    'or b^x of a state, or of a parameter that also enters elsewhere; '
                    'give the steady state through `trafo` or start from free '
                    'initial values'}
        exp_recast = _apply_exp_recast(S, per_cond, ev_per, tm_per, taken_names())
        if 'why' in exp_recast:
            return {'ok': False, 'why': exp_recast['why']}
        S, per_cond = exp_recast['S'], exp_recast['per_cond']
        ev_per, tm_per = exp_recast['ev_per'], exp_recast['tm_per']
        n_s = len(S)
        fixed_names |= set(exp_recast['consts'])
    if trig_recast is not None:
        exp_recast = _merge_recast(trig_recast, exp_recast)
    func_recast = None
    if any_in(_has_func):
        if equilibrate:
            return {'ok': False, 'why': 'equilibrate does not support logarithms, inverse '
                    'trigonometric or hyperbolic functions or non-integer powers of sums that no '
                    'chart makes rational; give the steady state through `trafo` or start from '
                    'free initial values'}
        func_recast = _apply_func_recast(S, per_cond, ev_per, tm_per, taken_names())
        if 'why' in func_recast:
            return {'ok': False, 'why': func_recast['why']}
        S, per_cond = func_recast['S'], func_recast['per_cond']
        ev_per, tm_per = func_recast['ev_per'], func_recast['tm_per']
        n_s = len(S)
        fixed_names |= set(func_recast['consts'])
    if func_recast is not None:
        exp_recast = _merge_recast(exp_recast, func_recast)

    nonrational = []
    rat_ok = set()            # deduped exprs recur across conditions; check each once

    def _rat(e):
        if e in rat_ok:
            return True
        ok = _is_rational_expr(e)
        if ok:
            rat_ok.add(e)
        return ok

    for c, (f_c, g_c, ic_c, _f_ss) in enumerate(per_cond):
        for X, e in zip(S, f_c):
            if not _rat(e):
                nonrational.append('d%s/dt = %s' % (X, e))
        for y, e in zip(obs_vars_per[c], g_c):
            if not _rat(e):
                nonrational.append('%s = %s' % (y, e))
        for X in S:
            if not (_rat(ic_c[_s(X)]) or _is_sqrt_rational(ic_c[_s(X)])):
                nonrational.append('%s(0) = %s' % (X, ic_c[_s(X)]))
    if nonrational:
        return {'ok': False, 'nonrational': nonrational}

    # a state has a free initial-value leaf when its initial value still contains the
    # state symbol (possibly under a dose); equilibrate-seeded states have none
    free_state = {_s(X): False for X in S}
    for c, (_f_c, _g_c, ic_c, _f_ss) in enumerate(per_cond):
        if seg_eq[c]:
            # joint mode: states stay free coordinates (seeded to x* by the analysis,
            # constraint as stacked df rows), keeping directions low-degree in (x, theta)
            if joint_steady_state:
                jfs = set(joint_fixed_states or [])
                for X in S:
                    if (_s(X) not in forcings and _s(X) not in jfs
                            and _s(X) not in held_map):
                        free_state[_s(X)] = True
            continue  # equilibrate-seeded: no state is a free coordinate (unless joint)
        for X in S:
            if X in spy.sympify(ic_c[_s(X)]).free_symbols:
                free_state[_s(X)] = True

    # boundary state-dose values (e.g. a second-dose parameter) may appear only in
    # an event, never in f/g/ic; collect their symbols so they become coordinates
    condition_events = condition_events or []
    event_vals = []
    for c in range(K):
        evs = list(condition_events[c]) if c < len(condition_events) else []
        event_vals.append([lp_sub(spy.sympify(pval(e['value']))) for e in evs])

    # f_ss is included so parameters the perturbed dynamics drop still count
    paramset = set()
    seen_ps = set()           # deduped exprs recur across conditions; walk each once
    for (f_c, g_c, ic_c, f_ss) in per_cond:
        for e in list(f_c) + list(g_c) + list(ic_c.values()) + list(f_ss):
            if e in seen_ps:
                continue
            seen_ps.add(e)
            paramset |= set(spy.sympify(e).free_symbols)
    for vs in event_vals:
        for e in vs:
            paramset |= set(e.free_symbols)
    for evs in ev_per:
        for e in evs:
            paramset |= set(spy.sympify(e['value']).free_symbols)
    if exp_recast is not None:
        by_name = {_s(x): x for x in paramset}
        for rl in exp_recast['rel']:
            paramset |= set(rl[2].free_symbols)
            paramset.add(by_name.get(rl[1], spy.Symbol(rl[1])))
    # a segment's left boundary may sit at a time given in the parameters
    for c, tm in enumerate(condition_times):
        if tm is not None and c < K:
            paramset |= set(lp_sub(pval(tm)).free_symbols)
    paramset -= set(S)
    paramset |= set(held_param_syms)   # held-variable initial-value parameters
    # coordinates a chart absorbed stay coordinates
    paramset |= {local.get(_s(n), spy.Symbol(_s(n))) for n in _as_list(keep_coords)}
    params = sorted(paramset, key=spy.default_sort_key)
    # a base seen only under a free exponent (Km in C^n/(Km^n+C^n)) vanished into its
    # recast atom; re-add it so the recast relation ties it and the Km co-scaling is
    # reported as an exact scaling
    if power_recast:
        known = set(_s(s) for s in S) | set(_s(s) for s in params) | set(forcings)
        for rc in power_recast:
            if rc['base'] not in known:
                params.append(spy.Symbol(rc['base']))
                known.add(rc['base'])
        params = sorted(params, key=spy.default_sort_key)

    free_states = [X for X in S if free_state[_s(X)]]
    leaf_names = [_s(X) for X in free_states] + [_s(s) for s in params]
    leaf_slot = {nm: i for i, nm in enumerate(leaf_names)}
    n_leaves = len(leaf_names)
    slot_of = dict(leaf_slot)
    for i in range(n_s):
        slot_of[_s(S[i])] = n_leaves + i
    base = n_leaves + n_s

    tapes = []
    reach_cache = {}  # structural Lie reach per (model, initial values, events)
    emit_cache = {}   # (f_c, g_c, obs names) recur across segments/conditions
    ss_sol_cache = {}  # linear resting-state solutions, keyed on the f_ss tuple
    ic_cache = {}     # emitted IC tapes, keyed on the seed-expression tuple
    for c, (f_c, g_c, ic_c, f_ss) in enumerate(per_cond):
        subs_map = subs_per[c]
        ekey = (tuple(f_c), tuple(g_c), tuple(_s(v) for v in obs_vars_per[c]))
        cached = emit_cache.get(ekey)
        if cached is None:
            try:
                emitted = _emit_tape_shared(f_c, g_c, slot_of, base)
            except _NotRationalError:
                return {'ok': False}
            # this segment's substituted model as lines for the scaling peel,
            # exponentials kept
            f0, g0 = scal_per_cond[c][0], scal_per_cond[c][1]
            e_num = {spy.E: spy.exp(1, evaluate=False)}
            m_lines = ['%s = %s' % (_s(scal_s[i]), spy.sympify(f0[i]).xreplace(e_num))
                       for i in range(len(scal_s))]
            o_lines = ['%s = %s' % (_s(obs_vars_per[c][j]), spy.sympify(g0[j]).xreplace(e_num))
                       for j in range(len(g0))]
            cached = (emitted, m_lines, o_lines)
            emit_cache[ekey] = cached
        (op, a, b, cnum, cden, outslots), m_lines, o_lines = cached
        f_out = outslots[:n_s]
        # structural first Lie order of every symbol in this segment's jets; the
        # saturation counts no flat step before it (a readout behind a transit chain is
        # flat at rank 0 until the chain has filled)
        reach_key = (ekey, tuple(sorted((_s(k), _s(v)) for k, v in ic_c.items())),
                     tuple((_s(e['var']), _s(e['value'])) for e in ev_per[c]),
                     bool(seg_eq[c]))
        reach = reach_cache.get(reach_key)
        if reach is None:
            reach = _lie_reach(f_c, g_c, S, ic_c, ev_per[c],
                               f_ss if seg_eq[c] else None)
            reach_cache[reach_key] = reach
        tape = {
            'op': op, 'a': a, 'b': b, 'cnum': cnum, 'cden': cden,
            'state_slots': [n_leaves + i for i in range(n_s)],
            'f_out': f_out,
            'g_out': outslots[n_s:n_s + len(g_c)],
            'ic_leaf': [-1] * n_s, 'ic_num': ['0'] * n_s, 'ic_den': ['1'] * n_s,
            'model_lines': m_lines,
            'obs_lines': o_lines,
            # initial values, boundary events and start time, for verify_scalings()
            'ver_ic': ['%s = %s' % (_s(X), _s(ic_c[_s(X)])) for X in S if _s(X) in ic_c],
            'ver_ev': [{'var': _s(e['var']), 'method': _s(e['method']),
                        'value': _s(e['value'])} for e in ev_per[c]],
            'ver_t0': [{'var': _s(e['var']), 'method': _s(e['method']),
                        'value': _s(e['value'])} for e in
                      (condition_t0_events[c] if condition_t0_events and
                       c < len(condition_t0_events) else [])],
            'ver_time': '' if tm_per[c] is None else _s(tm_per[c]),
        }
        if seg_eq[c] and joint_steady_state:
            # joint mode: each non-forcing state is an identity leaf seeded to x*;
            # the resting model is kept to solve x* and read [Jx|Jt]
            jfs = set(joint_fixed_states or [])
            # held pivots seed from their parameter (held_map), forcings and forced-zero
            # states from 0
            ic_map = {X: (held_map[_s(X)] if _s(X) in held_map
                          else X if (_s(X) not in forcings and _s(X) not in jfs)
                          else spy.Integer(0)) for X in S}
            # t0 events act on the pre-event coordinate x_ss (seeded from val_by);
            # the IC tape holds x0 = E(x_ss) with its chain-rule duals
            _compose_t0_events(ic_map, condition_t0_events[c]
                               if condition_t0_events and c < len(condition_t0_events)
                               else [], pval, subs_map)
            try:
                ic_op, ic_a, ic_b, ic_cnum, ic_cden, ic_out = _emit_tape_shared(
                    [ic_map[X] for X in S], [], leaf_slot, n_leaves)
            except _NotRationalError:
                return {'ok': False}
            tape.update({'ic_op': ic_op, 'ic_a': ic_a, 'ic_b': ic_b, 'ic_cnum': ic_cnum,
                         'ic_cden': ic_cden, 'ic_out': ic_out})
            tape['constraint_model'] = [
                '%s = %s' % (inv_solve_name.get(_s(S[i]), _s(S[i])),
                             spy.sympify(f_ss[i])) for i in range(n_real)]
        elif seg_eq[c]:
            # equilibrate-seeded first segment: a generically linear resting state (no
            # recast) is solved symbolically and emitted as an IC tape; otherwise it is
            # seeded numerically per point with its IFT duals (ic_seed)
            ss_ic = None
            if not power_recast and not _FORCE_CONSTRAINT_SEED:
                sskey = tuple(f_ss)
                if sskey in ss_sol_cache:
                    ss_sol = ss_sol_cache[sskey]
                else:
                    solve_s = [S[i] for i in range(n_real)
                               if _s(S[i]) not in forcings]
                    fss_polys = [spy.expand(spy.fraction(spy.together(
                                    spy.sympify(f_ss[i])))[0])
                                 for i in range(n_real) if _s(S[i]) not in forcings]
                    ss_sol = _linear_solution(fss_polys, solve_s)
                    ss_sol_cache[sskey] = ss_sol
                if ss_sol is not None:
                    # recast coordinates (beyond n_real) stay generic; forcings and
                    # dead states stay 0
                    ic_map = {X: (ss_sol[X] if X in ss_sol
                                  else X if i >= n_real else spy.Integer(0))
                              for i, X in enumerate(S)}
                    # t0 events compose onto the resting state
                    _compose_t0_events(ic_map, condition_t0_events[c]
                                       if condition_t0_events and
                                       c < len(condition_t0_events) else [],
                                       pval, subs_map)
                    try:
                        ic_op, ic_a, ic_b, ic_cnum, ic_cden, ic_out = _emit_tape_shared(
                            [ic_map[X] for X in S], [], leaf_slot, n_leaves)
                        ss_ic = {'ic_op': ic_op, 'ic_a': ic_a, 'ic_b': ic_b,
                                 'ic_cnum': ic_cnum, 'ic_cden': ic_cden, 'ic_out': ic_out}
                    except _NotRationalError:
                        ss_ic = None
            if ss_ic is not None:
                tape.update(ss_ic)
            else:
                tape['constraint_model'] = [
                    '%s = %s' % (inv_solve_name.get(_s(S[i]), _s(S[i])),
                                 spy.sympify(f_ss[i])) for i in range(n_real)]
        else:
            # IC tape: seed each state from its initial-value expression in the
            # leaves (free initial values, z coordinates, doses, resets).
            ickey = tuple(ic_c[_s(X)] for X in S)
            ic_emitted = ic_cache.get(ickey)
            if ic_emitted is None:
                try:
                    ic_emitted = _emit_tape_shared(list(ickey), [], leaf_slot,
                                                   n_leaves, allow_sqrt=True)
                except _NotRationalError:
                    return {'ok': False}
                ic_cache[ickey] = ic_emitted
            ic_op, ic_a, ic_b, ic_cnum, ic_cden, ic_out = ic_emitted
            tape.update({'ic_op': ic_op, 'ic_a': ic_a, 'ic_b': ic_b,
                         'ic_cnum': ic_cnum, 'ic_cden': ic_cden, 'ic_out': ic_out})
            # the symbolic jets of this segment, for jet_generator(): dynamics,
            # observables and initial values over the leaves
            tape['jet_f'] = ['%s = %s' % (_s(S[i]), _s(f_c[i])) for i in range(n_s)]
            tape['jet_g'] = [_s(e) for e in g_c]
            tape['jet_ic'] = ['%s = %s' % (_s(X), _s(ic_c[_s(X)])) for X in S]
            # a later segment's boundary events, and whether its start time is free of
            # the coordinates (the jets of a moving boundary need the time shift too)
            tape['jet_ev'] = [{'var': _s(e['var']), 'method': _s(e['method']),
                               'value': _s(e['value'])} for e in ev_per[c]]
            tape['jet_time_fixed'] = bool(tm_per[c] is None or
                                          not spy.sympify(tm_per[c]).free_symbols)
        # events at this segment's left boundary, applied by the kernel to the
        # propagated state; their values as an order-0 tape over the leaves
        evs = ev_per[c]
        if evs:
            idx_of_state = {_s(X): i for i, X in enumerate(S)}
            method_code = {'replace': 0, 'add': 1, 'multiply': 2}
            keep = [e for e in evs if _s(e['var']) in idx_of_state]
            if keep:
                vals = [e['value'] for e in keep]
                try:
                    ev_op, ev_a, ev_b, ev_cnum, ev_cden, ev_o = _emit_tape_shared(
                        [], vals, leaf_slot, n_leaves)
                except _NotRationalError:
                    return {'ok': False}
                tape.update({
                    'ev_op': ev_op, 'ev_a': ev_a, 'ev_b': ev_b, 'ev_cnum': ev_cnum,
                    'ev_cden': ev_cden, 'ev_out': ev_o,
                    'ev_var_idx': [idx_of_state[_s(e['var'])] for e in keep],
                    'ev_method': [method_code[_s(e['method'])] for e in keep]})
        # the segment's left boundary time as an order-0 tape: its value sets the gap
        # lengths, its duals hold a time that depends on the coordinates
        if tm_per[c] is not None:
            tv = tm_per[c]
            try:
                tm_op, tm_a, tm_b, tm_cnum, tm_cden, tm_o = _emit_tape_shared(
                    [], [tv], leaf_slot, n_leaves)
            except _NotRationalError:
                return {'ok': False}
            tape.update({'tm_op': tm_op, 'tm_a': tm_a, 'tm_b': tm_b, 'tm_cnum': tm_cnum,
                         'tm_cden': tm_cden, 'tm_out': tm_o})
        tape['_reach'] = reach
        tapes.append(tape)

    z_state_names = [_s(X) for X in free_states if _s(X) not in fixed_names]
    z_param_names = [_s(s) for s in params if _s(s) not in fixed_names]
    znames = z_state_names + z_param_names
    z_slots = [leaf_slot[nm] for nm in znames]
    # per tape, the first Lie order at which each coordinate can enter (-1: never)
    for t in tapes:
        r = t.pop('_reach', None) or {}
        t['lie_reach'] = [int(r.get(nm, -1)) for nm in znames]

    out = {
        'ok': True,
        'tapes': tapes,
        'n_leaves': n_leaves,
        'n_states': n_s,
        'z_slots': z_slots,
        'znames': znames,
        'leaf_names': leaf_names,
        'log_params': log_params,
    }
    if exp_recast is not None:
        # relation rows dW = W log(r) d(tau/L): one tape output per (W, leaf) pair
        rel = exp_recast['rel']
        try:
            r_op, r_a, r_b, r_cnum, r_cden, r_out = _emit_tape_shared(
                [rl[2] for rl in rel], [], leaf_slot, n_leaves, allow_sqrt=True)
        except _NotRationalError:
            return {'ok': False, 'why': 'an exponent that is not rational is not '
                    'supported by engine="modular"; try engine="symbolic"'}
        out['exp_relation'] = {'W': [rl[0] for rl in rel], 'z': [rl[1] for rl in rel],
                               'op': r_op, 'a': r_a, 'b': r_b, 'cnum': r_cnum,
                               'cden': r_cden, 'out': r_out}
        out['exp_atoms'] = exp_recast['atoms']
        out['exp_back'] = {'names': list(exp_recast['back'].keys()),
                           'values': list(exp_recast['back'].values())}
        out['exp_codim'] = exp_recast['codim']
    if equilibrate:
        out['equilibrate'] = True
        out['state_names'] = [_s(X) for X in S]
        out['param_names'] = [_s(s) for s in params]
        out['forcings'] = sorted(forcings)
        out['real_state_names'] = [inv_solve_name.get(_s(X), _s(X))
                                   for X in S[:n_real]]
        out['power_recast'] = power_recast
        out['held_state_params'] = held_state_params
        if joint_steady_state:
            out['joint_steady_state'] = True
            # state z-columns (the free states that entered znames), for the joint
            # system to seed on-manifold and stack the df constraint rows
            out['z_state_names'] = z_state_names
    elif power_recast:
        # transient free exponent: E and L are free-initial-value leaves; the analysis
        # stacks the recast relation rows and reports in real states + parameters
        out['recast_transient'] = True
        out['state_names'] = [_s(X) for X in S]
        out['param_names'] = [_s(s) for s in params]
        out['forcings'] = sorted(forcings)
        out['real_state_names'] = [_s(X) for X in S[:n_real]]
        out['power_recast'] = power_recast
        # the recast atom coordinates (E and L), auxiliary to the physical report
        out['recast_atom_names'] = ([rc['E'] for rc in power_recast] +
                                    sorted(set(rc['L'] for rc in power_recast)))
    return out
