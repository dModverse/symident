"""Resting states over GF(p) solved forward for turnover rates."""

import random
import sys

import sympy as spy

from .modp import _eval_terms, _poly_terms, _solve_mod
from .parse import _as_list
from .steady import _ss_compile


def _forward_rate_pick(rhs_by_name, solve_states, param_names, forcings, keep_free=None):
    """Match each state to a rate entering its balance linearly, preferring a negative
    turnover term (rate times a monomial containing the state) found in few balances.
    `keep_free` (the direction's support) is never solved for. Most-constrained states go
    first. Returns one rate name per solve state, or None if no matching exists."""
    keep_free = set(keep_free or [])
    # never solve for a recast coordinate (E, L): E multiplies a turnover rate, which
    # would make f bilinear in the rates
    paramset = {spy.Symbol(pn) for pn in param_names
                if pn not in keep_free and not pn.startswith('_E_') and not pn.startswith('_L_')}
    forc = {spy.Symbol(nm) for nm in forcings}
    cand, appear = {}, {}
    for s in solve_states:
        f = spy.sympify(rhs_by_name[s])
        s_sym = spy.Symbol(s)
        lst = []
        for r in sorted(f.free_symbols & paramset, key=str):
            if r in forc:
                continue
            try:
                pv = spy.Poly(f, r)
            except spy.PolynomialError:
                continue
            if pv.degree() != 1 or r in pv.nth(1).free_symbols:
                continue
            coef = pv.nth(1)
            turnover = s_sym in coef.free_symbols
            neg = coef.could_extract_minus_sign()
            lst.append((0 if (turnover and neg) else 1 if turnover else 2, str(r)))
            appear[str(r)] = appear.get(str(r), 0) + 1
        cand[s] = lst
    order = sorted(solve_states, key=lambda s: len(cand[s]))
    used, assign = set(), {}
    for s in order:
        opts = sorted((sc, appear.get(r, 99), r) for sc, r in cand[s] if r not in used)
        if not opts:
            return _rate_matching(cand, appear, order)
        assign[s] = opts[0][2]
        used.add(opts[0][2])
    return [assign[s] for s in solve_states]


def _rate_matching(cand, appear, order):
    """A complete matching of states to candidate rates by augmenting paths, each state
    trying its rates in preference order; None if there is none."""
    prefs = {s: [r for _, _, r in sorted((sc, appear.get(r, 99), r) for sc, r in cand[s])]
             for s in order}
    owner = {}
    def augment(s, seen):
        for r in prefs[s]:
            if r in seen:
                continue
            seen.add(r)
            if r not in owner or augment(owner[r], seen):
                owner[r] = s
                return True
        return False
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 10 * len(order) + 100))
    for s in order:
        if not augment(s, set()):
            return None
    by_state = {s: r for r, s in owner.items()}
    return [by_state[s] for s in cand]


def _forward_plan(rhs_by_name, solve_states, param_names, forcings, keep_free=None):
    """Forward plan when balances share rates: a balance without a rate of its own is
    solved for its own state, before the rates. Returns (state_solve, solve_rates), the
    state-solved balances in dependency order and the rates, or None. A balance whose
    rate coefficients depend on the others' at a random point is state-solved too."""
    pre = []
    for _ in range(len(solve_states)):
        r = _forward_plan_once(rhs_by_name, solve_states, param_names, forcings, keep_free, pre)
        if r is None or r[0] != 'dependent':
            return r
        new = [u for u in r[1] if u not in pre]
        if not new:
            return None
        pre = pre + new[:1]
    return None


def _forward_plan_once(rhs_by_name, solve_states, param_names, forcings, keep_free=None,
                       pre_bs=()):
    keep_free = set(keep_free or [])
    paramset = {spy.Symbol(pn) for pn in param_names
                if pn not in keep_free and not pn.startswith('_E_') and not pn.startswith('_L_')}
    forc = {spy.Symbol(nm) for nm in forcings}
    polys = {s: spy.sympify(rhs_by_name[s]) for s in solve_states}
    params = {s: polys[s].free_symbols & paramset for s in solve_states}
    cand = {}
    for s in solve_states:
        lst = []
        for r in sorted(params[s] - forc, key=str):
            try:
                pv = spy.Poly(polys[s], r)
            except spy.PolynomialError:
                continue
            if pv.degree() == 1 and r not in pv.nth(1).free_symbols:
                lst.append(str(r))
        cand[s] = lst
    state_set = set(solve_states)
    # a parameter of higher degree in any balance cannot be solved for linearly
    nonlin = set()
    for s_ in solve_states:
        for r in params[s_]:
            try:
                if spy.Poly(polys[s_], r).degree() > 1:
                    nonlin.add(str(r))
            except spy.PolynomialError:
                nonlin.add(str(r))
    cand = {s_: [r for r in cand[s_] if r not in nonlin] for s_ in solve_states}
    # parameters sharing a monomial; no two of them are solve rates
    pnames = {str(r) for r in paramset}
    partners = {}
    for s in solve_states:
        for mono in spy.Add.make_args(spy.expand(polys[s])):
            ps = [str(x) for x in mono.free_symbols if str(x) in pnames]
            for a in ps:
                partners.setdefault(a, set()).update(x for x in ps if x != a)
    banned = set()
    bs = list(pre_bs)                          # balances solved for their own state
    for u in bs:
        try:
            pu = spy.Poly(polys[u], spy.Symbol(u))
        except spy.PolynomialError:
            return None
        if pu.degree() != 1 or spy.Symbol(u) in pu.nth(1).free_symbols:
            return None
    while True:
        forbidden = {str(r) for u in bs for r in params[u]} | banned
        rest = [s for s in solve_states if s not in bs]
        match = {}                            # rate -> balance, by augmenting paths
        def augment(s, seen):
            for r in cand[s]:
                if r in forbidden or r in seen:
                    continue
                seen.add(r)
                if r not in match or augment(match[r], seen):
                    match[r] = s
                    return True
            return False
        for s in sorted(rest, key=lambda s: len(cand[s])):
            augment(s, set())
        # ban the solve rate with the most partners among the chosen, then rematch
        chosen = set(match)
        clash = sorted((r for r in chosen if partners.get(r, set()) & chosen),
                       key=lambda r: (-len(partners.get(r, ())), r))
        if clash:
            banned.add(clash[0])
            continue
        matched = set(match.values())
        unmatched = [s for s in rest if s not in matched]
        if not unmatched:
            break
        for u in unmatched:
            try:
                pu = spy.Poly(polys[u], spy.Symbol(u))
            except spy.PolynomialError:
                return None
            if pu.degree() != 1 or spy.Symbol(u) in pu.nth(1).free_symbols:
                return None
        bs.extend(unmatched)
    # dependency order: a state-solved balance reads the other states it contains
    deps = {u: {str(x) for x in polys[u].free_symbols if str(x) in state_set} & set(bs) - {u}
            for u in bs}
    order = []
    while len(order) < len(bs):
        ready = [u for u in bs if u not in order and deps[u] <= set(order)]
        if not ready:
            return None
        order.extend(sorted(ready))
    # rates as a column basis of the coefficient matrix at a random point, turnover
    # rates first, so the rate system is nonsingular
    rest = [s for s in solve_states if s not in bs]
    forbidden = {str(r) for u in bs for r in params[u]}
    Q = 2147483629
    rng = random.Random(len(rest))
    syms = set().union(*(polys[s].free_symbols for s in rest)) if rest else set()
    pt = {x: rng.randrange(2, Q - 1) for x in syms}
    def score(r):
        best = 2
        for s in rest:
            if r not in cand[s]:
                continue
            coef = spy.Poly(polys[s], spy.Symbol(r)).nth(1)
            if spy.Symbol(s) in coef.free_symbols:
                best = min(best, 0 if coef.could_extract_minus_sign() else 1)
        return best
    cols = sorted({r for s in rest for r in cand[s]} - forbidden,
                  key=lambda r: (score(r), sum(r in cand[s] for s in rest), r))
    col_vec = {}
    def vec_of(r):
        if r not in col_vec:
            col_vec[r] = [int(spy.Poly(polys[s], spy.Symbol(r)).nth(1).xreplace(pt)) % Q
                          if r in cand[s] else 0 for s in rest]
        return col_vec[r]
    def pick_basis(col_order):
        basis, chosen = [], []                # echelon rows (pivot, vector) mod Q
        for r in col_order:
            if partners.get(r, set()) & set(chosen):
                continue
            v = list(vec_of(r))
            for piv, b in basis:
                if v[piv]:
                    fct = v[piv]
                    v = [(x - fct * y) % Q for x, y in zip(v, b)]
            piv = next((i for i, x in enumerate(v) if x), None)
            if piv is None:
                continue
            inv = pow(v[piv], Q - 2, Q)
            basis.append((piv, [x * inv % Q for x in v]))
            chosen.append(r)
            if len(chosen) == len(rest):
                break
        return chosen
    chosen = pick_basis(cols)
    # the partner rule can block the greedy order: shuffles within each preference class
    key_of = {r: score(r) for r in cols}
    for att in range(40):
        if len(chosen) == len(rest):
            break
        rs = random.Random(att)
        shuffled = sorted(cols, key=lambda r: (key_of[r], rs.random()))
        cand2 = pick_basis(shuffled)
        if len(cand2) > len(chosen):
            chosen = cand2
    if len(chosen) < len(rest):
        # the balances whose coefficient rows depend on the earlier ones
        rows, dep = [], []
        for s in rest:
            v = [int(spy.Poly(polys[s], spy.Symbol(r)).nth(1).xreplace(pt)) % Q
                 if r in cand[s] else 0 for r in cols]
            for piv, b in rows:
                if v[piv]:
                    fct = v[piv]
                    v = [(x - fct * y) % Q for x, y in zip(v, b)]
            piv = next((i for i, x in enumerate(v) if x), None)
            if piv is None:
                dep.append(s)
                continue
            inv = pow(v[piv], Q - 2, Q)
            rows.append((piv, [x * inv % Q for x in v]))
        return ('dependent', dep)
    return order, chosen


_forward_cache = {}


def _forward_compile(model, state_names, param_names, forcings, solve_rates, state_solve=()):
    """Cached prime-independent compile of the forward solve: each solve rate's coefficient
    and the rate-free constant per balance as term lists over rgens = (non-solve params) +
    solve states, plus the steady-state Jacobian term lists. A balance in `state_solve` is
    compiled as its own state's coefficient and remainder instead. Returns {'bad': reason}
    if the balances are not linear in the chosen rates."""
    state_solve = tuple(state_solve)
    key = (tuple(model), tuple(state_names), tuple(param_names), tuple(sorted(forcings)),
           tuple(solve_rates), state_solve)
    c = _forward_cache.get(key)
    if c is not None:
        return c
    (param_syms, solve_states, polys, Jx, Jt, gens0, jx_terms, jt_terms,
     poly_bi, gen_lin, lin_terms, _pp) = _ss_compile(model, state_names, param_names, forcings)
    solve_set = set(solve_rates)
    r_syms = [spy.Symbol(r) for r in solve_rates]
    rgens = [th for th in param_syms if str(th) not in solve_set] + list(solve_states)
    rgenset = set(rgens)
    idx_of = {str(st): i for i, st in enumerate(solve_states)}
    state_t = []
    for u in state_solve:
        pu = spy.Poly(spy.sympify(polys[idx_of[u]]), spy.Symbol(u))
        c1, c0 = pu.nth(1).as_expr(), pu.nth(0).as_expr()
        if (c1.free_symbols | c0.free_symbols) - rgenset:
            return {'bad': 'state-solved balance %s holds a solve rate' % u}
        state_t.append((u, _poly_terms(c1, rgens), _poly_terms(c0, rgens)))
    rate_rows = [i for i in range(len(solve_states)) if str(solve_states[i]) not in set(state_solve)]
    coef_t, const_t = [], []
    for i in rate_rows:
        fexp = spy.sympify(polys[i])
        row = []
        for r in r_syms:
            dexp = spy.diff(fexp, r)
            if dexp.free_symbols - rgenset:
                return {'bad': 'not linear in the chosen rates at %s' % str(solve_states[i])}
            row.append(_poly_terms(dexp, rgens))
        cexp = fexp.subs({r: spy.Integer(0) for r in r_syms})
        if cexp.free_symbols - rgenset:
            return {'bad': 'unsubstituted symbol in balance %s' % str(solve_states[i])}
        coef_t.append(row)
        const_t.append(_poly_terms(cexp, rgens))
    c = {'param_names': [str(s) for s in param_syms], 'solve_states': [str(s) for s in solve_states],
         'rgens': [str(g) for g in rgens], 'coef_t': coef_t, 'const_t': const_t, 'state_t': state_t,
         'jx_terms': jx_terms, 'jt_terms': jt_terms, 'gens': [str(g) for g in gens0]}
    _forward_cache[key] = c
    return c


def solve_forward_modular(model, state_names, param_names, state_vals, param_vals, prime,
                        forcings=None, solve_rates=None, keep_free=None, backend='sympy',
                        state_solve=None):
    """Solve f = 0 over GF(prime) for a turnover subset of rates, given chosen resting
    `state_vals` and `param_vals` (residues; forcings held at 0). `solve_rates` (one per
    non-forcing state outside `state_solve`) is auto-picked avoiding `keep_free` if None;
    the balances in `state_solve` are solved for their own state first. Returns the
    joint-mode payload of the backward solve plus {'rates', 'solve_rates', 'state_solve'},
    or {'ok': False, 'why'}."""
    p = int(prime)
    model = _as_list(model)
    forcings = set(_as_list(forcings))
    state_names = _as_list(state_names)
    param_names = _as_list(param_names)
    svd = {str(k): int(v) % p for k, v in dict(state_vals).items()}
    pvd = {str(k): int(v) % p for k, v in dict(param_vals).items()}
    state_solve = [] if state_solve is None else _as_list(state_solve)
    picked = solve_rates is None
    if solve_rates is None:
        (_ps, _ss, _polys, *_r) = _ss_compile(model, state_names, param_names, forcings)
        rhs_by_name = {str(_ss[i]): _polys[i] for i in range(len(_ss))}
        ss_names = [str(s) for s in _ss]
        solve_rates = _forward_rate_pick(rhs_by_name, ss_names, param_names, forcings, keep_free=keep_free)
        state_solve = []
        if solve_rates is None:
            plan = _forward_plan(rhs_by_name, ss_names, param_names, forcings, keep_free=keep_free)
            if plan is None:
                return {'ok': False, 'why': 'no complete forward rate matching'}
            state_solve, solve_rates = plan
    solve_rates = _as_list(solve_rates)
    c = _forward_compile(model, state_names, param_names, forcings, solve_rates, state_solve)
    if 'bad' in c and picked:
        # the direct pick holds two rates of one product: use the planned matching
        plan = _forward_plan(rhs_by_name, ss_names, param_names, forcings, keep_free=keep_free)
        if plan is not None:
            state_solve, solve_rates = plan
            c = _forward_compile(model, state_names, param_names, forcings, solve_rates,
                                 state_solve)
    if 'bad' in c:
        return {'ok': False, 'why': c['bad']}
    solve_states = c['solve_states']
    n_s = len(solve_states)
    n_r = len(c['coef_t'])
    if len(solve_rates) != n_r:
        return {'ok': False, 'why': 'rate/state count mismatch (%d rates, %d balances)'
                % (len(solve_rates), n_r)}
    def cvfree(nm):
        return svd[nm] if nm in svd else (pvd[nm] if nm in pvd else 0)
    rg_idx = {nm: k for k, nm in enumerate(c['rgens'])}
    rgv = [cvfree(nm) for nm in c['rgens']]
    # state-solved balances first, each x = -c0/c1 at the values known so far
    for u, c1_t, c0_t in c['state_t']:
        c1 = _eval_terms(c1_t, rgv, p)
        if c1 == 0:
            return {'ok': False, 'why': 'vanishing state pivot mod p at %s' % u}
        xu = (-_eval_terms(c0_t, rgv, p)) * pow(c1, p - 2, p) % p
        if xu == 0:
            return {'ok': False, 'why': 'zero resting state %s' % u}
        svd[u] = xu
        rgv[rg_idx[u]] = xu
    A = [[_eval_terms(c['coef_t'][i][j], rgv, p) for j in range(n_r)] for i in range(n_r)]
    b = [[(-_eval_terms(c['const_t'][i], rgv, p)) % p] for i in range(n_r)]
    X = _solve_mod(A, b, p)
    if X is None:
        return {'ok': False, 'why': 'singular forward system mod p'}
    rates_dict = {solve_rates[j]: int(X[j][0]) % p for j in range(len(solve_rates))}
    def cvfull(nm):
        if nm in rates_dict:
            return rates_dict[nm]
        return svd[nm] if nm in svd else (pvd[nm] if nm in pvd else 0)
    ptvals = [cvfull(nm) for nm in c['gens']]
    df_jx = [[_eval_terms(c['jx_terms'][i][j], ptvals, p) for j in range(n_s)] for i in range(n_s)]
    df_jt = {th: [_eval_terms(c['jt_terms'][th][i], ptvals, p) for i in range(n_s)]
             for th in c['param_names']}
    return {'ok': True, 'rates': rates_dict, 'solve_rates': list(solve_rates),
            'state_solve': list(state_solve),
            'solve_states': list(solve_states), 'val_by': {s: cvfull(s) for s in solve_states},
            'df_jx': df_jx, 'df_jt': df_jt, 'df_state_cols': list(solve_states),
            'df_param_cols': list(c['param_names'])}
