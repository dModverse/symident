"""Resting states continued as power series along a parameter line."""

import sympy as spy

from .modp import _eval_terms, _mono_nz, _poly_terms, _solve_mod
from .parse import _as_list
from .steady import _ss_compile


def _ser_mul(a, b, p, T):
    out = [0] * (T + 1)
    for i, x in enumerate(a):
        if x:
            for j in range(T + 1 - i):
                if b[j]:
                    out[i + j] = (out[i + j] + x * b[j]) % p
    return out


def _ser_inv(a, p, T):
    """Inverse of a truncated power series with a unit constant term, or None."""
    if a[0] % p == 0:
        return None
    b = [0] * (T + 1)
    b[0] = pow(a[0] % p, p - 2, p)
    for k in range(1, T + 1):
        acc = sum(a[i] * b[k - i] for i in range(1, k + 1)) % p
        b[k] = (-acc) * b[0] % p
    return b


def _ser_terms(terms, gser, p, T, pw):
    """A term list at series values of the generators, truncated at eps^T; `pw` caches
    the powers of each generator."""
    out = [0] * (T + 1)
    for expo, (cn, cd) in terms:
        c = cn % p * pow(cd % p, p - 2, p) % p
        if not c:
            continue
        acc = [c] + [0] * T
        for g, e in _mono_nz(expo):
            key = (g, e)
            if key not in pw:
                s_ = [1] + [0] * T
                for _ in range(e):
                    s_ = _ser_mul(s_, gser[g], p, T)
                pw[key] = s_
            acc = _ser_mul(acc, pw[key], p, T)
        out = [(x + y) % p for x, y in zip(out, acc)]
    return out


def _ser_value(numden, gser, p, T, pw):
    num = _ser_terms(numden[0], gser, p, T, pw)
    den = _ser_terms(numden[1], gser, p, T, pw)
    if den == [1] + [0] * T:
        return num
    inv = _ser_inv(den, p, T)
    return None if inv is None else _ser_mul(num, inv, p, T)


_cont_cache = {}


def continue_resting_state(model, state_names, param_names, param_vals, dirs, rest_vals, prime,
                         order, forcings=None):
    """Resting state of `model` as a power series in eps along the parameters
    param_vals + eps dirs, continued from rest_vals, a resting state at eps = 0, by
    Newton steps: x_k = -J0^-1 [eps^k] f(x_<k). Returns {'ok', 'val_by': {state:
    [x_0, ..., x_T]}, 'df_jx': [[series]], 'df_jt': {param: [series per balance]},
    'df_state_cols', 'df_param_cols'} or {'ok': False, 'why'}."""
    p, T = int(prime), int(order)
    model = _as_list(model)
    forcings = set(_as_list(forcings))
    state_names = _as_list(state_names)
    param_names = _as_list(param_names)
    (param_syms, solve_states, polys, Jx, Jt, gens, jx_terms, jt_terms, poly_bi, gen_lin,
     lin_terms, point_plan) = _ss_compile(model, state_names, param_names, forcings)
    key = (tuple(model), tuple(state_names), tuple(param_names), tuple(sorted(forcings)))
    cached = _cont_cache.get(key)
    if cached is None:
        f_t = [_poly_terms(pl, gens) for pl in polys]
        # Newton linearises the cleared numerators, not f
        jn_t = [[_poly_terms(spy.diff(pl, s_), gens) for s_ in solve_states] for pl in polys]
        cached = (f_t, jn_t)
        _cont_cache[key] = cached
    f_t, jn_t = cached
    pv = {str(k): int(v) % p for k, v in dict(param_vals).items()}
    dv = {str(k): int(v) % p for k, v in dict(dirs).items()}
    rv = {str(k): int(v) % p for k, v in dict(rest_vals).items()}
    n_s = len(solve_states)
    pser = [[pv.get(str(th), 0)] + [dv.get(str(th), 0)] + [0] * (T - 1) if T >= 1
            else [pv.get(str(th), 0)] for th in param_syms]
    xser = [[rv.get(str(s_), 0)] + [0] * T for s_ in solve_states]

    def f_series():
        pw = {}
        out = []
        for t in f_t:
            v = _ser_value(t, pser + xser, p, T, pw)
            if v is None:
                return None
            out.append(v)
        return out

    f0 = f_series()
    if f0 is None or any(v[0] for v in f0):
        return {'ok': False, 'why': 'the start is not a resting state at eps = 0'}
    g0 = [pv.get(str(th), 0) for th in param_syms] + [rv.get(str(s_), 0) for s_ in solve_states]
    J0 = [[_eval_terms(jn_t[i][j], g0, p) for j in range(n_s)] for i in range(n_s)]
    for k in range(1, T + 1):
        fs = f_series()
        if fs is None:
            return {'ok': False, 'why': 'a pole along the continuation'}
        X = _solve_mod(J0, [[(-fs[i][k]) % p] for i in range(n_s)], p)
        if X is None:
            return {'ok': False, 'why': 'a singular resting Jacobian'}
        for i in range(n_s):
            xser[i][k] = X[i][0] % p
    pw = {}
    gser = pser + xser

    def ser_of(terms):
        v = _ser_value(terms, gser, p, T, pw)
        return v if v is not None else [0] * (T + 1)

    df_jx = [[ser_of(jx_terms[i][j]) for j in range(n_s)] for i in range(n_s)]
    df_jt = {str(th): [ser_of(jt_terms[str(th)][i]) for i in range(n_s)] for th in param_syms}
    return {'ok': True, 'val_by': {str(solve_states[i]): xser[i] for i in range(n_s)},
            'df_jx': df_jx, 'df_jt': df_jt,
            'df_state_cols': [str(x) for x in solve_states],
            'df_param_cols': [str(x) for x in param_syms]}

