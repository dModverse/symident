"""Polynomial systems over GF(p): linear plans, dense elimination, resultants and roots."""

import random

import numpy as np
import sympy as spy
from sympy.polys import galoistools as _gft
from sympy.polys.domains import ZZ as _ZZ

from .modp import _eval_terms, _poly_terms

_LINSOLVE_OPS_CAP = 2000  # bail to the numeric solver past this elimination size


def _linear_solution(polys, solve_states):
    """Solve polys = 0 for `solve_states` by generic linear elimination, each state a
    rational function of the parameters. Returns {state: expr}, or None when a
    coupled residual remains or an expression exceeds `_LINSOLVE_OPS_CAP`. A point
    where a pivot denominator vanishes mod p is caught at evaluation."""
    rem_set = set(solve_states)
    rem_p = [spy.sympify(pl) for pl in polys]
    elim = []
    progress = True
    while progress and rem_set:
        progress = False
        for idx in range(len(rem_p)):
            pl = rem_p[idx]
            if pl is None or not (pl.free_symbols & rem_set):
                continue
            if pl.count_ops() > _LINSOLVE_OPS_CAP:
                return None
            picked = None
            for v in pl.free_symbols & rem_set:
                pv = spy.Poly(pl, v)
                if pv.degree() != 1:
                    continue
                a = pv.nth(1)
                if (a.free_symbols & rem_set) or a.is_zero:
                    continue
                picked = (v, spy.cancel(-pv.nth(0) / a))
                break
            if picked is None:
                continue
            v, expr = picked
            elim.append((v, expr))
            rem_p[idx] = None
            for j in range(len(rem_p)):
                if rem_p[j] is not None and v in rem_p[j].free_symbols:
                    sub = rem_p[j].subs(v, expr)
                    if sub.count_ops() > _LINSOLVE_OPS_CAP:
                        return None
                    rem_p[j] = spy.fraction(spy.together(sub))[0]
            rem_set.discard(v)
            progress = True
    if rem_set:
        return None
    sol = {}
    for v, expr in reversed(elim):
        sol[v] = spy.cancel(expr.subs(sol))
    return sol


def _compile_linear_plan(polys, solve_states, param_syms):
    """Compile the linear-elimination solution of polys = 0 to per-state
    prime-independent (num, den) term lists in `solve_states` order. Returns
    (True, per_state_terms) when every state eliminates linearly, else (False, None)."""
    sol = _linear_solution(polys, solve_states)
    if sol is None:
        return False, None
    return True, [(str(s), _poly_terms(sol[s], param_syms)) for s in solve_states]


def _eval_bipoly_dict(bip, paramvals, p):
    """The state polynomial at numeric parameter values mod p, as a dict
    {state-exponent-tuple: coeff}."""
    d = {}
    for sm, ct in bip:
        c = _eval_terms(ct, paramvals, p)
        if c:
            d[sm] = (d.get(sm, 0) + c) % p
    return {m: c for m, c in d.items() if c}


def _dp_mul(a, b, nv, p):
    """Product of two dict polynomials mod p."""
    if len(a) * len(b) * nv < 4096:
        pairs = ((tuple(ma[k] + mb[k] for k in range(nv)), ca * cb)
                 for ma, ca in a.items() for mb, cb in b.items())
    else:
        A = np.array(list(a), dtype=np.int64).reshape(len(a), nv)
        B = np.array(list(b), dtype=np.int64).reshape(len(b), nv)
        keys = map(tuple, (A[:, None, :] + B[None, :, :]).reshape(-1, nv).tolist())
        ca = np.array([c % p for c in a.values()], dtype=np.int64)
        cb = np.array([c % p for c in b.values()], dtype=np.int64)
        pairs = zip(keys, (ca[:, None] * cb[None, :] % p).ravel().tolist())
    out = {}
    for key, c in pairs:
        v = (out.get(key, 0) + c) % p
        if v:
            out[key] = v
        elif key in out:
            del out[key]
    return out


def _dp_subst(d, i, expr, nv, p):
    """Substitute var i -> expr (a dict poly not containing var i) into d, mod p."""
    maxe = max((m[i] for m in d), default=0)
    pows = [{tuple([0] * nv): 1}]
    for _ in range(maxe):
        pows.append(_dp_mul(pows[-1], expr, nv, p))
    out = {}
    for m, c in d.items():
        term = pows[m[i]]
        base = tuple(0 if k == i else m[k] for k in range(nv))
        for mm, cc in term.items():
            key = tuple(base[k] + mm[k] for k in range(nv))
            v = (out.get(key, 0) + c * cc) % p
            if v:
                out[key] = v
            elif key in out:
                del out[key]
    return out


_REDUCE_TERMS_CAP = 2000    # past this polynomial size msolve is faster than the reduction


def _dp_add(a, b, p):
    """Sum of two dict polynomials mod p."""
    out = dict(a)
    for m, c in b.items():
        v = (out.get(m, 0) + c) % p
        if v:
            out[m] = v
        elif m in out:
            del out[m]
    return out


def _dp_coeffs(d, i):
    """Coefficients {k: dict poly free of var i} of d as a polynomial in var i."""
    cs = {}
    for m, c in d.items():
        cs.setdefault(m[i], {})[m[:i] + (0,) + m[i + 1:]] = c
    return cs


def _dp_subst_rat(d, i, a, b, nv, p):
    """d at var i = -b/a, times a^deg_i(d): a dict polynomial free of var i."""
    cs = _dp_coeffs(d, i)
    deg = max(cs)
    one = {tuple([0] * nv): 1}
    nb = {m: (-c) % p for m, c in b.items()}
    pow_a, pow_b = [one], [one]
    for _ in range(deg):
        pow_a.append(_dp_mul(pow_a[-1], a, nv, p))
        pow_b.append(_dp_mul(pow_b[-1], nb, nv, p))
    out = {}
    for k, ck in cs.items():
        out = _dp_add(out, _dp_mul(_dp_mul(ck, pow_b[k], nv, p), pow_a[deg - k], nv, p), p)
    return out


def _dp_value(d, vals, p):
    """Value of a dict polynomial at the point `vals` (residues mod p)."""
    acc = 0
    for m, c in d.items():
        t = c
        for k, e in enumerate(m):
            if e:
                t = t * pow(vals[k], e, p) % p
        acc = (acc + t) % p
    return acc


def _reduce_system(dps, nv, p):
    """Peel outputs and eliminate linear pivots. Returns (plan, sinks, resid, core):
    plan and sinks as (var, a, b) with var = -b/a, in elimination order; resid the
    remaining balances over the core states. None past _REDUCE_TERMS_CAP."""
    dps = [dict(d) for d in dps]
    alive = [j for j in range(len(dps)) if dps[j]]
    rem = set(range(nv))
    sinks = []
    changed = True
    while changed:
        changed = False
        for i in sorted(rem):
            js = [j for j in alive if any(m[i] for m in dps[j])]
            if len(js) == 1 and max(m[i] for m in dps[js[0]]) == 1:
                cs = _dp_coeffs(dps[js[0]], i)
                sinks.append((i, cs[1], cs.get(0, {})))
                alive.remove(js[0])
                rem.discard(i)
                changed = True
                break
    plan = []
    while True:
        # max degree of each remaining state per balance, and in how many it occurs
        deg_of = {}
        for j in alive:
            dg = {}
            for m in dps[j]:
                for i in rem:
                    if m[i] > dg.get(i, 0):
                        dg[i] = m[i]
            deg_of[j] = dg
        occ = {i: sum(1 for j in alive if i in deg_of[j]) for i in rem}
        # constant pivot coefficients first, then a state's own balance, then the
        # pivot touching the fewest balances with the lowest coefficient degree
        best = None
        for j in alive:
            for i, dg in deg_of[j].items():
                if dg != 1:
                    continue
                cs = _dp_coeffs(dps[j], i)
                a = cs[1]
                key = (0 if all(not any(m) for m in a) else 1, 0 if i == j else 1,
                       max(sum(m) for m in a) + occ[i] - 1,
                       len(a) + len(cs.get(0, {})), i, j)
                if best is None or key < best[0]:
                    best = (key, j, i, a, cs.get(0, {}))
        if best is None:
            break
        _, j, i, a, b = best
        plan.append((i, a, b))
        alive.remove(j)
        rem.discard(i)
        for k in alive:
            if any(m[i] for m in dps[k]):
                dps[k] = _dp_subst_rat(dps[k], i, a, b, nv, p)
                if len(dps[k]) > _REDUCE_TERMS_CAP:
                    return None
    return plan, sinks, [dps[j] for j in alive if dps[j]], sorted(rem)


def _pow_mod(base, e, f, p):
    """base^e modulo the monic dense polynomial f over GF(p), base of lower degree than f,
    all highest degree first; square and multiply on plain integer lists."""
    n = len(f) - 1
    low = [c % p for c in f[1:]]                 # x^n = -(low) in the quotient ring

    def mulmod(a, b):
        # a, b highest degree first, padded to n; the reduced product highest degree first
        a, b = a[::-1], b[::-1]
        prod = [0] * (2 * n - 1)
        for i, ai in enumerate(a):
            if ai:
                for j, bj in enumerate(b):
                    prod[i + j] += ai * bj
        for k in range(2 * n - 2, n - 1, -1):    # reduce the degrees >= n from the top
            c = prod[k] % p
            if c:
                for j, lj in enumerate(low):
                    prod[k - 1 - j] -= c * lj
        return [prod[i] % p for i in range(n)][::-1]

    res = [0] * (n - 1) + [1]
    b = ([0] * n + [c % p for c in base])[-n:]
    while e:
        if e & 1:
            res = mulmod(res, b)
        e >>= 1
        if e:
            b = mulmod(b, b)
    return _gft.gf_strip(res)


def _x_pow_mod(f, p):
    """x^p modulo the monic dense polynomial f (highest degree first) over GF(p)."""
    if len(f) == 2:
        return [(-f[1]) % p]
    return _pow_mod([1, 0], p, f, p)


def _split_roots(g, p, a=1):
    """The roots of the monic g over GF(p), a product of distinct linear factors: degree 2
    by the quadratic formula, higher degrees split by gcd(g, (x + a)^((p - 1)/2) - 1)."""
    from ..gfp import sqrt_mod
    if len(g) == 2:
        return [(-g[1]) % p]
    if len(g) == 3:
        b, c = g[1], g[2]
        r = sqrt_mod((b * b - 4 * c) % p, p)
        if r is None:
            return []
        i2 = pow(2, p - 2, p)
        return [((-b + r) * i2) % p, ((-b - r) * i2) % p]
    while True:
        h = _gft.gf_sub(_pow_mod([1, a % p], (p - 1) // 2, g, p), [1], p, _ZZ)
        d = _gft.gf_gcd(g, h, p, _ZZ)
        if 1 < len(d) < len(g):
            q = _gft.gf_quo(g, d, p, _ZZ)
            return _split_roots(_gft.gf_monic(d, p, _ZZ)[1], p, a + 1) + \
                _split_roots(_gft.gf_monic(q, p, _ZZ)[1], p, a + 1)
        a += 1


def _gf_roots(f, p):
    """Distinct nonzero roots in GF(p), ascending, of the dense univariate polynomial
    f (highest degree first)."""
    f = _gft.gf_strip([int(c) % p for c in f])
    if len(f) <= 1:
        return []
    f = _gft.gf_monic(f, p, _ZZ)[1]
    xp = _x_pow_mod(f, p) if len(f) <= 9 else _gft.gf_pow_mod([1, 0], p, f, p, _ZZ)
    h = _gft.gf_sub(xp, [1, 0], p, _ZZ)
    g = _gft.gf_gcd(f, h, p, _ZZ)
    if len(g) <= 1:
        return []
    if len(g) <= 9:
        return sorted(r for r in _split_roots(_gft.gf_monic(g, p, _ZZ)[1], p) if r)
    lin = _gft.gf_edf_zassenhaus(g, 1, p, _ZZ)
    return sorted(r for r in ((-fac[1]) % p for fac in lin) if r)


def _gf_resultant(f, g, p):
    """Resultant of two dense univariate polynomials over GF(p) by Euclid."""
    f, g = _gft.gf_strip(f), _gft.gf_strip(g)
    if not f or not g:
        return 0
    res = 1
    while True:
        df, dg = len(f) - 1, len(g) - 1
        if dg == 0:
            return res * pow(g[0], df, p) % p
        r = _gft.gf_rem(f, g, p, _ZZ)
        if not r:
            return 0
        if df % 2 and dg % 2:
            res = -res
        res = res * pow(g[0], df - (len(r) - 1), p) % p
        f, g = g, r


def _gf_interpolate(xs, ys, p):
    """Dense polynomial (highest degree first) through the points (xs, ys) over GF(p),
    by Newton's divided differences."""
    n = len(xs)
    c = list(ys)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            c[i] = (c[i] - c[i - 1]) * pow((xs[i] - xs[i - j]) % p, p - 2, p) % p
    poly = [c[-1]]
    for i in range(n - 2, -1, -1):
        poly = _gft.gf_add(_gft.gf_mul(poly, [1, (-xs[i]) % p], p, _ZZ), [c[i]], p, _ZZ)
    return _gft.gf_strip(poly)


def _bi_at_u(P, a, p):
    """Dense polynomial in v of the bivariate dict polynomial P{(du, dv): c} at u = a."""
    dv = max(k[1] for k in P)
    out = [0] * (dv + 1)
    for (i, j), c in P.items():
        out[dv - j] = (out[dv - j] + c * pow(a, i, p)) % p
    return _gft.gf_strip(out)


def _core_candidates(resid, core, p):
    """Candidate values (tuples aligned with `core`) of the core states, or None when
    the core has more than two states."""
    if not core:
        return [()] if not resid else []
    if len(core) > 2 or len(resid) != len(core):
        return None
    if len(core) == 1:
        i = core[0]
        g = None
        for d in resid:
            q = [0] * (max(m[i] for m in d) + 1)
            for m, c in d.items():
                q[len(q) - 1 - m[i]] = c
            q = _gft.gf_strip(q)
            g = q if g is None else _gft.gf_gcd(g, q, p, _ZZ)
        return [(r,) for r in _gf_roots(g, p)]
    # two states (u, v): the resultant in v by evaluation at u = a and interpolation.
    # A common factor g of the balances (a cleared denominator) is divided out at each
    # a through the gcd in v; the values stay polynomial in a (times a power of lc_v g)
    iu, iv = core
    P1, P2 = [{(m[iu], m[iv]): c for m, c in d.items()} for d in resid]
    if max(k[1] for k in P1) < 1 or max(k[1] for k in P2) < 1:
        return None
    bound = max(sum(k) for k in P1) * max(sum(k) for k in P2)
    rng = random.Random(p)
    dg = min(len(_gft.gf_gcd(_bi_at_u(P1, a, p), _bi_at_u(P2, a, p), p, _ZZ)) - 1
             for a in (rng.randrange(1, p), rng.randrange(1, p)))
    n1, n2 = max(k[1] for k in P1), max(k[1] for k in P2)
    xs, ys = [], []
    a = 0
    while len(xs) <= bound:
        a += 1
        f1, f2 = _bi_at_u(P1, a, p), _bi_at_u(P2, a, p)
        if len(f1) - 1 != n1 or len(f2) - 1 != n2:
            continue
        G = _gft.gf_gcd(f1, f2, p, _ZZ)
        if len(G) - 1 > dg:
            r = 0                      # a common root on top of g: a root of the resultant
        else:
            r = _gf_resultant(_gft.gf_quo(f1, G, p, _ZZ), _gft.gf_quo(f2, G, p, _ZZ), p)
        xs.append(a)
        ys.append(r)
    res = _gf_interpolate(xs, ys, p)
    if not res:
        return None
    out = []
    for a in _gf_roots(res, p):
        G = _gft.gf_gcd(_bi_at_u(P1, a, p), _bi_at_u(P2, a, p), p, _ZZ)
        out.extend((a, b) for b in _gf_roots(G, p))
    return out


def _solve_states_reduced(dps, solve_states, p):
    """Interior point of f = 0 over GF(p) by _reduce_system and a resultant on the
    core. Returns (sol, None), (None, fail-dict) when no candidate is an interior
    point, or None when the core is too large for this solve."""
    nv = len(solve_states)
    red = _reduce_system(dps, nv, p)
    if red is None:
        return None
    plan, sinks, resid, core = red
    try:
        cands = _core_candidates(resid, core, p)
    except Exception:
        return None
    if cands is None:
        return None
    steps = list(reversed(plan)) + list(reversed(sinks))
    for cp in cands:
        vals = [0] * nv
        for i, x in zip(core, cp):
            vals[i] = x
        ok = True
        for i, a, b in steps:
            av = _dp_value(a, vals, p)
            if av == 0:
                ok = False
                break
            vals[i] = (-_dp_value(b, vals, p)) * pow(av, p - 2, p) % p
            if vals[i] == 0:
                ok = False
                break
        if ok and all(_dp_value(d, vals, p) == 0 for d in dps):
            return {solve_states[i]: spy.Integer(vals[i]) for i in range(nv)}, None
    return None, {'ok': False, 'why': 'no consistent interior point'}
