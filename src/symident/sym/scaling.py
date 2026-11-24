"""Scaling symmetries from the integer kernel of the monomial exponents."""

import functools
import math
from fractions import Fraction

import sympy as spy

from .. import state
from ..errors import SYMPY_FAILS, InapplicableError, sympy_fail
from .charts import _exp_atom, _hyp_to_exp
from .linalg import exact_nullspace
from .parse import _as_list, _clean, _make_local_parse
from .recast import _TRIG, _TRIG_RECIPROCAL


def _ring_elem(x, R, gen):
    """x as an element of the polynomial ring R, None when it is not a polynomial in the symbols it reads."""
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
    """The terms of the expanded polynomial x, each as a tuple of powers, by sparse arithmetic where x is a
    polynomial."""
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
    """The terms of the expanded numerator and denominator of expr, each as a tuple of powers."""
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
                    raise InapplicableError(
                        "not a monomial in the coordinates: %s" % spy.Mul(*[b**e for b, e in powers])
                    )
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
            return spy.Dummy("exp")
        if isinstance(e, _TRIG) or type(e) in _TRIG_RECIPROCAL:
            exps.append(rec(e.args[0]))
            return spy.Dummy("trig")
        if logs and isinstance(e, spy.log):
            exps.append(rec(e.args[0]))
            return spy.Dummy("log")
        return e.func(*[rec(a) for a in e.args])

    return rec(_hyp_to_exp(expr)), exps


def _power_atoms(exprs, states, frozen):
    """{x**e: fresh symbol} for powers of a coordinate x with a non-numeric exponent e free of the states; under a
    scaling such a power has the weight e times the weight of x."""
    out = {}
    for e in exprs:
        for at in spy.sympify(e).atoms(spy.Pow):
            if at in out or at.exp.is_number or not at.base.is_Symbol or str(at.base) in frozen:
                continue
            if at.exp.free_symbols & states:
                continue
            out[at] = spy.Symbol("_P%d_%s" % (len(out), at.base))
    return out


def _scaling_rows(diff_equations, obs_functions, m, zvars, logs=False, seen=None):
    """(rows, skipped): sparse monomial-exponent rows of one (f, g) system for the scaling kernel."""
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
    obs_functions = [s[0] for s in split[: len(obs_new)]] + [u for s in split for u in s[1]]
    eq_split = {i: s[0] for i, s in zip(eq_new, split[len(obs_new) :])}

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
    return rows, skipped


def _materialize_rows(rows, ncols):
    """Dense matrix (list of lists) from sparse {col: coeff} rows."""
    out = []
    for r in rows:
        dense = [0] * ncols
        for c, v in r.items():
            dense[c] = v
        out.append(dense)
    return out


def _scaling_gens(rows, nz):
    """Primitive integer generators of the scaling kernel over the nz weight columns."""
    basis = exact_nullspace(rows, nz) if rows else []
    cvecs = [[v[j] for j in range(nz)] for v in basis if any(v[j] != 0 for j in range(nz))]
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
        return ds[0] if None not in ds and len({spy.expand(d) for d in ds}) == 1 else None
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
        # (lam^d b)^x = lam^(d x) b^x for lam > 0 on the principal branch
        return None if db is None or _weighted_degree(x, w) != 0 else spy.expand(db * x)
    if e.args:
        return 0 if all(_weighted_degree(a, w) == 0 for a in e.args) else None
    return None


def verify_scalings(scalings, tapes, shifts=()):
    """One bool per scaling: whether it leaves every tape's equations, observables, values and events invariant.

    Weights may be symbolic in coordinates the scaling leaves fixed. A coordinate in `shifts` (the logarithm of a
    scaled coordinate) is translated, not scaled: it may occur only in the constant equation and value of itself."""
    lines = []
    for t in tapes:
        lines += _as_list(t.get("model_lines", [])) + _as_list(t.get("obs_lines", []))
        lines += _as_list(t.get("ver_ic", [])) + _as_list(t.get("ver_rest", []))
        lines += ["_ = %s" % e["value"] for e in _as_list(t.get("ver_ev", [])) + _as_list(t.get("ver_t0", []))]
        if t.get("ver_time"):
            lines.append("_ = %s" % t["ver_time"])
    _, parse = _make_local_parse(lines)

    def split(line):
        lhs, rhs = _clean(line).split("=", 1)
        return lhs.strip(), spy.sympify(parse(rhs))

    checks = []  # (expression, target: state name, None for 0)
    for t in tapes:
        for l in _as_list(t.get("model_lines", [])) + _as_list(t.get("ver_ic", [])) + _as_list(t.get("ver_rest", [])):
            if "=" in l:
                x, e = split(l)
                checks.append((e, x))
        for l in _as_list(t.get("obs_lines", [])):
            if "=" in l:
                checks.append((split(l)[1], None))
        for ev in _as_list(t.get("ver_ev", [])) + _as_list(t.get("ver_t0", [])):
            e = spy.sympify(parse(str(ev["value"])))
            checks.append((e, None if str(ev["method"]) == "multiply" else str(ev["var"])))
        if t.get("ver_time"):
            checks.append((spy.sympify(parse(t["ver_time"])), None))
    # each distinct check once, with the symbols of the check; one free of the support has degree 0
    checks = list(dict.fromkeys(checks))
    syms = [{str(x) for x in e.free_symbols} for e, _ in checks]
    out = []
    shifts = set(shifts)
    for sc in scalings:
        w = {str(k): spy.sympify(parse(str(v))) for k, v in dict(sc).items()}
        sup = {k for k, v in w.items() if v != 0}
        moved = sup & shifts
        ok = all(not ({str(x) for x in w[k].free_symbols} & sup) for k in sup)
        for (e, target), es in zip(checks, syms):
            if not ok:
                break
            if e == 0:  # zero has every weight
                continue
            if es & moved:
                if target in moved and e.is_Symbol and str(e) == target:
                    continue
                ok = False
                break
            d = _weighted_degree(e, w) if es & sup else 0
            if d is None or spy.expand(d - (w.get(target, 0) if target else 0)) != 0:
                ok = False
                if state.env("LIEDIAG"):
                    print("[liediag] scaling fails at %s -> %s: degree %s" % (target, str(e)[:120], d), flush=True)
                break
        out.append(ok)
    return out


def _scaling_nonid(gens, znames, nz):
    """Scaling entries (support, integer vector, type) from generators."""
    non_id = []
    for c in gens:
        comp = {znames[j]: c[j] for j in range(nz) if c[j] != 0}
        non_id.append(
            {"support": sorted(comp.keys()), "vector": {k: str(v) for k, v in comp.items()}, "type": "scaling"}
        )
    return non_id


def _scaling_gens_recast(gens, znames, nz, recast):
    """Generators of span(gens) with c_E = exp * c_base imposed, weights possibly symbolic in exp."""
    if not gens:
        return []
    G = spy.Matrix(gens)  # rows = basis vectors, cols = coords
    idx = {nm: j for j, nm in enumerate(znames)}
    rel_rows = []
    for rc in recast:
        E, base, exp = str(rc["E"]), str(rc["base"]), str(rc["exp"])
        if E not in idx or base not in idx:
            continue
        exp_sym = spy.sympify(exp)
        j_e, j_b = idx[E], idx[base]
        # (G[:,E] - exp*G[:,base]) . alpha = 0 over the basis coefficients alpha
        rel_rows.append([G[i, j_e] - exp_sym * G[i, j_b] for i in range(G.rows)])
    if not rel_rows:
        return [[spy.Integer(x) for x in g] for g in gens]
    alphas = spy.Matrix(rel_rows).nullspace()  # over Q(exp)
    out = []
    for a in alphas:
        v = [spy.together(sum(a[i] * G[i, j] for i in range(G.rows))) for j in range(nz)]
        # L = log(base) shifts by c_base*log(lam) under the scaling; the kernel above
        # leaves it at 0, the joint nullspace contains the shift
        for rc in recast:
            base, L = str(rc["base"]), str(rc.get("L"))
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


def scaling_symmetries_per_condition(
    per_cond_model, per_cond_obs, inputs=None, fixed=None, extra_model=None, extra_obs=None
):
    """Scalings whose state weights may differ between conditions, as {'params': {name: w}, 'states': [{state: w}]}
    with one state dict per condition and integer weights.

    The rows of condition c over the states of c and the shared parameters are [A_c | B_c]. A parameter weight p
    extends to the states of condition c exactly when B_c p lies in the column space of A_c, i.e. when n B_c p = 0
    for every n with n A_c = 0; the state weights s_c then solve A_c s_c = -B_c p. A condition whose states do not
    move keeps weight 0 on them."""
    per_cond_model = [_as_list(m) for m in per_cond_model]
    per_cond_obs = [_as_list(o) for o in per_cond_obs]
    K = len(per_cond_model)
    extra_model = [_as_list(m) for m in (extra_model or [[] for _ in range(K)])]
    extra_obs = [_as_list(o) for o in (extra_obs or [[] for _ in range(K)])]
    if K < 2:
        return []
    inputset, fixedset = set(_as_list(inputs)), set(_as_list(fixed))
    local, parse = _make_local_parse([l for ls in per_cond_model + per_cond_obs + extra_model + extra_obs for l in ls])
    state_names = [_clean(l).split("=", 1)[0].strip() for l in per_cond_model[0]]
    state_syms = [local.get(nm, spy.Symbol(nm)) for nm in state_names]
    m = len(state_syms)
    systems, paramset = [], set()
    for c in range(K):
        rhs = {}
        for l in per_cond_model[c]:
            lhs, expr = _clean(l).split("=", 1)
            rhs[lhs.strip()] = spy.sympify(parse(expr))
        if set(rhs) != set(state_names):
            return []
        f = [rhs[nm] for nm in state_names]
        g = [spy.sympify(parse(_clean(l).split("=", 1)[1])) for l in per_cond_obs[c]]
        xf = []
        for l in extra_model[c]:
            lhs, expr = _clean(l).split("=", 1)
            if lhs.strip() in state_names:
                xf.append((state_names.index(lhs.strip()), spy.sympify(parse(expr))))
        xg = [spy.sympify(parse(_clean(l).split("=", 1)[1])) for l in extra_obs[c]]
        for e in f + g + [e for _, e in xf] + xg:
            paramset |= set(e.free_symbols)
        systems.append((f, g, xf, xg))
    paramset -= set(state_syms)
    paramset -= {local.get(nm, spy.Symbol(nm)) for nm in inputset}
    params = sorted(paramset, key=spy.default_sort_key)
    zvars = state_syms + params
    nz, npar = len(zvars), len(params)
    blocks = []
    for f, g, xf, xg in systems:
        seen, rows = set(), []
        sparse, _ = _scaling_rows(f, g, m, zvars, False, seen)
        rows += sparse
        pending = list(xf)
        while pending:
            fx, rest, used = [spy.Integer(0)] * m, [], set()
            for i, e in pending:
                if i in used:
                    rest.append((i, e))
                else:
                    fx[i] = e
                    used.add(i)
            pending = rest
            rows += _scaling_rows(fx, [], m, zvars, False, seen)[0]
        if xg:
            rows += _scaling_rows([spy.Integer(0)] * m, xg, m, zvars, False, seen)[0]
        M = spy.Matrix(_materialize_rows(rows, nz)) if rows else spy.zeros(0, nz)
        blocks.append((M[:, :m], M[:, m:]))
    cons = []
    for A, B in blocks:
        if A.rows == 0:
            continue
        # without states every row constrains the parameters directly
        lefts = A.T.nullspace() if m else [spy.eye(A.rows).col(i) for i in range(A.rows)]
        for n in lefts:
            row = n.T * B
            if any(x != 0 for x in row):
                cons.append([spy.Rational(x) for x in row])
    for j, p in enumerate(params):
        if str(p) in fixedset:
            cons.append([1 if k == j else 0 for k in range(npar)])
    out = []
    for pv in _scaling_gens(cons, npar) if npar else []:
        p = spy.Matrix([spy.Rational(x) for x in pv])
        sts, scale = [], 1
        for A, B in blocks:
            if A.rows == 0 or not m:
                sts.append([spy.Integer(0)] * m)
                continue
            try:
                sol, free = A.gauss_jordan_solve(-B * p)
            except ValueError:
                sts = None
                break
            sol = sol.subs({t: 0 for t in free})
            sts.append(list(sol))
            for x in sol:
                scale = spy.ilcm(scale, spy.Rational(x).q)
        if sts is None:
            continue
        out.append(
            {
                "params": {str(params[j]): int(pv[j] * scale) for j in range(npar) if pv[j] != 0},
                "states": [{state_names[i]: int(s[i] * scale) for i in range(m) if s[i] != 0} for s in sts],
            }
        )
    return out


def scaling_symmetries_multi(
    per_cond_model,
    per_cond_obs,
    inputs=None,
    fixed=None,
    recast=None,
    logs=False,
    extra_model=None,
    extra_obs=None,
    powers=False,
):
    """Scaling symmetries common to every condition; `inputs` and `fixed` do not scale. With `powers`, a power of a
    coordinate with a free exponent has the weight of the exponent times that of the coordinate."""
    per_cond_model = [_as_list(m) for m in per_cond_model]
    per_cond_obs = [_as_list(o) for o in per_cond_obs]
    extra_model = [_as_list(m) for m in (extra_model or [[] for _ in per_cond_model])]
    extra_obs = [_as_list(o) for o in (extra_obs or [[] for _ in per_cond_model])]
    inputset = set(_as_list(inputs))
    fixedset = set(_as_list(fixed))
    K = len(per_cond_model)
    if K == 0:
        return {"method": "scaling", "count": 0, "non_identifiable": [], "coordinates": []}

    all_lines = [l for lines in per_cond_model + per_cond_obs + extra_model + extra_obs for l in lines]
    local, parse = _make_local_parse(all_lines)

    state_names = [_clean(l).split("=", 1)[0].strip() for l in per_cond_model[0]]
    state_syms = [local.get(nm, spy.Symbol(nm)) for nm in state_names]

    per_f, per_g, paramset = [], [], set()
    for c in range(K):
        rhs = {}
        for l in per_cond_model[c]:
            lhs, expr = _clean(l).split("=", 1)
            rhs[lhs.strip()] = spy.sympify(parse(expr))
        f = [rhs[nm] for nm in state_names]
        g = [spy.sympify(parse(_clean(l).split("=", 1)[1])) for l in per_cond_obs[c]]
        per_f.append(f)
        per_g.append(g)
        for e in f + g:
            paramset |= set(spy.sympify(e).free_symbols)
    per_xf, per_xg = [], []
    for c in range(K):
        xf = []
        for l in extra_model[c]:
            lhs, expr = _clean(l).split("=", 1)
            if lhs.strip() in state_names:
                e = spy.sympify(parse(expr))
                xf.append((state_names.index(lhs.strip()), e))
                paramset |= set(e.free_symbols)
        xg = [spy.sympify(parse(_clean(l).split("=", 1)[1])) for l in extra_obs[c]]
        for e in xg:
            paramset |= set(e.free_symbols)
        per_xf.append(xf)
        per_xg.append(xg)

    paramset -= set(state_syms)
    paramset -= {local.get(nm, spy.Symbol(nm)) for nm in inputset}
    recast = list(_as_list(recast))
    powers = (
        _power_atoms(
            [e for f in per_f + per_g + per_xg for e in f] + [e for xf in per_xf for _, e in xf],
            set(state_syms),
            inputset | fixedset,
        )
        if powers
        else {}
    )
    if powers:

        def sub(e):
            return e.xreplace(powers) if e.has(*powers) else e

        per_f = [[sub(e) for e in f] for f in per_f]
        per_g = [[sub(e) for e in g] for g in per_g]
        per_xf = [[(i, sub(e)) for i, e in xf] for xf in per_xf]
        per_xg = [[sub(e) for e in xg] for xg in per_xg]
        paramset |= set(powers.values())
        recast += [{"E": str(P), "base": str(at.base), "exp": str(at.exp)} for at, P in powers.items()]
    params = sorted(paramset, key=spy.default_sort_key)
    # re-add free-exponent bases the recast eliminated, so _scaling_gens_recast can
    # impose c_E = exp*c_base on them
    for rc in recast:
        b = local.get(str(rc["base"]), spy.Symbol(str(rc["base"])))
        if b not in set(state_syms) and b not in params and str(b) not in inputset and str(b) not in fixedset:
            params.append(b)
    params = sorted(params, key=spy.default_sort_key)
    zvars = state_syms + params
    nz = len(zvars)
    znames = [str(s) for s in zvars]
    m = len(state_syms)

    rows, skipped, seen = [], 0, set()
    for c in range(K):
        sparse, sk = _scaling_rows(per_f[c], per_g[c], m, zvars, logs, seen)
        rows.extend(sparse)
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
            sparse, sk = _scaling_rows(fx, [], m, zvars, logs, seen)
            rows.extend(sparse)
            skipped += sk
        if per_xg[c]:
            sparse, sk = _scaling_rows([spy.Integer(0)] * m, per_xg[c], m, zvars, logs, seen)
            rows.extend(sparse)
            skipped += sk
    if skipped and state.env("LIEDIAG"):
        print(
            "[liediag] scaling_symmetries_multi: %d non-polynomial term(s) skipped "
            "(a scaling they would forbid may be over-reported)" % skipped
        )
    dense = _materialize_rows(rows, nz)
    # a coordinate in a free exponent does not scale: base^e is a monomial only for e invariant
    in_exp = {str(x) for at in powers for x in at.exp.free_symbols}
    for j in range(nz):
        if znames[j] in fixedset or znames[j] in in_exp:
            row = [0] * nz
            row[j] = 1
            dense.append(row)

    gens = _scaling_gens(dense, nz)
    # the integer kernel treats each recast E as free; recover the symbolic-weight scalings
    if recast:
        physical = _scaling_gens_recast(gens, znames, nz, recast)
        non_id = _scaling_nonid(physical, znames, nz)
    else:
        non_id = _scaling_nonid(gens, znames, nz)
    aux = {str(P) for P in powers.values()}
    if aux:
        for d in non_id:
            d["vector"] = {k: v for k, v in d["vector"].items() if k not in aux}
            d["support"] = [k for k in d["support"] if k not in aux]
        non_id = [d for d in non_id if d["vector"]]
    return {
        "method": "scaling",
        "count": len(non_id),
        "non_identifiable": non_id,
        "coordinates": [z for z in znames if z not in aux],
        "skipped": skipped,
    }
