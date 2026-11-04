"""Conserved moieties of a reaction network and states forced to zero at steady state."""

import math
from fractions import Fraction

import numpy as np
import sympy as spy

from ..errors import SYMPY_FAILS, sympy_fail
from ..rexpr import eval_numeric, get_symbols, replace_symbols
from ..sym.parse import _is_rational_expr, _make_local_parse


def _coef_matrix(totals, states):
    stset = set(states)
    sets = [[s for s in get_symbols(e) if s in stset] for e in totals.values()]
    species = list(dict.fromkeys(s for st in sets for s in st))
    # parameters in the coefficients, such as volumes, at a generic positive point
    others = sorted({s for e in totals.values() for s in get_symbols(e) if s not in stset})
    at = {o: 1.0 + 0.618 * (k + 1) for k, o in enumerate(others)}
    C = np.zeros((len(totals), len(species)))
    for i, e in enumerate(totals.values()):
        env0 = {s: 0.0 for s in species}
        env0.update(at)
        base = eval_numeric(e, env0)
        for s in sets[i]:
            env1 = dict(env0)
            env1[s] = 1.0
            C[i, species.index(s)] = eval_numeric(e, env1) - base
    return C, species, sets


def pivot_decomposition(totals, states, parameters=(), avoid=(), prefer=()):
    """One pivot species per conserved total, chosen by partial elimination."""
    names = list(totals)
    C, species, sets = _coef_matrix(totals, states)
    Cr = C.copy()
    pivots = [None] * len(names)
    avoid, prefer, parameters = set(avoid), list(prefer), set(parameters)
    for i in range(len(names)):
        cand = [s for s in sets[i] if s not in set(p for p in pivots if p)]
        if len(cand) < 2:
            continue
        nz = [s for s in cand if abs(Cr[i, species.index(s)]) > 1e-12]
        if not nz:
            continue
        keep = [s for s in nz if s not in avoid]
        if keep:
            nz = keep
        pref = [s for s in prefer if s in nz]
        if pref:
            e = sorted(pref)[0]
        else:
            future = {s for st in sets[i + 1:] for s in st}
            safe = [s for s in nz if s not in future]
            pool = safe if safe else nz
            nu = [s for s in pool if s not in parameters]
            e = sorted(nu)[0] if nu else sorted(pool)[0]
        if i < len(names) - 1:
            ce = species.index(e)
            piv = Cr[i, ce]
            for j in range(i + 1, len(names)):
                if abs(Cr[j, ce]) > 1e-12:
                    Cr[j] = Cr[j] - (Cr[j, ce] / piv) * Cr[i]
        pivots[i] = e
    return {"C": C, "species": species, "pivots": pivots, "sets": sets}


def _coef_expr(total, species, s):
    """The coefficient of species `s` in the linear `total`, an expression in its other symbols."""
    rest = [x for x in species if x != s]
    one = replace_symbols(rest + [s], ["0"] * len(rest) + ["1"], total)
    zero = replace_symbols(rest + [s], ["0"] * (len(rest) + 1), total)
    try:
        _, parse = _make_local_parse([one, zero])
        c = spy.simplify(parse(one.replace("^", "**")) - parse(zero.replace("^", "**")))
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    return None if c.free_symbols == set() else str(c).replace("**", "^")


def _num(x):
    x = float(x)
    if x == int(x):
        return str(int(x))
    return repr(x)


def substitute_cq(totals, f, states, parameters, avoid=()):
    """Each conserved total eliminates one pivot species, expressed through a `total` parameter."""
    parameters = list(parameters or [])
    if not totals:
        return {"f": f, "parameters": parameters, "cq_info": [], "elim_states": []}
    dec = pivot_decomposition(totals, states, parameters, avoid)
    names = list(totals)
    subs, cq_info, elim = {}, [], []
    for i, nm in enumerate(names):
        e = dec["pivots"][i]
        if e is None:
            continue
        coef = dec["C"][i, dec["species"].index(e)]
        rest = f"({replace_symbols([e], ['0'], totals[nm])})"
        held = _coef_expr(totals[nm], dec["sets"][i], e)
        if held is not None:
            recon = f"(({nm} - {rest}) / ({held}))"
        elif abs(coef - 1) < 1.5e-8:
            recon = f"({nm} - {rest})"
        else:
            recon = f"(({nm} - {rest}) / ({_num(coef)}))"
        subs[e] = recon
        elim.append(e)
        if nm not in parameters:
            parameters.append(nm)
        cq_info.append({"total_name": nm, "elim_state": e, "recon_expr": recon})
    keys, vals = list(subs), list(subs.values())
    for _ in range(len(subs)):
        f_new = replace_symbols(keys, vals, f)
        if f_new == f:
            break
        f = f_new
    for ci in cq_info:
        rec = ci["recon_expr"]
        for _ in range(len(subs)):
            rec_new = replace_symbols(keys, vals, rec)
            if rec_new == rec:
                break
            rec = rec_new
        ci["recon_expr"] = rec
    return {"f": f, "parameters": parameters, "cq_info": cq_info, "elim_states": elim}


def check_conserved(totals, f):
    """ValueError for a total whose time derivative under the right-hand sides `f` does not vanish."""
    try:
        _, parse = _make_local_parse([f"{k} = {v}" for k, v in f.items()] + [str(v) for v in totals.values()])
        rhs = {spy.Symbol(k): parse(str(v).replace("^", "**")) for k, v in f.items()}
        for name, e in totals.items():
            t = parse(str(e).replace("^", "**"))
            d = sum((spy.diff(t, x) * r for x, r in rhs.items() if x in t.free_symbols), spy.Integer(0))
            if spy.cancel(spy.together(d)) != 0:
                raise ValueError(f"the total {name} = {e} is not conserved: its time derivative under "
                                 f"the right-hand sides is {str(d).replace('**', '^')}, not zero")
    except SYMPY_FAILS as exc:
        sympy_fail(exc)


def conserved_totals(smatrix, species):
    """Conserved quantities of a stoichiometry (rows reactions, columns species) as totals."""
    from sympy.polys.domains import QQ
    from sympy.polys.matrices import DomainMatrix
    n = len(species)
    rows = [[QQ(_exact(x).numerator, _exact(x).denominator) for x in r] for r in smatrix]
    if not rows:
        return {}
    red, piv = DomainMatrix(rows, (len(rows), n), QQ).to_sparse().rref()
    R = red.rep
    out = {}
    for i, free in enumerate(c for c in range(n) if c not in set(piv)):
        # the basis of the nullspace from the reduced row echelon form, one vector per free column
        v = [Fraction(0)] * n
        v[free] = Fraction(1)
        for r, pc in enumerate(piv):
            x = R.get(r, {}).get(free)
            if x:
                v[pc] = -Fraction(int(QQ.numer(x)), int(QQ.denom(x)))
        v = [c * math.lcm(*(c.denominator for c in v)) for c in v]
        terms = [s if c == 1 else f"{int(c)}*{s}" for s, c in zip(species, v) if c]
        out[f"total{i + 1}"] = " + ".join(terms)
    return out


def _exact(x):
    """A stoichiometric coefficient as a rational; a float by its shortest decimal."""
    if isinstance(x, (int, Fraction)):
        return Fraction(x)
    if isinstance(x, (float, np.floating)):
        return Fraction(repr(float(x)))
    return Fraction(str(x))


def _expr_is_zero(expr, dead):
    """Whether `expr` vanishes with the symbols in `dead` at zero, decided exactly; undecided counts as nonzero."""
    try:
        _, parse = _make_local_parse([str(expr)])
        e = parse(str(expr))
        at = {x: 0 for x in e.free_symbols if str(x) in dead}
        if _is_rational_expr(e):
            num, den = spy.fraction(spy.cancel(spy.together(e)))
            return spy.expand(num.xreplace(at)) == 0 and spy.expand(den.xreplace(at)) != 0
        return spy.simplify(e.xreplace(at)) == 0
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return False


def _exact_lp(M, c):
    """A vertex minimizing c.w over M w <= 0, 0 <= w <= 1, by the simplex method over Q with Bland's rule."""
    m, n = len(M), len(c)
    zero, one = Fraction(0), Fraction(1)
    # columns: w, then the slacks of M w <= 0, then those of w <= 1, then the right-hand side
    T = [list(row) + [one if k == i else zero for k in range(m)] + [zero] * (n + 1) for i, row in enumerate(M)]
    T += [[one if k == j else zero for k in range(n)] + [zero] * m + [one if k == j else zero for k in range(n)] + [one]
          for j in range(n)]
    basis = list(range(n, 2 * n + m))
    z = list(c) + [zero] * (m + n + 1)
    while True:
        e = next((k for k in range(2 * n + m) if z[k] < 0), None)
        if e is None:
            break
        # every variable is bounded, so some row limits the entering one
        l = min((i for i in range(m + n) if T[i][e] > 0), key=lambda i: (T[i][-1] / T[i][e], basis[i]))
        piv = T[l][e]
        T[l] = [x / piv for x in T[l]]
        nz = [k for k, x in enumerate(T[l]) if x]
        for row in T + [z]:
            f = row[e]
            if f and row is not T[l]:
                for k in nz:
                    row[k] -= f * T[l][k]
        basis[l] = e
    w = [zero] * n
    for i, b in enumerate(basis):
        if b < n:
            w[b] = T[i][-1]
    return w


def _sink_cluster(M):
    """(zero, branches): the sole positive-weight reactant of each drained reaction, the reactants where several."""
    n_f, n_s = M.shape
    if n_f == 0 or n_s == 0:
        return [], []
    Mq = [[Fraction(float(a)) for a in row] for row in M]
    w = _exact_lp(Mq, [sum(row[j] for row in Mq) for j in range(n_s)])
    flux = [sum(a * x for a, x in zip(row, w)) for row in Mq]
    # a drained reaction lowers a weight that no reaction raises
    drained = [[j for j in range(n_s) if w[j] > 0 and Mq[i][j] < 0] for i, f in enumerate(flux) if f < 0]
    drained = [r for r in drained if r]
    zero = sorted({r[0] for r in drained if len(r) == 1})
    return zero, [r for r in drained if len(r) > 1 and not set(r) & set(zero)]


def _zero_states(smatrix, species, rates, forcings):
    """(zero, branches): the states forced to zero at rest with the forcings at zero, and _sink_cluster() branches."""
    S = np.nan_to_num(np.asarray(smatrix, dtype=float))
    if S.size == 0:
        return [], []
    rates = [str(r) for r in rates]
    vs = [set(get_symbols(r)) for r in rates]
    allv = set().union(*vs) if vs else set()
    dead = [f for f in forcings if f in set(species) | allv]
    live = [True] * len(rates)
    check = list(range(len(rates)))
    branches = []
    while True:
        for i in check:
            live[i] = not _expr_is_zero(rates[i], set(dead))
        L = np.array(live, dtype=bool)
        consumed = [sp for j, sp in enumerate(species) if np.any(S[L, j] < 0)]
        alive = [s for s in species if s not in dead]
        newdead = []
        for sp in [s for s in alive if s in consumed]:
            j = species.index(sp)
            prod = np.nonzero(S[:, j] > 0)[0]
            # consumed but no longer produced: zero where a consuming reaction has it as its single reactant
            single = any(live[i] and S[i, j] < 0 and
                         all(S[i, species.index(o)] >= 0 for o in alive if o != sp) for i in range(len(rates)))
            if len(prod) and not any(live[k] for k in prod) and single:
                newdead.append(sp)
        if not newdead:
            cl, br = _sink_cluster(S[L][:, [species.index(a) for a in alive]])
            branches = [[alive[k] for k in r] for r in br]
            newdead = [alive[k] for k in cl if alive[k] in consumed]
        newdead = [s for s in newdead if s not in dead]
        if not newdead:
            break
        dead = dead + newdead
        nd = set(newdead)
        check = [i for i in range(len(rates)) if live[i] and vs[i] & nd]
    return [d for d in dead if d not in set(forcings)], branches


def equil_zero_states(smatrix, species, rates, forcings):
    """States forced to zero at steady state with the forcings at zero."""
    return _zero_states(smatrix, species, rates, forcings)[0]


def equil_zero_branches(smatrix, species, rates, forcings):
    """(zero, notes): the forced zeros, then the first reactant of each branch of _zero_states(), a note per choice."""
    taken, notes = [], []
    while True:
        zero, br = _zero_states(smatrix, species, rates, list(forcings) + taken)
        if not br:
            return list(dict.fromkeys(taken + zero)), notes
        r = br[0]
        notes.append(f"the reaction of {' + '.join(r)} drains them: at rest one of them is zero "
                     f"({' or '.join(f'{x} = 0' for x in r)}); the result holds for {r[0]} = 0")
        taken.append(r[0])
