"""Conserved moieties of a reaction network and states forced to zero at steady state."""

from fractions import Fraction

import numpy as np
import sympy as spy

from .. import session
from ..errors import SYMPY_FAILS, sympy_fail
from ..gfp import verify_prime
from ..rexpr import eval_numeric, get_symbols, replace_symbols
from ..sym.modp import eval_rational_mod
from ..sym.parse import _is_rational_expr, _make_local_parse


def _coef_matrix(totals, states):
    stset = set(states)
    sets = [[s for s in get_symbols(e) if s in stset] for e in totals.values()]
    species = list(dict.fromkeys(s for st in sets for s in st))
    C = np.zeros((len(totals), len(species)))
    for i, e in enumerate(totals.values()):
        env0 = {s: 0.0 for s in species}
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
        if abs(coef - 1) < 1.5e-8:
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


def conserved_totals(smatrix, species):
    """Conserved quantities of a stoichiometry (rows reactions, columns species) as totals."""
    import sympy as spy
    S = spy.Matrix(np.asarray(smatrix).tolist())
    ns = S.nullspace()
    out = {}
    for i, v in enumerate(ns):
        v = v * spy.ilcm(*[spy.fraction(x)[1] for x in v])
        terms = []
        for s, c in zip(species, v):
            if c == 0:
                continue
            terms.append(s if c == 1 else f"{c}*{s}")
        name = f"total{i + 1}"
        out[name] = " + ".join(terms)
    return out


def _expr_is_zero(expr, dead, ntry=4):
    """Whether `expr` vanishes with the symbols in `dead` at zero; what is not shown to vanish counts as nonzero."""
    vars_ = get_symbols(expr)
    rng = session.current().rng("cq.zero", expr)
    q = verify_prime()
    try:
        _, parse = _make_local_parse([str(expr)])
        e = parse(str(expr))
        if _is_rational_expr(e):
            vals = [[0 if v in dead else rng.randrange(2, q) for v in vars_] for _ in range(ntry)]
            got = [eval_rational_mod(expr, vars_, pt, q) for pt in vals]
            if all(g is not None for g in got):
                return all(g == 0 for g in got)
        e = e.xreplace({x: 0 for x in e.free_symbols if str(x) in dead})
        return spy.simplify(e) == 0
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
    """Species zero at rest: reactants of positive weight in reactions that lower a weight no reaction raises."""
    n_f, n_s = M.shape
    if n_f == 0 or n_s == 0:
        return []
    Mq = [[Fraction(float(a)) for a in row] for row in M]
    w = _exact_lp(Mq, [sum(row[j] for row in Mq) for j in range(n_s)])
    flux = [sum(a * x for a, x in zip(row, w)) for row in Mq]
    drained = [i for i, f in enumerate(flux) if f < 0]
    return [j for j in range(n_s) if w[j] > 0 and any(Mq[i][j] < 0 for i in drained)]


def equil_zero_states(smatrix, species, rates, forcings):
    """States forced to zero at steady state with the forcings at zero."""
    S = np.nan_to_num(np.asarray(smatrix, dtype=float))
    if S.size == 0:
        return []
    rates = [str(r) for r in rates]
    vs = [set(get_symbols(r)) for r in rates]
    allv = set().union(*vs) if vs else set()
    dead = [f for f in forcings if f in set(species) | allv]
    live = [True] * len(rates)
    check = list(range(len(rates)))
    while True:
        for i in check:
            live[i] = not _expr_is_zero(rates[i], set(dead))
        L = np.array(live, dtype=bool)
        consumed = [sp for j, sp in enumerate(species) if np.any(S[L, j] < 0)]
        newdead = []
        for sp in [s for s in species if s not in dead and s in consumed]:
            j = species.index(sp)
            prod = np.nonzero(S[:, j] > 0)[0]
            if len(prod) and not any(live[k] for k in prod):
                newdead.append(sp)
        if not [s for s in newdead if s not in dead]:
            alive = [s for s in species if s not in dead]
            cl = _sink_cluster(S[L][:, [species.index(a) for a in alive]])
            cl = [alive[k] for k in cl if alive[k] in consumed]
            if cl:
                newdead = cl
            else:
                break
        newdead = [s for s in newdead if s not in dead]
        if not newdead:
            break
        dead = dead + newdead
        nd = set(newdead)
        check = [i for i in range(len(rates)) if live[i] and vs[i] & nd]
    return [d for d in dead if d not in set(forcings)]
