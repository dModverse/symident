"""Conserved moieties of a reaction network and states forced to zero at steady state."""

import math
from fractions import Fraction

import numpy as np
import sympy as spy

from ..errors import SYMPY_FAILS, InputError, sympy_fail
from ..rexpr import eval_numeric, get_symbols, replace_symbols
from ..sym.parse import _is_rational_expr, _make_local_parse

# the pivot sets that tie with the chosen one by structure, at most this many in all
PIVOT_CHOICES = 6
# rays of the cone of nonnegative conservation laws past which the order of the species decides the basis
_RAY_CAP = 400


def _coef_matrix(totals, states, shift=0.0):
    """(C, species, sets): the coefficients of the species in the linear totals, other symbols at a generic point."""
    stset = set(states)
    sets = [[s for s in get_symbols(e) if s in stset] for e in totals.values()]
    species = list(dict.fromkeys(s for st in sets for s in st))
    # parameters in the coefficients, such as volumes, at a generic positive point
    others = sorted({s for e in totals.values() for s in get_symbols(e) if s not in stset})
    at = {o: 1.0 + 0.618 * (k + 1) + shift * (k + 2) for k, o in enumerate(others)}
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


def _rank(M):
    """The numerical rank of a small matrix of coefficients at a generic point."""
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return 0
    return int(np.linalg.matrix_rank(M, tol=1e-9 * max(1.0, float(np.abs(M).max()))))


def _pivot_weight(C, C2, species, rows, avoid, extra):
    """A rank of each species as a pivot by structure, lower first: avoided, a coefficient that holds a symbol (it
    differs between the generic points of C and C2), the number of totals that hold it, then `extra`."""
    w = {}
    for j, s in enumerate(species):
        held = [i for i in rows if abs(C[i, j]) > 1e-12]
        symbolic = any(abs(C[i, j] - C2[i, j]) > 1e-9 * max(1.0, abs(C[i, j])) for i in held)
        w[s] = 1000 * (s in avoid) + 100 * bool(symbolic) + 10 * min(len(held), 9) + min(int(extra.get(s, 0)), 9)
    return w


def _greedy_basis(C, rows, species, order, start=()):
    """The columns `start`, then those of `order` that raise the rank of C[rows], until it is full."""
    target = _rank(C[rows]) if rows else 0
    cols = [species.index(s) for s in start]
    for s in order:
        if len(cols) >= target:
            break
        j = species.index(s)
        if j in cols:
            continue
        if _rank(C[np.ix_(rows, cols + [j])]) == len(cols) + 1:
            cols.append(j)
    return [species[j] for j in cols]


def _assign(C, rows, species, basis):
    """{row: pivot}: a matching of the totals `rows` to the species `basis` along nonzero coefficients."""
    cols = [species.index(s) for s in basis]
    match = {}

    def augment(r, seen):
        for j in cols:
            if abs(C[r, j]) > 1e-12 and j not in seen:
                seen.add(j)
                if j not in match or augment(match[j], seen):
                    match[j] = r
                    return True
        return False
    for r in rows:
        augment(r, set())
    return {r: species[j] for j, r in match.items()}


def _free_initial_error(prefer, C, rows, species, totals):
    """The InputError for states of `free_initial` whose resting values the totals do not leave independent."""
    names = list(totals)
    cols = [species.index(s) for s in prefer]
    shared = [names[i] for i in rows if sum(abs(C[i, j]) > 1e-12 for j in cols) >= 2]
    within = [names[i] for i in rows if all(abs(C[k, j]) <= 1e-12 for k in rows if k != i for j in cols)]
    if len(shared) == 1 and shared[0] in within:
        return InputError(f"`free_initial` names {', '.join(prefer)}, which the single total {shared[0]} = "
                          f"{totals[shared[0]]} holds; give one state per total")
    return InputError(f"`free_initial` names {', '.join(prefer)}, whose resting values the totals "
                      f"{', '.join(names[i] for i in rows)} do not leave independent; give one state per total, "
                      "with the coefficients of the chosen states linearly independent")


def pivot_decomposition(totals, states, parameters=(), avoid=(), prefer=(), extra=None):
    """One pivot species per independent conserved total, chosen by structure, and the pivot sets tied with it.

    The pivots are a basis of the columns of the coefficient matrix of the totals, chosen by the lowest
    _pivot_weight() (`extra` a further weight per species) in the order of `states`; `prefer` (``free_initial``)
    are taken first and checked. "ties" are the other bases of the same weight, "valid" further bases, each at most
    PIVOT_CHOICES - 1.
    """
    names = list(totals)
    C, species, sets = _coef_matrix(totals, states)
    # a total of a single species leaves that species constant: it needs no pivot
    rows = [i for i in range(len(names)) if sum(abs(C[i, j]) > 1e-12 for j in range(len(species))) >= 2]
    keep = []
    for i in rows:
        if _rank(C[keep + [i]]) == len(keep) + 1:
            keep.append(i)
    rows = keep
    avoid, parameters = set(avoid), set(parameters)
    w = _pivot_weight(C, _coef_matrix(totals, states, shift=0.377)[0], species, rows, avoid | parameters,
                      dict(extra or {}))
    pos = {s: k for k, s in enumerate(states)}
    order = sorted(species, key=lambda s: (w[s], pos.get(s, len(pos))))
    prefer = [str(s) for s in prefer or []]
    if prefer:
        dup = [s for k, s in enumerate(prefer) if s in prefer[:k]]
        if dup:
            raise InputError(f"`free_initial` names {dup[0]} twice; give one state per total")
        bad = [s for s in prefer if s not in species or
               not any(abs(C[i, species.index(s)]) > 1e-12 for i in rows)]
        if bad:
            held = ", ".join(f"{names[i]} = {totals[names[i]]}" for i in rows)
            raise InputError(f"`free_initial` names {bad[0]}, which is in no conserved total"
                             + (f" ({held})" if rows else ""))
        if _rank(C[np.ix_(rows, [species.index(s) for s in prefer])]) < len(prefer):
            raise _free_initial_error(prefer, C, rows, species, totals)
    basis = _greedy_basis(C, rows, species, order, prefer)

    def exchanges(b):
        """(position, species) pairs whose exchange keeps the columns `b` a basis."""
        X = np.linalg.solve(C[np.ix_(rows, [species.index(x) for x in b])], C[rows])
        return [(k, s) for s in order if s not in b for k in range(len(b))
                if abs(X[k, species.index(s)]) > 1e-9]
    ties, more = [], []
    if not prefer and rows:
        # bases one exchange apart, breadth first from the greedy one: those of equal weight tie with it
        seen, queue = {frozenset(basis)}, [basis]
        while queue and len(ties) < PIVOT_CHOICES - 1:
            b = queue.pop(0)
            for k, s in exchanges(b):
                alt = sorted(b[:k] + [s] + b[k + 1:], key=lambda x: pos.get(x, len(pos)))
                if frozenset(alt) in seen:
                    continue
                seen.add(frozenset(alt))
                if w[s] == w[b[k]]:
                    ties.append(alt)
                    queue.append(alt)
                elif len(more) < PIVOT_CHOICES - 1:
                    more.append(alt)
        more.sort(key=lambda b: sum(w[x] for x in b))
    basis = sorted(basis, key=lambda x: pos.get(x, len(pos)))
    ties = ties[:PIVOT_CHOICES - 1]
    more = more[:max(0, PIVOT_CHOICES - 1 - len(ties))]
    match = _assign(C, rows, species, basis)
    pivots = [match.get(i) for i in range(len(names))]
    return {"C": C, "species": species, "pivots": pivots, "sets": sets, "rows": rows, "basis": basis,
            "ties": ties, "valid": more}


def substitute_cq(totals, f, states, parameters, avoid=(), prefer=(), extra=None):
    """Each independent conserved total eliminates one pivot species, solved through the totals as parameters."""
    parameters = list(parameters or [])
    if not totals:
        return {"f": f, "parameters": parameters, "cq_info": [], "elim_states": [], "decomposition": None}
    dec = pivot_decomposition(totals, states, parameters, avoid, prefer, extra)
    names = list(totals)
    rows = [i for i, e in enumerate(dec["pivots"]) if e is not None]
    if not rows:
        return {"f": f, "parameters": parameters, "cq_info": [], "elim_states": [], "decomposition": dec}
    elim = [dec["pivots"][i] for i in rows]
    try:
        _, parse = _make_local_parse([str(totals[names[i]]) for i in rows] + [names[i] for i in rows])
        eqs = [parse(str(totals[names[i]]).replace("^", "**")) - parse(names[i]) for i in rows]
        unknowns = [parse(e) for e in elim]
        sol = spy.linsolve(eqs, unknowns)
        sol = next(iter(sol)) if sol else None
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        sol = None
    if sol is None or any(x.free_symbols & set(unknowns) for x in sol):
        raise InputError("the conserved totals " + ", ".join(names[i] for i in rows) +
                         " cannot be solved for the species " + ", ".join(elim))
    recon = {e: "(" + str(spy.cancel(x)).replace("**", "^") + ")" for e, x in zip(elim, sol)}
    cq_info = []
    for i in rows:
        nm, e = names[i], dec["pivots"][i]
        if nm not in parameters:
            parameters.append(nm)
        cq_info.append({"total_name": nm, "elim_state": e, "recon_expr": recon[e]})
    keys = list(recon)
    f = replace_symbols(keys, [recon[k] for k in keys], f)
    return {"f": f, "parameters": parameters, "cq_info": cq_info, "elim_states": elim, "decomposition": dec}


def total_coefficient(total, x):
    """The coefficient of the species `x` in the linear `total`, as an expression."""
    _, parse = _make_local_parse([str(total), str(x)])
    c = spy.diff(parse(str(total).replace("^", "**")), parse(str(x)))
    return str(c).replace("**", "^")


def cq_events(events, totals, cq_info):
    """The events of a model whose `cq_info` eliminated species through conserved totals, and the totals they change.

    An event on a species x of a total T adds a*(E(x) - x) to T, where a is the coefficient of x in T and E(x) the
    value of x after the events at that time; like every event value it reads the states before the events. An
    event on an eliminated species changes only the totals, and an event value reads an eliminated species
    through the totals.
    """
    from .conditions import compose_event
    recon = {ci["elim_state"]: ci["recon_expr"] for ci in cq_info}
    names = [ci["total_name"] for ci in cq_info]
    held = {t: set(get_symbols([str(totals[t])])) for t in names}
    keys, vals = list(recon), list(recon.values())

    def through(text):
        return replace_symbols(keys, vals, str(text)) if keys else str(text)
    out, changed, groups = [], [], {}
    for e in events:
        key = (str(e["time"]), tuple(e["cond"]) if e.get("cond") is not None else None, e.get("root"))
        groups.setdefault(key, []).append(e)
    for evs in groups.values():
        delta = {}
        for x in dict.fromkeys(e["var"] for e in evs):
            ops = [e for e in evs if e["var"] == x]
            old = recon.get(x, x)
            cur = old
            for e in ops:
                cur = compose_event(cur, e["method"], through(e["value"]))
            for t in names:
                if x in held[t]:
                    delta.setdefault(t, []).append(f"({total_coefficient(totals[t], x)})*(({cur}) - ({old}))")
            if x not in recon:
                out += [dict(e, value=through(e["value"]), time=through(e["time"])) for e in ops]
        e0 = evs[0]
        for t, parts in delta.items():
            try:
                _, parse = _make_local_parse(parts)
                v = spy.cancel(spy.together(sum((parse(d.replace("^", "**")) for d in parts), spy.Integer(0))))
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                v = " + ".join(parts)
            if v == 0:
                continue
            out.append({k: e0[k] for k in e0 if k not in ("var", "value", "method", "time")} |
                       {"var": t, "time": through(e0["time"]), "value": str(v).replace("**", "^"), "method": "add"})
            changed.append(t)
    return out, list(dict.fromkeys(changed))


def check_conserved(totals, f):
    """InputError for a total whose time derivative under the right-hand sides `f` does not vanish."""
    bad = None
    try:
        _, parse = _make_local_parse([f"{k} = {v}" for k, v in f.items()] + [str(v) for v in totals.values()])
        rhs = {spy.Symbol(k): parse(str(v).replace("^", "**")) for k, v in f.items()}
        for name, e in totals.items():
            t = parse(str(e).replace("^", "**"))
            d = sum((spy.diff(t, x) * r for x, r in rhs.items() if x in t.free_symbols), spy.Integer(0))
            if spy.cancel(spy.together(d)) != 0:
                bad = (name, e, d)
                break
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
    if bad is not None:
        name, e, d = bad
        raise InputError(f"the total {name} = {e} is not conserved: the time derivative under "
                         f"the right-hand sides is {str(d).replace('**', '^')}, not zero")


def _primitive(v):
    """The integer vector of the direction of the rational vector `v`, coprime entries."""
    v = [Fraction(x) for x in v]
    scale = math.lcm(*(x.denominator for x in v)) if v else 1
    z = [int(x * scale) for x in v]
    g = math.gcd(*z) if any(z) else 1
    return [x // g for x in z]


def _rref_rows(vectors, n):
    """The rows of the reduced row echelon form of the span of `vectors` over Q."""
    rows = [list(map(Fraction, v)) for v in vectors]
    out = []
    for col in range(n):
        piv = next((r for r in rows if r[col] != 0), None)
        if piv is None:
            continue
        rows.remove(piv)
        piv = [x / piv[col] for x in piv]
        rows = [[a - r[col] * b for a, b in zip(r, piv)] for r in rows]
        out = [[a - r[col] * b for a, b in zip(r, piv)] for r in out] + [piv]
    return [r for r in out if any(r)]


def _extreme_rays(basis, n):
    """The extreme rays of the nonnegative vectors in the span of `basis` (independent rows), by the double
    description method, or None past _RAY_CAP rays."""
    k = len(basis)
    A = [[basis[i][j] for i in range(k)] for j in range(n)]
    A = [a for a in A if any(a)]
    # a start from k independent constraints: the rays of a simplicial cone
    start, M = [], []
    for a in A:
        if len(_rref_rows(M + [a], k)) > len(M):
            M.append(a)
            start.append(a)
        if len(M) == k:
            break
    Minv = spy.Matrix(M).inv()
    rays = [[Fraction(int(spy.fraction(x)[0]), int(spy.fraction(x)[1])) for x in Minv.col(i)] for i in range(k)]
    done = list(start)

    def dot(a, y):
        return sum(x * z for x, z in zip(a, y))
    for a in A:
        if a in start:
            continue
        vals = [dot(a, r) for r in rays]
        pos = [r for r, v in zip(rays, vals) if v > 0]
        neg = [r for r, v in zip(rays, vals) if v < 0]
        new = [r for r, v in zip(rays, vals) if v >= 0]
        zero_of = [frozenset(i for i, c in enumerate(done) if dot(c, r) == 0) for r in rays]
        zp = {id(r): z for r, z in zip(rays, zero_of)}
        for p in pos:
            for q in neg:
                common = zp[id(p)] & zp[id(q)]
                # adjacent: no third ray is zero on every constraint both are zero on
                if any(common <= zp[id(r)] for r in rays if r is not p and r is not q):
                    continue
                ap, aq = dot(a, p), dot(a, q)
                new.append([ap * y - aq * x for x, y in zip(p, q)])
        rays = new
        done.append(a)
        if len(rays) > _RAY_CAP:
            return None
    out = []
    for r in rays:
        c = _primitive([sum(r[i] * basis[i][j] for i in range(k)) for j in range(n)])
        if any(c) and c not in out:
            out.append(c)
    return out


def canonical_laws(vectors, n):
    """A basis of the span of the conservation laws `vectors` (length n): the nonnegative extreme rays where they
    span it, few species and small coefficients first, completed by reduced rows; each primitive, of positive
    sum. Up to ties between species of the same role, the basis does not depend on the order of the species."""
    rows = _rref_rows(vectors, n)
    if not rows:
        return []
    k = len(rows)
    ints = [_primitive(r) for r in rows]
    supp = [frozenset(j for j, x in enumerate(r) if x) for r in ints]
    # the laws of species sets that no law joins are separate cones
    block = list(range(k))

    def root(i):
        while block[i] != i:
            block[i] = block[block[i]]
            i = block[i]
        return i
    for i in range(k):
        for j in range(i):
            if supp[i] & supp[j]:
                block[root(i)] = root(j)
    rays = []
    for b in dict.fromkeys(root(i) for i in range(k)):
        idx = [i for i in range(k) if root(i) == b]
        if len(idx) == 1:
            r = ints[idx[0]]
            if min(r) >= 0 or max(r) <= 0:
                rays.append([abs(x) for x in r])
            continue
        cols = sorted(set().union(*(supp[i] for i in idx)))
        sub = _extreme_rays([[rows[i][j] for j in cols] for i in idx], len(cols))
        for r in sub or []:
            full = [0] * n
            for j, x in zip(cols, r):
                full[j] = x
            rays.append(full)
    rays.sort(key=lambda r: (sum(1 for x in r if x), max(r), sum(r), [-abs(x) for x in r]))
    chosen, ech = [], []

    def independent(r):
        """Whether `r` is outside the span of the chosen laws; the echelon rows `ech` grow with it."""
        v = [Fraction(x) for x in r]
        for pc, row in ech:
            if v[pc]:
                c = v[pc]
                v = [a - c * b for a, b in zip(v, row)]
        pc = next((j for j, x in enumerate(v) if x), None)
        if pc is None:
            return False
        ech.append((pc, [x / v[pc] for x in v]))
        return True
    for r in rays:
        if len(chosen) == k:
            break
        if independent(r):
            chosen.append(r)
    for r in ints:
        if len(chosen) == k:
            break
        if independent(r):
            chosen.append(r if sum(r) > 0 or (sum(r) == 0 and next(x for x in r if x) > 0) else [-x for x in r])
    return sorted(chosen, key=lambda r: [j for j, x in enumerate(r) if x])


def _total_text(species, v):
    """The total sum_j v_j species_j with integer coefficients, as text."""
    out = ""
    for s, c in zip(species, v):
        if not c:
            continue
        term = s if abs(c) == 1 else f"{abs(c)}*{s}"
        out += (term if c > 0 else f"-{term}") if not out else (f" + {term}" if c > 0 else f" - {term}")
    return out


def conserved_totals(smatrix, species):
    """Conserved quantities of a stoichiometry (rows reactions, columns species) as totals."""
    from sympy.polys.domains import QQ
    from sympy.polys.matrices import DomainMatrix
    n = len(species)
    rows = {}
    for i, r in enumerate(smatrix):
        nz = {}
        for j, x in enumerate(r):
            if x:
                q = _exact(x)
                nz[j] = QQ(q.numerator, q.denominator)
        if nz:
            rows[i] = nz
    if not len(smatrix):
        return {}
    red, piv = DomainMatrix(rows, (len(smatrix), n), QQ).rref()
    R = red.rep
    laws = []
    for free in (c for c in range(n) if c not in set(piv)):
        # the basis of the nullspace from the reduced row echelon form, one vector per free column
        v = [Fraction(0)] * n
        v[free] = Fraction(1)
        for r, pc in enumerate(piv):
            x = R.get(r, {}).get(free)
            if x:
                v[pc] = -Fraction(int(QQ.numer(x)), int(QQ.denom(x)))
        laws.append(v)
    return {f"total{i + 1}": _total_text(species, v) for i, v in enumerate(canonical_laws(laws, n))}


def exact_totals(totals, odes):
    """The `totals` that the right-hand sides `odes` conserve exactly."""
    kept = {}
    for name, total in (totals or {}).items():
        try:
            check_conserved({name: total}, odes)
        except InputError:
            continue
        kept[name] = total
    return kept


def conserved_totals_of_odes(odes):
    """The linear conservation laws of the right-hand sides `odes` (state to expression) as totals."""
    from ..sym.exact import _conserved
    try:
        laws = _conserved([f"{k} = {v}" for k, v in odes.items()], {spy.Symbol(str(k)) for k in odes})
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return {}
    states = list(odes)
    vecs = [[Fraction(int(spy.Rational(law.get(spy.Symbol(str(k)), 0)).p),
                      int(spy.Rational(law.get(spy.Symbol(str(k)), 0)).q)) for k in states] for law in laws]
    out = {}
    for v in canonical_laws(vecs, len(states)):
        total = _total_text(states, v)
        try:
            check_conserved({"total": total}, odes)
        except InputError:
            continue
        out[f"total{len(out) + 1}"] = total
    return out


def _exact(x):
    """A stoichiometric coefficient as a rational; a float by the shortest decimal."""
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


def _one_way(smatrix, rates):
    """The reactions with each net rate whose numerator has terms of both signs split into a forward and a backward
    reaction, the terms of each sign over the denominator: every rate then runs one way."""
    S = np.nan_to_num(np.asarray(smatrix, dtype=float))
    rows, out = [], []
    for i, r in enumerate(rates):
        r = str(r)
        parts = None
        try:
            _, parse = _make_local_parse([r])
            e = parse(r)
            if _is_rational_expr(e):
                num, den = spy.fraction(spy.together(e))
                terms = spy.Add.make_args(spy.expand(num))
                pos = spy.Add(*[t for t in terms if t.as_coeff_Mul()[0] > 0])
                neg = -spy.Add(*[t for t in terms if t.as_coeff_Mul()[0] < 0])
                if pos != 0 and neg != 0:
                    parts = [str(pos / den), str(neg / den)]
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
        if parts is None:
            rows.append(S[i])
            out.append(r)
            continue
        rows += [S[i], -S[i]]
        out += parts
    return np.asarray(rows, dtype=float).reshape(-1, S.shape[1] if S.ndim == 2 else 0), out


def _zero_states(smatrix, species, rates, forcings):
    """(zero, branches): the states forced to zero at rest with the forcings at zero, and _sink_cluster() branches."""
    S = np.nan_to_num(np.asarray(smatrix, dtype=float))
    if S.size == 0:
        return [], []
    S, rates = _one_way(S, rates)
    vs = [set(get_symbols(r)) for r in rates]
    allv = set().union(*vs) if vs else set()
    dead = [f for f in forcings if f in set(species) | allv]
    live = [True] * len(rates)
    check = list(range(len(rates)))
    branches = []
    col = {sp: j for j, sp in enumerate(species)}
    while True:
        for i in check:
            live[i] = not _expr_is_zero(rates[i], set(dead))
        L = np.array(live, dtype=bool)
        consumed = {sp for j, sp in enumerate(species) if np.any(S[L, j] < 0)}
        alive = [s for s in species if s not in set(dead)]
        # the alive reactants of each reaction
        reactants = (S[:, [col[a] for a in alive]] < 0).sum(axis=1)
        newdead = []
        for sp in [s for s in alive if s in consumed]:
            j = col[sp]
            prod = np.nonzero(S[:, j] > 0)[0]
            # consumed but no longer produced: zero where it is the single reactant of a consuming reaction
            single = bool(np.any(L & (S[:, j] < 0) & (reactants == 1)))
            if len(prod) and not np.any(L[prod]) and single:
                newdead.append(sp)
        if not newdead:
            cl, br = _sink_cluster(S[L][:, [col[a] for a in alive]])
            branches = [[alive[k] for k in r] for r in br]
            newdead = [alive[k] for k in cl if alive[k] in consumed]
        newdead = [s for s in newdead if s not in dead]
        if not newdead:
            break
        dead = dead + newdead
        nd = set(newdead)
        check = [i for i in range(len(rates)) if live[i] and vs[i] & nd]
    return [d for d in dead if d not in set(forcings)], branches


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


def reduced_by_totals(odes, observables, events, totals, eliminated):
    """The model on the level sets of the conserved `totals`: the species `eliminated` solved through the totals as
    parameters, and an event on a species of a total changing that total, which becomes a state without dynamics.
    Returns the right-hand sides, the observables (a dict or a list of dicts), the events and the solved species."""
    cq = substitute_cq(totals, dict(odes), list(odes), [], prefer=list(eliminated))
    recon = {ci["elim_state"]: ci["recon_expr"] for ci in cq["cq_info"]}
    if set(recon) != set(eliminated):
        raise ValueError("the conserved totals do not solve for the eliminated species " + ", ".join(eliminated))
    keys, vals = list(recon), list(recon.values())

    def through(o):
        return None if o is None else {str(k): replace_symbols(keys, vals, str(v)) for k, v in dict(o).items()}
    f = {k: v for k, v in cq["f"].items() if k not in recon}
    obs = [through(o) for o in observables] if isinstance(observables, (list, tuple)) else through(observables)
    events, moved = cq_events(events, totals, cq["cq_info"]) if events else ([], [])
    for t in moved:
        f[t] = "0"
    return f, obs, events, recon
