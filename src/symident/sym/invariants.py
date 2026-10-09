"""Symbolic helpers of the reduction: module replay, Darboux cofactors, invariant checks."""

import sympy as spy

from .parse import _as_list, _make_local_parse


def _parse_generators(gens, extra_lines):
    """Shared symbol table + parsed xi dicts for a list of generator dicts."""
    gens = _as_list(gens)
    all_lines = list(extra_lines) + \
        [str(v) for g in gens for v in g.values()] + \
        [str(kk) for g in gens for kk in g.keys()]
    local, parse = _make_local_parse(all_lines)
    xis = [{str(kk): parse(str(v)) for kk, v in g.items()} for g in gens]
    return xis, parse


def module_reduce_replay(rows, cols, schedule):
    """Exact sympy replay of a module-reduction pivot schedule, one call for the
    whole block. `rows`: per-generator dicts {col: expr-string} (missing = 0)
    over the coordinate list `cols`; `schedule`: 1-based [row, col] pivot pairs.
    Returns {'rows': [{col: string} nonzero entries], 'combo': [[strings]]}, the
    combination coefficients tracking each reduced row over the originals."""
    cols = [str(c) for c in _as_list(cols)]
    rows = _as_list(rows)
    k = len(rows)
    all_lines = [str(v) for r in rows for v in r.values()] + cols
    local, parse = _make_local_parse(all_lines)
    E = [{c: (parse(str(r[c])) if c in r and str(r[c]) != "0"
              else spy.Integer(0)) for c in cols} for r in rows]
    C = [[spy.Integer(1 if i == j else 0) for j in range(k)] for i in range(k)]
    for st in _as_list(schedule):
        r = int(st[0]) - 1
        cc = cols[int(st[1]) - 1]
        piv = E[r][cc]
        if piv == 0:
            continue
        for j in range(k):
            if j == r or E[j][cc] == 0:
                continue
            m = spy.cancel(E[j][cc] / piv)
            for c2 in cols:
                E[j][c2] = spy.cancel(E[j][c2] - m * E[r][c2])
            for l in range(k):
                C[j][l] = spy.cancel(C[j][l] - m * C[r][l])
    return {'rows': [{c: str(E[j][c]) for c in cols if E[j][c] != 0}
                     for j in range(k)],
            'combo': [[str(x) for x in row] for row in C]}


def darboux_cofactors(candidates, generators):
    """Batch Darboux division test with proportionality dedup. For each candidate
    polynomial string, X(P)/P must be polynomial for EVERY generator (dicts
    {coord: xi-string}); duplicates proportional to an earlier keeper are
    dropped. Returns {'Ps': [strings], 'lams': [[cofactor strings per gen]]}."""
    cands = [str(c) for c in _as_list(candidates)]
    xis, parse = _parse_generators(generators, cands)
    Ps, lams, kept = [], [], []
    for cs in cands:
        try:
            P = parse(cs)
        except Exception:
            continue
        if getattr(P, "is_number", False):
            continue
        lam = []
        ok = True
        for xi in xis:
            XP = spy.Integer(0)
            for v, comp in xi.items():
                XP = XP + comp * spy.diff(P, spy.Symbol(v))
            q = spy.cancel(spy.expand(XP) / P)
            if str(spy.fraction(q)[1]) != "1":
                ok = False
                break
            lam.append(str(q))
        if not ok:
            continue
        if any(getattr(spy.cancel(P / Q), "is_number", False) for Q in kept):
            continue
        kept.append(P)
        Ps.append(str(P))
        lams.append(lam)
    return {'Ps': Ps, 'lams': lams}


def verify_invariants(invariants, generators):
    """Batch exact verification X(I) = 0: for each invariant string, the Lie
    derivative along every generator must cancel to literal 0 (a proof, not a
    sample). Returns a list of booleans."""
    invs = [str(i) for i in _as_list(invariants)]
    xis, parse = _parse_generators(generators, invs)
    out = []
    def _real(e):
        return e.subs({t: spy.Symbol(t.name, real=True) for t in e.free_symbols})

    for s in invs:
        try:
            # real symbols: the coordinates are real, and the atan/Abs forms of
            # the quadrature stage only cancel under that assumption
            I = _real(parse(s))
            ok = True
            for xi in xis:
                XI = spy.Integer(0)
                for v, comp in xi.items():
                    XI = XI + _real(comp) * spy.diff(I, spy.Symbol(v, real=True))
                XI = spy.cancel(spy.together(XI))
                if str(XI) != "0" and spy.simplify(XI) != 0:
                    # cancel() suffices for rational/exp invariants; the simplify
                    # fallback covers atan/Abs forms from the quadrature stage
                    ok = False
                    break
            out.append(ok)
        except Exception:
            out.append(False)
    return out


def integrating_factor_integral(Mstr, xi1, xi2, z1, z2):
    """First integral by quadrature from a 2D integrating factor:
    dI = M*(xi2 dz1 - xi1 dz2), so I = int M*xi2 dz1 + phi(z2) with phi fixed by
    the z2-part. Returns the integral string, or None when the quadrature stays
    unevaluated or the exactness check X(I) = 0 fails."""
    local, parse = _make_local_parse([str(Mstr), str(xi1), str(xi2),
                                      str(z1), str(z2)])
    try:
        M = parse(str(Mstr))
        f1 = parse(str(xi1))
        f2 = parse(str(xi2))
        # real symbols throughout: ratint then prefers real atan/log forms, and
        # a leftover complex-logarithm pair can be realified by re()
        real = {s: spy.Symbol(s.name, real=True)
                for s in (M.free_symbols | f1.free_symbols | f2.free_symbols)}
        M = M.subs(real)
        f1 = f1.subs(real)
        f2 = f2.subs(real)
        s1 = spy.Symbol(str(z1), real=True)
        s2 = spy.Symbol(str(z2), real=True)
        F = spy.integrate(M * f2, s1)
        rest = spy.cancel(spy.together(-M * f1 - spy.diff(F, s2)))
        phi = spy.integrate(rest, s2)
        I = F + phi
        if I.has(spy.Integral):
            return None
        if I.has(spy.I):
            # the imaginary remainder of a real first integral is constant
            I = spy.simplify(spy.re(spy.expand_complex(I)))
            if I.has(spy.I):
                return None
        XI = spy.together(f1 * spy.diff(I, s1) + f2 * spy.diff(I, s2))
        if spy.simplify(XI) != 0:
            return None
        return str(I)
    except Exception:
        return None
