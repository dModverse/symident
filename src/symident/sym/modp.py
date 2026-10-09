"""Evaluation of rational expressions modulo a prime and over the rationals."""

from fractions import Fraction

import sympy as spy

from .parse import _as_list, _make_local_parse


def _modp_rational(val, p):
    """Map a sympy Rational (or integer-valued expression) to its residue mod p."""
    r = spy.Rational(val)
    return (int(r.p) % p) * pow(int(r.q) % p, p - 2, p) % p


class _Mono(tuple):
    """Exponent tuple that also holds its nonzero (index, exponent) pairs."""

    def __new__(cls, expo):
        t = tuple.__new__(cls, expo)
        t.nz = tuple((k, e) for k, e in enumerate(expo) if e)
        return t


def _mono_nz(expo):
    nz = getattr(expo, 'nz', None)
    return nz if nz is not None else tuple((k, e) for k, e in enumerate(expo) if e)


def _poly_terms(expr, gens):
    """Compile a rational expression in `gens` to (numerator, denominator) term
    lists [(exponent tuple, (p, q))] with rational coefficients as integer pairs.
    Prime-independent, so the compile is cached and only the per-prime reduction
    (in _eval_terms) repeats."""
    if expr == 0:
        zero = _Mono([0] * len(gens))
        return [(zero, (0, 1))], [(zero, (1, 1))]
    num, den = spy.fraction(spy.together(expr))
    pos = {g: k for k, g in enumerate(gens)}
    def terms(e):
        # the Poly over the generators present only; a dense one over all of them
        # recurses through every level
        fs = e.free_symbols
        used = [g for g in gens if g in fs]
        if not used:
            r = spy.Rational(e)
            return [(_Mono([0] * len(gens)), (int(r.p), int(r.q)))]
        P = spy.Poly(e, *used)
        out = []
        for monom, coef in P.terms():
            r = spy.Rational(coef)
            full = [0] * len(gens)
            for g, m in zip(used, monom):
                full[pos[g]] = int(m)
            out.append((_Mono(full), (int(r.p), int(r.q))))
        return out
    return terms(num), terms(den)


def _bipoly(expr, state_gens, param_gens):
    """Compile a polynomial in (states, params) to a list of (state-monomial
    exponent tuple, param-coefficient term list), the latter from _poly_terms, so
    the per-point state-polynomial is rebuilt by evaluating the parameter
    coefficients mod p."""
    P = spy.Poly(expr, *state_gens)
    out = []
    for sm, coef in P.terms():
        ct = _poly_terms(coef, param_gens)
        out.append((tuple(int(m) for m in sm), ct))
    return out


def _solve_mod(A, B, p):
    """Solve A X = B over GF(p) by Gauss-Jordan (A: n x n, B: n x k, integer lists
    already reduced mod p). Returns X as a list of rows, or None if A is singular."""
    n = len(A)
    if n == 0:
        return [[] for _ in range(0)]
    k = len(B[0]) if B and B[0] else 0
    M = [list(A[i]) + list(B[i]) for i in range(n)]
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] % p), None)
        if piv is None:
            return None
        M[col], M[piv] = M[piv], M[col]
        inv = pow(M[col][col] % p, p - 2, p)
        M[col] = [(x * inv) % p for x in M[col]]
        for r in range(n):
            if r != col and M[r][col] % p:
                f = M[r][col] % p
                M[r] = [(M[r][c] - f * M[col][c]) % p for c in range(n + k)]
    return [M[i][n:] for i in range(n)]


_inv_mod_cache = {}


def _inv_mod(a, p):
    """Modular inverse of a mod p, memoised per (a, p): term-list denominators are
    a small fixed set. inv(0)=0 as with Fermat."""
    a %= p
    if a <= 1:
        return a
    key = (a, p)
    v = _inv_mod_cache.get(key)
    if v is None:
        v = pow(a, p - 2, p)
        _inv_mod_cache[key] = v
    return v


def _eval_poly_terms(terms, ptvals, p):
    """Value of one term list (from _poly_terms) at the integer point `ptvals`
    (gens order, already mod p) as a residue mod p."""
    acc = 0
    for monom, (cn, cd) in terms:
        if not cn:
            continue
        c = (cn % p) * _inv_mod(cd, p) % p          # cd is a compile-time constant
        for k, e in _mono_nz(monom):
            c = c * pow(ptvals[k], e, p) % p
        acc = (acc + c) % p
    return acc


def _eval_terms(numden, ptvals, p):
    """Residue num/den mod p of a (num, den) term list at the point `ptvals`."""
    nv = _eval_poly_terms(numden[0], ptvals, p)
    dv = _eval_poly_terms(numden[1], ptvals, p)
    if dv == 1:                                     # polynomial entry: no inverse
        return nv
    return nv * pow(dv % p, p - 2, p) % p


def _eval_terms_guarded(numden, ptvals, p):
    """Residue num/den mod p, or None when the denominator vanishes mod p (a Fermat
    inverse of 0 would silently return 0)."""
    dv = _eval_poly_terms(numden[1], ptvals, p)
    if dv % p == 0:
        return None
    nv = _eval_poly_terms(numden[0], ptvals, p)
    if dv == 1:
        return nv
    return nv * pow(dv, p - 2, p) % p


def eval_rational_mod(expr, names, vals, q):
    """Value of the rational expression `expr` at the integer point names -> vals,
    reduced modulo the prime q. Returns None if the denominator vanishes mod q.
    Used to certify a reconstructed identifiability direction at a fresh prime."""
    q = int(q)
    if not isinstance(names, (list, tuple)):
        names = [names]
    if not isinstance(vals, (list, tuple)):
        vals = [vals]
    local, parse = _make_local_parse([str(expr)] + [str(n) for n in names])
    e = spy.together(spy.sympify(parse(str(expr))).subs(
        {spy.Symbol(str(n)): spy.Integer(int(v))
         for n, v in zip(names, vals)}))
    num, den = spy.fraction(e)
    num, den = int(num), int(den) % q
    if den == 0:
        return None
    return int((num % q) * pow(den, q - 2, q) % q)


_eval_batch_cache = {}


def eval_rational_mod_batch(exprs, names, points, q):
    """Values of the rational expressions `exprs` (strings) at each integer point
    (rows of `points`, columns aligned with `names`), reduced modulo the prime q.
    The lambdified expressions are memoised on (exprs, names), since compiling
    dominates evaluation in the sampling loops; evaluation is exact (Fraction).
    Returns one list per point; -1 marks a vanishing denominator or a failure."""
    q = int(q)
    exprs = [str(e) for e in _as_list(exprs)]
    names = [str(n) for n in _as_list(names)]
    points = _as_list(points)
    if points and not isinstance(points[0], (list, tuple)):
        points = [points]
    key = (tuple(exprs), tuple(names))
    fns = _eval_batch_cache.get(key)
    if fns is None:
        local, parse = _make_local_parse(exprs + names)
        argsyms = [local.get(n, spy.Symbol(n)) for n in names]
        fns = []
        for ex in exprs:
            try:
                fns.append(spy.lambdify(argsyms, parse(ex), modules="math"))
            except Exception:
                fns.append(None)
        if len(_eval_batch_cache) > 512:
            _eval_batch_cache.clear()
        _eval_batch_cache[key] = fns
    out = []
    for pt in points:
        vals = [Fraction(int(v)) for v in pt]
        row = []
        for f in fns:
            v = None
            if f is not None:
                try:
                    v = Fraction(f(*vals))
                except Exception:
                    v = None
            if v is None or v.denominator % q == 0:
                row.append(-1)
            else:
                row.append(int(v.numerator % q) *
                           pow(v.denominator % q, q - 2, q) % q)
        out.append(row)
    return out


def eval_rational_batch(exprs, names, points):
    """Exact rational values of `exprs` at each integer point as decimal 'num'/'den'
    strings (den '0' marks a failure), for cofactor-matrix sampling. Shares the
    cache of eval_rational_mod_batch()."""
    exprs = [str(e) for e in _as_list(exprs)]
    names = [str(n) for n in _as_list(names)]
    points = _as_list(points)
    if points and not isinstance(points[0], (list, tuple)):
        points = [points]
    key = (tuple(exprs), tuple(names))
    fns = _eval_batch_cache.get(key)
    if fns is None:
        local, parse = _make_local_parse(exprs + names)
        argsyms = [local.get(n, spy.Symbol(n)) for n in names]
        fns = []
        for ex in exprs:
            try:
                fns.append(spy.lambdify(argsyms, parse(ex), modules="math"))
            except Exception:
                fns.append(None)
        if len(_eval_batch_cache) > 512:
            _eval_batch_cache.clear()
        _eval_batch_cache[key] = fns
    nums, dens = [], []
    for pt in points:
        vals = [Fraction(int(v)) for v in pt]
        nrow, drow = [], []
        for f in fns:
            v = None
            if f is not None:
                try:
                    v = Fraction(f(*vals))
                except Exception:
                    v = None
            if v is None:
                nrow.append("0")
                drow.append("0")
            else:
                nrow.append(str(v.numerator))
                drow.append(str(v.denominator))
        nums.append(nrow)
        dens.append(drow)
    return {'num': nums, 'den': dens}
