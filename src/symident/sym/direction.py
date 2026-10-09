"""Classification of a direction as a scaling or a general direction."""

import sympy as spy

from .parse import _make_local_parse


def _canon_int_vector(prim, gens):
    """Scale a polynomial vector by a rational constant to canonical integer-primitive
    form (clear denominators, divide by content, first nonzero LC positive), so
    gauge-equivalent representatives coincide for a fixed `gens` order."""
    coeffs = []
    for e in prim:
        if e == 0:
            continue
        for c in spy.Poly(e, *gens).coeffs():
            coeffs.append(spy.Rational(c))
    if not coeffs:
        return prim
    L = spy.Integer(1)
    for c in coeffs:
        L = spy.ilcm(L, c.q)
    scaled = [spy.expand(e * L) for e in prim]
    g = 0
    for e in scaled:
        if e == 0:
            continue
        for c in spy.Poly(e, *gens).coeffs():
            g = spy.igcd(g, int(c))
    if g and g != 1:
        scaled = [spy.expand(e / g) for e in scaled]
    for e in scaled:
        if e != 0:
            if spy.Poly(e, *gens).LC() < 0:
                scaled = [spy.expand(-x) for x in scaled]
            break
    return scaled


def _factor_map(e):
    """{irreducible factor: multiplicity} of a nonzero polynomial expression and its
    rational content, without expanding a product that is already factored."""
    c, fl = spy.factor_list(e)
    out = {}
    for f, m in fl:
        # a factor fixed up to sign: keep one orientation, move the sign to c
        if spy.Poly(f).LC() < 0:
            f = -f
            if m % 2:
                c = -c
        out[f] = out.get(f, 0) + int(m)
    return c, out


def _classify_factored(infis, gens):
    """classify_direction() for wide directions: lcm of the denominators and gcd of the
    numerators from factor lists (exact over Q), the canonical representative kept
    as a product. None when a component is not rational."""
    try:
        nz = [i for i, e in enumerate(infis) if e != 0]
        fr = [spy.fraction(spy.together(infis[i])) for i in nz]
        num = [_factor_map(a) for a, _ in fr]
        den = [_factor_map(b) for _, b in fr]
        # lcm of the denominators, gcd of the numerators times lcm/den
        L = {}
        for _, dm in den:
            for f, m in dm.items():
                L[f] = max(L.get(f, 0), m)
        cleared = []
        for (cn, nm), (cd, dm) in zip(num, den):
            fm = dict(nm)
            for f, m in L.items():
                fm[f] = fm.get(f, 0) + m - dm.get(f, 0)
            cleared.append((cn / cd, {f: m for f, m in fm.items() if m}))
        common = None
        for _, fm in cleared:
            common = dict(fm) if common is None else \
                {f: min(m, fm.get(f, 0)) for f, m in common.items() if fm.get(f, 0)}
        coeffs = [c for c, _ in cleared]
        # integer-primitive: clear the rational coefficients, divide by their gcd
        dl = 1
        for c in coeffs:
            dl = spy.ilcm(dl, spy.fraction(spy.nsimplify(c))[1])
        ints = [int(c * dl) for c in coeffs]
        g = 0
        for v in ints:
            g = spy.igcd(g, v)
        g = g or 1
        if ints[0] < 0:
            g = -g
        prim = {}
        degs = []
        for i, (c, fm) in zip(nz, cleared):
            rest = {f: m - common.get(f, 0) for f, m in fm.items() if m - common.get(f, 0)}
            e = spy.Integer(int(c * dl) // g)
            d = 0
            for f, m in rest.items():
                e = e * f**m
                d += m * spy.Poly(f, *gens).total_degree()
            prim[i] = e
            degs.append(d)
        is_scaling = all(degs[k] == 1 and prim[i].free_symbols == {gens[i]}
                         and spy.Poly(prim[i], gens[i]).total_degree() == 1
                         for k, i in enumerate(nz))
        comp_out = {str(gens[i]): str(prim[i]) for i in nz}
        if is_scaling:
            weights = {str(gens[i]): int(spy.Poly(prim[i], gens[i]).LC()) for i in nz}
            return {'type': 'scaling', 'weights': weights,
                    'components': comp_out, 'degree': 1}
        return {'type': 'general', 'weights': None, 'components': comp_out,
                'degree': int(max(degs)) if degs else 0}
    except Exception:
        return None


_CLASSIFY_MAX_CHARS = 20000


def classify_generator(vector):
    """Classify one non-identifiability direction and return its canonical
    poly-primitive differential generator.

    `vector` is {coordinate: component_str}, the components eta_i of
    X = sum_i eta_i d/dz_i (a scaling as {z_i: "w_i*z_i"}). X is defined up to a
    factor h(z); the gauge is fixed by clearing the common denominator, dividing by
    the polynomial content and normalising to integer-primitive form.

      scaling : every eta_i = w_i * z_i. The orbit is a ray, so any one coordinate
                can be fixed at any value.
      general : anything else. The orbit is curved, so fixing a coordinate is not
                free; the direction is removed by reparametrising onto invariants.

    Returns {'type', 'weights', 'components', 'degree'}: integer scaling weights
    (None unless scaling), the canonical generator {z_i: str(eta_i)} and its total
    degree (-1 when not polynomial)."""
    items = [(str(k), str(v)) for k, v in vector.items()]
    fallback = {'type': 'general', 'weights': None,
                'components': {k: v for k, v in items}, 'degree': -1}
    if not items:
        return {'type': 'general', 'weights': None, 'components': {}, 'degree': 0}
    # one moved coordinate: z*d/dz spans the same direction
    moved = [(k, v) for k, v in items if v.strip() != '0']
    if len(moved) == 1:
        z = moved[0][0]
        return {'type': 'scaling', 'weights': {z: 1}, 'components': {z: z}, 'degree': 1}
    # the canonical form of very long components costs more than it is worth
    if sum(len(v) for _, v in items) > _CLASSIFY_MAX_CHARS:
        return fallback

    # symbol table so names like E, I, S, N are model symbols, not sympy constants
    _, parse = _make_local_parse([k + " = " + v for k, v in items])
    comps = {}
    try:
        for k, v in items:
            comps[spy.Symbol(k)] = parse(v)
    except Exception:
        return fallback

    varset = set(comps.keys())
    for e in comps.values():
        varset |= set(spy.sympify(e).free_symbols)
    gens = sorted(varset, key=spy.default_sort_key)
    n = len(gens)
    infis = [comps.get(z, spy.Integer(0)) for z in gens]

    # a wide direction (many symbols) goes through factor lists: the multivariate gcd
    # of the expanded numerators costs minutes on an entry like
    # ksec*(Km + R1 + ... + R30)**2, its factor list is read off the product
    if n > 12:
        wide = _classify_factored(infis, gens)
        if wide is not None:
            return wide

    # canonical poly-primitive representative (gauge fix): clear the common
    # denominator, then divide by the polynomial content of the numerators
    try:
        dens = [spy.fraction(spy.together(e))[1] for e in infis]
        L = spy.Integer(1)
        for d in dens:
            L = spy.lcm(L, d)
        nums = [spy.expand(spy.cancel(e * L)) for e in infis]
        for e in nums:
            spy.Poly(e, *gens)                 # raises if not polynomial in gens
    except Exception:
        return fallback
    g = spy.Integer(0)
    for e in nums:
        g = spy.gcd(g, e)
    if g == 0:
        g = spy.Integer(1)
    prim = [spy.expand(spy.cancel(e / g)) for e in nums]
    prim = _canon_int_vector(prim, gens)

    # read the class off the canonical representative
    is_scaling, maxdeg = True, 0
    for i, e in enumerate(prim):
        if e == 0:
            continue
        P = spy.Poly(e, *gens)
        maxdeg = max(maxdeg, P.total_degree())
        monoms = list(P.as_dict().keys())
        if not (len(monoms) == 1 and sum(monoms[0]) == 1 and monoms[0][i] == 1):
            is_scaling = False

    comp_out = {str(gens[i]): str(prim[i]) for i in range(n) if prim[i] != 0}
    if is_scaling:
        weights = {}
        for i in range(n):
            if prim[i] != 0:
                key = tuple(1 if j == i else 0 for j in range(n))
                weights[str(gens[i])] = int(spy.Poly(prim[i], *gens).as_dict()[key])
        return {'type': 'scaling', 'weights': weights,
                'components': comp_out, 'degree': 1}
    return {'type': 'general', 'weights': None,
            'components': comp_out, 'degree': int(maxdeg)}
