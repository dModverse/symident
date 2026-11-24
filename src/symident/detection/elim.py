"""Input-output relations by elimination (docs/method/global.rst, "The exact count"): the states eliminated by
resultants over Z[theta] in the order of StructuralIdentifiability.jl (find_ioequations), each resultant factored, and
a factor dropped where it is nonzero at a point of a trajectory modulo a prime. The product of the factors kept
vanishes along every trajectory, so the relation is exact."""

import math
import types

import flint

from ..gfp import all_primes
from . import ioproj

# test points of the refutation of a factor
_POINTS = 3


class _Ring:
    """The polynomials over Z in the coordinates, the unknown states and their derivatives, the known states and the
    derivatives of the observables up to `order`."""

    def __init__(self, rel, order):
        self.rel = rel
        states = [j for j in range(rel.ns) if rel.X[j] is not None]
        self.unknown = [j for j in states if j not in rel.known]
        self.known = [j for j in states if j in rel.known]
        state_names = {str(rel.X[j]) for j in states}
        self.params = [n for n in rel.names if n not in state_names]
        self.order = order
        self.names = list(self.params) + [f"si_x{j}" for j in self.unknown] + [f"si_d{j}" for j in self.unknown] + \
            [f"si_u{j}" for j in self.known] + [f"si_y{r}_{k}" for r in range(len(rel.y)) for k in range(order + 1)]
        self.at = {n: i for i, n in enumerate(self.names)}
        self.ctx = flint.fmpz_mpoly_ctx.get(tuple(self.names), "lex")
        self.gens = self.ctx.gens()
        # the generators of the ring of the relations in this ring
        self.map = {i: self.at[n] for i, n in enumerate(rel.names) if n in self.at}
        for j in states:
            g = rel.names.index(str(rel.X[j]))
            self.map[g] = self.at[f"si_x{j}" if j in self.unknown else f"si_u{j}"]

    def x(self, j):
        return self.gens[self.at[f"si_x{j}"]]

    def d(self, j):
        return self.gens[self.at[f"si_d{j}"]]

    def y(self, r, k):
        return self.gens[self.at[f"si_y{r}_{k}"]]

    def poly(self, P):
        """A polynomial of the ring of the relations, over Q, as a primitive polynomial over Z, and the factor."""
        terms = list(P.terms())
        den = math.lcm(*[int(c.denominator) for _, c in terms]) if terms else 1
        n = len(self.names)
        d = {}
        for m, c in terms:
            e = [0] * n
            for i, k in enumerate(m):
                if k:
                    e[self.map[i]] += k
            d[tuple(e)] = int(c.numerator) * (den // int(c.denominator))
        return self.ctx.from_dict(d)

    def parts(self, q):
        """(numerator, denominator) over Z of a function of the ring of the relations."""
        if hasattr(q, "numer"):
            N, D = self.poly(q.numer), self.poly(q.denom)
            # the rational factors of numerator and denominator cleared apart
            fn = math.lcm(*[int(c.denominator) for _, c in q.numer.terms()]) if q.numer.terms() else 1
            fd = math.lcm(*[int(c.denominator) for _, c in q.denom.terms()]) if q.denom.terms() else 1
            return N * fd, D * fn
        return self.poly(q), self.ctx.from_dict({(0,) * len(self.names): 1})


def _deg(P, v):
    return P.degrees()[v] if not P.is_zero() else -1


def _coeff(P, v, k):
    """The coefficient of variable v to the power k, a polynomial in the others."""
    out = {}
    for m, c in P.to_dict().items():
        if m[v] == k:
            mm = list(m)
            mm[v] = 0
            out[tuple(mm)] = c
    return P.context().from_dict(out)


class Eliminator:
    """The elimination for one model: the equations, the derivation and the test points of the refutation."""

    def __init__(self, rel, rng, terms=math.inf):
        self.rel, self.rng, self.terms = rel, rng, terms
        ns = sum(1 for j in range(rel.ns) if rel.X[j] is not None and j not in rel.known)
        self.R = R = _Ring(rel, ns + 1)
        self.p = all_primes()[0]
        self.points = self._points()
        # x_j' den_j = num_j for the unknown states; the derivatives of the known states as polynomials
        self.x_eqs = {}
        for j in R.unknown:
            N, D = R.parts(rel.f[j])
            self.x_eqs[j] = D * R.d(j) - N
        self.known_d = {}
        for j in R.known:
            N, D = R.parts(rel.f[j])
            if not D.is_constant():
                raise ValueError("a known state with a rational right-hand side")
            self.known_d[j] = (N, D)
        self.y_eqs = {}
        for r, q in enumerate(rel.y):
            M, E = R.parts(q)
            self.y_eqs[r] = E * R.y(r, 0) - M
        self.steps = []
        # per state eliminated, (A, B) with A x + B = 0 along every trajectory, or None
        self.states = {}

    def _points(self):
        """Values of every variable at the start of trajectories at random coordinates modulo p."""
        R, rel, p = self.R, self.rel, self.p
        out = []
        tries = 0
        while len(out) < _POINTS:
            tries += 1
            if tries > 20:
                raise ValueError("no trajectory without a pole at random coordinates")
            vals = {s: self.rng.randrange(1, p) for s in rel.syms}
            got = ioproj.trajectory(rel, vals, p, R.order + 2)
            if got is None:
                continue
            obs, states = got
            pt = [0] * len(R.names)
            byname = {str(s): v for s, v in vals.items()}
            for n in R.params:
                pt[R.at[n]] = byname.get(n, self.rng.randrange(1, p)) % p
            for j in R.unknown:
                pt[R.at[f"si_x{j}"]] = int(states[j][0]) % p
                pt[R.at[f"si_d{j}"]] = int(states[j][1]) % p
            for j in R.known:
                pt[R.at[f"si_u{j}"]] = int(states[j][0]) % p
            for r in range(len(rel.y)):
                for k in range(R.order + 1):
                    pt[R.at[f"si_y{r}_{k}"]] = int(obs[r][k]) * math.factorial(k) % p
            out.append(pt)
        return out

    def value(self, P, pt):
        return int(P(*pt)) % self.p

    def derive(self, P):
        """The derivative along the flow: x_j -> d_j, u -> the right-hand side of u, y_r,k -> y_r,k+1."""
        R = self.R
        out = P.context().from_dict({})
        for j in R.unknown:
            v = R.at[f"si_x{j}"]
            if _deg(P, v) > 0:
                out += P.derivative(R.names[v]) * R.d(j)
        for j, (N, D) in self.known_d.items():
            v = R.at[f"si_u{j}"]
            if _deg(P, v) > 0:
                if j in getattr(self.rel, "cut", ()):
                    raise ValueError("the derivative of the last jet of a known input that the relations keep")
                c = int(D.coeffs()[0])
                if c != 1:
                    raise ValueError("a known state whose derivative has a rational coefficient")
                out += P.derivative(R.names[v]) * N
        for r in range(len(self.rel.y)):
            for k in range(R.order + 1):
                v = R.at[f"si_y{r}_{k}"]
                if _deg(P, v) > 0:
                    if k == R.order:
                        raise ValueError("the derivatives of an observable past the order of the ring")
                    out += P.derivative(R.names[v]) * R.y(r, k + 1)
        if any(_deg(P, R.at[f"si_d{j}"]) > 0 for j in R.unknown):
            raise ValueError("a derivative of a state in a polynomial to differentiate")
        return out

    def eliminate(self, f, g, v):
        """A polynomial free of variable v from f and g, both vanishing along every trajectory: their resultant in v
        after the reductions of the leading coefficients, factored, with the factors nonzero at a test point
        dropped."""
        self._within(f, g)
        while not f.is_zero() and not g.is_zero():
            if _deg(f, v) > _deg(g, v):
                f, g = g, f
            df, dg = _deg(f, v), _deg(g, v)
            lf, lg = _coeff(f, v, df), _coeff(g, v, dg)
            q, r = divmod(lg, lf)
            if r.is_zero():
                g = g - q * f * self.R.gens[v] ** (dg - df)
                continue
            if df == dg:
                q, r = divmod(lf, lg)
                if r.is_zero():
                    f = f - q * g
                    continue
            break
        if f.is_zero() or g.is_zero():
            return f + g
        if _deg(f, v) == 0:
            res = f
        elif _deg(g, v) == 0:
            res = g
        else:
            # polynomials in v^d: the resultant in v^d
            exps = {m[v] for P in (f, g) for m in P.monoms()}
            d = math.gcd(*exps)
            if d > 1:
                f, g = (_deflate(P, v, d) for P in (f, g))
            res, extra = _resultant(f, g, v)
            return self.keep(res, extra)
        return self.keep(res)

    def keep(self, P, extra=()):
        """The product of the irreducible factors of P and of the polynomials `extra`, whose product vanishes along
        every trajectory, that vanish at every test point: one of the factors vanishes along every trajectory, and a
        factor nonzero at a point of a trajectory is not it."""
        if P.is_zero():
            return P
        self._within(P, *extra)
        facs = []
        for Q in [P] + list(extra):
            for F, _ in Q.factor()[1]:
                if not F.is_constant() and all(F != G and F != -G for G in facs):
                    facs.append(F)
        kept = [F for F in facs if all(self.value(F, pt) == 0 for pt in self.points)]
        self.steps.append((len(P), [len(F) for F in facs], len(kept)))
        if not kept:
            raise ValueError("every factor of a resultant is nonzero at a point of a trajectory")
        out = kept[0]
        for F in kept[1:]:
            out = out * F
        return out

    def _within(self, *polys):
        """A ValueError where a polynomial has more terms than the cap."""
        n = max(len(P) for P in polys)
        if n > self.terms:
            raise ValueError(f"a polynomial of {n:,} terms, past the cap elimination_terms of {int(self.terms):,}")

    def run(self):
        """{observable: relation} in the derivatives of the observables, the known states and the coordinates."""
        R = self.R
        y_orders = {r: 0 for r in self.y_eqs}
        while True:
            degs = []
            for r, eq in self.y_eqs.items():
                ds = [_deg(eq, R.at[f"si_x{j}"]) for j in self.x_eqs if _deg(eq, R.at[f"si_x{j}"]) > 0]
                if ds:
                    degs.append((min(ds) * len(eq), min(ds), -ds.count(min(ds)) + len(eq) / 30, len(eq), r))
            if not degs:
                break
            r = sorted(degs)[0][-1]
            y_orders[r] += 1
            nxt = self.derive(self.y_eqs[r])
            for j, eq in self.x_eqs.items():
                nxt = self.eliminate(eq, nxt, R.at[f"si_d{j}"])
            for r2, eq in self.y_eqs.items():
                if r2 != r:
                    nxt = self.eliminate(nxt, eq, R.at[f"si_y{r2}_{y_orders[r2]}"])
            cand = sorted((_deg(self.y_eqs[r], R.at[f"si_x{j}"]), _deg(nxt, R.at[f"si_x{j}"]), j) for j in self.x_eqs
                          if _deg(self.y_eqs[r], R.at[f"si_x{j}"]) > 0)
            v = cand[0][2]
            vi = R.at[f"si_x{v}"]
            del self.x_eqs[v]
            for j in list(self.x_eqs):
                self.x_eqs[j] = self.eliminate(self.x_eqs[j], self.y_eqs[r], vi)
            for r2 in self.y_eqs:
                if r2 != r:
                    self.y_eqs[r2] = self.eliminate(self.y_eqs[r2], self.y_eqs[r], vi)
            self.states[v] = self.linear(self.y_eqs[r], nxt, vi)
            self.y_eqs[r] = self.eliminate(self.y_eqs[r], nxt, vi)
        return self.y_eqs, y_orders

    def linear(self, f, g, v):
        """(A, B) with A v + B in the ideal of f and g, A nonzero at a test point: one of them where it has degree one
        in v, else their first subresultant in v; None where A vanishes at the test points."""
        self._within(f, g)
        pair = None
        for P in (f, g):
            if _deg(P, v) == 1:
                c = _coeffs(P, v)
                pair = (c[1], c[0])
                break
        if pair is None and min(_deg(f, v), _deg(g, v)) >= 2:
            pair = _subresultant1(f, g, v)
        if pair is None or pair[0].is_zero() or all(self.value(pair[0], pt) == 0 for pt in self.points):
            return None
        return pair


def _deflate(P, v, d):
    """P(v^(1/d)) for a polynomial in v^d."""
    out = {}
    for m, c in P.to_dict().items():
        mm = list(m)
        mm[v] //= d
        out[tuple(mm)] = c
    return P.context().from_dict(out)


def _coeffs(P, v):
    """The coefficients of P in variable v, lowest degree first."""
    ctx = P.context()
    out = [{} for _ in range(_deg(P, v) + 1)]
    for m, c in P.to_dict().items():
        mm = list(m)
        k = mm[v]
        mm[v] = 0
        out[k][tuple(mm)] = c
    return [ctx.from_dict(d) for d in out]


def _resultant(f, g, v):
    """(D, factors): the determinant D of the Bezout matrix of f and g in v (the Sylvester matrix where the lower
    degree is one), the greatest common divisors of the rows and columns divided out first, and these divisors: D
    times their product is the resultant times a power of a leading coefficient, as in StructuralIdentifiability.jl."""
    if _deg(f, v) > _deg(g, v):
        f, g = g, f
    df, dg = _deg(f, v), _deg(g, v)
    zero = f.context().from_dict({})
    cf, cg = _coeffs(f, v), _coeffs(g, v)
    if df > 1:
        n = dg
        cf += [zero] * (n + 1 - len(cf))
        M = [[sum((cf[j + k + 1] * cg[i - k] - cg[j + k + 1] * cf[i - k] for k in range(min(i, n - 1 - j) + 1)), zero)
              for j in range(n)] for i in range(n)]
    else:
        n = df + dg
        M = [[zero] * n for _ in range(n)]
        for i in range(df):
            for j in range(dg + 1):
                M[i][i + j] = cg[j]
        for i in range(dg):
            for j in range(df + 1):
                M[df + i][i + j] = cf[j]
    removed = []
    for cells in [[(i, j) for j in range(n)] for i in range(n)] + [[(i, j) for i in range(n)] for j in range(n)]:
        nz = [(i, j) for i, j in cells if not M[i][j].is_zero()]
        if not nz:
            continue
        h = M[nz[0][0]][nz[0][1]]
        for i, j in nz[1:]:
            h = h.gcd(M[i][j])
        if not h.is_constant():
            for i, j in nz:
                M[i][j] = M[i][j] // h
            removed.append(h)
    return _det(M), removed


def _subresultant1(f, g, v):
    """(A, B): the first subresultant A v + B of f and g in v, of degrees at least two, a combination U f + V g."""
    m, n = _deg(f, v), _deg(g, v)
    cf, cg = _coeffs(f, v), _coeffs(g, v)
    zero = f.context().from_dict({})
    size = m + n - 2
    # the rows v^k f for k < n - 1 and v^k g for k < m - 1, the columns of the powers size .. 2 and then one power
    rows = [(k, cf) for k in range(n - 1)] + [(k, cg) for k in range(m - 1)]

    def entry(row, power):
        k, c = row
        d = power - k
        return c[d] if 0 <= d < len(c) else zero
    out = []
    for last in (1, 0):
        powers = list(range(size, 1, -1)) + [last]
        out.append(_det([[entry(row, q) for q in powers] for row in rows]))
    return out[0], out[1]


def _det(M):
    """The determinant by expansion along the rows, the minors cached."""
    n = len(M)
    cache = {}
    one = M[0][0].context().from_dict({(0,) * M[0][0].context().nvars(): 1})

    def minor(row, cols):
        if row == n:
            return one
        key = (row, cols)
        if key in cache:
            return cache[key]
        out, sign = None, 1
        for j in range(n):
            if cols >> j & 1:
                continue
            if not M[row][j].is_zero():
                t = M[row][j] * minor(row + 1, cols | 1 << j)
                t = t if sign > 0 else -t
                out = t if out is None else out + t
            sign = -sign
        out = out if out is not None else one - one
        cache[key] = out
        return out
    return minor(0, 0)


def split(P, R):
    """{(monomial in the derivatives of the observables and the known states): {monomial in the coordinates: integer}}
    of a relation."""
    npar = len(R.params)
    out = {}
    for m, c in P.to_dict().items():
        out.setdefault(tuple(m[npar:]), {})[tuple(m[:npar])] = int(c)
    return out


def relation_projection(rel, R, P):
    """(base, support, coefficients) of the relation P on the base functions of a projection of ioproj, the
    derivatives y_r^(k), k <= ms[r], and the known states read: the orders and the number of the base functions as
    ioproj.series_rank() reads them, the support in the exponents of that base and per monomial {monomial in the
    parameters: integer}; None where P reads a known state outside the base."""
    terms = split(P, R)
    ny = len(rel.y)
    ms = [-1] * ny
    for m in terms:
        for r in range(ny):
            for k in range(R.order + 1):
                if m[2 * len(R.unknown) + len(R.known) + r * (R.order + 1) + k]:
                    ms[r] = max(ms[r], k)
    inputs = rel.inputs(max(ms))
    # what ioproj.series_rank() reads of a projection: the orders and the number of base functions, no symbolic
    # derivatives
    pr = types.SimpleNamespace(ms=ms, nb=sum(k + 1 for k in ms if k >= 0) + len(inputs), mons=None)
    off_known = 2 * len(R.unknown)
    support, coefs = [], []
    for m, co in terms.items():
        if any(m[:off_known]):
            return None
        e = []
        for r in range(ny):
            if ms[r] >= 0:
                e += [m[off_known + len(R.known) + r * (R.order + 1) + k] for k in range(ms[r] + 1)]
        known = dict(zip(R.known, m[off_known:off_known + len(R.known)]))
        if any(k and j not in inputs for j, k in known.items()):
            return None
        e += [known.get(j, 0) for j in inputs]
        support.append(tuple(e))
        coefs.append(co)
    pr.mons = support
    return pr, support, coefs


def normalized(coefs):
    """The index of the reference coefficient, a monomial if one is, else one of fewest terms, and the order of the
    others by terms and degree."""
    ref = min(range(len(coefs)), key=lambda j: (len(coefs[j]), max(sum(m) for m in coefs[j]), j))
    order = sorted((j for j in range(len(coefs)) if j != ref),
                   key=lambda j: (len(coefs[j]), max(sum(m) for m in coefs[j]), j))
    return ref, order


def gradient_select(coefs, ref, order, nparams, p, rng):
    """The coefficients over the reference whose gradients in the parameters raise the rank at a random point modulo p,
    fewest terms first, then as many more that are no constant multiple of the reference."""
    theta = [rng.randrange(1, p) for _ in range(nparams)]

    def val(co, th):
        s = 0
        for m, c in co.items():
            t = c % p
            for x, k in zip(th, m):
                if k:
                    t = t * pow(x, k, p) % p
            s += t
        return s % p

    def grad(co):
        out = []
        for v in range(nparams):
            d = {}
            for m, c in co.items():
                if m[v]:
                    mm = list(m)
                    mm[v] -= 1
                    d[tuple(mm)] = d.get(tuple(mm), 0) + c * m[v]
            out.append(val(d, theta) if d else 0)
        return out
    r0, dr = val(coefs[ref], theta), grad(coefs[ref])
    if not r0:
        return [], []
    inv = pow(r0, -1, p)
    basis, piv, first, rest = [], [], [], []
    for j in order:
        cj, dj = val(coefs[j], theta), grad(coefs[j])
        # the gradient of c_j / c_ref
        row = [(a * r0 - cj * b) * inv * inv % p for a, b in zip(dj, dr)]
        if not any(row):
            continue
        for b, c in zip(basis, piv):
            if row[c]:
                row = [(x - row[c] * y) % p for x, y in zip(row, b)]
        c = next((i for i, x in enumerate(row) if x), None)
        if c is not None:
            ic = pow(row[c], -1, p)
            basis.append([x * ic % p for x in row])
            piv.append(c)
            first.append(j)
        else:
            rest.append(j)
    return first, first + rest[:len(first)]


def eliminate_relations(rel, rng, terms=math.inf):
    """The relations of the observables by elimination, {"relations": {observable: flint polynomial}, "orders",
    "ring", "steps", "states"}; a ValueError where a polynomial of the elimination has more than `terms` terms."""
    el = Eliminator(rel, rng, terms)
    eqs, orders = el.run()
    return {"relations": eqs, "orders": orders, "ring": el.R, "steps": el.steps, "states": el.states}
