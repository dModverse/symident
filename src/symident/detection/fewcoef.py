"""A few coefficients of a projection past ioproj.TERMS, verified (docs/method/global.rst, "The exact count"): the
relation is shown to exist by kernels at random points modulo random primes, the coefficients over a coefficient that is
a monomial in the parameters are Laurent polynomials, interpolated by Ben-Or and Tiwari and checked at the same random
points, each step with the error bound of the Schwartz-Zippel lemma."""

import itertools
import math

import numpy as np
import sympy as spy

from .. import _core
from ..gfp import PRIMES, all_primes, rref_modp, sieve
from . import ioproj, work

# the primes of the trials: uniform among the primes from _LOW to the largest prime of the engine arithmetic
_LOW = 1 << 29
# trials at random points, points per axis line at the start and at most, candidates per parameter
_TRIALS = 5
_AXIS = 24
_AXIS_MAX = 96
_CANDIDATES = 3
# the most terms of an interpolated coefficient and candidate exponent vectors of one
_TERMS = 40
_BOX = 200000


def _is_prime(n):
    """Miller-Rabin with the first twelve prime bases, deterministic below 3.3e24."""
    if n < 2:
        return False
    small = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    if n in small:
        return True
    if any(n % b == 0 for b in small):
        return False
    d, s = n - 1, 0
    while not d % 2:
        d, s = d // 2, s + 1
    for b in small:
        x = pow(b, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def random_prime(rng, hi):
    """A prime uniform among those from _LOW to `hi`: uniform integers until one is prime."""
    while True:
        n = rng.randrange(_LOW, hi + 1)
        if _is_prime(n):
            return n


def primes_at_least(lo, hi):
    """A lower bound on the number of primes from lo to hi: pi(x) > x / ln x for x >= 17 and pi(x) < 1.25506 x / ln x
    for x > 1 (Rosser and Schoenfeld 1962)."""
    return hi / math.log(hi) - 1.25506 * (lo - 1) / math.log(lo - 1)


class _Oracle:
    """The kernel vector of the projection at a parameter point modulo p, normalized at the last monomial of the
    support, cached per point; None where the kernel is not one vector with that entry free."""

    def __init__(self, pr):
        self.pr, self.cache, self.calls = pr, {}, 0

    def __call__(self, theta, p):
        key = (tuple(int(x) % p for x in theta), p)
        if key not in self.cache:
            self.calls += 1
            self.cache[key] = self.pr.vector(list(key[0]), p)
        return self.cache[key]


def _pure(poly):
    """The exponent of a polynomial (coefficients lowest first) with one term, None otherwise."""
    nz = [i for i, c in enumerate(poly) if c]
    return nz[0] if len(nz) == 1 else None


def _axis(orc, beta, p, npts, end):
    """Per parameter v the points t and the kernel vectors on the line beta with theta_v = t; None past the work
    count `end` (work.ticks()) or where the kernel is missing too often."""
    lines = []
    for v in range(len(beta)):
        ts, vals, t = [], [], 0
        while len(ts) < npts:
            t += 1
            if t > 3 * npts + 10 or work.ticks() > end:
                return None
            x = list(beta)
            x[v] = t
            vec = orc(x, p)
            if vec is not None:
                ts.append(t)
                vals.append(vec)
        lines.append((ts, np.array(vals, dtype=np.int64)))
    return lines


def _ranges(lines, p, r0, entries, beta=None):
    """{entry: per parameter (lowest, highest) exponent} of the entries over entry r0 that are Laurent polynomials on
    every axis line; with the number of entries found on every line that are not, and with `beta`, the point of the
    lines, {entry: gradient there}."""
    out, other, grads = {}, 0, {}
    for e in entries:
        rg, gr = [], []
        for ts, vals in lines:
            den = vals[:, r0]
            if not den.all():
                break
            w = [int(a) * pow(int(b), -1, p) % p for a, b in zip(vals[:, e], den)]
            got = ioproj.ratrec1(ts, w, p)
            if got is None:
                break
            a = _pure(got[1])
            nz = [i for i, c in enumerate(got[0]) if c]
            if a is None:
                other += 1
                break
            if not nz:
                break
            rg.append((nz[0] - a, nz[-1] - a))
            if beta is not None:
                # d/dt N(t) t^-a at t = beta_v
                t0 = beta[len(rg) - 1]
                N = sum(c * pow(t0, i, p) for i, c in enumerate(got[0])) % p
                dN = sum(i * c * pow(t0, i - 1, p) for i, c in enumerate(got[0]) if i) % p
                gr.append((dN * t0 - a * N) * pow(pow(t0, a + 1, p), -1, p) % p)
        else:
            out[e] = rg
            grads[e] = gr
    return (out, other, grads) if beta is not None else (out, other)


def _monomial_ref(lines, p, r):
    """The entries whose coefficient is a monomial in the parameters, least degree first: on every axis line the
    quotient by the normalizing entry has a numerator with one term, and every other entry over it is a Laurent
    polynomial there."""
    cands = []
    for e in range(r):
        tot = 0
        for ts, vals in lines:
            got = ioproj.ratrec1(ts, [int(x) for x in vals[:, e]], p)
            a = None if got is None else _pure(got[0])
            if a is None:
                break
            tot += a
        else:
            cands.append((tot, e))
    out = []
    for _, e in sorted(cands)[:8]:
        found, other = _ranges(lines, p, e, range(r))
        if not other and len(found) > 1:
            out.append((e, found))
    return out


class _BT:
    """The kernel vectors at the points c b^k modulo q of Ben-Or and Tiwari, b the first primes, kept for every
    entry."""

    def __init__(self, orc, n, q, rng):
        self.orc, self.q, self.rng, self.n = orc, q, rng, n
        self.bases = sieve(n)
        self._start()

    def _start(self):
        self.c = [self.rng.randrange(1, self.q) for _ in range(self.n)]
        self.pts, self.vecs = [], []

    def extend(self, m, end):
        """At least m points; False past the work count `end` or where the kernel is missing three times."""
        restarts = 0
        while len(self.vecs) < m:
            if work.ticks() > end:
                return False
            k = len(self.vecs)
            x = [ci * pow(b, k, self.q) % self.q for ci, b in zip(self.c, self.bases)]
            v = self.orc(x, self.q)
            if v is None:
                restarts += 1
                if restarts > 2:
                    return False
                self._start()
                continue
            self.pts.append(x)
            self.vecs.append(v)
        return True

    def sequence(self, e, r0, low):
        """The values of entry e over entry r0 at the points, times theta^-low; None where entry r0 is zero."""
        q = self.q
        out = []
        for x, v in zip(self.pts, self.vecs):
            if not v[r0]:
                return None
            s = int(v[e]) * pow(int(v[r0]), -1, q) % q
            for xi, L in zip(x, low):
                if L:
                    s = s * pow(xi, -L, q) % q
            out.append(s)
        return out

    def solve(self, e, r0, rg):
        """{exponents: residue} of the Laurent polynomial of entry e over entry r0 with the exponent ranges rg; None
        where the terms are not found."""
        q = self.q
        low = [L for L, _ in rg]
        seq = self.sequence(e, r0, low)
        if seq is None or 2 * int(_core.sym_b_morder(seq, q)) >= len(seq):
            return None
        cand = np.array(list(itertools.product(*[range(H - L + 1) for L, H in rg])), dtype=np.int64).reshape(-1, self.n)
        res = _core.sym_mono_residues(cand, self.bases, q)
        got = _core.sym_sparse_poly(seq, cand, res, q)
        if got.get("status") != "ok":
            return None
        exps = np.asarray(got["exps"], dtype=np.int64).reshape(-1, self.n)
        out = {}
        for m, a in zip(exps, np.atleast_1d(got["coeffs"])):
            m = tuple(int(x) for x in m)
            shift = math.prod(pow(ci, k, q) for ci, k in zip(self.c, m)) % q
            out[tuple(k + L for k, L in zip(m, low))] = int(a) * pow(shift, -1, q) % q
        return out


def _lift(per_prime, primes):
    """{exponents: Fraction} from the residues per prime, None where the supports differ or a coefficient has no
    bounded rational."""
    keys = sorted(per_prime[0])
    if any(sorted(d) != keys for d in per_prime[1:]):
        return None
    got = ioproj._ratrec([[d[k] for k in keys] for d in per_prime], primes)
    return None if got is None else dict(zip(keys, got))


def _value(co, theta, q):
    """The Laurent polynomial {exponents: Fraction} at theta modulo q; None where a denominator is zero there."""
    s = 0
    for m, c in co.items():
        if not c.denominator % q:
            return None
        t = c.numerator % q * pow(c.denominator, -1, q) % q
        for x, k in zip(theta, m):
            if k:
                t = t * pow(int(x), k, q) % q
        s += t
    return s % q


def _gradient(co, theta, q):
    """The gradient of the Laurent polynomial {exponents: Fraction} at theta modulo q; None entries as _value()."""
    out = []
    for v in range(len(theta)):
        d = {}
        for m, c in co.items():
            if m[v]:
                mm = list(m)
                mm[v] -= 1
                d[tuple(mm)] = d.get(tuple(mm), 0) + c * m[v]
        out.append(_value(d, theta, q) if d else 0)
    return out


def _rank_select(found, order, n, q, rng):
    """The entries of `order` whose gradients raise the rank at a random point modulo q, then as many more."""
    theta = [rng.randrange(1, q) for _ in range(n)]
    basis, piv, first = [], [], []
    for e in order:
        row = _gradient(found[e], theta, q)
        if any(x is None for x in row):
            continue
        for b, c in zip(basis, piv):
            if row[c]:
                row = [(x - row[c] * y) % q for x, y in zip(row, b)]
        c = next((j for j, x in enumerate(row) if x), None)
        if c is not None:
            inv = pow(row[c], -1, q)
            basis.append([x * inv % q for x in row])
            piv.append(c)
            first.append(e)
    return first, first + [e for e in order if e not in first][: len(first)]


class _Bounds:
    """Degree and height bounds of the maximal minors of the matrix of the monomials of the projection at r random
    states, the denominators of each row cleared (docs/method/global.rst, "The exact count")."""

    def __init__(self, pr):
        J = pr.base.J
        funcs = pr.base.funcs

        def deg(P):
            return max((sum(m) for m in P.itermonoms()), default=0)

        def height(P):
            # log2 of the 1-norm of P times the least common multiple of the denominators of the coefficients
            dens = [int(c.denominator) for c in P.itercoeffs()]
            L = math.lcm(*dens) if dens else 1
            return math.log2(max(1, sum(abs(int(c.numerator) * (L // int(c.denominator))) for c in P.itercoeffs())))

        dN = [deg(f.numer) for f in funcs]
        hN = [height(f.numer) for f in funcs]
        used = sorted({k for f in funcs for k, _ in f.exps})
        dF = {k: deg(J.fac[k]) for k in used}
        hF = {k: height(J.fac[k]) for k in used}
        t = [{k: sum(m[v] * dict(funcs[v].exps).get(k, 0) for v in range(len(funcs))) for k in used} for m in pr.mons]
        T = {k: max(tj[k] for tj in t) for k in used}
        self.r = len(pr.mons)
        self.degree = sum(
            sum(m[v] * dN[v] for v in range(len(funcs))) + sum((T[k] - tj[k]) * dF[k] for k in used)
            for m, tj in zip(pr.mons, t)
        )
        self.height = math.lgamma(self.r + 1) / math.log(2) + sum(
            sum(m[v] * hN[v] for v in range(len(funcs))) + sum((T[k] - tj[k]) * hF[k] for k in used)
            for m, tj in zip(pr.mons, t)
        )
        self.rows_degree = sum(dF.values())

    def trial(self, extra_degree=0, extra_height=0.0, hi=None):
        """The probability that one trial passes where the claim is false: a maximal minor, or a minor times the
        candidate less another, is zero at the random point modulo a random prime from _LOW to hi."""
        hi = hi or max(PRIMES)
        q = _LOW
        accept = 1 - (self.r + 6) * self.rows_degree / (q - 1)
        bad = math.ceil((self.height + extra_height + 1) / math.log2(_LOW))
        return (self.degree + extra_degree) / ((q - 1) * accept) + bad / primes_at_least(_LOW, hi)


def verified_functions(series, syms, large, cap, rng, log):
    """{"functions", "exist", "checks", "why", ...}: for the first projection of `large` ({"profile", "observable",
    "support"} from fieldproof), the coefficients over one that is a monomial in the parameters, interpolated as Laurent
    polynomials and verified, as many as raise the rank of their gradients and as many more, within `cap` entries of
    the matrices reduced. `exist` bounds the probability that no relation exists on the support, `checks` per
    function that it differs from the coefficient of the relation; both are those of _TRIALS independent trials."""
    from . import fieldproof

    end = work.ticks() + cap
    out = {"functions": [], "exist": None, "checks": {}, "why": None, "kernels": 0, "terms": None}
    rel = fieldproof._relations(series, syms)
    idx = [k for k, s_ in enumerate(syms) if s_ in set(series.params)]
    pos = [rel.names.index(str(syms[k])) for k in idx]
    params = [syms[k] for k in idx]
    n = len(idx)
    big = sorted(large, key=lambda g: len(g["support"]))[0]
    pr = ioproj.Projection.output(rel, big["profile"], big["observable"], pos, rng)
    if pr is None:
        out["why"] = "the projection is not rebuilt"
        return out
    pr.mons = [tuple(m) for m in big["support"]]
    r = len(pr.mons)
    out["terms"] = r
    orc = _Oracle(pr)
    p = all_primes()[0]

    def stop(why):
        out["why"] = why
        out["kernels"] = orc.calls
        log.append(f"projection of {r} terms: {why}")
        return out

    # the trajectory has the one relation: rank r - 1 of the Taylor coefficients at one point, exact
    rk = ioproj.series_rank(rel, pr, pr.mons, syms, rng, p)
    if rk != r - 1:
        return stop(f"rank {rk} of {r} terms along the trajectory")
    # the axis lines through one point: a coefficient that is a monomial, and the exponent ranges of the others
    beta = [rng.randrange(1, p) for _ in range(n)]
    npts, ref = _AXIS, []
    while not ref:
        lines = _axis(orc, beta, p, npts, end)
        if lines is None:
            return stop(f"the axis lines reached the cap of {cap:,} matrix entries")
        ref = _monomial_ref(lines, p, r)
        if not ref and npts >= _AXIS_MAX:
            return stop(f"no coefficient is a monomial in the parameters on axis lines of {npts} points")
        npts *= 2
    r0, _ = ref[0]
    ranges, _, grads = _ranges(lines, p, r0, range(r), beta)
    # the rank of the coefficients found on the axis lines: the rank the functions verified aim at
    target = rref_modp(np.array(list(grads.values()), dtype=np.int64).reshape(-1, n), p)["rank"]
    # the simplest entries first: the fewest exponent vectors in the ranges
    order = sorted(
        (
            e
            for e, rg in ranges.items()
            if e != r0 and any(L or H for L, H in rg) and math.prod(H - L + 1 for L, H in rg) <= _BOX
        ),
        key=lambda e: (math.prod(H - L + 1 for L, H in ranges[e]), sum(H - L for L, H in ranges[e]), e),
    )
    # Ben-Or and Tiwari modulo two primes, a third where the rational coefficients need it
    bts = {q: _BT(orc, n, q, rng) for q in all_primes()[:2]}
    found, tried, first, need = {}, 0, [], 8
    while len(first) < target:
        batch = order[tried : tried + _CANDIDATES * n]
        if not batch:
            return stop(f"the coefficients found raise the rank to {len(first)} of {target}")
        tried += len(batch)
        while True:
            if not all(bt.extend(need, end) for bt in bts.values()):
                return stop(f"the points of the interpolation reached the cap of {cap:,} matrix entries")
            per = {e: [bt.solve(e, r0, ranges[e]) for bt in bts.values()] for e in batch}
            done = [e for e in batch if all(g is not None for g in per[e])]
            if len(done) == len(batch) or need >= 2 * _TERMS + 2:
                break
            need = min(2 * need, 2 * _TERMS + 2)
        for e in done:
            got = _lift(per[e], list(bts))
            if got is None and len(bts) < 3:
                q = all_primes()[2]
                bts[q] = _BT(orc, n, q, rng)
                if not bts[q].extend(need, end):
                    return stop(f"the points of the interpolation reached the cap of {cap:,} matrix entries")
            if got is None and len(bts) == 3:
                more = bts[all_primes()[2]].solve(e, r0, ranges[e])
                got = None if more is None else _lift(per[e][:2] + [more], list(bts))
            if got:
                found[e] = got
        first, pins = _rank_select(found, [e for e in order if e in found], n, p, rng)
    # the trials: the kernel at a random point modulo a random prime is one vector, and each function is the
    # coefficient there
    bounds = _Bounds(pr)
    hi = max(PRIMES)
    for _ in range(_TRIALS):
        q = random_prime(rng, hi)
        theta = [rng.randrange(1, q) for _ in range(n)]
        if work.ticks() > end:
            return stop(f"the trials reached the cap of {cap:,} matrix entries")
        v = pr.vector(theta, q)
        if v is None or not v[r0]:
            return stop("the kernel at a random point modulo a random prime is not one vector")
        inv = pow(int(v[r0]), -1, q)
        for e in pins:
            if _value(found[e], theta, q) != int(v[e]) * inv % q:
                return stop(f"coefficient {e} differs from the kernel at a random point")
    out["exist"] = bounds.trial(hi=hi) ** _TRIALS
    for e in pins:
        co = found[e]
        expr = sum(
            (
                spy.Rational(c.numerator, c.denominator) * spy.prod([s**k for s, k in zip(params, m)])
                for m, c in co.items()
            ),
            spy.Integer(0),
        )
        # w = N / (L theta^a) with N a polynomial: the degree and height that the check adds to a minor
        L = math.lcm(*[c.denominator for c in co.values()])
        a = [max(0, -min(m[v] for m in co)) for v in range(n)]
        extra_d = max(max(sum(k + b for k, b in zip(m, a)) for m in co), sum(a), 1)
        extra_h = math.log2(sum(abs(c.numerator * (L // c.denominator)) for c in co.values()) + L)
        text = str(spy.cancel(expr))
        out["functions"].append(text)
        out["checks"][text] = bounds.trial(extra_d, extra_h, hi) ** _TRIALS
    out.update(
        kernels=orc.calls + _TRIALS,
        reference=str(pr.mons[r0]),
        degree=bounds.degree,
        height=round(bounds.height),
        trials=_TRIALS,
    )
    log.append(f"projection of {r} terms: {len(pins)} coefficients verified with {out['kernels']} kernels")
    return out
