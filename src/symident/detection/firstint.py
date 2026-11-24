"""No nonconstant rational first integral: a certificate at one point of the parameters, from the eigenvalues of the
Jacobian at an equilibrium (docs/method/global.rst, "Several experiments").

At an equilibrium whose eigenvalues are linearly independent over Z the system has no nonconstant rational first
integral (Cong, Llibre and Zhang 2011, Theorem 1), and a first integral of bounded degree at a generic point of the
parameters would give one at every point, since the points with one form a closed set defined over Q. The
independence is proven exactly: per diagonal block of the Jacobian the Galois group of the characteristic polynomial
is 2-transitive, shown by the cycle types of Frobenius elements at primes of degree one (Dedekind), the splitting
fields of the blocks differ, and the traces of the blocks are linearly independent over Q."""

import math
import time
from fractions import Fraction

import sympy as spy

from .. import session
from ..sym.numfield import NumberField, charpoly_mod, cycle_type, order_of, rank_over_q
from ..sym.parse import model_sympify

# the prime ideals of degree one a Galois test reads at most, and the first prime
_PRIME_IDEALS = 3000
_FIRST_PRIME = 10007
# the largest number field of an equilibrium a test takes, the most equilibria msolve may return, and the draws of
# the parameters
_MAX_DEGREE = 12
_MAX_POINTS = 120
_DRAWS = 3


class _OpenError(Exception):
    """The test does not settle the question at this equilibrium."""


def _diff(P, j):
    """The partial derivative of a sparse polynomial {exponents: Fraction} in the variable j."""
    out = {}
    for m, c in P.items():
        if m[j]:
            mm = m[:j] + (m[j] - 1,) + m[j + 1:]
            out[mm] = out.get(mm, 0) + c * m[j]
    return out


def _value(K, P, vals, pw):
    """A sparse polynomial at the values `vals` (elements of K), with the power table `pw`."""
    acc = K.zero
    for m, c in P.items():
        term = K.rational(c)
        for j, e in enumerate(m):
            if e:
                key = (j, e)
                if key not in pw:
                    pw[key] = vals[j] if e == 1 else K.mul(pw.get((j, e - 1)) or _power(K, vals[j], e - 1), vals[j])
                term = K.mul(term, pw[key])
        acc = K.add(acc, term)
    return acc


def _power(K, a, e):
    """a^e in K."""
    out = K.one
    for _ in range(e):
        out = K.mul(out, a)
    return out


def _sccs(n, edges):
    """The strongly connected components of the graph on range(n) with edges i -> j, iterative Tarjan."""
    order, low, stack, onst, comps = {}, {}, [], set(), []
    for root in range(n):
        if root in order:
            continue
        work = [(root, iter(edges[root]))]
        order[root] = low[root] = len(order)
        stack.append(root)
        onst.add(root)
        while work:
            v, it = work[-1]
            w = next(it, None)
            if w is not None:
                if w not in order:
                    order[w] = low[w] = len(order)
                    stack.append(w)
                    onst.add(w)
                    work.append((w, iter(edges[w])))
                elif w in onst:
                    low[v] = min(low[v], order[w])
                continue
            work.pop()
            if work:
                low[work[-1][0]] = min(low[work[-1][0]], low[v])
            if low[v] == order[v]:
                c = []
                while True:
                    w = stack.pop()
                    onst.discard(w)
                    c.append(w)
                    if w == v:
                        break
                comps.append(sorted(c))
    return comps


def _primes(start):
    """The primes above `start`, increasing."""
    p = start
    while True:
        p = int(spy.nextprime(p))
        yield p


def _galois(K, J, blocks, laws):
    """Per block of the Jacobian J over K: the multiplicity of the eigenvalue 0 and the cycle types that prove a
    2-transitive Galois group of the rest of the characteristic polynomial over K, and per pair of blocks with rests
    of degree at least two a prime at which the orders of the Frobenius elements differ, so that the splitting fields
    differ. The characteristic polynomials are taken modulo primes of degree one only: reduction commutes with them.
    The multiplicities are those at a prime where they add up to the `laws`, which give as many zero eigenvalues;
    raises _OpenError where the primes read do not show them."""
    zeros, need, pairs = None, {}, {}
    seen = 0
    for p in _primes(_FIRST_PRIME):
        if seen > _PRIME_IDEALS:
            break
        for tau in K.primes_of_degree_one(p):
            seen += 1
            red = [[K.residue(a, p, tau) for a in row] for row in J]
            if any(r is None for row in red for r in row):
                continue
            chis = [charpoly_mod([[red[i][j] for j in B] for i in B], p) for B in blocks]
            mult = [len(c) - 1 - max(k for k, x in enumerate(c) if x) for c in chis]
            if zeros is None:
                if sum(mult) != laws:
                    continue
                # the multiplicity over K is at least the one of the laws and at most the one modulo a prime
                zeros = mult
                need = {b: {"n": len(B) - zeros[b], "full": None, "minus": None if len(B) - zeros[b] > 2 else "n/a",
                            "transposition": None} for b, B in enumerate(blocks) if len(B) - zeros[b] >= 2}
                pairs = {(a, b): None for a in need for b in need if a < b}
            if mult != zeros:
                continue
            types = {}
            where = [p, int(tau)] if K.d > 1 else p
            for b, rec in need.items():
                ct = cycle_type(chis[b][:len(chis[b]) - zeros[b]], p)
                if ct is None:
                    continue
                types[b] = ct
                n = rec["n"]
                if rec["full"] is None and ct == [n]:
                    rec["full"] = where
                if rec["minus"] is None and ct == [n - 1, 1]:
                    rec["minus"] = where
                if rec["transposition"] is None and ct.count(2) == 1 and all(c % 2 for c in ct if c != 2):
                    rec["transposition"] = where
            for a, b in pairs:
                if pairs[(a, b)] is None and a in types and b in types and order_of(types[a]) != order_of(types[b]):
                    pairs[(a, b)] = where
        if zeros is not None and all(r["full"] is not None and r["minus"] is not None for r in need.values()) and \
                all(v is not None for v in pairs.values()):
            break
    if zeros is None:
        raise _OpenError(f"the Jacobian at the equilibrium has the eigenvalue 0 more often than the {laws} "
                         "conservation laws give, at every prime read" if laws else
                         "the Jacobian at the equilibrium has the eigenvalue 0 modulo every prime read, as under a "
                         "conservation law")
    for r in need.values():
        if r["full"] is None or r["minus"] is None:
            raise _OpenError(f"no prime of the {seen} read shows the cycles that make the Galois group of a block "
                             f"of degree {r['n']} 2-transitive")
    if any(v is None for v in pairs.values()):
        raise _OpenError("no prime of those read separates the splitting fields of two blocks of the Jacobian")
    return zeros, need, pairs


def _text(K, a):
    """An element of K as text in s, the generator of K."""
    if len(a) == 1:
        return str(a[0])
    s = spy.Symbol("s")
    return str(sum((spy.Rational(c.numerator, c.denominator) * s ** k for k, c in enumerate(a)),
                   spy.Integer(0))).replace("**", "^")


def _test(K, J, names, laws=0):
    """The certificate at one equilibrium with Jacobian J over K, or _OpenError with the reason. With `laws`
    conservation laws, the eigenvalue 0 of that multiplicity, the Jacobian on the level set of the laws, is left
    out."""
    blocks = _sccs(len(J), _edges(K, J))
    traces = [K.zero for _ in blocks]
    for b, B in enumerate(blocks):
        for i in B:
            traces[b] = K.add(traces[b], J[i][i])
    zeros, need, pairs = _galois(K, J, blocks, laws)
    keep = [b for b, B in enumerate(blocks) if len(B) > zeros[b]]
    if len(keep) == 1 and K.is_zero(traces[keep[0]]):
        raise _OpenError("the trace of the Jacobian at the equilibrium is 0")
    if rank_over_q([traces[b] for b in keep]) < len(keep):
        raise _OpenError(f"the traces of the {len(keep)} diagonal blocks are linearly dependent over Q in a field "
                         f"of degree {max(K.d, 1)}")
    out = []
    for b in keep:
        r = need.get(b)
        e = {"states": [names[i] for i in blocks[b]], "degree": len(blocks[b]) - zeros[b],
             "trace": _text(K, traces[b])}
        if zeros[b]:
            e["zero_eigenvalues"] = zeros[b]
        if r is not None:
            e["group"] = f"S_{r['n']}" if r["transposition"] not in (None, "n/a") or r["n"] == 2 else "2-transitive"
            e["primes"] = {"n-cycle": r["full"]}
            if r["minus"] != "n/a":
                e["primes"]["(n-1)-cycle"] = r["minus"]
            if r["transposition"] not in (None, "n/a"):
                e["primes"]["transposition"] = r["transposition"]
        out.append(e)
    return out, {f"{a},{b}": v for (a, b), v in pairs.items()}


def _jacobian(K, F, U, allv):
    """The Jacobian of the states U at the values `allv` over K, at an equilibrium: there each entry is the
    derivative of the numerator over the denominator."""
    pw, J = {}, []
    for i in U:
        N, D = F[i]
        dinv = K.inv(_value(K, D, allv, pw))
        J.append([K.mul(_value(K, _diff(N, j), allv, pw), dinv) for j in U])
    return J


def _edges(K, J):
    """The graph of the off-diagonal nonzero entries of J, i -> j."""
    return [[j for j in range(len(J)) if j != i and not K.is_zero(J[i][j])] for i in range(len(J))]


def _attempt(K, F, U, allv, names, laws, d):
    """(blocks and pairs of _test(), None, None) at the point `allv`, else (None, why, degree), the degree of a
    field of the parameters to draw again in, or None."""
    try:
        J = _jacobian(K, F, U, allv)
        return _test(K, J, names, laws), None, None
    except ZeroDivisionError:
        return None, "a denominator vanishes at the equilibrium", None
    except _OpenError as why:
        # traces of blocks rational in the parameters are dependent at a rational point: a field of degree nb
        # separates them
        nb = len(_sccs(len(U), _edges(K, J)))
        return None, str(why), nb if "linearly dependent over Q" in str(why) and d < nb else None


def _proven(K, point, alpha, sym, fixed, reads, st, equilibrium, laws, got, t0):
    """The fields of a proven certificate."""
    blocks, pairs = got
    return {"proven": True, "why": None, "point": point,
            "alpha": None if alpha is None else
            str(sum(c * sym ** k for k, c in enumerate(alpha))).replace("**", "^") + " = 0",
            "inputs": {st[i]: str(v) for i, v in fixed.items() if i in reads},
            "field": repr(K), "equilibrium": equilibrium,
            "laws": [" + ".join(f"({c})*{x}" for x, c in law.items()) for law in laws],
            "blocks": blocks, "splitting_fields": pairs,
            "seconds": round(time.monotonic() - t0, 2)}


def _field_poly(rng, d):
    """A monic irreducible integer polynomial of degree d, lowest degree first: irreducible modulo a prime."""
    while True:
        w = [rng.randint(-5, 5) for _ in range(d)] + [1]
        if w[0] == 0:
            continue
        for p in (101, 103, 107, 109, 113, 127):
            if cycle_type(list(reversed(w)), p) == [d]:
                return w


def _equilibria(F, U, fixed, theta, alpha, nonzero, level=()):
    """(equilibria, why): the equilibria of the states U over Q (the field of alpha when given), as
    [(K, {variable: element})], each Galois orbit one field, smallest first; None and the reason where msolve finds
    none or a family."""
    from .discrete import _msolve
    names = [f"z{i}" for i in range(len(U))]
    syms = [spy.Symbol(n) for n in names]
    a = spy.Symbol("a")
    gvals = dict(fixed)
    gvals.update(theta)
    for i, s in zip(U, syms):
        gvals[i] = s

    def expr(P):
        return sum((spy.Rational(c.numerator, c.denominator) * spy.prod([gvals[j] ** e for j, e in enumerate(m) if e])
                    for m, c in P.items()), spy.Integer(0))
    eqs, dens = [], spy.Integer(1)
    for i in U:
        N, D = F[i]
        num = spy.expand(expr(N))
        eqs.append(num)
        dens = dens * expr(D)
    sat = dens * (spy.prod(syms) if nonzero else 1)
    msnames = names + (["a"] if alpha is not None else [])
    allsyms = syms + ([a] if alpha is not None else [])
    gens = allsyms + [spy.Symbol("si_w")]

    def ms(e):
        e = spy.expand(e)
        if e == 0:
            return None
        P = spy.Poly(e, *gens)
        L = math.lcm(*[int(spy.Rational(c).q) for c in P.coeffs()])
        out = []
        for mono, c in P.terms():
            c = int(c * L)
            m = "*".join(f"{g}^{k}" if k > 1 else str(g) for g, k in zip(gens, mono) if k)
            out.append(f"{c}*{m}" if m else str(c))
        return "+".join(out).replace("+-", "-")
    # a balance that vanishes identically is a conservation law, whose level set enters below
    lines = [x for x in (ms(e) for e in eqs) if x is not None]
    lines += [ms(e) for e in level]
    if alpha is not None:
        lines.append(ms(sum(c * a ** k for k, c in enumerate(alpha))))
    lines.append(ms(spy.Symbol("si_w") * sat - 1))
    data = _msolve(msnames, 0, lines, ["-P", "1"])
    if data is None:
        from . import work
        why = work.msolve_too_large(msnames + ["si_w"], lines)
        return None, "msolve did not solve the balances" + (f": {why}" if why else "")
    if data[0] == 1:
        return None, "the equilibria form a family of positive dimension"
    if data[0] != 0:
        return None, "the balances have no solution"
    _, _, deg, vnames, lin, par = data[1]
    if deg <= 0 or par[0] != 1:
        return None, "msolve did not parametrize the equilibria"
    if deg > _MAX_POINTS:
        return None, f"the balances have {deg} solutions, more than {_MAX_POINTS}"
    T = spy.Symbol("t")
    elim = spy.Poly(list(reversed([int(c) for c in par[1][0][1]])), T)
    out = []
    for fac, _ in spy.factor_list(spy.sqf_part(elim))[1]:
        if fac.degree() > _MAX_DEGREE:
            continue
        K = NumberField([int(c) for c in reversed(fac.all_coeffs())])
        t = K.from_t([0, 1])
        den = K.from_t([int(c) for c in par[1][1][1]])
        try:
            dinv = K.inv(den)
        except ZeroDivisionError:
            continue
        vals = {}
        for j, q in enumerate(par[1][2]):
            vals[vnames[j]] = K.scale(K.mul(K.from_t([int(c) for c in q[0][1]]), dinv), Fraction(-1, int(q[1])))
        last = len(vnames) - 1
        rest = t
        for j in range(last):
            rest = K.sub(rest, K.scale(vals[vnames[j]], Fraction(int(lin[j]))))
        vals[vnames[last]] = K.scale(rest, Fraction(1, int(lin[last])))
        out.append((K, vals))
    out.sort(key=lambda kv: kv[0].d)
    return out, None if out else f"every equilibrium lies in a number field of degree above {_MAX_DEGREE}"


def _linear_point(F, U, fixed, nt, ns, rng, alpha):
    """(K, x, theta): an equilibrium x at random values of K away from zero and parameters theta that make it one:
    the parameters in products of parameters held at random values until the balances are linear in the others,
    those and the free ones of the others at random elements of K (Q, or Q(alpha)), the rest solved over K. None
    where the balances have no solution at x or force a parameter to zero."""
    K = NumberField(alpha) if alpha is not None else NumberField([0, 1])

    def draw():
        return K._reduce([Fraction(rng.randint(2, 40))] + [Fraction(rng.randint(-9, 9)) for _ in range(K.d - 1)]) \
            if K.d > 1 else (Fraction(rng.randint(2, 40)),)
    # the states in K as well: identities of the balances can tie traces with a rational factor of the states
    xs = {i: K.add(K.rational(Fraction(rng.randint(1, 30), rng.randint(1, 7))),
                   K._reduce([Fraction(0)] + [Fraction(rng.randint(-5, 5), rng.randint(1, 7)) for _ in range(K.d - 1)])
                   if K.d > 1 else K.zero) for i in U}
    vals = dict(xs)
    vals.update({i: K.rational(int(v)) for i, v in fixed.items()})
    held = {}
    while True:
        # the parameter in most products of parameters is held at a value
        hits = {}
        for i in U:
            for m in F[i][0]:
                ks = [j for j, e in enumerate(m[ns:], start=ns) if e and j not in held]
                if sum(m[j] for j in ks) > 1:
                    for j in ks:
                        hits[j] = hits.get(j, 0) + 1
        if not hits:
            break
        held[max(hits, key=lambda j: (hits[j], -j))] = draw()
    vals.update(held)

    def at(m):
        # the value of the states and held parameters of a monomial
        out = K.one
        for j, e in enumerate(m):
            if e and (j < ns or j in held):
                for _ in range(e):
                    out = K.mul(out, vals[j])
        return out
    # rows: sum_k A_ik theta_k = b_i over K
    rows = []
    for i in U:
        row, b = [K.zero] * nt, K.zero
        for m, c in F[i][0].items():
            k = next((j - ns for j, e in enumerate(m[ns:], start=ns) if e and j not in held), None)
            if k is None:
                b = K.sub(b, K.scale(at(m), c))
            else:
                row[k] = K.add(row[k], K.scale(at(m), c))
        rows.append(row + [b])
    # the solved parameters in a random order: a fixed order can tie the traces of several blocks to one parameter
    piv, r, cols = [], 0, list(range(nt))
    rng.shuffle(cols)
    for col in cols:
        p = next((i for i in range(r, len(rows)) if not K.is_zero(rows[i][col])), None)
        if p is None:
            continue
        rows[r], rows[p] = rows[p], rows[r]
        inv = K.inv(rows[r][col])
        rows[r] = [K.mul(x, inv) for x in rows[r]]
        for i in range(len(rows)):
            if i != r and not K.is_zero(rows[i][col]):
                f = rows[i][col]
                rows[i] = [K.sub(x, K.mul(f, y)) for x, y in zip(rows[i], rows[r])]
        piv.append(col)
        r += 1
    if any(all(K.is_zero(x) for x in row[:nt]) and not K.is_zero(row[nt]) for row in rows):
        return None
    theta = {j - ns: v for j, v in held.items()}
    for k in range(nt):
        if k not in piv and k not in theta:
            theta[k] = draw()
    for j, k in enumerate(piv):
        acc = rows[j][nt]
        for f, v in theta.items():
            if f not in piv and not K.is_zero(rows[j][f]):
                acc = K.sub(acc, K.mul(v, rows[j][f]))
        if K.is_zero(acc):
            # a parameter the equilibrium forces to zero: a degenerate point
            return None
        theta[k] = acc
    return K, xs, theta


def certificate(series, laws=()):
    """{"proven": True, ...} where the states of the first condition of `series` with free starts have no nonconstant
    rational first integral at a generic point of the parameters, constants and constant inputs, proven at one point
    by the eigenvalues at an equilibrium; {"proven": False, "why": ...} where the test does not settle it. With the
    conservation `laws` ({state: coefficient}, as from conservation_laws()), the system on a generic level set of
    the laws: every rational first integral is then a function of the laws."""
    t0 = time.monotonic()
    part = series.parts[0]
    seg = part["segs"][0]
    st = [str(x) for x in part["st"]]
    ns = len(st)
    known = set(part["known"])
    F = seg["F"]
    U = [i for i in range(ns) if i not in known]
    out = {"proven": False, "why": None, "theorem": "Cong, Llibre and Zhang 2011, Theorem 1"}
    # a known series that grows at a constant rate is a clock: the flow from a later start is another model
    clocks = [i for i in known if F[i][1] and all(not any(m) for P in F[i] for m in P) and any(F[i][0].values())]
    reads = {j for i in U for P in F[i] for m in P for j, e in enumerate(m[:ns]) if e}
    if reads & set(clocks):
        out["why"] = "the right-hand sides read the clock"
        return out
    rng = session.current().rng("first integrals", tuple(series.coords))
    names = [st[i] for i in U]
    params = [str(p) for p in series.params]
    last = None
    # equilibria away from zero first, over Q; a field of degree r where r blocks have rational traces
    plan = [(1, nonzero) for nonzero in (True, False) for _ in range(_DRAWS)]
    while plan:
        d, nonzero = plan.pop(0)
        # the known series (inputs) at constant values, the parameters at a point of Q or of Q(alpha)
        fixed = {i: spy.Integer(rng.randint(2, 30)) for i in known}
        alpha = _field_poly(rng, d) if d > 1 else None
        a = spy.Symbol("a")
        theta = {}
        for k in range(len(params)):
            v = spy.Integer(rng.randint(2, 40))
            if alpha is not None:
                v = v + sum(rng.randint(-9, 9) * a ** j for j in range(1, d))
            theta[ns + k] = v
        # an equilibrium chosen first, the parameters solved, where the balances are linear in them
        lin = _linear_point(F, U, fixed, len(params), ns, rng, alpha) if nonzero else None
        if lin is not None:
            K, xv, tv = lin
            allv = [K.zero] * (ns + len(params))
            for i, v in fixed.items():
                allv[i] = K.rational(int(v))
            for i, v in xv.items():
                allv[i] = v
            for k, v in tv.items():
                allv[ns + k] = v
            got, why, retry = _attempt(K, F, U, allv, names, len(laws), d)
            if got is None:
                last = why
                if retry is not None:
                    plan.insert(0, (retry, nonzero))
                continue
            out.update(_proven(K, {p: _text(K, tv[k]) for k, p in enumerate(params)}, alpha, spy.Symbol("s"),
                               fixed, reads, st, {names[j]: _text(K, xv[i]) for j, i in enumerate(U)}, laws, got,
                               t0))
            return out
        # a level set of the laws, at random values
        level = []
        for law in laws:
            g = sum((model_sympify(str(c)).xreplace(series.known).xreplace(
                {spy.Symbol(p): theta[ns + k] for k, p in enumerate(params)}) * spy.Symbol(f"z{U.index(st.index(x))}")
                for x, c in law.items()), spy.Integer(0))
            level.append(spy.fraction(spy.together(g - rng.randint(2, 30)))[0])
        found, why = _equilibria(F, U, fixed, theta, alpha, nonzero, level)
        if not found:
            last = last or f"no isolated equilibrium: {why}"
            continue
        for K, vals in found:
            x = [vals[f"z{j}"] for j in range(len(U))]
            gv = {}
            for k, v in theta.items():
                if alpha is None:
                    gv[k] = K.rational(Fraction(int(v.p), int(v.q)))
                    continue
                acc = K.zero
                for (e,), c in spy.Poly(v, a).terms():
                    acc = K.add(acc, K.scale(_power(K, vals["a"], e), Fraction(int(c.p), int(c.q))))
                gv[k] = acc
            for i, v in fixed.items():
                gv[i] = K.rational(int(v))
            for i, xi in zip(U, x):
                gv[i] = xi
            allv = [gv.get(j, K.zero) for j in range(ns + len(params))]
            got, why, retry = _attempt(K, F, U, allv, names, len(laws), d)
            if got is None:
                last = why
                if retry is not None:
                    plan.insert(0, (retry, nonzero))
                    break
                continue
            al = spy.Symbol("alpha")
            point = {p: str(theta[ns + k].xreplace({a: al})).replace("**", "^") for k, p in enumerate(params)}
            out.update(_proven(K, point, alpha, al, fixed, reads, st,
                               {names[j]: _text(K, xj) for j, xj in enumerate(x)}, laws, got, t0))
            return out
    out["why"] = last or "no equilibrium of the draws settles the test"
    out["seconds"] = round(time.monotonic() - t0, 2)
    return out
