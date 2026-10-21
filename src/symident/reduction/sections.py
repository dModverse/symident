"""Face sections and carriers of the invariant map."""

import itertools
import math
import re

import numpy as np
import sympy as spy

from .. import session
from ..errors import SYMPY_FAILS, RouteError, sympy_fail
from ..rexpr import sym_sort
from .darboux import apply_x
from .expr import free_syms, red_locals, red_sympify, tri_solve
from .scaling import mono_string
from .sign import DOMAIN, is_real, num_sign, positive_set, set_positive, sgn, sgn_real, time_up


def mono_parts(m):
    """Coefficient and exponents of the monomial string `m`, or None unless it is a monomial with positive powers."""
    coef, ex = 1, {}
    for f in [x.strip() for x in m.split("*")]:
        if re.match(r"^[0-9]+$", f):
            coef *= int(f)
            continue
        sp = f.split("^")
        if len(sp) > 2 or not re.match(r"^[A-Za-z.][0-9A-Za-z._]*$", sp[0]):
            return None
        try:
            k = int(sp[1]) if len(sp) == 2 else 1
        except ValueError:
            return None
        if k < 1:
            return None
        ex[sp[0]] = ex.get(sp[0], 0) + k
    if not ex:
        return None
    return {"coef": coef, "ex": ex}


def balance_products(p, q):
    """The pairs [p0*q0, p1*q1] and [p0*q1, p1*q0] of reduced monomials if they differ; [] unless all are monomial."""
    parts = [mono_parts(x) for x in list(p) + list(q)]
    if any(x is None for x in parts):
        return []

    def mul(u, v):
        ex = dict(u["ex"])
        for n, k in v["ex"].items():
            ex[n] = ex.get(n, 0) + k
        return {"coef": u["coef"] * v["coef"], "ex": ex}

    def join(u):
        ex = {k: v for k, v in u["ex"].items() if v != 0}
        if not ex:
            return str(u["coef"])
        nms = sym_sort(list(ex))
        mono = mono_string([ex[n] for n in nms], nms)
        return mono if u["coef"] == 1 else f"{u['coef']}*{mono}"
    out = []
    for flip in (False, True):
        l = mul(parts[0], parts[3 if flip else 2])
        r = mul(parts[1], parts[2 if flip else 3])
        for n in list(dict.fromkeys(list(l["ex"]) + list(r["ex"]))):
            g = min(l["ex"].get(n, 0), r["ex"].get(n, 0))
            if g > 0:
                l["ex"][n] -= g
                r["ex"][n] -= g
        ls, rs = join(l), join(r)
        if ls != rs:
            out.append([ls, rs])
    return out


def _bound_pins(support):
    """Pins v = r/2 for a positive v of `support` below the root r of a declared bound linear in v alone."""
    out = []
    for h in DOMAIN.get("bounds") or []:
        try:
            e = red_sympify(str(h).replace("^", "**"), red_locals([str(h)]))
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            continue
        syms = [str(x) for x in e.free_symbols]
        if len(syms) != 1 or syms[0] not in support or is_real(syms[0]):
            continue
        v = spy.Symbol(syms[0])
        c = spy.diff(e, v)
        if c.free_symbols or not c.is_negative or spy.diff(e, v, 2) != 0:
            continue
        r = spy.Rational(-e.xreplace({v: 0}), c)
        if r > 0:
            out.append([syms[0], str(r / 2)])
    return out


def section_cands(support, extra=()):
    """Gauge-section candidates of a curved block: pins, summand balances (those in `support` first), monomial pairs."""
    def key(pr):
        return " = ".join(sym_sort(list(pr)))
    head, groups = [], []
    for tm in extra:
        tm = list(dict.fromkeys(tm))
        if len(tm) < 2:
            continue
        g = [[a, b] for a, b in itertools.combinations(tm, 2)]
        groups.append(g)
        head += g
    if len(groups) > 1:
        for ia, ib in itertools.combinations(range(len(groups)), 2):
            A, B = groups[ia], groups[ib]
            for u in A[:3]:
                for v in B[:3]:
                    head += balance_products(u, v)
            if len(head) >= 40:
                break
    seen_k, h2 = set(), []
    for pr in head:
        k = key(pr)
        if k not in seen_k:
            seen_k.add(k)
            h2.append(pr)
    sup = set(support)
    ident = re.compile(r"[A-Za-z_.][A-Za-z0-9_.]*")
    own = [pr for pr in h2 if all(x in sup for t in pr for x in ident.findall(t))]
    foreign = [pr for pr in h2 if pr not in own]
    pins = _bound_pins(support) + [[v, "0" if is_real(v) else "1"] for v in support]
    # every summand balance precedes the monomial pairs, so the cap on kept candidates cannot drop it
    head = pins + own + foreign
    mons = [(m, 1) for m in support]
    if len(support) > 1:
        mons += [(f"{a}*{b}", 2) for a, b in itertools.combinations(support, 2)]
    if len(mons) < 2:
        return head
    pairs = list(itertools.combinations(range(len(mons)), 2))
    out = [[mons[i][0], mons[j][0]] for i, j in pairs]
    order = sorted(range(len(pairs)), key=lambda k: mons[pairs[k][0]][1] + mons[pairs[k][1]][1])
    out = [out[k] for k in order]
    # products of two coordinates at one, after the balances between coordinates
    out += [[f"{a}*{b}", "1"] for a, b in itertools.combinations(support, 2)]
    head_k = {key(pr) for pr in head}
    return head + [pr for pr in out if key(pr) not in head_k]


def inv_summands(Ie):
    """The distinct unsigned terms of the numerator of `Ie`, sorted; [] for fewer than two."""
    try:
        num = spy.expand(spy.fraction(spy.together(Ie))[0])
        tm = [str(x) for x in spy.Add.make_args(num)]
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return []
    if len(tm) < 2:
        return []
    return sym_sort(list(dict.fromkeys(re.sub(r"^-", "", t.replace("**", "^")) for t in tm)))


def section_monotone(pr, preps, xi_of, loc, pts):
    """Exact values of the numerator of X(log(m1/m2)) at the rational points, None unless it has one sign."""
    ms = [red_sympify(m.replace("^", "**"), loc) for m in pr]
    # X(log(m1/m2)) = (X(m1) m2 - m1 X(m2)) / (m1 m2), so the denominator needs one sign as well
    if sgn(ms[0] * ms[1]) == 0:
        return None
    row, moved = [], False
    for g in range(len(preps)):
        Xm = [apply_x(xi_of[g], m, loc) for m in ms]
        try:
            d = spy.expand(Xm[0] * ms[1] - ms[0] * Xm[1])
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            return None
        if d.is_zero:
            row += [spy.Integer(0)] * len(pts)
            continue
        if d.is_number:
            if num_sign(d) is None:
                return None
            moved = True
            row += [d] * len(pts)
            continue
        sd = sgn(d)
        if sd == 0:
            return None
        moved = True
        for pt in pts:
            v = d.xreplace({spy.Symbol(k): x for k, x in pt.items()})
            row.append(v if v.is_finite else spy.Integer(sd))
    if not moved:
        return None
    return row


def _monomial_section(pairs, sym):
    """The positive points with m1 = m2 for each pair of Laurent monomials as theta0 * u^K, or None."""
    rows, cs, names = [], [], []
    for m1, m2 in pairs:
        c, rest = spy.cancel(m1 / m2).as_coeff_Mul()
        if not c.is_Rational or c <= 0:
            return None
        ex = {}
        for base, k in rest.as_powers_dict().items():
            if base == 1:
                continue
            if not base.is_Symbol or not k.is_Integer:
                return None
            ex[str(base)] = int(k)
            if str(base) not in names:
                names.append(str(base))
        if not ex and c != 1:
            return None
        if ex:
            rows.append(ex)
            cs.append(c)
    if not names:
        return {}
    if any(not sym(v).is_positive for v in names):
        return None
    A = spy.Matrix([[r.get(v, 0) for v in names] for r in rows])
    if A.rank() < A.rows:
        return None
    x0 = [spy.Integer(1)] * len(names)
    if any(c != 1 for c in cs):
        P = A.T * (A * A.T).inv()
        x0 = [spy.Mul(*[cs[i] ** -P[j, i] for i in range(len(cs))]) for j in range(len(names))]
    K = A.nullspace()
    us = [spy.Symbol(f"si_sec_u{j}", positive=True) for j in range(len(K))]
    param = {}
    for j, v in enumerate(names):
        e = x0[j]
        for i, k in enumerate(K):
            den = spy.ilcm(*[spy.fraction(x)[1] for x in k])
            e *= us[i] ** int(k[j] * den)
        param[sym(v)] = e
    return param


def _owned(extra=()):
    """Symbols by name with the sign of the declared domain, and a map of expressions onto them."""
    pos = positive_set()
    cache = {}

    def sym(name):
        if name not in cache:
            real = name in set(DOMAIN["real_carriers"]) or (pos is not None and name not in set(pos) and
                                                              name not in set(extra))
            cache[name] = spy.Symbol(name, real=True) if real else spy.Symbol(name, positive=True)
        return cache[name]

    def own(e):
        e = spy.sympify(e)
        return e.xreplace({s: sym(str(s)) for s in e.free_symbols})
    return sym, own


def _vanishes(e):
    return spy.cancel(e) == 0 or spy.simplify(e) == 0


def _linear_section(pairs):
    """The points m1 = m2 per pair, each solved for a linear coordinate with a coefficient of fixed sign, or None."""
    param = {}
    for m1, m2 in pairs:
        eq = spy.fraction(spy.cancel(spy.together((m1 - m2).xreplace(param))))[0]
        if eq.is_zero:
            continue
        hit = None
        for v in sorted(eq.free_symbols, key=str):
            P = spy.Poly(eq, v)
            if P.degree() == 1 and _sign_fixed(P.coeff_monomial(v)) != 0:
                hit = (v, spy.cancel(-P.coeff_monomial(1) / P.coeff_monomial(v)))
                break
        if hit is None:
            return None
        param = {k: spy.cancel(e.xreplace({hit[0]: hit[1]})) for k, e in param.items()}
        param[hit[0]] = hit[1]
    return param


def section_injective(invariants, sol, tmp, pins, pairs=()):
    """Whether the solution `sol` in the invariant symbols `tmp` gives back every point of the section."""
    sym, own = _owned()
    try:
        sub = {sym(str(k)): own(v) for k, v in pins}
        inv = [own(e).xreplace(sub) for e in invariants]
        param = {}
        if pairs:
            own_pairs = [(own(a).xreplace(sub), own(b).xreplace(sub)) for a, b in pairs]
            param = _monomial_section(own_pairs, sym)
            if param is None:
                param = _linear_section(own_pairs)
            if param is None:
                return False
        back = {sym(t): e for t, e in zip(tmp, inv)}
        for v, e in sol.items():
            if sym(v) in sub:
                continue
            if not _vanishes((own(e).xreplace(sub).xreplace(back) - sym(v)).xreplace(param)):
                return False
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return False
    return True


def _sign_fixed(e):
    return sgn_real(e, DOMAIN["real_carriers"])


def _constraints(e):
    """The conditions num/den > 0 equivalent to e > 0, one per square root factor and one for the rest."""
    out, plain = [], []
    for f in spy.Mul.make_args(spy.cancel(spy.together(e))):
        base, ex = f.as_base_exp()
        if ex == spy.Rational(1, 2) or ex == spy.Rational(-1, 2):
            out.append(spy.fraction(spy.cancel(spy.together(base))))
        else:
            plain.append(f)
    out.append(spy.fraction(spy.cancel(spy.together(spy.Mul(*plain)))))
    return out


def _concave(poly, g):
    """Whether the quadratic `poly` in the gauge `g` is concave: -Hessian positive definite."""
    H = spy.hessian(-poly.as_expr(), g)
    return all(_sign_fixed(spy.cancel(H[:k, :k].det())) == 1 for k in range(1, len(g) + 1))


def _signed_factors(polys):
    """The conditions p > 0 as (sign, factors), with every factor of known sign removed."""
    reps = []
    for p in polys:
        lead, facs = spy.factor_list(p)
        reps.append([1 if lead > 0 else -1 if lead < 0 else 0, list(facs), False])
    changed = True
    while changed:
        changed = False
        for i, rep in enumerate(reps):
            if rep[2]:
                continue
            keep = []
            for f, m in rep[1]:
                sf = _sign_fixed(f)
                if sf == 0:
                    for j, o in enumerate(reps):
                        if j != i and len(o[1]) == 1 and o[1][0][1] % 2 == 1:
                            r = spy.cancel(f / o[1][0][0])
                            if r.is_number and r != 0:
                                sf = o[0] * (1 if r > 0 else -1)
                                o[2] = True
                                break
                if sf == 0:
                    keep.append((f, m))
                else:
                    rep[0] *= sf ** m
                    changed = True
            rep[1] = keep
    return [(s, fs) for s, fs, _ in reps]


def _convex_kind(p, g):
    """Whether {p > 0} is convex in the gauge `g`: affine, a concave quadratic, or an interval in one coordinate."""
    poly = spy.Poly(p, *g)
    if poly.total_degree() <= 1:
        return True
    moved = [x for x in g if poly.degree(x) > 0]
    if len(moved) == 1 and _interval(spy.Poly(p, moved[0]), not is_real(str(moved[0]))):
        return True
    return poly.total_degree() == 2 and _concave(poly, moved)


def _connected(polys, g):
    """Whether {p > 0 for all polys} is connected over the gauge `g`, by convexity or by interval fibers."""
    reps = _signed_factors(polys)
    if any(sg <= 0 and not fs for sg, fs in reps):
        return True
    conds = [sg * spy.Mul(*[f ** m for f, m in fs]) for sg, fs in reps if fs]
    if not g or all(_convex_kind(c, g) for c in conds):
        return True
    facts = [(fs[0][0], sg) for sg, fs in reps if len(fs) == 1 and fs[0][1] % 2 == 1]

    def sign_free_of(a, x):
        lead, facs = spy.factor_list(a)
        s = 1 if lead > 0 else -1 if lead < 0 else 0
        for f, m in facs:
            sf = _sign_fixed(f)
            if sf == 0:
                for k, sk in facts:
                    r = spy.cancel(f / k)
                    if spy.Symbol(str(x)) not in k.free_symbols and r.is_number and r != 0:
                        sf = sk * (1 if r > 0 else -1)
                        break
            s *= sf ** m
        return s
    best = None
    for x in g:
        lower, upper, keep, ok = [], [], [], True
        for c in conds:
            P = spy.Poly(c, x)
            if P.degree() == 0:
                keep.append(c)
                continue
            a, b = P.coeff_monomial(x), P.coeff_monomial(1)
            sa = sign_free_of(a, x) if P.degree() == 1 else 0
            if sa == 0:
                ok = False
                break
            (lower if sa > 0 else upper).append(spy.cancel(-b / a))
        if not ok:
            continue
        if not is_real(str(x)):
            lower.append(spy.Integer(0))
        if best is None or len(lower) * len(upper) < len(best[1]) * len(best[2]):
            best = (x, lower, upper, keep)
    if best is None:
        return False
    # one elimination suffices: the fibers are intervals, so the set is connected iff its projection is
    x, lower, upper, keep = best
    proj = keep + [spy.Mul(*spy.fraction(spy.cancel(spy.together(u - lo)))) for lo in lower for u in upper]
    return _connected(proj, [y for y in g if y != x])


def _rational_gauges(sol, gauge):
    """`sol` and the gauge after g = c*u^2 or g = u^2/c, for each choice that leaves no root in the gauge."""
    real = set(DOMAIN["real_carriers"])
    sym, own = _owned([str(x) for e in sol.values() for x in spy.sympify(e).free_symbols
                       if str(x).startswith("si_red_c") and str(x) not in real])
    outs = [({k: own(e) for k, e in sol.items()}, list(gauge))]
    for i, v in enumerate(gauge):
        x = sym(v)
        nxt = []
        for cur, names in outs:
            roots = {w for e in cur.values() for w in e.atoms(spy.Pow)
                     if w.exp.is_Rational and w.exp.q == 2 and w.has(x)}
            if not roots:
                nxt.append((cur, names))
                continue
            if is_real(v):
                nxt.append((cur, names))
                continue
            u = spy.Symbol(f"si_root_{v}", positive=True)
            n0 = len(nxt)
            for w in roots:
                c = spy.cancel(w.base / x)
                if c.has(x):
                    c = spy.cancel(w.base * x)
                if c.has(*[sym(y) for y in gauge]) or _sign_fixed(c) != 1:
                    continue
                for to in (c * u**2, u**2 / c):
                    new = {k: spy.simplify(spy.cancel(e.subs(x, to))) for k, e in cur.items()}
                    if any(w2.has(u) for e in new.values() for w2 in e.atoms(spy.Pow)
                           if w2.exp.is_Rational and w2.exp.q != 1):
                        continue
                    nxt.append((new, names[:i] + [str(u)] + names[i + 1:]))
            if len(nxt) == n0:
                nxt.append((cur, names))
        outs = nxt
    done = []
    for cur, names in outs:
        # t = s^2, s > 0, maps t > 0 one to one and leaves no root of a single symbol
        sq = {w.base for e in cur.values() for w in e.atoms(spy.Pow)
              if w.exp.is_Rational and w.exp.q == 2 and w.base.is_Symbol and w.base.is_positive}
        cur = {k: spy.factor(spy.cancel(spy.together(e.xreplace(
            {t: spy.Symbol(f"si_sq_{t}", positive=True)**2 for t in sq})))) for k, e in cur.items()}
        done.append(({k: e.xreplace({y: spy.Symbol(str(y)) for y in e.free_symbols}) for k, e in cur.items()},
                     names))
    return done


def _convex_set(carriers, sol, gauge):
    """Whether the gauge values at which every positive carrier is positive form a connected set."""
    for sol2, gauge2 in _rational_gauges(sol, gauge):
        new = [v for v in gauge2 if v.startswith("si_root_")] + \
            [str(y) for e in sol2.values() for y in e.free_symbols if str(y).startswith("si_sq_")]
        old = positive_set()
        if new and old is not None:
            set_positive(list(old) + new)
        try:
            cons = [k for c in carriers if not is_real(c) and _sign_fixed(sol2[c]) != 1
                    for k in _constraints(sol2[c])]
            if _connected([spy.expand(num * den) for num, den in cons], [spy.Symbol(v) for v in gauge2]):
                return True
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
        finally:
            if new and old is not None:
                set_positive(old)
    return False


def _max_variations(opts):
    """The most sign changes of a sequence whose entries take a sign from the sets `opts`."""
    best = {}
    for o in opts:
        best = {sg: max([0] if not best else [v + (sg != last) for last, v in best.items()]) for sg in o}
    return max(best.values(), default=0)


def _interval(poly, positive):
    """Whether {x: poly(x) > 0} is an interval, on x > 0 when `positive`, for every sign of undecided coefficients."""
    coeffs = poly.all_coeffs()
    signs = [_sign_fixed(c) for c in coeffs]
    nz = [i for i, c in enumerate(coeffs) if c != 0]
    if not nz or len(signs) - 1 - nz[0] <= 1:
        return True
    opts = [{signs[i]} if signs[i] else {1, -1} for i in nz]
    if len(signs) - 1 - nz[0] == 2:
        # only a positive leading coefficient leaves the concave case
        if 1 not in opts[0]:
            return True
        opts[0] = {1}
    return positive and _max_variations(opts) <= 1


def _gauge_minor(gens, gauge, sub, loc):
    """The sign of the minor of the generators on `gauge`, with denominators cleared if needed."""
    X = [[red_sympify(str(g.get(v, "0")).replace("^", "**"), loc).subs(sub) for v in gauge] for g in gens]
    s = sgn(spy.cancel(spy.Matrix(X).det()))
    if s != 0:
        return s
    # D*X has the invariants and orbits of X off the pole set D = 0 and is defined across it
    clear = []
    for row in X:
        L = spy.Integer(1)
        for e in row:
            L = spy.lcm(L, spy.fraction(spy.cancel(spy.together(e)))[1])
        clear.append([spy.cancel(e * L) for e in row])
    return sgn(spy.cancel(spy.Matrix(clear).det()))


def orbits_reach(invariants, branches, carriers, gauge, tmp, gens, pins):
    """The carrier branch over which every level set of the invariants is one orbit, or None."""
    if len(gens) != len(gauge) or not gauge or any(g is None for g in gens):
        return None
    loc = red_locals([str(x) for g in gens for x in g.values()] + [str(v) for _, v in pins])
    try:
        sub = [(spy.Symbol(str(k)), spy.sympify(str(v), locals=loc)) for k, v in pins]
        if _gauge_minor(gens, gauge, sub, loc) == 0:
            return None
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    for br in branches:
        if any(br.get(c) is None for c in carriers):
            continue
        sol = {c: br[c] for c in carriers}
        if not section_injective(invariants, sol, tmp, pins) or not _graph(invariants, sol, tmp, pins):
            continue
        try:
            sol = {c: spy.sympify(e).subs(sub) for c, e in sol.items()}
            if _convex_set(carriers, sol, gauge):
                return sol
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
    return None


def face_reach(b, pins, Z, gens, limit_only=False):
    """The carriers over the face gauge `Z` at Z = 0 when every level set is one orbit, or None."""
    supp = b["support"]
    U = [v for v in supp if v not in Z]
    loc = red_locals(list(b["invariants"]) + supp + [str(v) for v in pins.values()])
    tmp = [f"si_red_c{l + 1}" for l in range(len(b["invariants"]))]
    try:
        sub = [(spy.Symbol(k), red_sympify(str(v).replace("^", "**"), loc)) for k, v in pins.items()]
        Ies = [red_sympify(iv.replace("^", "**"), loc).subs(sub) for iv in b["invariants"]]
        branches = [{str(k): v for k, v in d.items()} for d in
                    spy.solve([e - spy.Symbol(t) for e, t in zip(Ies, tmp)], [spy.Symbol(u) for u in U], dict=True)]
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    old = DOMAIN["real_carriers"]
    DOMAIN["real_carriers"] = [t for t, e in zip(tmp, Ies) if sgn(e) != 1]
    try:
        if limit_only:
            ok = [{c: br[c] for c in U} for br in branches if all(br.get(c) is not None for c in U)
                  and section_injective(Ies, {c: br[c] for c in U}, tmp, list(pins.items()))
                  and _graph(Ies, {c: br[c] for c in U}, tmp, list(pins.items()))]
            sol = ok[0] if len(ok) == 1 else None
        else:
            sol = orbits_reach(Ies, branches, U, Z, tmp, gens, list(pins.items()))
        if sol is None:
            return None
        at0 = {c: spy.cancel(e.subs([(spy.Symbol(z), 0) for z in Z])) for c, e in sol.items()}
        # the face point of the orbit through theta, at q = I(theta), on the whole domain
        back = [(spy.Symbol(t), e) for t, e in zip(tmp, Ies)]
        for c, e in at0.items():
            th = spy.cancel(spy.together(e.subs(back)))
            if _bad(e) or _bad(th) or (sgn(th) != 1 if not is_real(c) else
                                       sgn(spy.fraction(th)[1]) == 0):
                return None
        return at0
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    finally:
        DOMAIN["real_carriers"] = old


def face_regular(Z, exprs, rhs, equilibrate, coords, support=None, rest=None, inputs=()):
    """Why the model is not continuous up to the face Z = 0 (None when it is), and the level of that claim."""
    loc = red_locals([str(x) for x in exprs] + list(rhs or {}) + [str(x) for x in (rhs or {}).values()])
    sub = [(spy.Symbol(z), spy.Integer(0)) for z in Z]
    states = set(rhs or {})
    for d in model_denominators(exprs):
        try:
            d0 = spy.cancel(d.subs(sub))
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            return "a model denominator does not simplify at the face", None
        if d0.is_zero:
            return "a model denominator vanishes at the face", None
        if support is not None and not set(free_syms(d)) & set(support):
            continue
        if rhs is None or not set(free_syms(d)) & states:
            if set(free_syms(d)) <= set(coords) and sgn(d0) == 0:
                return f"the denominator {d0} has no fixed sign at the face", None
    for x in exprs:
        try:
            if _bad(red_sympify(str(x).replace("^", "**"), loc).subs(sub)):
                return "a model expression has a pole at the face", None
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            return "a model expression does not simplify at the face", None
    if not equilibrate or rest:
        return None, "exact"
    if rhs is None:
        return "the resting state at the face is not known to be continuous", None
    xs = [x for x, f in rhs.items() if spy.sympify(f.replace("^", "**"), locals=loc) != 0]
    try:
        F = [red_sympify(rhs[x].replace("^", "**"), loc) for x in xs]
        J = spy.Matrix(F).jacobian([spy.Symbol(x) for x in xs])
        if all(not e.has(*[spy.Symbol(x) for x in xs]) for e in J) and sgn(spy.cancel(J.subs(sub).det())) != 0:
            return None, "exact"
        if rest_checked(rhs, Z, inputs):
            return None, "checked"
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
    return "the resting state of the face model is not found regular at random points of the face", None


def face_state_limit(b, pins, Z, gens, states, degrees):
    """Whether the states the lifted fields `gens` move have a positive limit at the face."""
    from ..sym.invariants import verify_invariants
    from .blocks import solve_block
    from .expr import gen_prep
    preps = [gen_prep({"generator": g}) for g in gens]
    if any(not p_["ok"] for p_ in preps):
        return False
    moved = [x for x in states if any(x in p_["support"] for p_ in preps)]
    if not moved:
        return True
    old = positive_set()
    if old is not None:
        set_positive(list(old) + list(states))
    try:
        found = solve_block(preps, *degrees).get("invariants") or []
        supp = list(dict.fromkeys(list(b["support"]) + moved))
        if len(found) != len(supp) - len(Z):
            return False
        if not all(verify_invariants([f.replace("^", "**") for f in found], [dict(p_["comps"]) for p_ in preps])):
            return False
        return face_reach({"support": supp, "invariants": found}, pins, Z, gens, limit_only=True) is not None
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return False
    finally:
        if old is not None:
            set_positive(old)


def rest_checked(rhs, Z, inputs, points=3):
    """Whether the face model has a resting state with a regular Jacobian at `points` random points over F_q."""
    # a check without an error bound, with the inputs at 0; a point without a resting state is redrawn
    from ..gfp import verify_prime
    from ..sym.steady import solve_steady_state_modular
    states = [x for x, f in rhs.items() if str(f).strip() not in ("0", "0.0")]
    syms = {str(s) for f in rhs.values() for s in spy.sympify(str(f).replace("^", "**")).free_symbols}
    params = sorted(syms - set(states) | {x for x in rhs if x not in states})
    q = verify_prime()
    rng = session.current().rng("face rest", tuple(sorted(Z)), tuple(states))
    lines = [f"{x} = {str(rhs[x]).replace('^', '**')}" for x in states]
    found = 0
    # an algebraic resting state has a point over F_q at a fixed positive fraction of the draws
    for _ in range(8 * points):
        vals = {v: (0 if v in set(Z) or v in set(inputs or ()) else rng.randrange(1, q)) for v in params}
        res = solve_steady_state_modular(lines, states, params, vals, q, forcings=list(inputs or ()))
        if res.get("ok"):
            found += 1
            if found == points:
                return True
        elif res.get("why") != "no consistent interior point":
            return False
    return False


def _graph(invariants, sol, tmp, pins):
    """Whether the invariants take the values `tmp` at the solved carriers, for every gauge value."""
    sym, own = _owned([t for t in tmp if t not in set(DOMAIN["real_carriers"])])
    try:
        sub = {sym(str(k)): own(v) for k, v in pins}
        at = {sym(c): own(e).xreplace(sub) for c, e in sol.items()}
        return all(_vanishes(own(e).xreplace(sub).xreplace(at) - sym(t)) for e, t in zip(invariants, tmp))
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return False


def model_denominators(exprs):
    """The distinct non-constant factors of the denominators of `exprs`."""
    if not exprs:
        return []
    exprs = [str(x).replace("^", "**") for x in exprs]
    loc = red_locals(exprs)
    out, keys = [], set()
    for x in exprs:
        try:
            den = spy.fraction(spy.together(red_sympify(x, loc)))[1]
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            continue
        if not free_syms(den):
            continue
        for fct in spy.Mul.make_args(den):
            if fct.is_Pow:
                fct = fct.base
            if free_syms(fct) and str(fct) not in keys:
                keys.add(str(fct))
                out.append(fct)
    return out


def _bad(e):
    return e is None or e.has(spy.zoo, spy.nan, spy.oo)


def face_section(b, pins, dens=()):
    """Face section: pin some coordinates to 0, certified to meet every orbit once, within the search budget."""
    out = {"solved": False}
    if not b.get("invariants") or any(re.search(r"exp\(|log\(", s) for s in b["invariants"]):
        return out
    if b.get("face_invariants") is not None and len(b["face_invariants"]) == len(b["invariants"]) \
            and callable(b.get("face_back")):
        b2 = {k: v for k, v in b.items() if k not in ("invariants", "face_invariants", "face_back")}
        b2["invariants"] = b["face_invariants"]
        res = face_section(b2, pins, dens)
        if res.get("solved"):
            nm = list(res["meaning"])
            res["meaning"] = dict(zip(nm, b["face_back"](list(res["meaning"].values()))))
        return res
    supp = b["support"]
    r = len(supp) - len(b["invariants"])
    if r < 1 or (b.get("target") is not None and len(b["invariants"]) < b["target"]):
        return out
    z_cand = [s for s in supp if s not in pins]
    if positive_set() is not None:
        z_cand = [s for s in z_cand if s in set(positive_set())]
    if len(z_cand) < r:
        return out
    loc = red_locals(list(b["invariants"]) + supp + list(pins) + list(pins.values()) +
                     [f"{s}_si_f" for s in supp])
    try:
        Ies = [red_sympify(iv.replace("^", "**"), loc) for iv in b["invariants"]]
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return out
    if pins:
        pin_pairs = [(spy.Symbol(nm), red_sympify(v, loc)) for nm, v in pins.items()]
        Ies = [e.subs(pin_pairs) for e in Ies]

    def zero_ok(sub):
        for Ie in Ies:
            try:
                e = spy.cancel(Ie.subs(sub))
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                return False
            if e.is_zero or _bad(e):
                return False
        return True
    z_cand = [v for v in z_cand if zero_ok([(spy.Symbol(v), spy.Integer(0))])]
    if len(z_cand) < r:
        return out
    z_ord = list(reversed(z_cand))
    conditional = []
    pos = positive_set()
    for s in itertools.combinations(range(len(z_cand)), r):
        if time_up():
            break
        Z = [z_ord[i] for i in s]
        sub_z = [(spy.Symbol(v), spy.Integer(0)) for v in Z]
        if not zero_ok(sub_z):
            continue
        ok_d = True
        for d in dens:
            try:
                e = spy.expand(d.subs(sub_z))
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                ok_d = False
                break
            if e.is_zero:
                ok_d = False
                break
        if not ok_d:
            continue
        U = [v for v in supp if v not in Z]
        face = sub_z + [(spy.Symbol(v), spy.Symbol(f"{v}_si_f")) for v in U]
        eqs, ok_z = [], True
        for Ie in Ies:
            try:
                Ief = spy.together(Ie.subs(face))
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                Ief = None
            if _bad(Ief):
                ok_z = False
                break
            eqs.append(spy.together(Ief - Ie))
        if not ok_z:
            continue
        info = {"complete": True}
        brs = tri_solve(eqs, [f"{v}_si_f" for v in U], info)
        good, n_open = None, 0 if info["complete"] else 1
        pos_u = [True] * len(U) if pos is None else [v in set(pos) for v in U]
        for br in brs:
            if len(br) < len(U):
                continue
            vals = [spy.cancel(br[f"{v}_si_f"]) for v in U]
            if any(_bad(e) or any(x.endswith("_si_f") for x in free_syms(e)) for e in vals):
                continue
            back = [(spy.Symbol(f"{v}_si_f"), vals[i]) for i, v in enumerate(U)]
            ok_eq = True
            for e in eqs:
                try:
                    fr = spy.fraction(spy.together(e.subs(back)))
                    if not (spy.cancel(fr[0]).is_zero and not spy.cancel(fr[1]).is_zero):
                        ok_eq = False
                        break
                except SYMPY_FAILS as exc:
                    sympy_fail(exc)
                    ok_eq = False
                    break
            if not ok_eq:
                continue
            sg_all = [sgn(e) for e in vals]
            if any(x == -1 for x in sg_all):
                continue
            if all(x == 1 for x in sg_all) and good is None:
                good = vals
                continue
            n_open += 1
            sg = [1 if not pos_u[i] else x for i, x in enumerate(sg_all)]
            if any(x == -1 for x in sg):
                continue
            if len(brs) == 1:
                cond = []
                for i, e in enumerate(vals):
                    if sg[i] != 0:
                        continue
                    hit = None
                    for k, Ie in enumerate(Ies):
                        try:
                            q = sgn(spy.cancel(e / Ie))
                        except SYMPY_FAILS as exc:
                            sympy_fail(exc)
                            q = 0
                        if q != 0:
                            hit = (k, q)
                            break
                    cond.append(hit)
                if cond and all(c is not None for c in cond):
                    if len({c[0] for c in cond}) == 1 and len({c[1] for c in cond}) == 1:
                        conditional.append({"face": Z, "inv": cond[0][0], "sign": cond[0][1]})
        if good is None or n_open > 0:
            continue
        id_face = True
        for i, v in enumerate(U):
            try:
                e = spy.cancel(good[i].subs(sub_z) - spy.Symbol(v))
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                id_face = False
                break
            if not e.is_zero:
                id_face = False
                break
        if not id_face:
            continue
        res = dict(out)
        res.update(solved=True, pins={z: "0" for z in Z}, gauge=Z, section=[f"{z} = 0" for z in Z],
                   face=True, coverage="total", solved_domain={}, inv_names=[], shifted=[],
                   meaning={v: str(spy.factor(e)).replace("**", "^") for v, e in zip(U, good)})
        return res
    for a in conditional:
        for c2 in conditional:
            if a["inv"] == c2["inv"] and a["sign"] == 1 and c2["sign"] == -1:
                out["split"] = {"invariant": str(Ies[a["inv"]]).replace("**", "^"),
                                "positive": [f"{z} = 0" for z in a["face"]],
                                "negative": [f"{z} = 0" for z in c2["face"]]}
                return out
    return out


def face_carriers(sol, coords, inv_start):
    """`sol` with each coordinate the face moves pinned to a fresh carrier q_<k> holding its value there."""
    m = sol["meaning"]
    moved = [u for u, v in m.items() if re.sub(r"\s", "", v) != u and u in set(coords)]
    m = {u: m[u] for u in moved}
    if not moved:
        sol["meaning"] = m
        return sol
    qn, k = [], inv_start
    for _ in moved:
        while True:
            k += 1
            if f"q_{k}" not in set(coords):
                break
        qn.append(f"q_{k}")
    sol["pins"] = dict(sol["pins"], **dict(zip(moved, qn)))
    sol["inv_names"] = qn
    sol["meaning"] = dict(zip(qn, [m[u] for u in moved]))
    sol["solved_domain"] = {q: "positive" for q in qn}
    return sol


def prep_degree(comps):
    """Maximal total degree of the components `comps`, or None."""
    loc = red_locals(list(comps.values()))
    degs = []
    for x in comps.values():
        try:
            degs.append(int(spy.total_degree(red_sympify(str(x).replace("^", "**"), loc))))
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            pass
    return max(degs) if degs else None


def translation_compress(preps):
    """A moved coordinate entering together with unmoved ones only through their sum."""
    supp = list(dict.fromkeys(c for pr in preps for c in pr["support"]))
    vars_ = list(dict.fromkeys(v for pr in preps for v in pr["vars"]))
    unmoved = [v for v in vars_ if v not in set(supp)]
    if not unmoved:
        return None
    loc = red_locals([x for pr in preps for x in pr["comps"].values()] + vars_)
    E = [{k: red_sympify(str(x).replace("^", "**"), loc) for k, x in pr["comps"].items()} for pr in preps]

    def dz(e, a, b):
        return spy.expand(spy.diff(e, spy.Symbol(a)) - spy.diff(e, spy.Symbol(b))).is_zero

    def seen(v):
        return any(v in free_syms(e) for ex in E for e in ex.values())
    groups, used = {}, set()
    for v in supp:
        if not seen(v):
            continue
        mem = [w for w in unmoved if w not in used and
               all(dz(e, v, w) for ex in E for e in ex.values())]
        if mem:
            groups[v] = mem
            used |= set(mem)
    if not groups:
        return None
    taken = set(vars_) | set(loc)
    U = {}
    for k, v in enumerate(groups, start=1):
        u = f"si_u{k}"
        while u in taken:
            u += "_"
        U[v] = u
    fwd = [(spy.Symbol(v), spy.Symbol(U[v]) - sum(spy.Symbol(w) for w in groups[v])) for v in groups]
    all_mem = {w for m in groups.values() for w in m}
    new_preps = []
    for i, pr in enumerate(preps):
        ex = {k: spy.expand(e.subs(fwd)) for k, e in E[i].items()}
        left = {s for e in ex.values() for s in free_syms(e)}
        if all_mem & left:
            raise RouteError("translation group did not separate")
        keys = [U.get(k, k) for k in pr["comps"]]
        comps = dict(zip(keys, [str(e) for e in ex.values()]))
        pr = dict(pr, comps=comps, support=keys, vars=sym_sort(list(dict.fromkeys(keys + list(left)))),
                  degree=prep_degree(comps))
        new_preps.append(pr)
    bwd = [(spy.Symbol(U[v]), sum((spy.Symbol(x) for x in [v] + groups[v]), spy.Integer(0))) for v in groups]
    extra_names = list(U.values()) + list(groups) + [w for m in groups.values() for w in m]

    def back(invs):
        out = []
        for iv in invs:
            l2 = red_locals([iv] + extra_names)
            e = spy.sympify(iv.replace("^", "**"), locals=l2).subs(bwd)
            out.append(str(spy.factor(spy.together(e))).replace("**", "^"))
        return out
    key_back = {u: v for v, u in U.items()}

    def back_prep(pr):
        l2 = red_locals(list(pr["comps"].values()) + extra_names)
        comps = {key_back.get(k, k): str(spy.expand(spy.sympify(str(x).replace("^", "**"), locals=l2).subs(bwd)))
                 for k, x in pr["comps"].items()}
        keys = list(comps)
        vs = {s for x in comps.values() for s in free_syms(spy.sympify(x, locals=l2))}
        return dict(pr, comps=comps, support=keys, vars=sym_sort(list(dict.fromkeys(keys + list(vs)))),
                    degree=prep_degree(comps))
    return {"preps": new_preps, "back": back, "back_prep": back_prep,
            "note": [" + ".join([v] + groups[v]) for v in groups]}


def unmoved_compress(preps):
    """Unmoved coordinates entering only through a few monomials, searched as single symbols."""
    supp = list(dict.fromkeys(c for pr in preps for c in pr["support"]))
    vars_ = list(dict.fromkeys(v for pr in preps for v in pr["vars"]))
    unmoved = sym_sort([v for v in vars_ if v not in set(supp)])
    if len(unmoved) < 2:
        return None
    loc = red_locals([x for pr in preps for x in pr["comps"].values()] + vars_)
    usyms = [spy.Symbol(u) for u in unmoved]
    terms = []
    for pr in preps:
        tp = []
        for x in pr["comps"].values():
            P = spy.Poly(red_sympify(str(x).replace("^", "**"), loc), *usyms)
            tp.append([{"e": [int(a) for a in t[0]], "c": t[1]} for t in P.terms()])
        terms.append(tp)
    all_e = []
    for tp in terms:
        for tl in tp:
            for t in tl:
                if t["e"] not in all_e:
                    all_e.append(t["e"])
    E = np.array([e for e in all_e if sum(e) > 0], dtype=np.int64)
    if E.size == 0:
        return None

    def gcdv(v):
        g = 0
        for x in v:
            g = math.gcd(g, abs(int(x)))
        return g
    prim = []
    for row in E:
        p_ = list(row // gcdv(row))
        if p_ not in prim:
            prim.append(p_)
    prim = sorted(prim, key=sum)
    B = []

    def coef_of(v, B):
        # the unique exact solution of B^T c = v when it is a nonnegative integer vector, else None
        if not B:
            return [] if not any(int(x) for x in v) else None
        try:
            sol, par = spy.Matrix(B).T.gauss_jordan_solve(spy.Matrix([int(x) for x in v]))
        except ValueError:
            return None
        if par.shape[0] or any(not (x.is_Integer and x >= 0) for x in sol):
            return None
        return [int(x) for x in sol]
    for p_ in prim:
        if coef_of(p_, B) is None:
            B.append(p_)
    if len(B) > 3 or len(B) >= int(np.sum(E.sum(axis=0) > 0)):
        return None
    if any(coef_of(E[i], B) is None for i in range(E.shape[0])):
        return None
    taken = set(vars_) | set(loc)
    W = []
    for k in range(1, len(B) + 1):
        w = f"si_w{k}"
        while w in taken:
            w += "_"
        W.append(w)
    wsyms = [spy.Symbol(w) for w in W]
    used_u = [unmoved[j] for j in range(len(unmoved)) if E[:, j].sum() > 0]
    new_preps = []
    for i, pr in enumerate(preps):
        comps = {}
        for j, key in enumerate(pr["comps"]):
            e = spy.Integer(0)
            for tm in terms[i][j]:
                k = coef_of(tm["e"], B)
                mono = spy.Integer(1)
                for ws, kk in zip(wsyms, k or []):
                    mono = mono * spy.Pow(ws, spy.Integer(kk))
                e = e + tm["c"] * mono
            comps[key] = str(spy.expand(e))
        vs = set(pr["support"]) | (set(pr["vars"]) - set(used_u))
        for x in comps.values():
            vs |= set(free_syms(red_sympify(x, red_locals([x] + W))))
        new_preps.append(dict(pr, comps=comps, degree=prep_degree(comps), vars=sym_sort(list(vs))))
    Bm = np.array(B, dtype=np.int64)
    bwd = []
    for k in range(len(B)):
        e = spy.Integer(1)
        for j in np.nonzero(Bm[k] > 0)[0]:
            e = e * spy.Pow(usyms[j], spy.Integer(int(Bm[k, j])))
        bwd.append((wsyms[k], e))

    def back(invs):
        out = []
        for iv in invs:
            l2 = red_locals([iv] + W + unmoved)
            e = spy.sympify(iv.replace("^", "**"), locals=l2).subs(bwd)
            out.append(str(spy.factor(spy.together(e))).replace("**", "^"))
        return out

    def back_prep(pr):
        l2 = red_locals(list(pr["comps"].values()) + W + unmoved)
        comps = {k: str(spy.expand(spy.sympify(str(x).replace("^", "**"), locals=l2).subs(bwd)))
                 for k, x in pr["comps"].items()}
        vs = set(pr["support"]) | {s for x in comps.values() for s in free_syms(spy.sympify(x, locals=l2))}
        return dict(pr, comps=comps, vars=sym_sort(list(vs)), degree=prep_degree(comps))
    note = []
    for k in range(len(B)):
        parts = [unmoved[j] if Bm[k, j] == 1 else f"{unmoved[j]}^{int(Bm[k, j])}"
                 for j in np.nonzero(Bm[k] > 0)[0]]
        note.append("*".join(parts))
    return {"preps": new_preps, "back": back, "back_prep": back_prep, "note": note}


def match_carriers(adm):
    """One distinct carrier per entry from its candidate list in `adm`, or None."""
    n = len(adm)
    pick = [None] * n

    def go(l, used):
        if l >= n:
            return True
        for v in adm[l]:
            if v in used:
                continue
            pick[l] = v
            if go(l + 1, used | {v}):
                return True
        return False
    return pick if go(0, frozenset()) else None
