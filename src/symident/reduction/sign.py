"""Signs and positivity over the declared domain."""

import math
import re
import time
import warnings

import numpy as np
import sympy as spy

from .expr import free_syms, red_locals, red_sympify


def entry_class(e):
    """'yes', 'root' (a non-integer rational power) or 'no' for an entry outside rational/exp/log."""
    try:
        fnames = {type(f).__name__ for f in e.atoms(spy.Function)}
    except Exception:
        return "no"
    if fnames and not fnames <= {"exp", "log"}:
        return "no"
    root = False
    for pw in e.atoms(spy.Pow):
        if not pw.exp.is_Rational:
            return "no"
        if not pw.exp.is_Integer:
            root = True
    return "root" if root else "yes"


def positive_set_of(positive, coords):
    """The positive coordinates among `coords` from `positive`, None for all; unknown names warn."""
    if positive is True:
        return None
    if positive is False:
        return []
    if isinstance(positive, str):
        positive = [positive]
    positive = list(dict.fromkeys(positive))
    unknown = [p for p in positive if p not in set(coords)]
    if unknown:
        warnings.warn(f"no effect: {', '.join(unknown)}: not a coordinate of the analysis.",
                      stacklevel=2)
    return [p for p in positive if p in set(coords)]


DOMAIN = {"positive": None, "real_carriers": [], "no_roots": False, "sgn_cache": None, "deadline": None}


def positive_set():
    """The declared positive coordinates; None for the positive orthant."""
    return DOMAIN["positive"]


def domain_name():
    """Name of the declared domain for reports."""
    return "positive orthant" if DOMAIN["positive"] is None else "declared domain"


def set_positive(x):
    """Declare the positive coordinates `x` and clear the sign cache; returns the previous ones."""
    old = DOMAIN["positive"]
    DOMAIN["positive"] = x
    DOMAIN["sgn_cache"] = {}
    return old


def _num_val(e):
    try:
        return float(spy.N(e))
    except Exception:
        return None


def sgn_poly(pp):
    """Sign of a radical-free polynomial on the declared domain; 0 when undecided."""
    try:
        ex = spy.expand(pp)
    except Exception:
        return 0
    if ex.is_number:
        v = _num_val(ex)
        if v is None or v == 0 or not math.isfinite(v):
            return 0
        return 1 if v > 0 else -1
    try:
        P = spy.Poly(ex)
        tm = P.terms()
    except Exception:
        return 0
    if not tm:
        return 0
    expts = np.array([list(t[0]) for t in tm], dtype=float)
    cf = []
    for t in tm:
        v = _num_val(t[1])
        if v is None or not math.isfinite(v):
            return 0
        cf.append(v)
    cf = np.array(cf)
    pos = positive_set()
    if pos is not None:
        try:
            gens = [str(g) for g in P.gens]
        except Exception:
            return 0
        if len(gens) != expts.shape[1]:
            return 0
        pos_s = set(pos)
        real = set(DOMAIN["real_carriers"])
        free = [g not in pos_s and not (re.match(r"^si_red_c[0-9]+$", g) and g not in real) for g in gens]
        for j, fr in enumerate(free):
            if fr and np.any(expts[:, j] % 2 != 0):
                return 0
    if np.all(cf > 0):
        return 1
    if np.all(cf < 0):
        return -1
    if absorb(expts, cf):
        return 1
    if absorb(expts, -cf):
        return -1
    return 0


def absorb(expts, cf):
    """AM-GM absorption of every negative term by two positive ones with matching exponents."""
    bud = np.where(cf > 0, cf, 0.0)
    tol = 1e-9
    n = len(cf)
    for i in np.nonzero(cf < 0)[0]:
        need_v = -cf[i]
        tgt = 2 * expts[i]
        for u in range(n):
            if bud[u] <= tol:
                continue
            for v in range(u + 1, n):
                if bud[v] <= tol:
                    continue
                if np.any(expts[u] + expts[v] != tgt):
                    continue
                take = min(2 * math.sqrt(bud[u] * bud[v]), need_v)
                bu, bv = bud[u], bud[v]
                bud[u] = max(0, bu - (take / 2) * math.sqrt(bu / bv))
                bud[v] = max(0, bv - (take / 2) * math.sqrt(bv / bu))
                need_v -= take
                if need_v <= tol * abs(cf[i]):
                    break
            if need_v <= tol * abs(cf[i]):
                break
        if need_v > tol * abs(cf[i]):
            return False
    return bud.sum() > tol * np.abs(cf).sum()


def time_up():
    """Whether the deadline of the domain has passed."""
    d = DOMAIN["deadline"]
    return d is not None and time.monotonic() > d


def sgn(pp, depth=0):
    """Sign on the declared domain by structural recursion; memoised per domain."""
    cache = DOMAIN["sgn_cache"]
    if depth > 0 or cache is None:
        return sgn_raw(pp, depth)
    try:
        key = f"{bool(DOMAIN['no_roots'])}:{','.join(sorted(DOMAIN['real_carriers']))}|{spy.srepr(pp)}"
    except Exception:
        return sgn_raw(pp, depth)
    if key in cache:
        return cache[key]
    v = sgn_raw(pp, depth)
    cache[key] = v
    return v


def sgn_raw(pp, depth=0):
    """Sign of `pp` on the declared domain by structural recursion, uncached; 0 when undecided."""
    if depth > 8:
        return 0
    try:
        ex = spy.expand(pp)
    except Exception:
        return 0
    if ex.is_number:
        v = _num_val(ex)
        if v is None or v == 0 or not math.isfinite(v):
            return 0
        return 1 if v > 0 else -1
    if ex.is_Function and type(ex).__name__ == "exp":
        return 1
    if ex.is_Pow and ex.exp.is_Rational:
        sb = sgn(ex.base, depth + 1)
        if sb == 1:
            return 1
        if sb == -1 and ex.exp.is_Integer:
            return 1 if int(ex.exp) % 2 == 0 else -1
        return 0
    if ex.is_Mul:
        s = 1
        for arg in spy.Mul.make_args(ex):
            sa = sgn(arg, depth + 1)
            if sa == 0:
                return 0
            s *= sa
        return s
    try:
        rads = [w for w in ex.atoms(spy.Pow) if w.exp.is_Rational and not w.exp.is_Integer
                and not w.base.is_number]
    except Exception:
        rads = []
    if not rads:
        try:
            fr = spy.fraction(spy.cancel(spy.together(ex)))
        except Exception:
            return 0
        s1, s2 = sgn_poly(fr[0]), sgn_poly(fr[1])
        if s1 == 0 or s2 == 0:
            return 0
        return s1 * s2
    for s in sorted(rads, key=lambda w: -len(str(w))):
        if not spy.simplify(s.exp - spy.Rational(1, 2)).is_zero:
            continue
        try:
            a, rest = ex.as_independent(s, as_Add=True)
            cc = spy.cancel(rest / s)
        except Exception:
            continue
        if cc.has(s):
            continue
        if sgn(s.base, depth + 1) != 1:
            continue
        sc = sgn(cc, depth + 1)
        if a.is_zero:
            return sc
        sa = sgn(a, depth + 1)
        if sc == 0:
            continue
        if sa == 0 and DOMAIN["no_roots"]:
            continue
        if sa == 0:
            try:
                d2 = spy.expand(cc ** 2 * s.base - a ** 2)
            except Exception:
                d2 = None
            if d2 is not None and sgn(d2, depth + 1) == 1:
                return sc
            continue
        if sa == sc:
            return sa
        try:
            d = spy.expand(a ** 2 - cc ** 2 * s.base)
        except Exception:
            continue
        sdd = sgn(d, depth + 1)
        if sdd == 0:
            continue
        return sa if sdd > 0 else sc
    return 0


def shifted_meaning(invariants, nms, tmp_n, sh):
    """Meaning of each invariant name, the shifted meaning where `sh` shifts it."""
    m = dict(zip(nms, invariants))
    if sh is None:
        return m
    for l, nm in enumerate(nms):
        tl = tmp_n[l]
        if sh["shift"].get(tl) is not None:
            m[nm] = str(sh["meaning"][tl]).replace("**", "^")
    return m


def _rads(e, half_only=False, symbol_base=False):
    out = []
    try:
        for w in e.atoms(spy.Pow):
            if not w.exp.is_Rational or w.exp.is_Integer:
                continue
            if symbol_base:
                if not w.base.is_Symbol:
                    continue
            elif w.base.is_number:
                continue
            if half_only and not spy.simplify(w.exp - spy.Rational(1, 2)).is_zero:
                continue
            out.append(w)
    except Exception:
        return []
    return out


def rad_norm(e, rounds=3):
    """Radical denominators rationalised by conjugate multiplication."""
    for _ in range(rounds):
        try:
            num, den = spy.fraction(spy.cancel(spy.together(e)))
        except Exception:
            return e
        rads = _rads(den, half_only=True)
        if not rads:
            return spy.cancel(num / den)
        s = sorted(rads, key=lambda w: -len(str(w)))[0]
        try:
            a, b = den.as_independent(s, as_Add=True)
            conj = a - b
            e = spy.cancel(spy.expand(num * conj) / spy.expand(den * conj))
        except Exception:
            return e
    return e


def half_pow(e):
    """cancel() modulo s^2 = z: symbol radicands substituted by squares of positive symbols."""
    rads = _rads(e, symbol_base=True)
    bases = list(dict.fromkeys(str(w.base) for w in rads))
    bases = [v for v in bases if all((spy.Integer(2) * w.exp).is_Integer for w in rads if str(w.base) == v)]
    if not bases:
        return spy.cancel(spy.together(e))
    ws = [spy.Symbol(f"si_red_h_{v}", positive=True) for v in bases]
    fwd = [(spy.Symbol(v), w * w) for v, w in zip(bases, ws)]
    back = [(w, spy.sqrt(spy.Symbol(v))) for v, w in zip(bases, ws)]
    out = spy.cancel(spy.together(e.subs(fwd)))
    return spy.cancel(out.subs(back))


def _forms(e):
    out = []
    for fn in (lambda x: spy.cancel(spy.together(x)), half_pow, rad_norm):
        try:
            out.append(fn(e))
        except Exception:
            out.append(None)
    return out


def _dom_key(prefix, e):
    return f"{prefix}:{bool(DOMAIN['no_roots'])}:{','.join(sorted(DOMAIN['real_carriers']))}|{spy.srepr(e)}"


def pos_form(e):
    """A form of e certified positive on the domain, or None."""
    cache = DOMAIN["sgn_cache"]
    key = None
    if cache is not None:
        try:
            key = _dom_key("pos", e)
        except Exception:
            key = None
    if key is not None and key in cache:
        return cache[key]
    f = pos_form_raw(e)
    if key is not None:
        cache[key] = f
    return f


def pos_form_raw(e):
    """A form of `e` whose numerator and denominator have the same certified sign, or None; uncached."""
    seen = set()
    for f in _forms(e):
        if f is None:
            continue
        k = str(f)
        if k in seen:
            continue
        seen.add(k)
        try:
            num, den = spy.fraction(f)
        except Exception:
            continue
        s1 = sgn(num)
        if s1 != 0 and s1 == sgn(den):
            return f
    return None


def pos_cert(e):
    """Whether `e` is certified positive on the domain."""
    return pos_form(e) is not None


def shift_fix(es, tmp_n, Ies):
    """Indefinite factors linear in a placeholder, made definite by a certified shift."""
    keys = list(es) if isinstance(es, dict) else None
    es = list(es.values()) if keys is not None else list(es)
    shift = {}
    mean_e = dict(zip(tmp_n, Ies))
    tsym = {t: spy.Symbol(t) for t in tmp_n}
    cache = DOMAIN["sgn_cache"] if DOMAIN["sgn_cache"] is not None else {}

    def fparts(part):
        try:
            fl = spy.factor_list(part)
        except Exception:
            return []
        return [fm[0] for fm in fl[1]]

    def factor_bases(e0):
        try:
            key = "fb:" + spy.srepr(e0)
        except Exception:
            key = None
        if key is not None and key in cache:
            return cache[key]
        out, seen = [], set()

        def push(f):
            k = str(f)
            if k not in seen:
                seen.add(k)
                out.append(f)
        form_seen = set()
        for e in [e0] + _forms(e0)[1:]:
            if e is None:
                continue
            ek = str(e)
            if ek in form_seen:
                continue
            form_seen.add(ek)
            try:
                fr = spy.fraction(spy.cancel(spy.together(e)))
            except Exception:
                continue
            lvl1 = fparts(fr[0]) + fparts(fr[1])
            for f in lvl1:
                push(f)
            for f in lvl1:
                for f2 in fparts(f):
                    push(f2)
            for w in _rads(e):
                for f2 in fparts(w.base):
                    push(f2)
        if key is not None:
            cache[key] = out
        return out

    def find_one():
        for e in es:
            if pos_cert(e):
                continue
            for f in factor_bases(e):
                if sgn(f) != 0:
                    continue
                fs = free_syms(f)
                for tl in [t for t in tmp_n if t not in shift and t in fs]:
                    if set(fs) - {tl}:
                        try:
                            dg = int(spy.Poly(f, tsym[tl]).degree())
                        except Exception:
                            dg = None
                        if dg != 1:
                            continue
                    try:
                        sk = f"sv:{tl}:{spy.srepr(f)}"
                    except Exception:
                        sk = None
                    rt = cache.get(sk) if sk is not None else None
                    if rt is None:
                        try:
                            rt = spy.solve(f, tsym[tl])
                        except Exception:
                            rt = []
                        if sk is not None:
                            cache[sk] = rt
                    if len(rt) != 1:
                        continue
                    s = rad_norm(rt[0])
                    if sgn(s) != 1:
                        continue
                    sub = [(tsym[tj], mean_e[tj]) for tj in tmp_n if tj != tl]
                    try:
                        cov = rad_norm(mean_e[tl] - s.subs(sub))
                    except Exception:
                        continue
                    if sgn(cov) != 1:
                        continue
                    return {"tl": tl, "s": s, "cov": cov}
        return None

    def norm(e):
        try:
            return spy.cancel(spy.together(e))
        except Exception:
            return e
    es = [norm(e) for e in es]
    for _ in tmp_n:
        hit = find_one()
        if hit is None:
            break
        shift[hit["tl"]] = hit["s"]
        mean_e[hit["tl"]] = hit["cov"]
        pr = [(tsym[hit["tl"]], tsym[hit["tl"]] + hit["s"])]
        es = [norm(e.subs(pr)) for e in es]
    if keys is not None:
        es = dict(zip(keys, es))
    return {"es": es, "shift": shift, "meaning": mean_e}


def sgn_real(pp, real_syms):
    """Sign with `real_syms` ranging over R and the rest over the positive orthant."""
    if not real_syms:
        return sgn(pp)
    try:
        ex = spy.expand(pp)
    except Exception:
        return 0
    present = [s for s in real_syms if s in set(free_syms(ex))]
    if not present:
        return sgn(ex)
    try:
        sq = [w for w in ex.atoms(spy.Pow) if w.exp == spy.Rational(1, 2) and not w.base.is_number]
    except Exception:
        sq = []
    if len(sq) == 1 and not DOMAIN["no_roots"]:
        w = sq[0]
        try:
            a, rest = ex.as_independent(w, as_Add=True)
            cc = spy.cancel(rest / w)
        except Exception:
            return 0
        if cc.has(w):
            return 0
        if sgn_real(w.base, real_syms) != 1:
            return 0
        sc = sgn_real(cc, real_syms)
        if sc == 0:
            return 0
        if a.is_zero:
            return sc
        try:
            d2 = spy.expand(cc ** 2 * w.base - a ** 2)
        except Exception:
            d2 = None
        if d2 is not None and sgn_real(d2, real_syms) == 1:
            return sc
        return sc if sgn_real(a, real_syms) == sc else 0
    if len(sq) > 1:
        return 0
    rest = ex
    corr = spy.Integer(0)
    rset = set(real_syms)
    for t in present:
        ts = spy.Symbol(t)
        try:
            A = spy.expand(rest.coeff(ts, 2))
            B = spy.expand(rest.coeff(ts, 1))
            C = spy.expand(rest.coeff(ts, 0))
            chk = str(spy.expand(rest - (A * ts ** 2 + B * ts + C)))
        except Exception:
            return 0
        if chk != "0":
            return 0
        if (set(free_syms(A)) | set(free_syms(B))) & rset:
            return 0
        if sgn(A) != 1:
            return 0
        corr = corr + B ** 2 / (4 * A)
        rest = C
    try:
        red = spy.cancel(spy.together(rest - corr))
    except Exception:
        return 0
    return 1 if sgn(red) == 1 else 0


def pos_form_real(e, real_syms):
    """A form of `e` with numerator and denominator certified positive, `real_syms` over R, or None."""
    if not real_syms:
        return pos_form(e)
    seen = set()
    for f in _forms(e):
        if f is None:
            continue
        k = str(f)
        if k in seen:
            continue
        seen.add(k)
        try:
            num, den = spy.fraction(f)
        except Exception:
            continue
        if sgn_real(num, real_syms) == 1 and sgn_real(den, real_syms) == 1:
            return f
    return None


def is_real(v):
    """Whether `v` ranges over R, outside the declared positive coordinates."""
    pos = positive_set()
    return pos is not None and v not in set(pos)


def translation_gauge(b, gauge):
    """Whether `gauge` has a real coordinate and block `b` moves every real coordinate of it."""
    if not b.get("preps") or not any(is_real(v) for v in gauge):
        return False
    for v in gauge:
        if not is_real(v):
            continue
        xs = []
        for pr in b["preps"]:
            cv = pr["comps"].get(v)
            try:
                xs.append(0.0 if cv is None else float(cv))
            except ValueError:
                return False
        if not any(x != 0 for x in xs):
            return False
    return True


def translates(b, v, pinned):
    """Whether a direction of block `b` on real coordinates, free of `pinned` but `v`, translates `v`."""
    for pr in b["preps"]:
        if v not in pr["comps"] or not all(is_real(s) for s in pr["support"]) or \
                (set(pinned) - {v}) & set(pr["support"]):
            continue
        loc = red_locals(list(pr["comps"].values()))
        try:
            ex = {k: red_sympify(x, loc) for k, x in pr["comps"].items()}
            g = None
            for e in ex.values():
                g = e if g is None else spy.gcd(g, e)
            ex = {k: spy.cancel(e / g) for k, e in ex.items()}
        except Exception:
            continue
        if any(set(free_syms(e)) & set(pr["support"]) for e in ex.values()):
            continue
        if sgn(ex[v]) != 0:
            return True
    return False


def real_form(e, real_syms):
    """`e` cancelled if rational or a real root with a denominator of certified sign over `real_syms`, else None."""
    try:
        f = spy.cancel(spy.together(e))
    except Exception:
        return None
    cls = entry_class(f)
    if cls == "root" and DOMAIN["no_roots"]:
        return None
    if cls == "root":
        rads = [w for w in f.atoms(spy.Pow) if not w.exp.is_Integer and not w.base.is_number]
        if not rads or any(sgn_real(w.base, real_syms) != 1 for w in rads):
            return None
    elif cls != "yes":
        return None
    if f.atoms(spy.log):
        return None
    den = spy.fraction(f)[1]
    if sgn_real(den, real_syms) != 1 and sgn_real(-den, real_syms) != 1:
        return None
    return f
