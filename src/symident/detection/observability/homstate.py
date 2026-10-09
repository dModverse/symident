"""Kernel entries as rational functions of the parameters and the resting state of one
condition: :math:`e = N(\\theta, x)/D(\\theta, x)` from the series identity
:math:`N - e\\,D \\equiv 0 \\bmod t^T` along lines :math:`\\theta_0 + t\\,d`, with the
monomials reduced modulo the leading terms of the resting equations."""

import itertools
import re

import numpy as np
import sympy as spy

from ... import _core
from ...gfp import PRIMES, VERIFY_PRIME, invmod, mulmod, random_point
from ...sym.linalg import sym_rat_recon_big
from ..closedform import poly_string
from ..fields import ser_mul
from ..series import series_null

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")
_N_PRIMES = 12
_MAX_UNKNOWNS = 1200
_N_BASES = 32


def _candidate_primes():
    """The analysis primes followed by further primes below them, the verification prime left out."""
    out = list(PRIMES)
    q = min(PRIMES)
    while len(out) < _N_PRIMES:
        q = int(spy.prevprime(q))
        if q != VERIFY_PRIME:
            out.append(q)
    return out


def _monomials(nv, deg, banned):
    """Exponent vectors of total degree at most deg, none divisible by a banned one."""
    out = []
    for d in range(deg + 1):
        for c in itertools.combinations_with_replacement(range(nv), d):
            e = [0] * nv
            for i in c:
                e[i] += 1
            if not any(all(e[k] >= b[k] for k in range(nv)) for b in banned):
                out.append(tuple(e))
    return out


def _leading(lines, names):
    """Leading exponents (graded reverse lexicographic, states first) of the numerators of
    the resting equations `lhs = rhs` in the variables `names` alone."""
    out = []
    syms = [spy.Symbol(n) for n in names]
    for ln in lines:
        rhs = ln.split("=", 1)[1] if "=" in ln else ln
        loc = {t: spy.Symbol(t) for t in _IDENT.findall(rhs)}
        try:
            e = spy.sympify(rhs.replace("^", "**"), locals=loc)
            num = spy.fraction(spy.together(e))[0]
            P = spy.Poly(spy.expand(num), *syms)
        except Exception:
            continue
        if P.is_zero or P.total_degree() < 1 or (num.free_symbols - set(syms)):
            continue
        out.append(tuple(int(x) for x in P.monoms(order="grevlex")[0]))
    return out


class StateFit:
    """Fits of the entries of one homotopy over (relevant parameters, resting states)."""

    def __init__(self, hom):
        self.hom, self.sys = hom, hom.sys
        self.lines = {}
        self.good_base = {}
        self.mis = []
        n = len(hom.base)
        self.pool0 = hom.rec.pool_next
        hom.rec.pool_next += _N_BASES * (n + 1)

    def use(self, mis):
        """Resting states of the conditions `mis` as variables, suffixed _c<k> for several."""
        sys = self.sys
        self.states, self.var = [], {}
        for mi in mis:
            ci = sys.equil_conds[mi]
            zero = set(sys.zero_of(ci))
            st = sys.hom_states[mi] if len(getattr(sys, "hom_states", [])) > mi else {}
            for nm in sys.real_state_names:
                if nm in st and nm not in zero and st[nm][0] % PRIMES[0] != 0:
                    v = nm if len(mis) == 1 else f"{nm}_c{mi + 1}"
                    self.states.append(v)
                    self.var[v] = (mi, nm)
        self.mis = list(mis)

    def state_sets(self, fc):
        """The condition of a free state column (else the first), then all conditions."""
        sys = self.sys
        mi = 0 if fc < sys.n_shared or not sys.n_st else (fc - sys.n_shared) // sys.n_st
        sets = [[mi]]
        if sys.kc > 1:
            sets.append(list(range(sys.kc)))
        return sets

    def names(self, reli):
        """The variables of a fit: the resting states, then the relevant parameters `reli`."""
        return list(self.states) + list(reli)


    def resting_subs(self):
        """The resting states of the used conditions as rational functions of the parameters,
        where their equations are linear in the states; an empty dict otherwise."""
        subs = {}
        for mi in self.mis:
            ren = {nm: v for v, (m, nm) in self.var.items() if m == mi}
            if not ren:
                continue
            syms = {v: spy.Symbol(v) for v in ren.values()}
            eqs = []
            for ln in self.sys.models[self.sys.equil_conds[mi]]:
                rhs = ln.split("=", 1)[1] if "=" in ln else ln
                rhs = _IDENT.sub(lambda m: ren.get(m.group(0), m.group(0)), rhs)
                loc = {t: spy.Symbol(t) for t in _IDENT.findall(rhs)}
                try:
                    num = spy.fraction(spy.together(spy.sympify(rhs.replace("^", "**"), locals=loc)))[0]
                    P = spy.Poly(spy.expand(num), *syms.values())
                except Exception:
                    return {}
                if P.total_degree() > 1:
                    return {}
                if P.total_degree() == 1:
                    eqs.append(P.as_expr())
            sol = spy.solve(eqs, list(syms.values()), dict=True)
            if len(sol) != 1 or set(sol[0]) != set(syms.values()):
                return {}
            subs.update(sol[0])
        return subs

    def substitute(self, expr):
        """`expr` with the resting states replaced by `resting_subs()`, unchanged without them."""
        subs = self.resting_subs()
        if not subs:
            return expr
        loc = {t: spy.Symbol(t) for t in _IDENT.findall(expr)}
        e = spy.sympify(expr.replace("^", "**"), locals=loc)
        return str(spy.factor(spy.cancel(e.subs(subs)))).replace("**", "^")

    def banned(self, names):
        """Leading terms of the resting equations of the used conditions, states renamed."""
        out = []
        for mi in self.mis:
            ren = {nm: v for v, (m, nm) in self.var.items() if m == mi}
            lines = []
            for ln in self.sys.models[self.sys.equil_conds[mi]]:
                rhs = ln.split("=", 1)[1] if "=" in ln else ln
                lines.append(_IDENT.sub(lambda m: ren.get(m.group(0), m.group(0)), rhs))
            out += _leading(lines, names)
        return out

    def _series(self, ns, nm, reli, d, p, T, base):
        """Series of a variable along the line: a state from the continuation, a parameter
        linear in t."""
        if nm in self.var:
            mi, sn = self.var[nm]
            v = ns["states"][mi][sn]
            return [int(x) % p for x in (list(v) + [0] * T)[:T]]
        out = [0] * T
        out[0] = int(base[self.hom.an.run.slot(nm)]) % p
        if T > 1:
            out[1] = int(d.get(nm, 0)) % p
        return out

    def _rows(self, a, i, reli, mons, dg, p, T, n_rows, seed0):
        """Rows [m(t)_k | -(e m)(t)_k] of the series identity, from lines until n_rows."""
        hom = self.hom
        names = self.names(reli)
        rows = []
        self.den_at0 = []
        line = 0
        while len(rows) < n_rows:
            line += 1
            if line > 4 * n_rows:
                return None
            d = hom.zero_line()
            for nm, x in zip(reli, random_point(len(reli), seed0 + 31 * line + 7 * dg)):
                d[nm] = int(x) % p
            ns, base = self._line(d, T - 1, p, seed0 + 31 * line + 7 * dg)
            if ns is None:
                return None
            es = hom.entry(ns, a, hom.cols[a][i], p)
            if es is None:
                return None
            es = [int(x) % p for x in es[:T]]
            var = [self._series(ns, nm, reli, d, p, T, base) for nm in names]
            pw = [[[1] + [0] * (T - 1)] for _ in names]
            mono = []
            for m in mons:
                s = [1] + [0] * (T - 1)
                for v, k in enumerate(m):
                    while len(pw[v]) <= k:
                        pw[v].append(ser_mul(pw[v][-1], var[v], p))
                    if k:
                        s = ser_mul(s, pw[v][k], p)
                mono.append(s)
            em = [ser_mul(es, s, p) for s in mono]
            self.den_at0.append([s[0] for s in mono])
            for k in range(T):
                rows.append([s[k] for s in mono] + [(p - x[k]) % p for x in em])
        return np.asarray(rows, dtype=np.int64)

    def _den_alive(self, dv, p):
        """Whether the denominator is nonzero at the base of some line (not in the ideal)."""
        M = np.asarray(self.den_at0, dtype=object)
        return any(int(x) % p for x in M @ np.asarray(dv, dtype=object))

    def _line(self, d, T, p, seed):
        """The series kernel along base + t d from the last base point that served at p,
        else from fresh points until every condition has a resting state modulo p."""
        hom = self.hom
        key = ("tx", T, p, seed)
        if key in self.lines:
            return self.lines[key]
        n = len(hom.base)
        out = (None, None)
        for j in range(_N_BASES + 1):
            base = self.good_base.get(p, hom.base) if j == 0 else \
                np.array(hom.an.sc["pool"](self.pool0 + (j - 1) * (n + 1) + np.arange(n)), dtype=np.int64)
            Sw = self.sys.hom_line(base, d, T, p, hom.nt)
            if Sw is None:
                continue
            ns = series_null(Sw, self.sys.nz_wide, T + 1, p)
            if any(v is None for v in ns["V"]):
                continue
            ns["states"] = list(self.sys.hom_states)
            out = (ns, base)
            self.good_base[p] = base
            break
        self.lines[key] = out
        return out

    def _null(self, R, p, order):
        """The canonical solution: the nullspace in row echelon form over the columns in
        `order` (state monomials first), its last row whose denominator does not vanish
        on the resting states; None without one."""
        rr = _core.sym_rref_mod(R.astype(np.float64), p)
        n = R.shape[1]
        piv = [int(x) for x in np.atleast_1d(rr["piv"])]
        free = [c for c in range(n) if c not in set(piv)]
        self.nullity = len(free)
        if not free:
            return None
        Rm = np.asarray(rr["R"], dtype=np.int64).reshape(len(piv), n)
        B = np.zeros((len(free), n), dtype=np.int64)
        for j, fc in enumerate(free):
            B[j, fc] = 1
            for r, c in enumerate(piv):
                B[j, c] = (p - Rm[r, fc]) % p
        Bo = B[:, order]
        rb = _core.sym_rref_mod(Bo.astype(np.float64), p)
        rk = int(rb["rank"])
        Ro = np.asarray(rb["R"], dtype=np.int64).reshape(rk, n)
        inv = np.argsort(order)
        half = n // 2
        for r in range(rk - 1, -1, -1):
            v = Ro[r][inv] % p
            if np.any(v[half:] != 0) and self._den_alive(v[half:], p):
                return v
        return None

    def fit(self, a, i, reli):
        """Entry i of direction a over (states, reli), or None."""
        for mis in self.state_sets(self.hom.fcs[a]):
            self.use(mis)
            if self.states:
                r = self._fit(a, i, reli)
                if r is not None:
                    return r
        return None

    def _fit(self, a, i, reli):
        names = self.names(reli)
        nv = len(names)
        banned = self.banned(names)
        cands = _candidate_primes()
        ns_ = len(self.states)
        for dg in range(1, self.hom.d_cap + 1):
            mons = _monomials(nv, dg, banned)
            nm = len(mons)
            if 2 * nm > _MAX_UNKNOWNS:
                break
            has_x = [any(m[:ns_]) for m in mons] * 2
            order = [c for c in range(2 * nm) if has_x[c]] + [c for c in range(2 * nm) if not has_x[c]]
            T = 2 * dg + 4
            need = 3 * nm + 12
            vecs, primes, ref = [], [], None
            for p in cands:
                R = self._rows(a, i, reli, mons, dg, p, T, need, 9973)
                v = None if R is None else self._null(R, p, order)
                self.hom.diag(f"states {','.join(self.states)}: degree {dg}, {nm} monomials "
                              f"({len(banned)} leading terms out), rows {None if R is None else R.shape[0]}, "
                              f"nullity {getattr(self, 'nullity', None) if R is not None else None}")
                if R is None:
                    continue
                if v is None:
                    break
                nz_den = [k for k in range(nm, 2 * nm) if v[k] % p]
                if not nz_den:
                    break
                if ref is None:
                    ref = nz_den[0]
                if v[ref] % p == 0:
                    break
                vecs.append(mulmod(v, invmod(int(v[ref]), p), p))
                primes.append(p)
                try:
                    rec = sym_rat_recon_big(np.stack(vecs, axis=1), primes)
                except Exception:
                    rec = None
                lifted = rec is not None and not any(dd == "0" for dd in rec["den"])
                good = lifted and self._verified(a, i, reli, rec, mons, T + 4)
                self.hom.diag(f"degree {dg}, {len(primes)} prime(s): " +
                              ("verified" if good else "lifted, not verified" if lifted else "no lift"))
                if good:
                    return {"expr": self._expr(rec, mons, self.names(reli)), "reli": reli}
                if len(primes) >= 6:
                    break
        return None

    @staticmethod
    def _expr(rec, mons, names):
        nm = len(mons)
        M = np.asarray(mons, dtype=np.int64).reshape(nm, len(names))
        num = poly_string(rec["num"][:nm], rec["den"][:nm], M, names)
        den = poly_string(rec["num"][nm:], rec["den"][nm:], M, names)
        return num if den == "1" else f"({num})/({den})"

    def _verified(self, a, i, reli, rec, mons, T):
        """The identity on fresh lines with a longer series, at a prime outside the lift."""
        p = VERIFY_PRIME
        nm = len(mons)
        R = self._rows(a, i, reli, mons, 0, p, T, 2 * nm + 8, 424243)
        if R is None:
            return False
        coef = []
        for n, dd in zip(rec["num"], rec["den"]):
            dv = int(dd) % p
            if dv == 0:
                return False
            coef.append(int(n) % p * invmod(dv, p) % p)
        c = np.asarray(coef, dtype=object)
        return all(int(x) % p == 0 for x in (np.asarray(R, dtype=object) @ c))
