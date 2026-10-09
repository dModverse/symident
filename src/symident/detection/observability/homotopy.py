"""Residual directions along lines through the base point: the resting state continued
as a power series in t along :math:`\\theta_0 + t\\,d`, kernel entries as series, closed
forms fitted to their Pade values on a star of lines."""

import re

import numpy as np

from ... import _core, state
from ...gfp import PRIMES, VERIFY_PRIME, invmod, matinv_modp, mono_table, mulmod, random_point
from ...rexpr import sym_sort
from ...sym.linalg import sym_rat_recon_big
from ...sym.modp import eval_rational_mod
from ..closedform import eval_modq, poly_string, simplify
from ..fields import ser_inv, ser_mul
from ..series import pade_at, pade_mod, series_null
from ..verify import in_kernel
from .homstate import StateFit

_EXTRA_PRIMES = (1518500131, 1518500101)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")


def _inv_mod(M, p):
    try:
        return np.asarray(matinv_modp(np.asarray(M, dtype=np.int64) % p, p), dtype=np.int64) % p
    except Exception:
        return None


def _matmul_mod(A, B, p):
    return (np.asarray(A, dtype=object) @ np.asarray(B, dtype=object) % p).astype(np.int64)


def _pade(es, dn, dmax, p):
    for dd in range(dmax + 1):
        pq = pade_mod(es, dn, dd, p)
        if pq is not None:
            return pq
    return None


class Homotopy:
    """Closed forms of residual directions of the joint kernel along parameter lines."""

    def __init__(self, rec, base, nt):
        an = rec.an
        self.an, self.sys, self.rec = an, an.sys, rec
        self.base = np.asarray(base, dtype=np.int64)
        self.nt = nt
        run = an.run
        self.pars = [nm for nm in self.sys.solve_param_names if run.slot(nm) is not None]
        self.d_cap = max(1, min(3, int(an.ctrl["degree_cap"])))
        self.lines = {}

    def diag(self, msg):
        """Print `msg` when the HOMDIAG switch is set."""
        if state.env("HOMDIAG"):
            print(f"[hom] {msg}")

    def line_null(self, d, T, p):
        """The series kernel along base + t d, truncated at order T."""
        Sw = self.sys.hom_line(self.base, d, T, p, self.nt)
        if Sw is None:
            return None
        ns = series_null(Sw, self.sys.nz_wide, T + 1, p)
        if any(v is None for v in ns["V"]):
            return None
        ns["states"] = list(self.sys.hom_states)
        return ns

    def line_at(self, key, d, T, p):
        """The series kernel along `d` at order `T` mod `p`, memoised by `key`."""
        k = (key, T, p)
        if k not in self.lines:
            self.lines[k] = self.line_null(d, T, p)
        return self.lines[k]

    def zero_line(self):
        """A line direction with zero in every parameter."""
        return dict.fromkeys(self.pars, 0)

    def basis(self, ns, p):
        """The residual directions along the line as series vectors, one per direction, each
        one on its gauge column and zero on the other gauge columns; None if singular."""
        key = ("W", self.mode, tuple(self.fcs))
        if key in ns:
            return ns[key]
        ns[key] = None
        if any(fc not in ns["free"] for fc in self.fcs):
            return None
        V = [np.asarray(ns["V"][ns["free"].index(fc)], dtype=np.int64) % p for fc in self.fcs]
        if self.mode == "free":
            W = []
            for a, v in enumerate(V):
                if int(v[self.anchor[a], 0]) % p == 0:
                    return None
                inv = ser_inv([int(x) for x in v[self.anchor[a], :]], p)
                w = np.zeros_like(v)
                for c in np.nonzero(np.any(v != 0, axis=1))[0]:
                    w[c, :] = ser_mul([int(x) for x in v[c, :]], inv, p)
                W.append(w)
            ns[key] = W
            return W
        k, T = len(V), V[0].shape[1]
        M = [np.array([[int(V[b][g, t]) for b in range(k)] for g in self.gauge], dtype=np.int64)
             for t in range(T)]
        X0 = _inv_mod(M[0], p)
        if X0 is None:
            return None
        X = [X0]
        for t in range(1, T):
            acc = np.zeros((k, k), dtype=np.int64)
            for j in range(1, t + 1):
                acc = (acc + _matmul_mod(M[j], X[t - j], p)) % p
            X.append((p - _matmul_mod(X0, acc, p)) % p)
        W = []
        for a in range(k):
            w = np.zeros_like(V[0])
            for t in range(T):
                for j in range(t + 1):
                    for b in range(k):
                        c = int(X[t - j][b, a])
                        if c:
                            w[:, t] = (w[:, t] + mulmod(V[b][:, j], c, p)) % p
            W.append(w)
        ns[key] = W
        return W

    def entry(self, ns, a, col, p):
        """The series of column `col` of residual direction a along the line."""
        W = self.basis(ns, p)
        return None if W is None else [int(x) for x in W[a][col, :]]

    def value(self, nm, p):
        """The base point value of parameter `nm` mod `p`."""
        return int(self.base[self.an.run.slot(nm)]) % p

    # ---- relevance ----------------------------------------------------------------------

    def _derivative(self, grp, seed):
        """For each entry, whether the first-order series along random combinations of
        the parameters in `grp` moves it."""
        P1 = PRIMES[0]
        d = self.zero_line()
        for nm, x in zip(grp, random_point(len(grp), seed)):
            d[nm] = int(x) % P1
        ns = self.line_null(d, 1, P1)
        if ns is None:
            return None
        out = []
        for a in range(len(self.fcs)):
            hits = []
            for c in self.cols[a]:
                es = self.entry(ns, a, c, P1)
                hits.append(es is None or es[1] != 0)
            out.append(hits)
        return out

    def _relevance(self):
        """The parameters each entry depends on, by groups of eight, then one by one."""
        self.rel = [[[] for _ in c] for c in self.cols]
        groups = [self.pars[i:i + 8] for i in range(0, len(self.pars), 8)]
        for gi, g in enumerate(groups, start=1):
            hit = self._derivative(g, 7919 * gi + 1)
            if hit is None:
                return False
            if not any(any(h) for h in hit):
                continue
            for nm in g:
                h = self._derivative([nm], 104729 + self.pars.index(nm) + 1)
                if h is None:
                    return False
                for a in range(len(self.fcs)):
                    for i, x in enumerate(h[a]):
                        if x:
                            self.rel[a][i].append(nm)
        return True

    # ---- fit --------------------------------------------------------------------------

    def _samples(self, a, i, reli, dg, p):
        """Samples of entry i of direction a at prime p from the Pade forms on a star
        of lines through the base."""
        k = len(reli)
        n_mon = mono_table(k, dg).shape[0]
        need, per, T = 2 * n_mon + 4, 2 * dg + 1, 2 * dg + 2
        U, R = [], []
        for line in range(1, -(-need // per) + 1):
            d = self.zero_line()
            for nm, x in zip(reli, random_point(k, 31 * line + 7 * dg + 1)):
                d[nm] = int(x) % p
            ns = self.line_at((tuple(reli), 31 * line + 7 * dg + 1), d, T, p)
            if ns is None:
                self.diag(f"no line at prime {p}")
                return None
            es = self.entry(ns, a, self.cols[a][i], p)
            if es is None:
                self.diag(f"no gauged basis at prime {p}")
                return None
            pq = _pade(es, dg, dg, p)
            if pq is None:
                if p != PRIMES[0]:
                    self.diag(f"no Pade at prime {p}")
                return None
            for t in range(1, per + 1):
                val = pade_at(pq, t, p)
                if val is None:
                    continue
                U.append([(self.value(nm, p) + t * (d[nm] % p)) % p for nm in reli])
                R.append(val)
        return np.asarray(U, dtype=np.int64).reshape(len(U), k), np.asarray(R, dtype=np.int64)

    def _constant(self, a, i):
        v = self.entry(self.ns0, a, self.cols[a][i], PRIMES[0])
        try:
            rc = sym_rat_recon_big(np.array([[int(v[0])]], dtype=np.int64), [PRIMES[0]])
        except Exception:
            return None
        if rc is None or rc["den"][0] == "0":
            return None
        expr = rc["num"][0] if rc["den"][0] == "1" else f"{rc['num'][0]}/{rc['den'][0]}"
        return {"expr": expr, "reli": []}

    def _lift(self, samples, primes, mons):
        """Coefficients of the rational fit lifted from several primes; None if not."""
        n_mon = mons.shape[0]
        coef = np.zeros((2 * n_mon, len(primes)), dtype=np.int64)
        ref_free = None
        for jp, (sm, pr) in enumerate(zip(samples, primes)):
            fit = _core.sym_fit_rational(sm[0], mons, sm[1], pr)
            if fit.get("status") != "ok":
                if len(primes) > 1:
                    self.diag(f"fit {fit.get('status')} at prime {pr}")
                return False
            raw = np.asarray(fit["coeffs"], dtype=np.int64)
            if ref_free is None:
                ref_free = int(fit["free_col"])
            dn = int(raw[ref_free]) % pr
            if dn == 0:
                return False
            coef[:, jp] = mulmod(raw, invmod(dn, pr), pr)
        try:
            return sym_rat_recon_big(coef, list(primes))
        except Exception:
            return None

    def fit_entry(self, a, i):
        """Entry i of direction a as a rational function of its relevant parameters."""
        reli = self.rel[a][i]
        if not reli:
            return self._constant(a, i)
        cands = list(dict.fromkeys(list(PRIMES) + list(_EXTRA_PRIMES)))
        for dg in range(self.d_cap + 1):
            mons = mono_table(len(reli), dg)
            n_mon = mons.shape[0]
            for np_ in (1, 2, 3):
                primes, samples = [], []
                for pc in cands:
                    if len(primes) == np_:
                        break
                    sm = self._samples(a, i, reli, dg, pc)
                    if sm is not None and len(sm[1]) >= 2 * n_mon:
                        primes.append(pc)
                        samples.append(sm)
                    elif pc == cands[0]:
                        break
                if len(primes) < np_:
                    break
                rec = self._lift(samples, primes, mons)
                if rec is False:
                    break
                if rec is None or any(d == "0" for d in rec["den"]):
                    self.diag(f"entry {self.cols[a][i]} degree {dg}: no lift at {np_} prime(s)")
                    continue
                num = poly_string(rec["num"][:n_mon], rec["den"][:n_mon], mons, reli)
                den = poly_string(rec["num"][n_mon:], rec["den"][n_mon:], mons, reli)
                return {"expr": num if den == "1" else f"({num})/({den})", "reli": reli}
        return None

    # ---- directions -------------------------------------------------------------------

    def _anchors(self, fcs):
        """The gauge: for the k residual directions the first k shared columns on which
        their base vectors are independent; each direction one on its own gauge column."""
        n_sh = self.sys.n_shared
        free = self.ns0["free"]
        self.fcs = [fc for fc in fcs if fc in free]
        if not self.fcs:
            return
        P1 = PRIMES[0]
        B = np.array([[int(self.ns0["V"][free.index(fc)][c, 0]) % P1 for c in range(n_sh)] for fc in self.fcs],
                     dtype=np.int64)
        rr = _core.sym_rref_mod(B.astype(np.float64), P1)
        piv = [int(x) for x in np.atleast_1d(rr["piv"])]
        if len(piv) < len(self.fcs):
            self.diag("the residual directions are dependent on the shared columns")
            self.fcs = []
            return
        self.gauge = piv[:len(self.fcs)]
        if self.mode == "free":
            self.anchor = []
            for fc in self.fcs:
                v = self.ns0["V"][free.index(fc)]
                moved = [c for c in range(n_sh) if int(v[c, 0]) % P1]
                self.anchor.append(fc if fc < n_sh else moved[0])
        W = self.basis(self.ns0, P1)
        if W is None:
            self.fcs = []
            return
        if self.mode != "free":
            self.anchor = list(self.gauge)
        skip = [set(self.gauge) if self.mode != "free" else {self.anchor[a]} for a in range(len(self.fcs))]
        self.cols = [[c for c in range(n_sh) if c not in skip[a] and int(W[a][c, 0]) % P1]
                     for a in range(len(self.fcs))]

    def _verified(self, a, i, e):
        """The fitted entry against the Pade value at t = 1 on a fresh line at a prime
        outside the lift."""
        nsv, dv, P2 = self.nsv, self.dv, VERIFY_PRIME
        if nsv is None:
            return False
        es = self.entry(nsv, a, self.cols[a][i], P2)
        if es is None:
            return False
        pq = _pade(es, self.d_cap, self.d_cap, P2)
        if pq is None:
            return False
        at1 = [(self.value(nm, P2) + dv[nm]) % P2 for nm in e["reli"]]
        try:
            pred = eval_rational_mod(e["expr"], list(e["reli"]), at1, P2)
        except Exception:
            pred = None
        if pred is None or (int(pred) - pade_at(pq, 1, P2)) % P2 != 0:
            self.diag(f"entry {self.cols[a][i]} failed verification: {e['expr']}")
            return False
        return True

    def direction(self, a):
        """Direction `a` in closed form from its fitted entries, or None when an entry is not
        fitted or the direction fails the kernel check at a fresh point."""
        names = self.an.znames
        entries = {names[self.anchor[a]]: "1"}
        used = set()
        for i, col in enumerate(self.cols[a]):
            e = self.fit_entry(a, i)
            if e is not None and not self._verified(a, i, e):
                return None
            if e is None and not state.env("NOHOMSTATES"):
                e = self.state_fit.fit(a, i, self.rel[a][i])
                if e is not None:
                    e["expr"] = self.state_fit.substitute(e["expr"])
                    if set(_IDENT.findall(e["expr"])) & set(self.state_fit.states):
                        used.add(tuple(self.state_fit.mis))
            if e is None:
                self.diag(f"fc {self.fcs[a]}: entry {col} not fitted ({','.join(self.rel[a][i])})")
                return None
            entries[names[col]] = simplify(e["expr"])
        single = sorted(m[0] for m in used if len(m) == 1)
        if len(single) > 1 or not self._in_kernel(entries, single[0] if single else None):
            self.diag(f"fc {self.fcs[a]}: not in the kernel at a fresh point")
            return None
        route = "homotopy fit"
        if used:
            route += " in the parameters and the resting states" + \
                (f" of condition {single[0] + 1}" if single and len(used) == 1 else "")
        return {"support": sym_sort(list(entries)), "vector": entries, "type": "general",
                "closed_form": True, "route": route}

    def _in_kernel(self, entries, mi_plain, tries=4):
        """Whether the shared part of the kernel at a fresh point modulo the verification prime
        holds `entries`, a state read as its resting value (unsuffixed in condition `mi_plain`)."""
        an, sys, q = self.an, self.sys, VERIFY_PRIME
        n = len(self.base)
        idx = {nm: c for c, nm in enumerate(an.znames)}
        start = self.rec.pool_next
        self.rec.pool_next += tries * (n + 1)
        for k in range(tries):
            pt = np.asarray(an.sc["pool"](start + k * (n + 1) + np.arange(n)), dtype=np.int64)
            rq = an.kcall(pt, q, self.nt)
            if not rq.get("ok"):
                continue
            env = {nm: int(pt[i]) for i, nm in enumerate(an.leaf_names_aug) if i < n}
            for mi, ci in enumerate(sys.equil_conds):
                sol = sys.solve_cond(pt, q, ci)
                if sol is None:
                    break
                for nm, v in sol["sol"]["val_by"].items():
                    env[f"{nm}_c{mi + 1}"] = int(v) % q
                    if mi == mi_plain:
                        env[nm] = int(v) % q
            else:
                v = np.zeros(an.nz, dtype=np.int64)
                for nm, ex in entries.items():
                    got = eval_modq(ex, env, q)
                    if got is None:
                        return False
                    v[idx[nm]] = got % q
                return in_kernel(rq, v, list(range(sys.n_shared)), an.nz, q)
        return False

    def run(self, fcs):
        """(free columns, directions) with None where a direction does not close; None
        without a series kernel at the base. Each direction is first normalised on one
        shared column, the rest then jointly on a gauge of shared columns."""
        self.ns0 = self.line_null(self.zero_line(), 1, PRIMES[0])
        if self.ns0 is None:
            self.diag("no series kernel at the base")
            return None
        self.diag(f"residual {','.join(map(str, fcs))}; base free {','.join(map(str, self.ns0['free']))}; "
                  f"shared {self.sys.n_shared}")
        P2 = VERIFY_PRIME
        self.dv = {nm: int(x) % P2 for nm, x in zip(self.pars, random_point(len(self.pars), 424242))}
        self.nsv = self.line_null(self.dv, 2 * self.d_cap + 2, P2)
        self.state_fit = StateFit(self)
        done = {}
        todo = list(fcs)
        for mode in ("free", "gauge"):
            if not todo or (mode == "gauge" and state.env("NOHOMGAUGE")):
                break
            self.mode = mode
            self._anchors(todo)
            if not self.fcs or not self._relevance():
                continue
            for a, fc in enumerate(self.fcs):
                d = self.direction(a)
                if d is not None:
                    done[fc] = d
            todo = [fc for fc in todo if fc not in done]
        fcs_all = [fc for fc in fcs if fc in set(self.ns0["free"])]
        return fcs_all, [done.get(fc) for fc in fcs_all]
