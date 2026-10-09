"""Monomial, polynomial and rational invariants from modular nullspaces."""

import math
import re

import numpy as np
import sympy as spy

from .. import _core
from ..gfp import PRIMES, Pool, invmod, mulmod, rat_recon, rref_modp
from ..rexpr import sym_sort
from .expr import eval_batch, red_locals, red_sympify
from .scaling import mono_string, primitive_rows


def qr_rank(M, tol=1e-7):
    """Numerical rank by pivoted QR with R's convention |R_ii| > tol |R_11|."""
    import scipy.linalg
    M = np.asarray(M, dtype=float)
    if M.size == 0:
        return 0
    R = scipy.linalg.qr(M, mode="r", pivoting=True)[0]
    d = np.abs(np.diag(R))
    if not d.size or d[0] == 0:
        return 0
    return int(np.sum(d > tol * d[0]))


def eta_rows(preps, vars_, pt, p):
    """Log-derivative rows eta_i = xi_i(z)/z_i of a block at one point mod p."""
    idx = [[i for i, v in enumerate(vars_) if v in pr["comps"]] for pr in preps]
    exprs = [preps[g]["comps"][vars_[i]] for g in range(len(preps)) for i in idx[g]]
    vals = eval_batch(exprs, pt, p)
    if any(v is None for v in vals):
        return None
    rows = np.zeros((len(preps), len(vars_)), dtype=np.int64)
    off = 0
    for g in range(len(preps)):
        for k, i in enumerate(idx[g]):
            rows[g, i] = vals[off + k] * invmod(int(pt[vars_[i]]) % p, p) % p
        off += len(idx[g])
    return rows


def gcd(a, b):
    """Greatest common divisor of the absolute values of `a` and `b`."""
    return math.gcd(int(abs(a)), int(abs(b)))


def modular_nullspace(rows_at, n, eval_vars, primes=PRIMES):
    """Common nullspace of sampled rows over the primes as primitive integer columns."""
    basis_per, pivots_per, n_rows = {}, {}, 0
    nv = len(eval_vars)
    for pi, p in enumerate(primes, start=1):
        pool = Pool()
        rows = np.zeros((0, n), dtype=np.int64)
        prev, flat = -1, 0
        off, draws = 50 * pi, 0
        chunk = max(1, min(16, n // 8))
        red = np.zeros((0, n), dtype=np.int64)
        pend = np.zeros((0, n), dtype=np.int64)
        while True:
            pt = dict(zip(eval_vars, [int(x) for x in pool(np.arange(1, nv + 1) + off)]))
            off += nv
            draws += 1
            new = rows_at(pt, p)
            if new is None:
                if draws > 25:
                    break
                continue
            new = np.asarray(new, dtype=np.int64).reshape(-1, n)
            rows = np.vstack([rows, new])
            pend = np.vstack([pend, new])
            if pend.shape[0] < chunk and rows.shape[0] <= 4 * n + 20:
                continue
            rr = rref_modp(np.vstack([red, pend]) % p, p)
            red = rr["R"][:rr["rank"]]
            pend = np.zeros((0, n), dtype=np.int64)
            r = rr["rank"]
            if r == n:
                break
            flat = flat + 1 if r == prev else 0
            prev = r
            if flat >= 2 or rows.shape[0] > 4 * n + 20:
                break
        n_rows = max(n_rows, rows.shape[0])
        if rows.shape[0] == 0:
            return {"basis": None, "rows": 0}
        ref = rref_modp(rows % p, p)
        piv = [int(x) for x in ref["piv"]]
        free = [c for c in range(n) if c not in set(piv)]
        if not free:
            return {"basis": np.zeros((n, 0), dtype=np.int64), "rows": n_rows}
        pivots_per[pi] = piv
        B = np.zeros((n, len(free)), dtype=np.int64)
        for j, fc in enumerate(free):
            v = np.zeros(n, dtype=np.int64)
            v[fc] = 1
            for ri, pc in enumerate(piv):
                v[pc] = (p - int(ref["R"][ri, fc])) % p
            B[:, j] = v
        basis_per[pi] = B
    first = pivots_per.get(1)
    use = [pi for pi in sorted(pivots_per) if pivots_per[pi] == first]
    if len(use) < 2:
        return {"basis": None, "rows": n_rows}
    nf = basis_per[use[0]].shape[1]
    residues = np.stack([basis_per[pi].flatten(order="F") for pi in use], axis=1)
    rec = rat_recon(residues, [primes[pi - 1] for pi in use])
    if any(d == "0" for d in rec["den"]):
        return {"basis": None, "rows": n_rows}
    num = [int(x) for x in rec["num"]]
    den = [int(x) for x in rec["den"]]
    if any(abs(x) > 2 ** 40 for x in num) or any(x > 2 ** 40 for x in den):
        return {"basis": None, "rows": n_rows}
    B = np.zeros((n, nf), dtype=object)
    for j in range(nf):
        dj = den[j * n:(j + 1) * n]
        L = 1
        for d in set(dj):
            L = L * d // math.gcd(L, d)
        for i in range(n):
            B[i, j] = num[j * n + i] * (L // dj[i])
    B = B.astype(np.int64)
    return {"basis": primitive_rows(B.T).T, "rows": n_rows}


def monomial_invariants(preps):
    """Laurent-monomial invariants: the integer kernel of the log-derivative rows."""
    vars_ = sym_sort(list(dict.fromkeys(v for pr in preps for v in pr["support"])))
    eval_vars = sym_sort(list(dict.fromkeys(vars_ + [v for pr in preps for v in pr["vars"]])))
    n = len(vars_)
    ns = modular_nullspace(lambda pt, p: eta_rows(preps, vars_, pt, p), n, eval_vars)
    cert = f"monomial stage: exact modular nullspace over {len(PRIMES)} primes, {ns['rows']} sampling rows"
    if ns["basis"] is None:
        return {"ok": False, "invariants": [], "exps": None,
                "cert": cert + "; reconstruction failed (inconclusive)"}
    if ns["basis"].shape[1] == 0:
        return {"ok": False, "invariants": [], "exps": None,
                "cert": f"no Laurent-monomial invariant ({cert})"}
    inv = [mono_string(ns["basis"][:, j], vars_) for j in range(ns["basis"].shape[1])]
    return {"ok": True, "invariants": inv, "exps": ns["basis"], "vars": vars_, "cert": cert}


def poly_string(cf, expts, vars_):
    """The integer polynomial with coefficients `cf` on the monomials `expts` as a string."""
    terms = []
    for k in np.nonzero(np.asarray(cf))[0]:
        mono = mono_string(expts[k], vars_)
        a = int(cf[k])
        if mono == "1":
            terms.append(str(a))
        elif a == 1:
            terms.append(mono)
        elif a == -1:
            terms.append(f"-{mono}")
        else:
            terms.append(f"{a}*{mono}")
    out = terms[0]
    for t in terms[1:]:
        out += (" - " if t.startswith("-") else " + ") + re.sub(r"^-", "", t)
    return out


def _rows_mono(preps, vars_, expts):
    def rows_at(pt, p):
        eta = eta_rows(preps, vars_, pt, p)
        if eta is None:
            return None
        mv = np.asarray(_core.sym_mono_residues(expts, [int(pt[v]) % p for v in vars_], p), dtype=np.int64)
        inner = (expts @ eta.T) % p
        return np.stack([mulmod(mv, inner[:, g], p) for g in range(eta.shape[0])])
    return rows_at


def independent_poly(B, expts, vars_, moved, preps):
    """A functionally independent subset by a greedy Jacobian rank test."""
    from ..rexpr import eval_numeric
    pool = Pool()
    ptv = [float(x) for x in pool(np.arange(1, len(vars_) + 1) + 5)]
    pt = dict(zip(vars_, ptv))
    Xi = np.array([[eval_numeric(pr["comps"][v], pt) if v in pr["comps"] else 0.0 for v in moved]
                   for pr in preps], dtype=float)
    target = len(moved) - qr_rank(Xi, tol=1e-9)
    mv = np.exp(expts @ np.log(np.array(ptv)))
    m_idx = [vars_.index(m) for m in moved]

    def grad(cf):
        return np.array([np.sum(cf * mv * expts[:, i] / ptv[i]) for i in m_idx])
    deg = expts.sum(axis=1)
    key = [float(np.max(deg[B[:, j] != 0])) * 1e4 + float(np.sum(B[:, j] != 0)) for j in range(B.shape[1])]
    J = np.zeros((0, len(moved)))
    sel = []
    for j in sorted(range(B.shape[1]), key=lambda j: key[j]):
        if len(sel) >= target and target > 0:
            break
        Jc = np.vstack([J, grad(B[:, j].astype(float))])
        if qr_rank(Jc, tol=1e-9) > J.shape[0]:
            J = Jc
            sel.append(j)
    if not sel:
        sel = list(range(min(1, B.shape[1])))
    return sel


def poly_invariants(preps, d_poly):
    """Polynomial invariants of total degree <= d_poly."""
    from ..gfp import mono_table
    moved = sym_sort(list(dict.fromkeys(v for pr in preps for v in pr["support"])))
    vars_ = sym_sort(list(dict.fromkeys(moved + [v for pr in preps for v in pr["vars"]])))
    expts = mono_table(len(vars_), int(d_poly))
    mi = [vars_.index(m) for m in moved]
    expts = expts[expts[:, mi].sum(axis=1) > 0]
    N = expts.shape[0]
    if not N:
        return {"ok": False, "invariants": [], "cert": "polynomial stage skipped: empty ansatz"}
    ns = modular_nullspace(_rows_mono(preps, vars_, expts), N, vars_)
    cert = (f"polynomial stage (degree <= {d_poly}, {N} monomials): exact modular nullspace over "
            f"{len(PRIMES)} primes, {ns['rows']} sampling rows")
    if ns["basis"] is None:
        return {"ok": False, "invariants": [], "cert": cert + "; reconstruction failed (inconclusive)"}
    if ns["basis"].shape[1] == 0:
        return {"ok": False, "invariants": [],
                "cert": f"no polynomial invariant of total degree <= {d_poly} beyond functions of "
                        f"the unmoved coordinates ({cert})"}
    sel = independent_poly(ns["basis"], expts, vars_, moved, preps)
    inv = [poly_string(ns["basis"][:, j], expts, vars_) for j in sel]
    return {"ok": True, "invariants": inv, "coefs": ns["basis"][:, sel], "expts": expts,
            "vars": vars_,
            "cert": cert + f"; {len(sel)} functionally independent of {ns['basis'].shape[1]}"}


RATIONAL_CAP = 700


def rational_invariants(preps, d_poly):
    """Rational invariants with a single-coordinate denominator (Laurent ansatz)."""
    from ..gfp import mono_table
    moved = sym_sort(list(dict.fromkeys(v for pr in preps for v in pr["support"])))
    vars_ = sym_sort(list(dict.fromkeys(moved + [v for pr in preps for v in pr["vars"]])))

    def skip(cert):
        return {"ok": False, "invariants": [], "cert": cert}
    if d_poly < 1:
        return skip("rational stage skipped (d_poly = 0)")
    pool = Pool()
    divisible = []
    for v in moved:
        xs = [pr["comps"][v] for pr in preps if v in pr["comps"]]
        if not xs:
            divisible.append(True)
            continue
        ok = True
        for t in (1, 2):
            pt = dict(zip(vars_, [int(x) for x in pool(np.arange(1, len(vars_) + 1) + 31 * t)]))
            pt[v] = 0
            val = eval_batch(xs, pt, PRIMES[0])
            if any(x is None for x in val) or not all(x == 0 for x in val):
                ok = False
                break
        divisible.append(ok)
    den_c = [m for m, d in zip(moved, divisible) if d]
    if not den_c:
        return skip("rational stage skipped: no moved coordinate divides its own generator "
                    "component (no admissible denominator)")
    moved_idx = [vars_.index(m) for m in moved]
    T0 = mono_table(len(vars_), int(d_poly))
    keep0 = T0[:, moved_idx].sum(axis=1) > 0
    laur = []
    for v in den_c:
        i = vars_.index(v)
        Ti = T0[T0[:, i] == 0].copy()
        Ti[:, i] = -1
        laur.append(Ti)
    expts = np.vstack([T0[keep0]] + laur)
    N = expts.shape[0]
    if N > RATIONAL_CAP:
        return skip(f"rational stage skipped ({N} Laurent monomials over the {RATIONAL_CAP}-unknown budget)")
    ns = modular_nullspace(_rows_mono(preps, vars_, expts), N, vars_)
    cert = (f"rational stage (numerator degree <= {d_poly}, denominator one of {{{', '.join(den_c)}}}; "
            f"{N} Laurent monomials): exact modular nullspace over {len(PRIMES)} primes, "
            f"{ns['rows']} sampling rows")
    if ns["basis"] is None:
        return skip(cert + "; reconstruction failed (inconclusive)")
    if ns["basis"].shape[1] == 0:
        return skip(f"no rational invariant with a single-coordinate denominator and numerator "
                    f"degree <= {d_poly} ({cert})")
    sel = independent_poly(ns["basis"], expts, vars_, moved, preps)
    inv = [poly_string(ns["basis"][:, j], expts, vars_) for j in sel]
    loc = red_locals(inv)
    out = []
    for s in inv:
        try:
            e = spy.cancel(spy.together(red_sympify(s.replace("^", "**"), loc)))
            out.append(str(e).replace("**", "^"))
        except Exception:
            out.append(s)
    return {"ok": True, "invariants": out,
            "cert": cert + f"; {len(sel)} functionally independent of {ns['basis'].shape[1]}"}
