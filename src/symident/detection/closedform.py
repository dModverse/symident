"""Closed-form entries: monomial strings, rational fits from residues, back-substitution."""

import re
import time

import numpy as np
import sympy as spy

from .. import _core
from ..gfp import PRIMES, rat_recon
from ..sym.charts import recast_backsub_expr
from ..sym.modp import eval_rational_mod

RP8 = list(dict.fromkeys(list(PRIMES) + [1518500141, 1518500131, 1518500101, 1518500077]))


def expired(ctrl):
    """Whether the deadline in `ctrl` has passed."""
    d = ctrl.get("deadline")
    return d is not None and time.monotonic() > d


def mono_string(expo, vars_):
    """The monomial with exponents `expo` in `vars_` as a string, "1" for the constant."""
    terms = []
    for k, v in enumerate(vars_):
        e = int(expo[k])
        if e != 0:
            terms.append(v if e == 1 else f"{v}^{e}")
    return "*".join(terms) if terms else "1"


def poly_string(numc, denc, mons, vars_):
    """The polynomial with coefficients `numc`/`denc` on the monomials `mons` as a string."""
    parts = []
    for j in range(mons.shape[0]):
        n, d = numc[j], denc[j]
        if n == "0":
            continue
        coef = n if d == "1" else f"({n}/{d})"
        ms = mono_string(mons[j], vars_)
        parts.append(coef if ms == "1" else f"{coef}*{ms}")
    return " + ".join(parts) if parts else "0"


def reconstruct_entry(sample_u, mons, residues, primes):
    """One nullspace entry as a rational function; None without a bounded-degree fit."""
    n_mon = mons.shape[0]
    coef_res = np.zeros((2 * n_mon, len(primes)), dtype=np.int64)
    free_col = None
    for j, p in enumerate(primes):
        fit = _core.sym_fit_rational(sample_u, mons, np.asarray(residues[:, j], dtype=np.int64), p)
        if fit.get("status") != "ok":
            return None
        if free_col is None:
            free_col = fit["free_col"]
        elif fit["free_col"] != free_col:
            return None
        coef_res[:, j] = fit["coeffs"]
    rec = rat_recon(coef_res, list(primes))
    if any(d == "0" for d in rec["den"]):
        return None
    return {"num_coef_n": rec["num"][:n_mon], "num_coef_d": rec["den"][:n_mon],
            "den_coef_n": rec["num"][n_mon:], "den_coef_d": rec["den"][n_mon:]}


def simplify(expr):
    """`expr` cancelled by SymPy as a string; `expr` itself when that fails."""
    try:
        out = str(spy.cancel(spy.sympify(expr)))
    except Exception:
        return expr
    return out


def recast_backsub(vector, recast):
    """The entries of `vector` with the recast coordinates substituted back; an entry unchanged on failure."""
    e_n = [str(r["E"]) for r in recast]
    l_n = [str(r["L"]) for r in recast]
    b_n = [str(r["base"]) for r in recast]
    x_n = [str(r["exp"]) for r in recast]
    out = {}
    for k, x in vector.items():
        try:
            v = recast_backsub_expr(str(x), e_n, l_n, b_n, x_n)
            out[k] = v if isinstance(v, str) else str(x)
        except Exception:
            out[k] = str(x)
    return out


def eval_modq(expr, env, q):
    """Rational expression at the integer point env modulo q; None if its denominator vanishes."""
    try:
        v = eval_rational_mod(str(expr), list(env.keys()), [int(x) for x in env.values()], int(q))
    except Exception:
        return None
    return None if v is None else int(v)


def _strip_name(nm):
    return re.sub(r"\|c[0-9]+$", "", nm)
