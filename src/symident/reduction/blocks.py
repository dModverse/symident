"""Curved blocks: invariant search per block, fixed parameters, block verification."""

import re
from contextlib import contextmanager

import mpmath
import numpy as np
import sympy as spy

from ..errors import SYMPY_FAILS, RouteError, swallow, sympy_fail
from ..gfp import Pool
from ..rexpr import sym_sort
from ..sym.invariants import verify_invariants
from .darboux import _moved_vars, darboux, exp_factors, int_factor, separable
from .expr import gen_prep, group_by, module_reduce, red_locals, red_sympify
from .invariants import (
    _direction_rank,
    exact_rank,
    monomial_invariants,
    poly_invariants,
    rational_invariants,
)
from .linear import linear_flow
from .scaling import coords_of, scaling_block
from .sections import translation_compress, unmoved_compress
from .sign import sgn


def block_corank(preps):
    """(corank, generic): moved coordinates minus the direction rank at a point, and whether that rank is generic."""
    moved, vars_ = _moved_vars(preps)
    ipt = [int(x) for x in Pool()(np.arange(1, len(vars_) + 1) + 5)]
    r = _direction_rank(preps, moved, vars_, ipt)
    return len(moved) - r, r >= min(len(preps), len(moved))


_IV_PREC = 200
# the functions mpmath has in interval arithmetic; an invariant with another one is evaluated exactly
_IV_MODULE = {
    "exp": mpmath.iv.exp,
    "log": mpmath.iv.log,
    "sqrt": mpmath.iv.sqrt,
    "sin": mpmath.iv.sin,
    "cos": mpmath.iv.cos,
    "tan": mpmath.iv.tan,
    "pi": mpmath.iv.pi,
    "E": mpmath.iv.e,
}


@contextmanager
def _ivprec():
    """The precision of mpmath's interval context set to _IV_PREC bits for the block."""
    old = mpmath.iv.prec
    mpmath.iv.prec = _IV_PREC
    try:
        yield
    finally:
        mpmath.iv.prec = old


_IV_FUNCS = {"exp", "log", "sin", "cos", "tan"}


def _iv_value(d, syms, point):
    """`d` at the interval `point` of `syms` in interval arithmetic, or None where a function or constant has no
    interval version; a non-integer number enters as an interval that encloses it, never as a float."""
    if {type(f).__name__ for f in d.atoms(spy.Function)} - _IV_FUNCS:
        return None
    consts = {spy.pi: mpmath.iv.pi, spy.E: mpmath.iv.e}
    if d.atoms(spy.NumberSymbol) - set(consts):
        return None
    nums = sorted((a for a in d.atoms(spy.Number) if not a.is_Integer), key=str)
    vals = []
    for a in nums:
        if not (a.is_Rational or a.is_Float):
            return None
        q = spy.Rational(a)
        vals.append(mpmath.iv.mpf(int(q.p)) / int(q.q))
    cs = sorted(d.atoms(spy.NumberSymbol), key=str)
    dums = [spy.Dummy() for _ in nums + cs]
    d2 = d.xreplace(dict(zip(nums + cs, dums)))
    f = spy.lambdify(list(syms) + dums, d2, modules=[_IV_MODULE])
    return mpmath.iv.mpf(f(*(list(point) + vals + [consts[c] for c in cs])))


def _iv_det(rows):
    """The determinant of a square interval matrix by elimination with pivots away from zero;
    None if no pivot interval excludes zero."""
    a = [list(r) for r in rows]
    n, det = len(a), mpmath.iv.mpf(1)
    for k in range(n):
        piv = next((i for i in range(k, n) if not (a[i][k].a <= 0 <= a[i][k].b)), None)
        if piv is None:
            return None
        if piv != k:
            a[k], a[piv] = a[piv], a[k]
            det = -det
        det *= a[k][k]
        for i in range(k + 1, n):
            f = a[i][k] / a[k][k]
            a[i] = [a[i][j] - f * a[k][j] for j in range(n)]
    return det


def independent_set(invs, vars_, target, moved=None):
    """A functionally independent subset of invariants, in order, by a nonzero Jacobian minor certified at a point."""
    if moved is None:
        moved = vars_
    if not invs or target <= 0:
        return []
    ipool = [int(x) for x in Pool()(np.arange(1, len(vars_) + 1) + 13)]
    pool = np.array([float(x) for x in ipool])
    if any("exp(" in s for s in invs):
        pool = 1 + pool / (pool.max() + 1)
    ipt = {spy.Symbol(k): spy.Integer(v) for k, v in zip(vars_, ipool)}
    loc = {v: spy.Symbol(v) for v in vars_}

    def grad_exact(iv):
        # the gradient as exact rationals at the integer point; None where one is not rational
        try:
            e = spy.sympify(iv.replace("^", "**"), locals=dict(loc, **red_locals(iv)))
            out = [spy.diff(e, spy.Symbol(v)).xreplace(ipt) for v in moved]
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            return None
        return out if all(x.is_Rational for x in out) else None

    rat = [spy.Rational(v).limit_denominator(10**6) for v in pool]
    rpt = {spy.Symbol(k): r for k, r in zip(vars_, rat)}
    with _ivprec():
        ivpt = [mpmath.iv.mpf(int(r.p)) / int(r.q) for r in rat]
    syms = [spy.Symbol(v) for v in vars_]

    def grad(iv):
        # the gradient at a rational point, with a float copy to order the minors: in interval
        # arithmetic where every function has an interval version, else from an exact value
        # widened to an interval
        try:
            e = spy.sympify(iv.replace("^", "**"), locals=dict(loc, **red_locals(iv)))
            ds = [spy.diff(e, spy.Symbol(v)) for v in moved]
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            return None
        try:
            with _ivprec():
                ex = [_iv_value(d, syms, ivpt) for d in ds]
            if all(x is not None and mpmath.isfinite(x.a) and mpmath.isfinite(x.b) for x in ex):
                return ex, np.array([float(x.mid) for x in ex])
        except (NameError, TypeError, ValueError, ZeroDivisionError, OverflowError):
            pass
        try:
            ex = [d.xreplace(rpt) for d in ds]
            with _ivprec():
                ivx = [mpmath.iv.mpf(str(spy.N(x, 80, strict=True))) for x in ex]
            # a point value from 80 digits is widened to an interval of that accuracy
            ivx = [x + mpmath.iv.mpf([-1, 1]) * (abs(x).b * mpmath.mpf(10) ** -70 + mpmath.mpf(10) ** -70) for x in ivx]
            return ivx, np.array([float(x.mid) for x in ivx])
        except spy.core.evalf.PrecisionExhausted:
            return None
        except SYMPY_FAILS + (ZeroDivisionError, OverflowError) as exc:
            sympy_fail(exc)
            return None

    def certified(rows, fl):
        # a nonzero maximal minor: columns chosen by pivoted QR, the interval of the minor free of zero
        from scipy.linalg import qr

        cols = sorted(qr(fl, pivoting=True)[2][: len(rows)])
        with _ivprec():
            det = _iv_det([[r[c] for c in cols] for r in rows])
        return det is not None and not (det.a <= 0 <= det.b)

    cand = list(dict.fromkeys(invs))

    def select(exact):
        # exact ranks while every gradient is rational, else the float rank throughout
        J, Jx, Jn, sel = np.zeros((0, len(moved))), [], [], []
        for iv in cand:
            if len(sel) >= target:
                break
            if exact:
                gx = grad_exact(iv)
                r = None if gx is None else exact_rank(Jx + [gx])
                if r is None:
                    return None
                if r > len(Jx):
                    Jx.append(gx)
                    sel.append(iv)
                continue
            gr = grad(iv)
            if gr is None or not np.all(np.isfinite(gr[1])):
                continue
            Jc = np.vstack([J, gr[1]])
            if certified(Jn + [gr[0]], Jc):
                J = Jc
                Jn.append(gr[0])
                sel.append(iv)
        return sel

    out = select(True)
    return out if out is not None else select(False)


def solve_block(preps, d_poly, d_darboux, d_exp, sep=True):
    """The invariant stages on one module-reduced sub-block."""
    moved, vars_ = _moved_vars(preps)
    target, count_exact = block_corank(preps)
    certs, found, stage_ = [], [], "monomial"
    certs.append(
        f"invariant count {target}: "
        + (
            "exact, the directions are independent at a point"
            if count_exact
            else "an upper bound, from a rank at one point"
        )
    )
    moved_re = re.compile(r"(?<![0-9A-Za-z_.])(" + "|".join(re.escape(m) for m in moved) + r")(?![0-9A-Za-z_.])")
    sgn_cache = {}

    def definite(iv):
        if iv in sgn_cache:
            return sgn_cache[iv]
        try:
            e = red_sympify(iv.replace("^", "**"), red_locals(iv))
            v = sgn(e) != 0
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            v = False
        sgn_cache[iv] = v
        return v

    def pick(cand):
        cand = list(dict.fromkeys(c for c in cand if moved_re.search(c)))
        is_exp = ["exp(" in c for c in cand]
        tier = [3 if ie else (1 if definite(c) else 2) for c, ie in zip(cand, is_exp)]
        order = sorted(range(len(cand)), key=lambda i: (tier[i], len(cand[i])))
        return independent_set([cand[i] for i in order], vars_, target, moved)

    def short():
        return len(found) < target or not all(definite(f) for f in found)

    def better(got):
        return len(got) > len(found) or (len(got) == len(found) and sum(map(definite, got)) > sum(map(definite, found)))

    mono = monomial_invariants(preps)
    certs.append(mono["cert"])
    if mono["ok"]:
        found = pick(mono["invariants"])
    if short():
        poly = poly_invariants(preps, d_poly)
        certs.append(poly["cert"])
        if poly["ok"]:
            got = pick(found + poly["invariants"])
            if got != found:
                if better(got):
                    stage_ = "polynomial"
                found = got
    if sep and short():
        s = separable(preps)
        certs.append(s["cert"])
        if s["ok"]:
            got = pick(found + s["invariants"])
            if got != found:
                if better(got):
                    stage_ = "separable"
                found = got
    elif not sep:
        certs.append("separable-characteristics stage skipped (separable=False)")
    if short():
        rat = rational_invariants(preps, d_poly)
        certs.append(rat["cert"])
        if rat["ok"]:
            got = pick(found + rat["invariants"])
            if got != found:
                if better(got):
                    stage_ = "rational"
                found = got
    dar = None
    if short():
        dar = darboux(preps, d_darboux, extactic_ok=len(found) < target)
        certs.append(dar["cert"])
        if dar["ok"]:
            got = pick(found + dar["invariants"])
            if got != found:
                if better(got):
                    stage_ = "darboux"
                found = got
    if short() and dar is not None:
        if d_exp > 0:
            ex = exp_factors(preps, dar, d_exp)
            certs.append(ex["cert"])
            if ex["ok"]:
                got = pick(found + ex["invariants"])
                if got != found:
                    if better(got):
                        stage_ = "exp"
                    found = got
            if len(found) < target and len(preps) == 1 and len(moved) == 2:
                itf = int_factor(preps, dar, ex["factors"])
                certs.append(itf["cert"])
                if itf["ok"]:
                    found = found + [itf["invariants"][0]]
                    stage_ = "intfactor"
        else:
            certs.append("exp stage skipped (d_exp = 0)")
    reason = (
        None
        if len(found) >= target
        else (
            f"found {len(found)} of {target} independent invariants (poly/rational degree <= {d_poly}, "
            f"Darboux degree <= {d_darboux}, exp numerator degree <= {d_exp}); gauge pinning needs the full set"
        )
    )
    return {
        "stage": stage_,
        "invariants": found,
        "target": target,
        "certificates": certs,
        "count": "exact" if count_exact else "upper bound",
        "reason": reason,
    }


def fixed_reduce(preps, fixed):
    """Combinations of a curved group that leave every fixed coordinate unmoved."""
    k = len(preps)
    cols = sym_sort(list(dict.fromkeys(c for pr in preps for c in pr["support"])))
    fixed_cols = [c for c in fixed if c in cols]
    if not fixed_cols or not k:
        return {"preps": preps, "sources": [[i] for i in range(k)], "removed": 0, "fixed_cols": []}
    loc = red_locals([x for pr in preps for x in pr["comps"].values()])
    E = [{cc: red_sympify(pr["comps"].get(cc, "0"), loc) for cc in cols} for pr in preps]
    sources = [[i] for i in range(k)]
    used_pivot = [False] * k
    for fc in fixed_cols:
        piv = None
        for r in range(k):
            if not used_pivot[r] and str(E[r][fc]) != "0":
                piv = r
                break
        if piv is None:
            continue
        for j in range(k):
            if j == piv or str(E[j][fc]) == "0":
                continue
            m = spy.cancel(E[j][fc] / E[piv][fc])
            for c2 in cols:
                E[j][c2] = spy.cancel(E[j][c2] - m * E[piv][c2])
            sources[j] = sorted(set(sources[j]) | set(sources[piv]))
        used_pivot[piv] = True
    out_p, out_s = [], []
    for j in [i for i in range(k) if not used_pivot[i]]:
        comps = {cc: str(E[j][cc]) for cc in cols}
        pr = gen_prep({"generator": {c: v for c, v in comps.items() if v != "0"}})
        if not pr["ok"] or not pr["support"]:
            continue
        out_p.append(pr)
        out_s.append(sources[j])
    return {
        "preps": out_p,
        "sources": out_s,
        "removed": sum(used_pivot),
        "removed_idx": [i for i in range(k) if used_pivot[i]],
        "fixed_cols": fixed_cols,
    }


def _axis_signed(v, reduced, originals):
    """Whether `v` has a reduced component of proven sign on the domain, with every original denominator of one sign."""
    try:
        if any(sgn(red_sympify(pr.get("den") or "1", red_locals([pr.get("den") or "1"]))) == 0 for pr in originals):
            return False
        for pr in reduced:
            c = (pr.get("comps") or {}).get(v)
            if c is not None and sgn(red_sympify(c, red_locals([c]))) != 0:
                return True
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
    return False


def _input_absorbed(d, prefer):
    """The reason a direction of a model with unknown inputs has no chart in the parameters, or None."""
    unknown = (prefer or {}).get("unknown_inputs") or {}
    if not unknown or d.get("generator") is None or d.get("input_proof"):
        return None
    varying = set((prefer or {}).get("states") or ()) | {j for js in unknown.values() for j in js}
    comps = [str(v) for v in d["generator"].values()]
    used = {x for x in red_locals(comps)}
    hit = sym_sort(used & varying)
    if not hit and not d.get("input_refusal"):
        return None
    moved = [c for c, v in d["generator"].items() if str(v).strip() not in ("0", "0.0")]
    inputs = sym_sort([w for w, js in unknown.items() if set(js) & (set(moved) | used)])
    params = sym_sort([c for c in moved if c not in varying])
    w = ", ".join(inputs) or "the unknown inputs"
    p = ", ".join(params) or "the moved parameters"
    verb = "is" if len(params) == 1 else "are"
    if d.get("input_refusal") == "setting":
        return (
            f"the direction changes the trajectory of {w} together with {p}; the proofs that unknown inputs absorb "
            "a direction cover models with one condition, without events and equilibrate, so the reduction has "
            f"no section for it; {p} {verb} non-identifiable"
        )
    if d.get("input_refusal"):
        return (
            f"the direction changes the trajectory of {w} together with {p}; neither the inversion of the inputs "
            f"from the observables nor a pointwise lift of the parameter field proves that {w} "
            f"{'absorbs' if len(inputs) == 1 else 'absorb'} it, so the reduction has no section for it; {p} {verb} "
            "non-identifiable"
        )
    return (
        f"the direction changes the trajectory of {w} together with {p}, with components that depend on "
        f"{', '.join(hit)}, which vary in time: it is no vector field on the parameters, so the reduction "
        f"has no chart for it; {p} {verb} non-identifiable because {w} "
        f"{'absorbs' if len(inputs) == 1 else 'absorb'} the change"
    )


def curved(syms, idx, labels, fixed, d_poly, d_darboux, d_exp, sep=True, pins=None, prefer=None):
    """Curved directions: prep, group, fixed elimination, linear flows, compressions, module reduction, search.

    `prefer` holds the structure of the model the reduction uses: ``observation_only`` and ``states`` (for
    the choice of pins) and ``unknown_inputs`` (input names and jets)."""
    blocks = []

    def unresolved(labs, support, reason):
        return {
            "labels": labs,
            "type": "curved",
            "support": support,
            "stage": "none",
            "status": "unresolved",
            "invariants": [],
            "certificates": [],
            "reason": reason,
            "module_combos": None,
            "transversal": None,
            "new_parameters": None,
            "pins": None,
        }

    scal_labs = {labels[i] for i, d in enumerate(syms) if d.get("type") == "scaling"}

    def kind_of(labs):
        return "scaling" if labs and all(l in scal_labs for l in labs) else "curved"

    has_gen = []
    for i in idx:
        if syms[i].get("generator") is None:
            why = (
                "no closed-form generator: the reconstruction of the detection found none (level open)"
                if (prefer or {}).get("reconstructed")
                else "no closed-form generator; re-run the detection with reconstruct=True"
            )
            blocks.append(unresolved([labels[i]], coords_of(syms[i]), why))
            continue
        why = _input_absorbed(syms[i], prefer)
        if why is not None:
            blk = unresolved([labels[i]], coords_of(syms[i]), why)
            blk["absorbed_by_inputs"] = True
            blocks.append(blk)
            continue
        has_gen.append(i)
    gi = has_gen
    preps = [gen_prep(syms[i], pins) for i in gi]
    for w, pr in enumerate(preps):
        if not pr["ok"]:
            blocks.append(unresolved([labels[gi[w]]], coords_of(syms[gi[w]]), pr.get("reason")))
    keep = [w for w, pr in enumerate(preps) if pr["ok"]]
    gi = [gi[w] for w in keep]
    preps = [preps[w] for w in keep]
    if not gi:
        return blocks
    for grp in group_by([pr["support"] for pr in preps]):
        glab = [labels[gi[g]] for g in grp]
        fx = fixed_reduce([preps[g] for g in grp], fixed)
        fix_cert = (
            [
                f"fixed-coordinate elimination: {fx['removed']} of {len(grp)} direction(s) cannot "
                f"combine away from {{{', '.join(fx['fixed_cols'])}}}"
            ]
            if fx["removed"] > 0
            else []
        )
        fixed_labs = []
        if fx["removed"] > 0:
            used = {s for src in fx["sources"] for s in src}
            fixed_labs = [glab[i] for i in fx["removed_idx"] if i not in used]
        if fixed_labs or not fx["preps"]:
            fl = fixed_labs if fx["preps"] else glab
            blocks.append(
                {
                    "labels": fl,
                    "type": "curved",
                    "kind": kind_of(fl),
                    "support": sym_sort(list(dict.fromkeys(c for g in grp for c in preps[g]["support"]))),
                    "stage": "fixed",
                    "status": "fixed",
                    "removed_by_fixed": fx["removed"],
                    "invariants": [],
                    "certificates": fix_cert,
                    "reason": None,
                    "module_combos": None,
                    "transversal": None,
                    "new_parameters": None,
                    "pins": None,
                    "gauge_note": "removed entirely by the user's fixed coordinates",
                }
            )
        if not fx["preps"]:
            continue
        # the generators as detected, for the exact check of the invariants; those that move a
        # fixed coordinate act only through the combinations that do not
        originals = [preps[g] for g in grp if not set(preps[g]["support"]) & set(fixed or [])]
        glab_k = [glab[s[0]] if len(s) == 1 else "(" + "+".join(glab[i] for i in s) + ")" for s in fx["sources"]]
        # a block that translates linear and logarithmic coordinates needs no invariant search
        lf = linear_flow(fx["preps"], fixed, prefer)
        if lf is not None:
            own = sym_sort([l for l in glab if l not in set(fixed_labs)])[: len(fx["preps"])]
            lf.update(
                labels=own,
                kind=kind_of(own),
                originals=originals,
                removed_by_fixed=fx["removed"] if fx["removed"] > 0 and not fixed_labs else None,
            )
            lf["certificates"] = fix_cert + lf["certificates"]
            blocks.append(lf)
            continue
        try:
            tc = translation_compress(fx["preps"])
        except RouteError:
            tc = None
        except Exception as exc:
            swallow(exc, "translation compression")
            tc = None
        p1 = fx["preps"] if tc is None else tc["preps"]
        try:
            uc = unmoved_compress(p1)
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            uc = None
        except Exception as exc:
            swallow(exc, "unmoved compression")
            uc = None
        p2 = p1 if uc is None else uc["preps"]

        def decomp(pr):
            if uc is not None:
                pr = uc["back_prep"](pr)
            if tc is not None:
                pr = tc["back_prep"](pr)
            return pr

        mr = module_reduce(p2, glab_k)
        subs = group_by([pr["support"] for pr in mr["preps"]])
        # the sub-blocks share the group's labels: each label goes to one sub-block, the
        # sub-blocks with fewer sources first
        src_of = [{s for j in sub for li in mr["labels_of"][j] for s in fx["sources"][li]} for sub in subs]
        taken = set()
        for k in sorted(range(len(subs)), key=lambda k: len(src_of[k])):
            own = sym_sort([glab[s] for s in src_of[k] if glab[s] not in taken])[: len(subs[k])]
            taken.update(own)
            src_of[k] = own
        for sub, sub_labs in zip(subs, src_of):
            supp = sym_sort(list(dict.fromkeys(c for j in sub for c in mr["preps"][j]["support"])))
            if len(supp) == 1:
                # a direction along one coordinate: the orbit is the axis where the component keeps one sign
                blk = scaling_block(np.array([[1]], dtype=np.int64), supp, sub_labs, fixed)
                if blk.get("pins") and not _axis_signed(
                    supp[0], [decomp(mr["preps"][j]) for j in sub], [preps[g] for g in grp]
                ):
                    blk.update(
                        reach=False,
                        coverage="partial",
                        gauge_note=f"the component on {supp[0]} is not proven of one sign on the domain; "
                        f"the orbits between the zeros miss the pin",
                    )
                blocks.append(blk)
                continue
            sol = solve_block([mr["preps"][j] for j in sub], d_poly, d_darboux, d_exp, sep)
            face_inv = None
            if sol["invariants"]:
                if uc is not None:
                    face_inv = sol["invariants"] if tc is None else tc["back"](sol["invariants"])
                    sol["invariants"] = uc["back"](sol["invariants"])
                    sol["certificates"].append(
                        "searched in the unmoved monomial(s) "
                        + "; ".join(uc["note"])
                        + " (constants along every orbit), invariants substituted back"
                    )
                if tc is not None:
                    sol["invariants"] = tc["back"](sol["invariants"])
                    sol["certificates"].append(
                        "searched in the translation-group sum(s) "
                        + "; ".join(tc["note"])
                        + ", invariants substituted back"
                    )
            preps_orig = [mr["preps"][j] for j in sub]
            if uc is not None or tc is not None:
                preps_orig = [decomp(pr) for pr in preps_orig]
            blocks.append(
                {
                    "labels": sub_labs,
                    "type": "curved",
                    "kind": kind_of(sub_labs),
                    "support": sym_sort(list(dict.fromkeys(c for pr in preps_orig for c in pr["support"]))),
                    "stage": sol["stage"],
                    "target": sol["target"],
                    "count": sol.get("count"),
                    "status": "invariant_only" if sol["invariants"] else "unresolved",
                    "removed_by_fixed": fx["removed"] if fx["removed"] > 0 and not fixed_labs else None,
                    "invariants": sol["invariants"],
                    "certificates": fix_cert + sol["certificates"],
                    "reason": sol["reason"],
                    "module_combos": mr["combos"],
                    "transversal": None,
                    "new_parameters": None,
                    "pins": None,
                    "preps": preps_orig,
                    "originals": originals,
                    "face_invariants": face_inv,
                    "face_back": uc["back"] if uc is not None else None,
                }
            )
    return blocks


def verify_blocks(blocks):
    """X(I) = 0 for every invariant and every generator, exactly."""

    def chk(inv, preps):
        try:
            return [
                bool(x)
                for x in verify_invariants([s.replace("^", "**") for s in inv], [dict(pr["comps"]) for pr in preps])
            ]
        except Exception as exc:
            swallow(exc, "invariant verification")
            return [False] * len(inv)

    ident = re.compile(r"[A-Za-z._][A-Za-z0-9._]*")
    for bi, b in enumerate(blocks):
        if b["type"] == "scaling":
            if (
                b.get("Wres") is not None
                and b.get("inv_exps") is not None
                and b["inv_exps"].shape[1]
                and b["Wres"].shape[0]
            ):
                cols = b["Wcols"]
                sel = [cols.index(c) for c in b["support"]]
                if np.any(b["Wres"][:, sel] @ b["inv_exps"] != 0):
                    raise RuntimeError("internal error: scaling invariant not in the weight kernel.")
                b["certificates"].append("verified: integer weight annihilation, exact")
            continue
        if not b.get("invariants") or b.get("preps") is None:
            continue
        keep = chk(b["invariants"], b["preps"])
        if not all(keep):
            b["certificates"].append(f"verification DROPPED {sum(not k for k in keep)} invariant(s): X(I) != 0")
            b["invariants"] = [i for i, k in zip(b["invariants"], keep) if k]
            if not any(keep):
                b["status"] = "unresolved"
        else:
            b["certificates"].append("verified: X(I) = 0 exactly (sympy cancel) for every generator")
        if b.get("originals") and b["invariants"]:
            keep = chk(b["invariants"], b["originals"])
            if not all(keep):
                b["certificates"].append(
                    f"verification DROPPED {sum(not k for k in keep)} invariant(s): "
                    "X(I) != 0 for a generator as detected"
                )
                b["invariants"] = [i for i, k in zip(b["invariants"], keep) if k]
                if not any(keep):
                    b["status"] = "unresolved"
            else:
                b["certificates"].append("verified: X(I) = 0 exactly for every generator as detected")
        inv = b["invariants"]
        inv_syms = {m for s in inv for m in ident.findall(s)}
        other = [pr for j, ob in enumerate(blocks) if j != bi for pr in (ob.get("preps") or [])]
        if not other or not inv:
            continue
        other = [pr for pr in other if set(pr["support"]) & inv_syms]
        if not other:
            b["certificates"].append(
                "verified: X(I) = 0 for every direction of every other block too, "
                "none of which moves a symbol of these invariants"
            )
            continue
        keep = chk(inv, other)
        if not all(keep):
            b["certificates"].append(
                f"verification DROPPED {sum(not k for k in keep)} invariant(s): "
                "X(I) != 0 for a direction of another block"
            )
            b["invariants"] = [i for i, k in zip(inv, keep) if k]
            if not any(keep):
                b["status"] = "unresolved"
        else:
            b["certificates"].append("verified: X(I) = 0 for every direction of every other block too")
    return blocks
