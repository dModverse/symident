"""Whether the resting balances have a solution, and one in the declared domain, at random parameter values."""

import math
import os
import re
import tempfile
from fractions import Fraction

import sympy as spy

from .. import session
from ..errors import SYMPY_FAILS, InputError, sympy_fail
from .parse import _make_local_parse

# the check runs on systems up to this many resting states, each msolve call within the short caps of work.CAPS
_MAX_STATES = 12
_POINTS = 3


def _run_msolve(names, eqs):
    """The parsed msolve output over Q for `eqs` in `names`, or None on failure or past the short caps of msolve
    (work.quick_msolve())."""
    from ..detection import work
    from .msolve import MsolveMissingError, _msolve_binary

    try:
        exe = _msolve_binary()
    except MsolveMissingError:
        return None
    with work.quick_msolve():
        if work.msolve_too_large(names, eqs):
            return None
        with tempfile.TemporaryDirectory() as td:
            fi, fo = os.path.join(td, "in.ms"), os.path.join(td, "out.ms")
            with open(fi, "w") as fh:
                fh.write(",".join(names) + "\n0\n" + ",\n".join(eqs) + "\n")
            r = work.run_msolve([exe, "-f", fi, "-o", fo, "-t", "1"])
            if r.returncode != 0 or not os.path.exists(fo):
                return None
            with open(fo) as fh:
                txt = fh.read().strip().rstrip(":")
    if not txt:
        return None
    from ..detection.discrete import parse_rational

    try:
        return parse_rational(txt)
    except (SyntaxError, ValueError, TypeError, NameError):
        return None


def _systems(multi, zero_of):
    """(balances, unknowns, held) per distinct resting model of the equilibrated conditions."""
    held = {str(k): str(v) for k, v in (multi.get("held_state_params") or {}).items()}
    forc = set(str(x) for x in (multi.get("forcings") or []))
    real = [str(x) for x in (multi.get("real_state_names") or multi.get("state_names") or [])]
    out, seen = [], set()
    for ti, t in enumerate(multi.get("tapes") or []):
        lines = [str(x) for x in (t.get("constraint_model") or [])]
        zero = set(zero_of(ti))
        key = (tuple(lines), tuple(sorted(zero)))
        if not lines or key in seen:
            continue
        seen.add(key)
        _, parse = _make_local_parse(lines)
        rhs = {}
        for ln in lines:
            lhs, r = ln.split("=", 1)
            rhs[lhs.strip()] = spy.sympify(parse(r))
        sub = {spy.Symbol(x): spy.Integer(0) for x in real if x in forc or x in zero}
        sub.update({spy.Symbol(x): spy.Symbol(p) for x, p in held.items()})
        unknowns = [x for x in real if x not in forc and x not in zero and x not in held]
        eqs = [rhs[x].xreplace(sub) for x in unknowns if x in rhs]
        out.append((eqs, unknowns))
    return out


def _positive_names(positive):
    """The names declared positive, or True for every symbol."""
    if positive is True:
        return True
    if not positive:
        return set()
    return {str(p) for p in positive if re.fullmatch(r"[A-Za-z_.][A-Za-z0-9_.]*", str(p))}


def _cleared(e, gens):
    """The numerator and the denominator of `e` as polynomials in `gens` with coprime integer coefficients."""
    num, den = spy.fraction(spy.together(e))
    out = []
    for q in (num, den):
        P = spy.Poly(spy.expand(q), *gens, domain="QQ")
        if P.is_zero:
            out.append(P)
            continue
        _, P = P.clear_denoms(convert=True)
        out.append(P.primitive()[1])
    return out


def modular_decision(eqs, unknowns, pos, rng, draws=_POINTS):
    """{"exists", "bound", "draws"} of the resting balances `eqs` in `unknowns` over Q(theta)
    (docs/method/conditions.rst, "Resting states"): "proven" where a simple point modulo the count prime at an integer
    point lifts by Hensel's lemma, "no" with the probability `bound` of a resting state for generic parameter values
    where none exists modulo the prime at every point drawn, else "open"."""
    from ..detection import work
    from ..detection.discrete import _gb_count
    from ..gfp import PRIMES

    if not unknowns or not eqs:
        return {"exists": "open", "bound": None, "draws": []}
    xs = [spy.Symbol(x) for x in unknowns]
    th = sorted(set().union(*[e.free_symbols for e in eqs]) - set(xs), key=str)
    gens = th + xs
    nums, dens = [], []
    for e in eqs:
        n, d = _cleared(e, gens)
        if not n.is_zero:
            nums.append(n)
        if not d.is_ground:
            dens.append(d)
    deg_theta = [n.total_degree() for n in nums] + [1 + sum(d.total_degree() for d in dens)]
    names = [f"x{i}" for i in range(len(xs))]
    ren = dict(zip(xs, [spy.Symbol(n) for n in names]))
    square = len(nums) == len(xs)
    M = 2**64
    Dt = math.prod(max(1, int(d)) for d in deg_theta)
    bound, verdicts = 1.0, []
    for _ in range(draws):
        at = {}
        for s in th:
            v = rng.randint(1, M)
            at[s] = spy.Integer(-v if not (pos is True or str(s) in pos) and rng.random() < 0.5 else v)
        polys = [spy.Poly(n.as_expr().xreplace(at).xreplace(ren), *[spy.Symbol(x) for x in names]) for n in nums]
        den = spy.prod([d.as_expr().xreplace(at).xreplace(ren) for d in dens]) if dens else spy.Integer(1)
        if den == 0:
            verdicts.append("open")
            continue
        w = spy.Poly(spy.Symbol("si_w") * den - 1, *[spy.Symbol(x) for x in names], spy.Symbol("si_w"))
        whole = all(not q.is_zero for q in polys)
        polys = [q for q in polys if not q.is_zero]
        if any(q.is_ground for q in polys):
            # a nonzero constant balance: no solution at this point over the complex numbers
            bound *= Dt / M
            verdicts.append("no")
            continue
        p = PRIMES[0]
        text = [str(q.as_expr()).replace("**", "^").replace(" ", "") for q in polys + [w]]
        dims = []
        with work.quick_msolve():
            k = _gb_count(names, p, text, names, exact=True, dims=dims)
        if k == 0:
            # none modulo p; the empty set over Q at the point decides, without a bad prime
            data = _run_msolve(names + ["si_w"], text)
            if data is not None and data[0] == -1:
                bound *= Dt / M
                verdicts.append("no")
            else:
                verdicts.append("open")
        elif (
            k
            and square
            and whole
            and dims
            and k == dims[0]
            and all(any(int(c) % p for c in q.coeffs()) for q in polys + [w])
        ):
            # as many distinct values as the dimension of the quotient, no equation zero modulo p: every point is simple
            verdicts.append("proven")
        else:
            verdicts.append("open")
    if "proven" in verdicts:
        return {"exists": "proven", "bound": None, "draws": verdicts}
    if verdicts and all(v == "no" for v in verdicts):
        return {"exists": "no", "bound": bound if bound < 1 else None, "draws": verdicts}
    return {"exists": "open", "bound": None, "draws": verdicts}


def rest_check(multi, zero_of, positive, points=_POINTS):
    """{"points", "consistent", "in_domain", "undecided", "checked"} of the resting balances at random
    parameter values: an InputError where the balances have no solution over the complex numbers at any of them."""
    systems = _systems(multi, zero_of)
    if not systems:
        return None
    if multi.get("rest_aux") or multi.get("power_recast") or multi.get("exp_atoms"):
        return {"checked": False, "why": "the resting model holds leaves of non-rational terms"}
    if any(len(u) > _MAX_STATES for _, u in systems):
        return {"checked": False, "why": f"more than {_MAX_STATES} resting states"}
    pos = _positive_names(positive)
    rng = session.current().rng("rest check")
    report = {
        "points": 0,
        "consistent": 0,
        "in_domain": 0,
        "undecided": 0,
        "checked": True,
        "how": "msolve over Q at random rational parameter values, real roots isolated; a check at "
        "random points, not a proof. exists: a Groebner basis modulo a random prime at random "
        "integer points, proven by a simple point, or no with the error bound",
    }
    constant = set()
    for _ in range(points):
        # per system: "none" (no complex solution), "out" (none in the domain), "unsure" or "in"
        states = []
        for eqs, unknowns in systems:
            syms = sorted(set().union(*[e.free_symbols for e in eqs]) - {spy.Symbol(x) for x in unknowns}, key=str)
            at = {}
            for s in syms:
                v = Fraction(rng.randint(1, 40), rng.randint(1, 7))
                if not (pos is True or str(s) in pos) and rng.random() < 0.5:
                    v = -v
                at[s] = spy.Rational(v.numerator, v.denominator)
            try:
                num, den = [], spy.Integer(1)
                for x, e in zip(unknowns, eqs):
                    n, d = spy.fraction(spy.together(e.xreplace(at)))
                    n = spy.expand(n)
                    if n.is_number and n != 0:
                        # a balance that no resting value of the states solves
                        constant.add(x)
                    if n != 0:
                        num.append(n)
                        den = den * d
                den = spy.expand(den)
            except SYMPY_FAILS:
                return {"checked": False, "why": "the balances do not clear to polynomials"}
            if any(n.is_number for n in num):
                states.append("none")
                continue
            if not num:
                states.append("unsure")
                continue
            names = [f"x{i}" for i in range(len(unknowns))]
            ren = {spy.Symbol(x): spy.Symbol(n) for x, n in zip(unknowns, names)}
            polys = [n.xreplace(ren) for n in num]
            if not den.is_number:
                # a resting state is no pole of the right-hand sides
                polys.append(spy.expand(spy.Symbol("xw") * den.xreplace(ren) - 1))
                names = names + ["xw"]
            # msolve reads expanded polynomials without spaces
            polys = [str(spy.expand(q)).replace("**", "^").replace(" ", "") for q in polys]
            data = _run_msolve(names, polys)
            if data is None:
                from .msolve import MsolveMissingError, _msolve_binary

                try:
                    _msolve_binary()
                except MsolveMissingError:
                    return dict(report, checked=False, why="msolve is not installed (symident.install_msolve())")
                return dict(report, checked=False, why="msolve did not finish")
            if data[0] == -1:
                states.append("none")
                continue
            if data[0] != 0:
                # a resting set of positive dimension is not decided
                states.append("unsure")
                continue
            verdicts = []
            for sol in data[1][1]:
                v = "in"
                for x, (lo, hi) in zip(unknowns, sol[: len(unknowns)]):
                    if pos is True or x in pos:
                        if hi <= 0:
                            v = "out"
                            break
                        if lo <= 0:
                            v = "unsure"
                verdicts.append(v)
            states.append("in" if "in" in verdicts else "unsure" if "unsure" in verdicts else "out")
        report["points"] += 1
        if "none" not in states:
            report["consistent"] += 1
            if all(v == "in" for v in states):
                report["in_domain"] += 1
            elif "out" not in states:
                report["undecided"] += 1
    try:
        decisions = [modular_decision(eqs, unknowns, pos, rng) for eqs, unknowns in systems]
    except SYMPY_FAILS + (ValueError,) as exc:
        sympy_fail(exc)
        decisions = []
    if decisions:
        verdict = [d["exists"] for d in decisions]
        report["exists"] = "no" if "no" in verdict else "proven" if all(v == "proven" for v in verdict) else "open"
        report["bound"] = (
            max((d["bound"] for d in decisions if d["exists"] == "no" and d["bound"] is not None), default=None)
            if report["exists"] == "no"
            else None
        )
        if report["exists"] == "no" and any(d["exists"] == "no" and d["bound"] is None for d in decisions):
            report["bound"] = None
    if report["points"] and not report["consistent"] and report.get("exists") != "proven":
        states = sorted(constant) or sorted({x for _, u in systems for x in u})
        what = (
            f"the balance{'s' if len(states) > 1 else ''} of {', '.join(states)} at rest, with the inputs "
            f"at zero, {'are nonzero constants' if len(states) > 1 else 'is a nonzero constant'}"
            if constant
            else "the resting balances of " + ", ".join(states) + " have no common solution over the complex numbers"
        )
        sure = f" (error bound {report['bound']:.3g})" if report.get("bound") is not None else ""
        raise InputError(
            f"equilibrate: {what} at {report['points']} random parameter values, "
            f"so the model has no resting state for generic parameters{sure}; check the right-hand "
            "sides, the inputs at rest and the conserved totals"
        )
    return report
