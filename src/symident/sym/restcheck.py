"""Whether the resting balances have a solution, and one in the declared domain, at random parameter values."""

import os
import re
import subprocess
import tempfile
from fractions import Fraction

import sympy as spy

from .. import session
from ..errors import SYMPY_FAILS, InputError
from .parse import _make_local_parse

# the check runs on systems up to this many resting states, each msolve call up to this many seconds
_MAX_STATES = 12
_TIMEOUT = 10.0
_POINTS = 3


def _run_msolve(names, eqs, timeout):
    """The parsed msolve output over Q for `eqs` in `names`, or None on failure or timeout."""
    from .msolve import MsolveMissingError, _msolve_binary
    try:
        exe = _msolve_binary()
    except MsolveMissingError:
        return None
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.ms"), os.path.join(td, "out.ms")
        with open(fi, "w") as fh:
            fh.write(",".join(names) + "\n0\n" + ",\n".join(eqs) + "\n")
        try:
            r = subprocess.run([exe, "-f", fi, "-o", fo, "-t", "1"], capture_output=True, text=True,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            return None
        if r.returncode != 0 or not os.path.exists(fo):
            return None
        with open(fo) as fh:
            txt = fh.read().strip().rstrip(":")
    if not txt:
        return None
    txt = re.sub(r"(-?\d+)\s*/\s*2\^(\d+)", r"Fraction(\1, 2**\2)", txt)
    try:
        return eval(txt, {"Fraction": Fraction, "__builtins__": {}})
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


def rest_check(multi, zero_of, positive, points=_POINTS, timeout=_TIMEOUT):
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
    report = {"points": 0, "consistent": 0, "in_domain": 0, "undecided": 0, "checked": True,
              "how": "msolve over Q at random rational parameter values, real roots isolated; a check at "
                     "random points, not a proof"}
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
            data = _run_msolve(names, polys, timeout)
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
                for x, (lo, hi) in zip(unknowns, sol[:len(unknowns)]):
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
    if report["points"] and not report["consistent"]:
        states = sorted(constant) or sorted({x for _, u in systems for x in u})
        what = (f"the balance{'s' if len(states) > 1 else ''} of {', '.join(states)} at rest, with the inputs "
                f"at zero, {'are nonzero constants' if len(states) > 1 else 'is a nonzero constant'}" if constant
                else "the resting balances of " + ", ".join(states) + " have no common solution over the complex "
                "numbers")
        raise InputError(f"equilibrate: {what} at {report['points']} random parameter values, "
                         "so the model has no resting state for generic parameters; check the right-hand sides, "
                         "the inputs at rest and the conserved totals")
    return report
