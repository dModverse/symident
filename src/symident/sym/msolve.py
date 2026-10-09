"""Points of polynomial systems over GF(p) from msolve."""

import contextlib
import itertools
import os

import sympy as spy

from ..install import find_msolve
from .gfpoly import _dp_coeffs, _dp_subst_rat, _dp_value, _gf_roots

_MSOLVE_TIMEOUT = 600


class MsolveMissingError(RuntimeError):
    """Raised when the msolve executable is not available."""


@contextlib.contextmanager
def msolve_timeout(seconds):
    """Time limit of one msolve call in seconds within the block."""
    global _MSOLVE_TIMEOUT
    old, _MSOLVE_TIMEOUT = _MSOLVE_TIMEOUT, seconds
    try:
        yield
    finally:
        _MSOLVE_TIMEOUT = old


def _msolve_binary():
    """Path of the msolve executable."""
    path, _ = find_msolve()
    if path is None:
        raise MsolveMissingError(
            "equilibrate: this steady state needs msolve; run symident.install_msolve(), or "
            "set SYMIDENT_MSOLVE to an msolve executable.")
    return path


def _const_reduce(dps, nv, p):
    """Eliminate pivots x_i = -b/a0 with constant a0. Returns (plan, rest), plan as
    (i, b, 1/a0) in elimination order."""
    dps = [dict(d) for d in dps if d]
    plan = []
    while True:
        pick = None
        for j, d in enumerate(dps):
            for i in range(nv):
                if not any(m[i] for m in d) or max(m[i] for m in d) != 1:
                    continue
                cs = _dp_coeffs(d, i)
                a = cs[1]
                if len(a) == 1 and not any(next(iter(a))):
                    pick = (j, i, cs.get(0, {}), pow(next(iter(a.values())) % p, p - 2, p))
                    break
            if pick:
                break
        if pick is None:
            return plan, dps
        j, i, b, inv = pick
        dps.pop(j)
        plan.append((i, b, inv))
        a1 = {tuple([0] * nv): 1}
        b1 = {m: c * inv % p for m, c in b.items()}      # x_i = -(b * inv)
        dps = [(_dp_subst_rat(d, i, a1, b1, nv, p) if any(m[i] for m in d) else d) for d in dps]
        dps = [d for d in dps if d]


def _msolve_points(polys, vars_idx, nv, p):
    """Interior points of `polys` over GF(p). Returns a list of {index: value} or a
    fail-dict."""
    import ast
    import subprocess
    import tempfile
    names = {i: "x%d" % i for i in vars_idx}

    def pstr(d):
        out = []
        for m, c in d.items():
            mono = "*".join(names[i] + ("^%d" % m[i] if m[i] > 1 else "") for i in vars_idx if m[i])
            out.append("%d*%s" % (c % p, mono) if mono else "%d" % (c % p))
        return "+".join(out) or "0"

    eqs = [pstr(d) for d in polys] + ["w*" + "*".join(names[i] for i in vars_idx) + "-1"]
    vs = [names[i] for i in vars_idx] + ["w"]
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.ms"), os.path.join(td, "out.ms")
        with open(fi, "w") as fh:
            fh.write(", ".join(vs) + "\n" + str(p) + "\n" + ",\n".join(eqs) + "\n")
        r = subprocess.run([_msolve_binary(), "-f", fi, "-o", fo, "-P", "1", "-t", "1"],
                           capture_output=True, text=True, timeout=_MSOLVE_TIMEOUT)
        txt = ""
        if os.path.exists(fo):
            with open(fo) as fh:
                txt = fh.read()
    if r.returncode != 0 or "too large" in (r.stdout + r.stderr) or not txt.strip():
        return {'ok': False, 'why': 'msolve failed: %s' % (r.stdout + r.stderr).strip()[-200:]}
    data = ast.literal_eval(txt.strip().rstrip(":"))
    if data[0] != 0:
        return {'ok': False, 'why': 'positive-dimensional steady state'}
    _, nvar, deg, vnames, lin, par = data[1]
    if deg <= 0 or par[0] != 1:
        return {'ok': False, 'why': 'no consistent interior point'}
    elim, den, params = par[1][0][1], par[1][1][1], par[1][2]
    lin = [int(c) % p for c in lin]
    last = len(vnames) - 1
    if lin[last] == 0:
        return {'ok': False, 'why': 'msolve: unexpected linear form'}
    idx_of = {names[i]: i for i in vars_idx}

    def ev(coeffs, x):          # coefficients lowest degree first
        acc = 0
        for c in reversed(coeffs):
            acc = (acc * x + c) % p
        return acc

    out = []
    for t in _gf_roots(list(reversed([int(c) % p for c in elim])), p):
        dv = ev(den, t)
        if dv == 0:
            continue
        dinv = pow(dv, p - 2, p)
        vals = {}
        for k, q in enumerate(params):
            vals[vnames[k]] = (-ev(q[0][1], t)) * dinv % p
        # last variable from the linear form
        rest = (t - sum(lin[k] * vals[vnames[k]] for k in range(last))) % p
        vals[vnames[last]] = rest * pow(lin[last], p - 2, p) % p
        out.append({idx_of[nm]: v for nm, v in vals.items() if nm in idx_of})
    return out


def _component_points(polys, nv, p):
    """Interior points of `polys` over GF(p), solved per block of polynomials that share
    variables; a block in one variable by its roots. Returns an iterator of
    {index: value} or a fail-dict."""
    vs = [{i for m in d for i in range(nv) if m[i]} for d in polys]
    parent = list(range(len(polys)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    owner = {}
    for j, v in enumerate(vs):
        for i in v:
            if i in owner:
                parent[find(j)] = find(owner[i])
            else:
                owner[i] = j
    blocks = {}
    for j in range(len(polys)):
        blocks.setdefault(find(j), []).append(j)
    lists = []
    for js in blocks.values():
        bv = sorted(set().union(*(vs[j] for j in js)))
        if not bv:
            return {'ok': False, 'why': 'no consistent interior point'}
        if len(bv) == 1:
            i = bv[0]
            roots = None
            for j in js:
                deg = max(m[i] for m in polys[j])
                co = [0] * (deg + 1)
                for m, c in polys[j].items():
                    co[deg - m[i]] = (co[deg - m[i]] + c) % p
                r = set(_gf_roots(co, p))
                roots = r if roots is None else roots & r
            pts = [{i: t} for t in sorted(roots)]
        else:
            pts = _msolve_points([polys[j] for j in js], bv, nv, p)
            if isinstance(pts, dict):
                return pts
        if not pts:
            return {'ok': False, 'why': 'no consistent interior point'}
        lists.append(pts)

    def merged():
        for combo in itertools.product(*lists):
            out = {}
            for c in combo:
                out.update(c)
            yield out
    return merged()


def _solve_states_msolve(dps, solve_states, p):
    """Interior point of f = 0 over GF(p), checked against f."""
    nv = len(solve_states)
    plan, rest = _const_reduce(dps, nv, p)
    # outputs: a state in a single remaining balance, linearly, is solved after msolve
    sinks = []
    while True:
        occ = {}
        for j, d in enumerate(rest):
            for i in {i for m in d for i in range(nv) if m[i]}:
                occ.setdefault(i, []).append(j)
        pick = next(((i, js[0]) for i, js in sorted(occ.items()) if len(js) == 1 and
                     max(m[i] for m in rest[js[0]]) == 1 and len(occ) > 1), None)
        if pick is None:
            break
        i, j = pick
        cs = _dp_coeffs(rest[j], i)
        sinks.append((i, cs[1], cs.get(0, {})))
        rest = rest[:j] + rest[j + 1:]
    vars_left = sorted({i for d in rest for m in d for i in range(nv) if m[i]})
    if rest and not vars_left:
        return None, {'ok': False, 'why': 'no consistent interior point'}
    cands = _component_points(rest, nv, p) if rest else iter([{}])
    if isinstance(cands, dict):
        return None, cands
    for cand in itertools.islice(cands, 4096):
        vals = [0] * nv
        for i, v in cand.items():
            vals[i] = v
        ok = True
        for i, a, b in reversed(sinks):
            av = _dp_value(a, vals, p)
            if av == 0:
                ok = False
                break
            vals[i] = (-_dp_value(b, vals, p)) * pow(av, p - 2, p) % p
        if not ok:
            continue
        for i, b, inv in reversed(plan):
            vals[i] = (-_dp_value(b, vals, p)) * inv % p
        if all(vals) and all(_dp_value(d, vals, p) == 0 for d in dps):
            return {solve_states[i]: spy.Integer(vals[i]) for i in range(nv)}, None
    return None, {'ok': False, 'why': 'no consistent interior point'}
