"""One tape for conditions that differ only in the values of symbols: the model is compiled once over
placeholders, and each condition gives the placeholders as leaf values (a number or another leaf)."""

import collections
import re
from fractions import Fraction

import numpy as np
import sympy as spy

from ..errors import SYMPY_FAILS, sympy_fail

_NUM = re.compile(r"^\(?\s*(-?\d+)\s*(?:/\s*(\d+))?\s*\)?$")
_SYM = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# the arrays the kernel reads of a single-segment tape
_KERNEL_KEYS = (
    "op",
    "a",
    "b",
    "cnum",
    "cden",
    "state_slots",
    "f_out",
    "g_out",
    "ic_leaf",
    "ic_num",
    "ic_den",
    "ic_op",
    "ic_a",
    "ic_b",
    "ic_cnum",
    "ic_cden",
    "ic_out",
    "ic_seed",
)
_LINE_KEYS = ("model_lines", "obs_lines", "ver_ic", "jet_f", "jet_g", "jet_ic")


def _value(v):
    """A substitution value as a Fraction or a symbol name, None for anything else."""
    v = str(v).strip()
    m = _NUM.match(v)
    if m:
        return Fraction(int(m.group(1)), int(m.group(2) or 1))
    return v if _SYM.match(v) else None


def _text(v):
    return v if isinstance(v, str) else "(%d/%d)" % (v.numerator, v.denominator)


class _Zeroed:
    """The symbols of an expression after the placeholders that are 0 vanish, as SymPy evaluates a product with a
    factor 0 and a sum with a term 0."""

    def __init__(self, e, ph):
        self.tree = self._node(e, ph)

    def _node(self, e, ph):
        if e.is_Symbol:
            return ("p", str(e)) if str(e) in ph else ("s", str(e))
        if e.is_Number:
            return ("z",) if e == 0 else ("c",)
        if e.is_Add or e.is_Mul:
            return ("+" if e.is_Add else "*", [self._node(x, ph) for x in e.args])
        if e.is_Pow and e.exp.is_Integer and e.exp > 0:
            return ("*", [self._node(e.base, ph)])
        # any other function keeps every symbol, also of a zero placeholder
        return ("f", sorted(str(s) for s in e.free_symbols))

    def symbols(self, val):
        """The symbols with `val`: placeholder -> Fraction or symbol name; None for an expression that vanishes."""

        def go(n):
            k = n[0]
            if k == "s":
                return {n[1]}
            if k == "p":
                v = val[n[1]]
                if isinstance(v, str):
                    return {v}
                return None if v == 0 else set()
            if k == "z":
                return None
            if k == "c":
                return set()
            if k == "f":
                return {
                    val[x] if x in val and isinstance(val[x], str) else x
                    for x in n[1]
                    if not (x in val and not isinstance(val[x], str))
                }
            parts = [go(x) for x in n[1]]
            if k == "*":
                if any(p is None for p in parts):
                    return None
                return set().union(*parts)
            live = [p for p in parts if p is not None]
            return set().union(*live) if live else None

        return go(self.tree)


def _eligible(kw):
    """Whether the conditions differ only in the values of symbols and nothing in them needs a tape per condition."""
    K = len(kw.get("condition_subs") or [])
    if K < 2 or kw.get("equilibrate") or kw.get("joint_steady_state") or kw.get("forcings"):
        return False
    if any(kw.get("seg_equilibrate") or []) or kw.get("condition_chain") is not None:
        return False
    if any(kw.get("condition_events") or []) or any(kw.get("condition_t0_events") or []):
        return False
    if kw.get("condition_times") is not None and any(t is not None for t in kw["condition_times"]):
        return False
    if any(dict(d or {}) for d in (kw.get("condition_rest_subs") or [])):
        return False
    if kw.get("held_state_params") or kw.get("joint_fixed_states"):
        return False
    ic0 = [dict(d) for d in (kw.get("condition_ic0") or [])]
    return len(ic0) == K and all(d == ic0[0] for d in ic0)


def shared_tape_compile(compile_fn, kw):
    """compile_fn(**kw) as one compile of the model over placeholders, the tape of each condition holding the
    values of the placeholders as the leaves it sets; None where the conditions do not allow it."""
    if not _eligible(kw):
        return None
    subs = [{str(k): str(v) for k, v in dict(d).items()} for d in kw["condition_subs"]]
    K = len(subs)
    keys = sorted(set().union(*subs))
    vals = {}
    for k in keys:
        col = [_value(d.get(k, k)) for d in subs]
        if any(v is None for v in col):
            return None
        vals[k] = col
    varying = [k for k in keys if len(set(vals[k])) > 1]
    observation = list(kw.get("observation") or [])
    cobs = list(kw.get("condition_obs") or [])
    obs = [list(cobs[c]) if c < len(cobs) and cobs[c] is not None else observation for c in range(K)]
    if len({len(o) for o in obs}) != 1:
        return None
    model = [str(x) for x in (kw.get("model") or [])]
    states = {line.split("=", 1)[0].strip() for line in model}
    ov = _observable_values(obs)
    if ov is None:
        return None
    obs_val, obs_ids = ov
    if any(a in states or a in vals for a in obs_ids):
        return None
    if not varying and not obs_ids:
        return None
    taken = set(_IDENT.findall("\n".join(model + [x for o in obs for x in o]))) | set(keys)
    for k in keys:
        taken |= {v for v in vals[k] if isinstance(v, str)}
    ph = {}
    for k in varying + obs_ids:
        nm = k + "_sv"
        while nm in taken:
            nm += "_"
        taken.add(nm)
        ph[k] = nm
    # the value of each placeholder per condition
    pval = [{ph[k]: vals[k][c] for k in varying} for c in range(K)]
    for c in range(K):
        for a in obs_ids:
            pval[c][ph[a]] = obs_val[c].get(a, a)
    value_syms = sorted({v for d in pval for v in d.values() if isinstance(v, str)})
    if any(v in states for v in value_syms):
        return None
    # a value that is itself substituted in the condition would be substituted again
    for c in range(K):
        if any(isinstance(v, str) and v in vals and vals[v][c] != v for v in pval[c].values()):
            return None
    t_subs = {k: (ph[k] if k in ph else _text(vals[k][0])) for k in keys}
    t_obs = [
        _IDENT.sub(lambda m: ph.get(m.group(0), m.group(0)) if m.group(0) in obs_ids else m.group(0), line)
        for line in obs[0]
    ]
    kw1 = dict(
        kw,
        condition_subs=[t_subs],
        condition_ic0=[dict(kw["condition_ic0"][0])],
        condition_obs=[t_obs],
        observation=t_obs,
        fixed=list(kw.get("fixed") or []) + sorted(ph.values()),
        keep_coords=list(kw.get("keep_coords") or []) + value_syms,
        seg_equilibrate=[False],
        condition_events=[[]],
        condition_t0_events=[[]],
        condition_rest_subs=[{}],
        condition_times=None,
    )
    out = compile_fn(**kw1)
    if not out.get("ok") or len(out.get("tapes") or []) != 1:
        return None
    if out.get("exp_relation") or out.get("power_recast") or out.get("recast_transient") or out.get("log_params"):
        return None
    tpl = out["tapes"][0]
    op = np.asarray(tpl["op"], dtype=np.int64)
    if np.any(op > 3) or tpl.get("ic_seed") is not None or tpl.get("constraint_model") is not None:
        return None
    nl, ns = int(out["n_leaves"]), int(out["n_states"])
    slot = {nm: i for i, nm in enumerate(out["leaf_names"])}
    if any(v not in slot for v in value_syms):
        return None
    used = [nm for nm in sorted(ph.values()) if nm in slot]
    if not _reciprocals_free_of(tpl, nl + ns, nl, {slot[nm] for nm in used}):
        return None
    parsed = _template_exprs(tpl)
    if parsed is None:
        return None
    phs = set(ph.values())
    f_tr = {x: _Zeroed(e, phs) for x, e in parsed["f"].items()}
    g_tr = [_Zeroed(e, phs) for e in parsed["g"]]
    ic_tr = {x: _Zeroed(e, phs) for x, e in parsed["ic"].items()}
    znames = list(out["znames"])
    pat = re.compile(r"\b(" + "|".join(sorted(map(re.escape, used), key=len, reverse=True)) + r")\b") if used else None
    tapes, seen = [], set()
    for c in range(K):
        val = pval[c]
        t = dict(tpl)
        t["leaf_slot"] = [slot[nm] for nm in used]
        t["leaf_src"] = [slot[val[nm]] if isinstance(val[nm], str) else -1 for nm in used]
        t["leaf_num"] = ["0" if isinstance(val[nm], str) else str(val[nm].numerator) for nm in used]
        t["leaf_den"] = ["1" if isinstance(val[nm], str) else str(val[nm].denominator) for nm in used]
        if pat is not None:
            txt = {nm: _text(val[nm]) for nm in used}

            def sub(lines):
                return [pat.sub(lambda m: txt[m.group(0)], str(x)) for x in lines]

            for key in _LINE_KEYS:
                if tpl.get(key) is not None:
                    t[key] = sub(tpl[key])
        reach, syms = _reach(f_tr, g_tr, ic_tr, val)
        seen |= syms
        t["lie_reach"] = [int(reach.get(nm, -1)) for nm in znames]
        tapes.append(t)
    # a coordinate that every condition drops is no coordinate, as in a compile per condition
    keep = [i for i, nm in enumerate(znames) if nm in seen]
    if len(keep) < len(znames):
        out["znames"] = [znames[i] for i in keep]
        out["z_slots"] = [out["z_slots"][i] for i in keep]
        for t in tapes:
            t["lie_reach"] = [t["lie_reach"][i] for i in keep]
    out["tapes"] = tapes
    out["shared_tape"] = True
    sg = _scaling_groups(tpl, [{nm: pval[c][nm] for nm in used} for c in range(K)])
    if sg is not None:
        groups, canon = sg
        out["scaling_groups"] = groups
        # per group the lines with a nonzero number at 1 where no two terms meet: the same monomials and
        # degrees, and lines that repeat between the groups
        out["scaling_tapes"] = []
        for g, cv in zip(groups, canon):
            t = {k: v for k, v in tapes[g[0]].items() if k.startswith("ver_")}
            if pat is not None:
                txt = {nm: _text(cv[nm]) for nm in used}
                for key in ("model_lines", "obs_lines", "ver_ic"):
                    if tpl.get(key) is not None:
                        t[key] = [pat.sub(lambda m: txt[m.group(0)], str(x)) for x in tpl[key]]
            out["scaling_tapes"].append(t)
    return out


def _observable_values(obs):
    """(per condition {identifier: value}, the identifiers) of the identifiers in which the observables `obs` of the
    conditions differ, None where the observables are not one template."""
    tmpl = [(_IDENT.split(line), _IDENT.findall(line)) for line in obs[0]]
    obs_val = [{} for _ in obs]
    for c, lines in enumerate(obs):
        for li, line in enumerate(lines):
            seps, ids = _IDENT.split(line), _IDENT.findall(line)
            if seps != tmpl[li][0] or len(ids) != len(tmpl[li][1]):
                return None
            for a, b in zip(tmpl[li][1], ids):
                if a != b and obs_val[c].setdefault(a, b) != b:
                    return None
    obs_ids = sorted(set().union(*[set(d) for d in obs_val]))
    # every occurrence of an identifier that differs takes the value of the condition
    for c, lines in enumerate(obs):
        for li, line in enumerate(lines):
            for a, b in zip(tmpl[li][1], _IDENT.findall(line)):
                if a in obs_ids and b != obs_val[c].get(a, a):
                    return None
    return obs_val, obs_ids


def _scaling_groups(tpl, pval):
    """(groups, canonical values): the conditions whose expressions have the same monomials, None where unexpanded."""
    # no placeholder enters a denominator, so a value 0 drops a term, a symbol renames a factor and terms that meet
    # add: equal zero patterns, symbols and vanishing sums give equal monomials and equal scaling rows
    try:
        exprs = []
        for key in ("model_lines", "obs_lines", "ver_ic"):
            for line in tpl.get(key) or []:
                num = spy.expand(spy.fraction(spy.together(spy.sympify(str(line).split("=", 1)[1])))[0])
                exprs.append(num)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    phs = set().union(*[set(v) for v in pval]) if pval else set()
    terms = []
    for ei, num in enumerate(exprs):
        for t in spy.Add.make_args(num):
            c, rest = t.as_coeff_Mul()
            pw = rest.as_powers_dict()
            if not any(str(b) in phs for b in pw):
                continue
            if not all(b.is_Symbol and e.is_Integer for b, e in pw.items()):
                return None
            terms.append(
                (ei, Fraction(int(c.p), int(c.q)) if c.is_Rational else None, {str(b): int(e) for b, e in pw.items()})
            )
    if any(c is None for _, c, _ in terms):
        return None
    # the terms of every expression without placeholders, which a renamed term may meet
    plain = set()
    for ei, num in enumerate(exprs):
        for t in spy.Add.make_args(num):
            _, rest = t.as_coeff_Mul()
            pw = rest.as_powers_dict()
            if not any(str(b) in phs for b in pw):
                plain.add((ei, tuple(sorted((str(b), str(e)) for b, e in pw.items()))))

    def pattern(val):
        return tuple((nm, ("Z" if v == 0 else "N") if isinstance(v, Fraction) else v) for nm, v in sorted(val.items()))

    meets = {}
    out, canon = {}, {}
    for c, val in enumerate(pval):
        pat = pattern(val)
        if pat not in meets:
            # the monomials of the terms after renaming, with a term at 0 left out; those that two terms share
            mon = {}
            for ti, (ei, _, pw) in enumerate(terms):
                if any(isinstance(val.get(b), Fraction) and val[b] == 0 for b in pw):
                    continue
                m = {}
                for b, e in pw.items():
                    v = val.get(b, b)
                    if isinstance(v, Fraction):
                        continue
                    m[v] = m.get(v, 0) + e
                key = (ei, tuple(sorted((b, str(e)) for b, e in m.items())))
                mon.setdefault(key, []).append(ti)
            meets[pat] = [(k, ts) for k, ts in mon.items() if len(ts) > 1 or k in plain]
        vanish = []
        for gi, (k, ts) in enumerate(meets[pat]):
            if k in plain:
                # a sum with a term free of placeholders, whose coefficient is not kept: the values set the group
                vanish.append((gi, tuple(str(val[b]) for ti in ts for b in terms[ti][2] if b in val)))
                continue
            tot = Fraction(0)
            for ti in ts:
                x = terms[ti][1]
                for b, e in terms[ti][2].items():
                    v = val.get(b)
                    if isinstance(v, Fraction):
                        x *= v**e
                tot += x
            if tot == 0:
                vanish.append((gi, ()))
        key = (pat, tuple(vanish))
        if key not in out:
            canon[key] = (
                val
                if meets[pat]
                else {nm: (Fraction(1) if isinstance(v, Fraction) and v != 0 else v) for nm, v in val.items()}
            )
        out.setdefault(key, []).append(c)
    return list(out.values()), [canon[k] for k in out]


def _reciprocals_free_of(tpl, base, nl, slots):
    """Whether no reciprocal of the tape reads a value that depends on the leaves `slots`."""
    dep = np.zeros(base + len(tpl["op"]), dtype=bool)
    for s in slots:
        dep[s] = True
    ops, a, b = list(tpl["op"]), list(tpl["a"]), list(tpl["b"])
    for i, o in enumerate(ops):
        if o == 0:
            continue
        if o == 3:
            if dep[a[i]]:
                return False
            continue
        dep[base + i] = bool(dep[a[i]] or (o in (1, 2) and dep[b[i]]))
    ic_ops = list(tpl.get("ic_op") or [])
    icd = np.zeros(nl + len(ic_ops), dtype=bool)
    for s in slots:
        icd[s] = True
    for i, o in enumerate(ic_ops):
        ai, bi = tpl["ic_a"][i], tpl["ic_b"][i]
        if o == 0:
            continue
        if o != 1 and o != 2 and icd[ai]:
            return False
        icd[nl + i] = bool(icd[ai] or (o in (1, 2) and icd[bi]))
    return True


def _template_exprs(tpl):
    """The right-hand sides, observables and initial values of the template as SymPy expressions, None where they do
    not parse."""

    def rhs(lines):
        out = {}
        for line in lines or []:
            x, e = str(line).split("=", 1)
            out[x.strip()] = spy.sympify(e)
        return out

    try:
        f = rhs(tpl.get("jet_f") or tpl.get("model_lines"))
        g = [spy.sympify(str(x)) for x in tpl.get("jet_g") or []]
        if not g:
            g = list(rhs(tpl.get("obs_lines")).values())
        ic = rhs(tpl.get("jet_ic") or tpl.get("ver_ic"))
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return None
    return {"f": f, "g": g, "ic": ic}


def _reach(f_tr, g_tr, ic_tr, val):
    """(_lie_reach of one condition, the symbols of the expressions of the condition) with the placeholders at
    `val`."""
    rhs = {x: (tr.symbols(val) or set()) for x, tr in f_tr.items()}
    syms = set().union(*rhs.values()) if rhs else set()
    dist = {}
    queue = collections.deque()
    for tr in g_tr:
        s = tr.symbols(val) or set()
        syms |= s
        for x in sorted(s):
            if x not in dist:
                dist[x] = 0
                queue.append(x)
    while queue:
        u = queue.popleft()
        for s in rhs.get(u, ()):
            if s not in dist:
                dist[s] = dist[u] + 1
                queue.append(s)
    out = dict(dist)
    for x, tr in ic_tr.items():
        s = tr.symbols(val) or set()
        syms |= s
        if x in dist:
            for y in s:
                if dist[x] < out.get(y, 10**9):
                    out[y] = dist[x]
    return out, syms


def kernel_tapes(tapes):
    """The tapes for a kernel call: a tape that holds the same kernel arrays as an earlier one names that one as
    `base` and gives only the leaves it sets."""
    tapes = list(tapes)
    if not any("leaf_slot" in t for t in tapes):
        return tapes
    first, out = {}, []
    for i, t in enumerate(tapes):
        if "leaf_slot" not in t:
            out.append(t)
            continue
        key = tuple(id(t.get(k)) for k in _KERNEL_KEYS)
        j = first.get(key)
        if j is None:
            first[key] = i
            out.append(t)
            continue
        out.append(
            {
                "base": j,
                "leaf_slot": t["leaf_slot"],
                "leaf_src": t["leaf_src"],
                "leaf_num": t["leaf_num"],
                "leaf_den": t["leaf_den"],
            }
        )
    return out
