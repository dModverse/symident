"""The conditions that change the rank: chosen at a random point, with the directions proven on the others."""

from fractions import Fraction

import numpy as np
import sympy as spy

from ... import _core, session, state
from ...errors import SYMPY_FAILS, swallow, sympy_fail
from ...gfp import PRIMES, random_point
from ...par import process_map
from ...sym.parse import model_sympify
from ...sym.shared import kernel_tapes
from ..closedform import eval_modq
from ..conditions import scaling_component
from ..fields import exact_direction, exact_scaling_set, field_setup, lie_reach_max, restricted_setup
from .context import ints, ok, tape_fields

# conditions below which every condition is analyzed
MIN_CONDITIONS = 24
# rounds of conditions added where a direction fails on a condition left out
_ROUNDS = 3
# conditions of a direction proven one at a time before the rest of them are added
_SINGLE = 16
# directions past which the lifts run in forked workers
_PAR_LIFTS = 16


def resting(multi, implicit_steady_state):
    """Whether every condition starts at rest (equilibrate)."""
    return bool(implicit_steady_state and multi.get("joint_steady_state"))


def eligible(multi, n_gaps, implicit_steady_state):
    """Whether the conditions are chosen before the analysis: many conditions of one segment each, without unknown
    inputs or recast atoms, each with the Lie orders at which the coordinates reach the output; at rest without
    conserved totals and without leaves of non-rational terms."""
    tapes = multi.get("tapes") or []
    if len(tapes) < MIN_CONDITIONS or state.get("inner") or state.env("NOSELECT") or n_gaps:
        return False
    if multi.get("equilibrate") and not resting(multi, implicit_steady_state):
        return False
    if resting(multi, implicit_steady_state) and (multi.get("held_state_params") or multi.get("rest_aux") or
                                                  multi.get("rest_inverted")):
        return False
    if multi.get("nuisance") or multi.get("nuisance_order") is not None or multi.get("power_recast") or \
            multi.get("exp_atoms"):
        return False
    nz = len(multi.get("z_slots") or [])
    return all(t.get("ic_seed") is None and len(t.get("lie_reach") or []) == nz for t in tapes)


def _reach(t):
    """The coordinates that reach the output of the tape `t` at some Lie order."""
    return np.asarray(t["lie_reach"], dtype=np.int64) >= 0


def taking_order(tapes):
    """(order, first): the conditions one of each structure in turn, a structure being the set of coordinates that
    reach the output; `first` marks the first condition in the order that reaches a coordinate."""
    groups = {}
    for i, t in enumerate(tapes):
        groups.setdefault(_reach(t).tobytes(), []).append(i)
    lists = list(groups.values())
    order = [g[k] for k in range(max(len(g) for g in lists)) for g in lists if k < len(g)]
    first = np.zeros(len(tapes), dtype=np.int64)
    seen = np.zeros(len(tapes[0]["lie_reach"]), dtype=bool)
    for i in order:
        r = _reach(tapes[i])
        if np.any(r & ~seen):
            first[i] = 1
            seen |= r
    return order, first, len(lists)


def choose(multi, cores, nt=None, kept=()):
    """The conditions whose rows at a random point raise the rank of the rows of the conditions before them, and the
    first reader of each coordinate (sym_obs_cover), the conditions `kept` first; None where the kernel fails at
    every point tried."""
    tapes = multi["tapes"]
    order, first, n_struct = taking_order(tapes)
    if kept:
        order = list(kept) + [c for c in order if c not in kept]
        first[list(kept)] = 1
    memo = {}
    kt = kernel_tapes([tape_fields(t, memo) for t in tapes])
    nz = len(multi["z_slots"])
    nt = lie_reach_max(tapes) + 2 if nt is None else int(nt)
    for k in range(4):
        pt = random_point(int(multi["n_leaves"]), seed=session.current().seed("condition selection", k) % (1 << 31))
        r = _core.sym_obs_cover(kt, int(multi["n_leaves"]), int(multi["n_states"]),
                                np.asarray(multi["z_slots"], dtype=np.int64), pt, PRIMES[0], nt, nz, cores,
                                np.asarray(order, dtype=np.int64), first)
        if ok(r):
            return {"kept": sorted(ints(r["conditions"])), "rank": int(r["rank"]), "lie_order": nt,
                    "structures": n_struct, "taken": int(r["taken"])}
    return None


def choose_resting(an, order, first, n_struct):
    """choose() for conditions at rest, from the Analysis `an` of all of them: the rows of each condition on the
    shared coordinates once the resting states of the condition, solved at the point itself, are eliminated; None
    where a resting state is not solved."""
    sys, run = an.sys, an.run
    if sys.equil_conds != list(range(len(run.tapes))):
        return None
    if sys.chained:
        sys.unchain()
    P = PRIMES[0]
    nt = lie_reach_max(run.multi["tapes"]) + 2
    for k in range(4):
        pt = random_point(run.n_leaves, seed=session.current().seed("condition selection", k) % (1 << 31))
        if sys.fwd_on:
            pt = sys.fwd_seed(pt, P)
        if pt is None:
            continue
        rows, owner = [], []
        for c in order:
            sc0 = sys.solve_direct(pt, P, c)
            if sc0 is None:
                rows = None
                break
            loc = sys.cond_local(c, sys.cond_obs(c, sc0["ptc"], P, nt, 0), sc0, P, scale=len(sys.recast_rel) > 0)
            if loc is None:
                rows = None
                break
            Q = sys._project([loc], P, run.cores)[1]
            for q in Q:
                rows += [np.asarray(r, dtype=np.int64) for r in q]
                owner += [c] * len(q)
        if rows is None:
            continue
        nsh = len(sys.shared_idx)
        gain = {}
        rank = 0
        if rows:
            r = _core.sym_independent_rows(np.vstack(rows).reshape(-1, nsh).astype(np.float64), P, run.cores)
            rank = int(r["rank"])
            for i in ints(r["rows"]):
                gain[owner[i]] = gain.get(owner[i], 0) + 1
        kept = sorted({c for c in order if gain.get(c) or first[c]})
        return {"kept": kept, "rank": rank, "lie_order": nt, "structures": n_struct, "taken": len(order)}
    return None


def restrict(multi, kept):
    """`multi` with the tapes of the conditions `kept`; the scaling groups of a shared tape are kept whole, so that an
    exact scaling holds for every condition."""
    out = dict(multi, tapes=[multi["tapes"][i] for i in kept])
    if multi.get("scaling_groups") and not multi.get("scaling_tapes"):
        pos = {c: j for j, c in enumerate(kept)}
        out["scaling_groups"] = [g for g in ([pos[c] for c in grp if c in pos] for grp in multi["scaling_groups"]) if g]
    return out


def _field(d):
    """The generator {coordinate: component} of the closed form of `d`."""
    vec = {str(k): str(v) for k, v in d["vector"].items()}
    if d.get("type") == "scaling":
        vec = {k: scaling_component(w, k) for k, w in vec.items()}
    return {k: v for k, v in vec.items() if v.strip() not in ("0", "")}


def _weights(d):
    """The weights of `d` where it is a scaling, the component of each coordinate a rational multiple of it; None
    otherwise."""
    try:
        if d.get("type") == "scaling":
            w = {k: Fraction(str(x)) for k, x in d["vector"].items()}
        else:
            w = {}
            for k, x in d["vector"].items():
                q = spy.cancel(model_sympify(str(x)) / spy.Symbol(k))
                if not q.is_Rational:
                    return None
                w[k] = Fraction(int(q.p), int(q.q))
    except (TypeError, ValueError, ZeroDivisionError, *SYMPY_FAILS):
        return None
    w = {k: x for k, x in w.items() if x != 0}
    return w or None


def _proportional_scaling(d, ws, coords):
    """Whether `d` is a scaling proportional to one of the exact scalings `ws` of the whole model on `coords`."""
    v = _weights(d)
    if v is None:
        return False
    k0 = next(iter(v))
    for w in ws:
        w = {k: Fraction(str(x)) for k, x in w.items() if k in coords and Fraction(str(x)) != 0}
        if not w.get(k0):
            continue
        c = v[k0] / w[k0]
        if set(v) == set(w) and all(v[k] == c * w[k] for k in v):
            return True
    return False


def _annihilated(multi, conds, fields, nt, cores):
    """Per condition of `conds`, per field, whether the field annihilates every row of the condition up to Lie order
    `nt` at a random point; None for a field that does not evaluate there."""
    znames = list(multi["znames"])
    leaves = list(multi["leaf_names"])
    P = PRIMES[1]
    pt = random_point(int(multi["n_leaves"]), seed=session.current().seed("selection check") % (1 << 31))
    env = {nm: int(v) for nm, v in zip(leaves, pt)}
    V = np.zeros((len(znames), len(fields)), dtype=np.int64)
    col = {z: i for i, z in enumerate(znames)}
    for j, f in enumerate(fields):
        for k, e in f.items():
            x = eval_modq(e, env, P)
            if x is None or k not in col:
                return None
            V[col[k], j] = x % P
    memo, out = {}, []
    for c in conds:
        t = kernel_tapes([tape_fields(multi["tapes"][c], memo)])
        o = _core.sym_obs_annihilates(t, int(multi["n_leaves"]), int(multi["n_states"]),
                                      np.asarray(multi["z_slots"], dtype=np.int64), pt, P, int(nt), V, cores)
        out.append(None if not ok(o) else [bool(x) for x in np.atleast_1d(o["zero"])])
    return out


def cover(multi, result, kept, cores, rest=None):
    """The directions of `result`, found on the conditions `kept`, on every other condition of `multi`.

    A direction is a symmetry of a condition whose output no coordinate of the support reaches, of every condition
    where the direction is proportional to an exact scaling of the whole model, and where the lift of the closed form
    is proven on the condition.
    {"ok": True, "how"} where every direction holds on every condition; {"ok": False, "add"} with the conditions
    where a direction is not shown; None where a direction has no closed form or no proof on `kept`. With `rest`, the
    conditions start at rest, and `rest(conds, d)` proves the output identities of `d` at the resting states of the
    conditions `conds` where a direction has no lift."""
    dropped = [c for c in range(len(multi["tapes"])) if c not in set(kept)]
    how = {"support": 0, "scaling": 0, "lift": 0, "rest": 0}
    dirs = result.get("non_identifiable") or []
    if int(result["rank"]) == int(result["dim"]) or not dropped:
        return {"ok": True, "how": how}
    if any(not d.get("vector") or not d.get("exact") for d in dirs):
        return None
    znames = list(multi["znames"])
    col = {z: i for i, z in enumerate(znames)}
    fields = [_field(d) for d in dirs]
    if any(not f or any(k not in col for k in f) for f in fields):
        return None
    reach = np.vstack([_reach(multi["tapes"][c]) for c in dropped])
    fixed = [x for x in multi["leaf_names"] if x not in set(znames)]
    ws = None
    pending = {}
    for j, (d, f) in enumerate(zip(dirs, fields)):
        hit = np.any(reach[:, [col[k] for k in f]], axis=1)
        how["support"] += int(np.sum(~hit))
        if not np.any(hit):
            continue
        if _weights(d) is not None:
            if ws is None:
                ws = exact_scaling_set(multi, fixed)
            if _proportional_scaling(d, ws, set(znames) - set(multi.get("z_state_names") or ())):
                how["scaling"] += int(np.sum(hit))
                continue
        pending[j] = [dropped[i] for i in np.nonzero(hit)[0]]
    if not pending:
        return {"ok": True, "how": how}
    lifted = _lifted(multi, pending, dirs, cores, rest is not None)
    left = {j: cs for j, cs in pending.items() if j not in lifted}
    how["lift"] += sum(len(cs) for j, cs in pending.items() if j in lifted)
    if not left:
        return {"ok": True, "how": how}
    if rest is not None:
        add = set()
        for j, cs in left.items():
            bad = _bisect(cs, lambda part: _holds(rest(part, dirs[j])))
            how["rest"] += len(cs) - len(bad)
            add |= set(bad)
        return {"ok": not add, "add": sorted(add), "how": how}
    # the conditions where a direction without a lift on all of them leaves the kernel at a random point; a lift per
    # condition for the others
    conds = sorted(set().union(*left.values()))
    checked = _annihilated(multi, conds, [fields[j] for j in left], int(result.get("lie_order_used") or 0), cores)
    if checked is None:
        return None
    add, single = set(), {}
    for ci, c in enumerate(conds):
        for k, j in enumerate(left):
            if c not in left[j]:
                continue
            if checked[ci] is None or not checked[ci][k]:
                add.add(c)
            else:
                single.setdefault(j, []).append(c)
    for j, cs in single.items():
        for c in cs[:_SINGLE]:
            if _lifted(multi, {j: [c]}, dirs, cores):
                how["lift"] += 1
            else:
                add.add(c)
        add |= set(cs[_SINGLE:])
    if add:
        return {"ok": False, "add": sorted(add), "how": how}
    return {"ok": True, "how": how}


def _holds(out):
    """Whether a proof result is a success."""
    return isinstance(out, dict) and out.get("ok") is True


def _bisect(conds, proven, cap=_SINGLE):
    """The conditions of `conds` where `proven(part)` fails on the singleton, by halving the parts that fail; past
    `cap` failing parts, all conditions of the parts left."""
    bad, todo = [], [list(conds)]
    while todo:
        part = todo.pop()
        if proven(part):
            continue
        if len(part) == 1:
            bad += part
            continue
        if len(todo) + len(bad) >= cap:
            bad += part
            continue
        todo += [part[:len(part) // 2], part[len(part) // 2:]]
    return bad


def _lifted(multi, pending, dirs, cores, at_rest=False):
    """The directions of `pending` ({direction: conditions}) whose lift of the closed form is proven on all of their
    conditions at once, `at_rest` where the conditions start at rest."""
    from .proof import representatives
    out = set()
    conds = sorted({c for cs in pending.values() for c in cs})
    if not conds:
        return out
    sub = dict(multi, tapes=[multi["tapes"][c] for c in conds])
    pos = {c: i for i, c in enumerate(conds)}
    try:
        setup = field_setup(sub, [[i] for i in range(len(conds))], cores, fields=[{}])
    except Exception as exc:
        swallow(exc, "selection proof")
        return out
    try:
        light = restricted_setup(setup, [[i] for i in range(len(conds))], at_rest)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return out

    def attempt(j, known):
        d = dirs[j]
        chains = [[pos[c]] for c in pending[j]]
        # the lift proven on the kept conditions first
        proven = [dict(d["lift_form"], **d["state_components"])] if d.get("lift_form") and \
            d.get("state_components") else []
        for vec in proven + representatives(d):
            try:
                ex = exact_direction(light, vec, chains, at_rest, known=known)
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                continue
            if ex.get("ok") is True:
                return ex["fields"]
        return None

    known, todo = [], list(pending)
    # a pass over many directions runs in forked workers with the fields proven before it
    if len(todo) >= _PAR_LIFTS and cores > 1:
        got = process_map(lambda j: attempt(j, []), todo, cores, on_error=lambda j: None)
        for j, f in zip(todo, got):
            if f is not None:
                out.add(j)
                known.append(f)
        todo = [j for j in todo if j not in out]
    for j in todo:
        f = attempt(j, known)
        if f is not None:
            out.add(j)
            known.append(f)
    return out
