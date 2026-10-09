"""Gap series helpers, exact scalings, field analyses per regime and the Lie-order
certificate (manuscript, Supplementary Note S3)."""

import json
import re

import numpy as np

from .. import _core, state
from ..gfp import PRIMES, invmod, mulmod, random_point
from ..par import process_map
from ..rexpr import get_symbols, sym_sort
from ..sym.fields import (
    extend_fields,
    lift_start,
    static_fields_symbolic,
    verify_distribution,
)
from ..sym.modp import eval_rational_mod_batch
from ..sym.scaling import scaling_symmetries_multi, verify_scalings


def series_embed(S, N, col_map, nz_out, p, scale=None):
    """Series rows with column c scaled by scale[c] and moved to column col_map[c] (0-based)."""
    S = np.asarray(S, dtype=np.int64)
    out = np.zeros((S.shape[0], nz_out * N), dtype=np.int64)
    for c, cm in enumerate(col_map):
        v = S[:, c * N:(c + 1) * N]
        if scale is not None:
            v = mulmod(v, scale[c], p)
        out[:, cm * N:(cm + 1) * N] = v
    return out


def ser_mul(a, b, p):
    """Product of the truncated power series `a` and `b` mod `p`."""
    N = len(a)
    out = [0] * N
    for k in range(N):
        acc = 0
        for j in range(k + 1):
            acc = (acc + int(a[j]) * int(b[k - j])) % p
        out[k] = acc
    return out


def ser_inv(b, p):
    """Inverse of the truncated power series `b` mod `p`; `b[0]` must be nonzero."""
    N = len(b)
    out = [0] * N
    out[0] = invmod(b[0], p)
    for k in range(1, N):
        acc = 0
        for j in range(1, k + 1):
            acc = (acc + int(b[j]) * out[k - j]) % p
        out[k] = (p - out[0] * acc % p) % p
    return out


def series_const(M, N):
    """Series rows of the constant matrix `M`, `N` coefficients per column, zero beyond order 0."""
    M = np.asarray(M, dtype=np.int64)
    out = np.zeros((M.shape[0], M.shape[1] * N), dtype=np.int64)
    if M.shape[0]:
        out[:, np.arange(M.shape[1]) * N] = M
    return out


def series_stack(res, sblocks, nz, N, p, cores=1):
    """Series rank of the stacked series blocks attached to a reduced kernel result."""
    S = np.vstack(sblocks).astype(np.int64)
    sr = _core.sym_series_rank(S, int(nz), int(N), p, [], at_one_below=int(res["rank"]),
                               cores=int(cores))
    res = dict(res)
    res["rank_s"] = sr["rank"]
    res["S"] = sr["S"]
    res["N"] = N
    res["at_one"] = sr.get("R") is not None
    if res["at_one"]:
        res["R"] = sr["R"]
        res["pivots"] = np.atleast_1d(sr["pivots"]).astype(np.int64)
        res["rank"] = len(res["pivots"])
    return res


def rank_of(r):
    """The series rank of a kernel result, else its rank."""
    return r["rank"] if r.get("rank_s") is None else r["rank_s"]


def rank_score(r):
    """Rank plus series rank of a kernel result, the saturation score."""
    return r["rank"] + rank_of(r)


def lie_reach_max(tapes):
    """The largest first Lie order at which a coordinate enters some tape; 0 when the tapes do not tell."""
    R = [np.atleast_1d(np.asarray(t.get("lie_reach"), dtype=np.int64)) for t in tapes
         if t.get("lie_reach") is not None]
    R = [r for r in R if r.size]
    if not R or len({r.size for r in R}) != 1:
        return 0
    M = np.vstack(R).astype(float)
    M[M < 0] = np.nan
    if np.all(np.isnan(M)):
        return 0
    with np.errstate(all="ignore"):
        first = np.nanmin(np.where(np.isnan(M), np.inf, M), axis=0)
    first = first[np.isfinite(first)]
    return int(first.max()) if first.size else 0


def _uniq(items):
    seen, out = set(), []
    for x in items:
        k = json.dumps(x, sort_keys=True, default=str)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _lines(x):
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(v) for v in x]


def _evs(x):
    if x is None:
        return []
    if isinstance(x, dict):
        return [x]
    return list(x)


def exact_scaling_set(multi, fixed):
    """Weight vectors of the scalings of the whole model, exactly checked."""
    ver_keys = ["model_lines", "obs_lines", "ver_ic", "ver_ev", "ver_t0", "ver_time"]
    ver_tapes = _uniq([{k: t[k] for k in ver_keys if k in t} for t in multi["tapes"]])

    def ev_lines(t, add):
        out = []
        for e in _evs(t.get("ver_ev")) + _evs(t.get("ver_t0")):
            if (e["method"] != "multiply") == add:
                out.append(f"{e['var']} = {e['value']}" if add else f"_ = {e['value']}")
        return out
    try:
        full = scaling_symmetries_multi(
            per_cond_model=[_lines(t.get("model_lines")) for t in ver_tapes],
            per_cond_obs=[_lines(t.get("obs_lines")) for t in ver_tapes],
            inputs=list(multi["forcings"]) if multi.get("forcings") else None,
            fixed=list(fixed) if fixed else None,
            extra_model=[_lines(t.get("ver_ic")) + ev_lines(t, True) for t in ver_tapes],
            extra_obs=[ev_lines(t, False) + ([f"_ = {t['ver_time']}"]
                                           if t.get("ver_time") else []) for t in ver_tapes])
    except Exception:
        full = None
    if full is None:
        return []
    ws = [d["vector"] for d in full.get("non_identifiable", [])]
    ws = [w for w in ws if all(re.match(r"^-?[0-9]+$", str(v)) for v in w.values())]
    if not ws:
        return []
    try:
        ok = [bool(x) for x in verify_scalings(ws, ver_tapes)]
    except Exception:
        ok = [False] * len(ws)
    if state.env("LIEDIAG"):
        print(f"[liediag] exact scalings: {len(ws)} candidate(s), verified {ok}")
    return [w for w, k in zip(ws, ok) if k]


def regime_keys(tapes):
    """Key per tape from its model and observation lines; equal keys share one regime."""
    return ["\n".join(_lines(t.get("model_lines")) + ["\x1f"] + _lines(t.get("obs_lines")))
            for t in tapes]


def _as_eqn(lines):
    out = {}
    for l in lines:
        lhs, rhs = l.split("=", 1)
        out[lhs.strip()] = rhs.strip()
    return out


def field_setup(multi, chains, cores, fixed=None, fields=None):
    """Generators of each regime with a closed form, and the joins of every chain."""
    from .core import symmetry_detection
    tapes = multi["tapes"]
    key_of = regime_keys(tapes)
    rkeys = list(dict.fromkeys(key_of))
    regime_of = [rkeys.index(k) + 1 for k in key_of]
    regimes = []
    for k in rkeys:
        t = tapes[key_of.index(k)]
        regimes.append({"f": _lines(t.get("model_lines")), "g": _lines(t.get("obs_lines"))})
    par = (fields is None and len(regimes) >= max(2, cores // 2) and cores > 1
           and not state.env("SERIALBLOCKS"))
    cores_r = 1 if par else cores

    def one(rg):
        fe = _as_eqn(rg["f"])
        ge = _as_eqn(rg["g"])
        syms = set(get_symbols(list(fe.values()) + list(ge.values())))
        fx = [s for s in (fixed or []) if s in syms and s not in fe]
        try:
            with state.options(inner=True, order_hint=None):
                r = symmetry_detection(fe, ge, reconstruct=True,
                                       verify=False, cores=cores_r, verbose=False,
                                       fixed=fx if fx else None)
        except Exception as e:
            if state.env("LIEDIAG"):
                print(f"[liediag] regime analysis failed: {e!r}")
            return []
        return [dict(d["generator"]) for d in r["symmetries"] if d.get("generator") is not None]

    if fields is not None:
        gens = [fields for _ in regimes]
    elif par:
        gens = process_map(one, regimes, min(cores, len(regimes)), on_error=lambda x: [])
    else:
        gens = [one(rg) for rg in regimes]
    joins = []
    for ci, ch in enumerate(chains, start=1):
        for j in range(1, len(ch)):
            a = regime_of[ch[j - 1]]
            b = regime_of[ch[j]]
            ev = tapes[ch[j]].get("ver_ev")
            if a != b or _evs(ev):
                joins.append({"from": a - 1, "to": b - 1, "ev": _evs(ev), "chain": ci})
    states = list(dict.fromkeys(l.split("=", 1)[0].strip() for rg in regimes for l in rg["f"]))
    return {"regimes": regimes, "gens": gens, "regime_of": regime_of, "joins": joins,
            "states": states, "tapes": tapes}


def own_fields(dirs):
    """Closed forms of the directions as fields; a scaling with weight w moves z by w*z."""
    if not dirs:
        return None
    out = []
    for d in dirs:
        if not d.get("closed_form") or not d.get("vector"):
            return None
        v = {k: str(x) for k, x in d["vector"].items()}
        if d.get("type") == "scaling":
            v = {k: f"({x})*{k}" for k, x in v.items()}
        out.append(v)
    return out


_CSUFFIX = re.compile(r"\|c[0-9]+$")


def field_check(setup, V, znames, leaf_names, point, chains, joint):
    """Certificate that the kernel vectors V lie in the kernel of every order."""
    P = PRIMES[0]
    leaf_val = {nm: int(point[i]) % P for i, nm in enumerate(leaf_names)}
    V = np.asarray(V, dtype=np.int64).reshape(len(znames), -1)
    lifts = []
    lift_of = [None] * len(chains)
    for ci, ch in enumerate(chains):
        t1 = setup["tapes"][ch[0]]
        t0 = [] if joint is None else _evs(t1.get("ver_t0"))
        if joint is None:
            base = dict(leaf_val)
            state_col = None
        else:
            if ch[0] not in joint["conds"]:
                continue
            mi = joint["conds"].index(ch[0])
            sol = joint["solve"](point, P, joint["conds"][mi])
            if sol is None:
                raise RuntimeError("no resting state at the evaluation point")
            xs = sol["sol"]["val_by"]
            base = dict(leaf_val)
            base.update({k: int(v) % P for k, v in xs.items()})
            state_col = {f"{k}|c{mi + 1}" for k in xs}
        vecs = []
        for k in range(V.shape[1]):
            v = V[:, k] % P
            d = {}
            for j in np.nonzero(v)[0]:
                nm = znames[j]
                if state_col is not None and nm in state_col:
                    st = _CSUFFIX.sub("", nm)
                    d[st] = int(v[j]) * int(base[st]) % P
                elif not _CSUFFIX.search(nm):
                    d[nm] = int(v[j])
            lf = lift_start(_lines(t1.get("ver_ic")), t0, base, d, P, joint=joint is not None)
            if lf is None:
                raise RuntimeError("the lift has a pole")
            vecs.append(lf["vector"])
        lf0 = lift_start(_lines(t1.get("ver_ic")), t0, base, {}, P, joint=joint is not None)
        times = [t["ver_time"] for t in (setup["tapes"][i] for i in ch) if t.get("ver_time")]
        lift_of[ci] = len(lifts)
        lifts.append({"regime": setup["regime_of"][ch[0]] - 1, "point": lf0["point"],
                      "vectors": vecs, "times": times})
    joins = []
    for e in setup["joins"]:
        e = dict(e)
        li = lift_of[e["chain"] - 1]
        e["lift"] = li
        del e["chain"]
        joins.append(e)
    return verify_distribution(setup["regimes"], setup["gens"], joins, lifts, P)


def field_proof(multi, N, znames, leaf_names, point, chains, joint, cores, setup=None, fixed=None):
    """The field certificate of the kernel vectors `N`, from `setup` or a fresh field setup."""
    if setup is None:
        setup = field_setup(multi, chains, cores, fixed=fixed)
    return field_check(setup, N, znames, leaf_names, point, chains, joint)


_IDENT = re.compile(r"[A-Za-z._][A-Za-z0-9._]*")


def field_directions(setup, N, n_res, scal_rows, znames, leaf_names, point, chains, joint,
                     partial=False):
    """Closed forms of the residual kernel from the fields of the first regime."""
    P = PRIMES[0]
    reg1 = setup["regime_of"][chains[0][0]]
    stset = set(setup["states"])
    p_cols = [i for i, z in enumerate(znames) if not _CSUFFIX.search(z) and z not in stset]
    p_names = [znames[i] for i in p_cols]
    li = {nm: i for i, nm in enumerate(leaf_names)}
    vals = [int(point[li[n]]) % P for n in p_names]
    t1 = setup["tapes"][chains[0][0]]
    groups = [_evs(t1.get("ver_t0"))] + [_evs(setup["tapes"][i].get("ver_ev")) for i in chains[0][1:]]
    groups = [g for g in groups if g]
    ext = extend_fields(setup["gens"][reg1 - 1], _lines(t1.get("ver_ic")), groups,
                          setup["states"])
    cand = []
    for g in ext:
        th = {k: g[k] for k in g if k in p_names}
        if not th:
            continue
        syms = {m for v in th.values() for m in _IDENT.findall(str(v))}
        if syms & stset:
            continue
        try:
            ev = eval_rational_mod_batch([str(v) for v in th.values()], p_names, [vals], P)
            ev = [int(x) for x in np.asarray(ev).ravel()]
        except Exception:
            ev = None
        if ev is None or any(x < 0 for x in ev):
            continue
        x = np.zeros(len(p_names), dtype=np.int64)
        for k, e in zip(th.keys(), ev):
            x[p_names.index(k)] = e
        cand.append({"expr": th, "val": x})
    if not cand:
        return None
    N = np.asarray(N, dtype=np.int64)
    Np = N[p_cols, :] % P
    scal_rows = np.asarray(scal_rows, dtype=np.int64).reshape(-1, len(znames))
    Sp = scal_rows[:, p_cols] % P if scal_rows.shape[0] else np.zeros((0, len(p_cols)), dtype=np.int64)

    def matrix_rank(M):
        if M.shape[0] == 0:
            return 0
        return int(_core.sym_rref_mod(M.astype(np.float64), P)["rank"])
    basis = Sp
    chosen = []
    r_np = matrix_rank(Np.T)
    for c0 in cand:
        in_ker = matrix_rank(np.vstack([Np.T, c0["val"]])) == r_np
        if not in_ker or matrix_rank(np.vstack([basis, c0["val"]])) == matrix_rank(basis):
            continue
        basis = np.vstack([basis, c0["val"]])
        chosen.append(c0)
        if len(chosen) == n_res:
            break
    if state.env("LIEDIAG"):
        print(f"[liediag] global symmetries: {len(ext)} field(s), {len(cand)} candidate(s), "
              f"{len(chosen)} chosen of {n_res}")
    if not chosen or (len(chosen) < n_res and not partial):
        return None
    Vw = np.zeros((N.shape[0], len(chosen)), dtype=np.int64)
    for j, c0 in enumerate(chosen):
        cc = _core.sym_solve_mod(Np, c0["val"], P)
        if cc is None:
            raise RuntimeError("a direction is not in the kernel")
        acc = np.zeros(N.shape[0], dtype=np.int64)
        for k, ck in enumerate(np.atleast_1d(cc)):
            acc = (acc + mulmod(N[:, k], int(ck), P)) % P
        Vw[:, j] = acc
    chk = field_check(setup, Vw, znames, leaf_names, point, chains, joint)
    if not chk.get("ok"):
        return None
    out = [{"support": sym_sort(list(c0["expr"])), "vector": c0["expr"], "type": "general",
            "closed_form": True, "route": "global symmetry", "certified": True} for c0 in chosen]
    if len(chosen) < n_res:
        return {"partial": True, "dirs": out, "vectors": Vw}
    return out


def static_fields(multi, models, conds, chains, p_names, leaf_names, point, forcings=None,
                  zeros=None):
    """Fields of the static subsystems of the resting conditions with values at point."""
    tp = multi["tapes"]
    cl = []
    for i, ci in enumerate(conds):
        idx = chains[i]
        cl.append({"rest": _lines(models[ci]),
                   "zero": [str(z) for z in zeros[i]] if zeros is not None and len(zeros) > i and zeros[i] else [],
                   "segs": [{"f": _lines(tp[t].get("model_lines")), "g": _lines(tp[t].get("obs_lines")),
                             "ev": _evs(tp[t].get("ver_ev"))} for t in idx],
                   "t0": _evs(tp[idx[0]].get("ver_t0"))})
    try:
        sf = static_fields_symbolic(cl, list(p_names), list(forcings) if forcings else None)
    except Exception:
        sf = None
    if state.env("STATICDIAG"):
        print(f"[staticdiag] {sf}")
    if sf is None or not sf.get("fields"):
        return None
    P = PRIMES[0]
    li = {nm: i for i, nm in enumerate(leaf_names)}
    vals = [int(point[li[n]]) % P for n in p_names]
    out = []
    for f in sf["fields"]:
        try:
            ev = eval_rational_mod_batch([str(v) for v in f.values()], list(p_names), [vals], P)
            ev = [int(x) for x in np.asarray(ev).ravel()]
        except Exception:
            ev = None
        if ev is None or any(x < 0 for x in ev):
            continue
        x = np.zeros(len(p_names), dtype=np.int64)
        for k, e in zip(f.keys(), ev):
            x[list(p_names).index(k)] = e
        lx = np.array([0 if x[i] == 0 else int(x[i]) * invmod(vals[i], P) % P
                       for i in range(len(x))], dtype=np.int64)
        out.append({"support": sym_sort(list(f)), "vector": dict(f), "val": x, "lval": lx,
                    "type": "general", "closed_form": True, "route": "static subsystem",
                    "certified": True})
    return out or None


def _flat_at(r):
    r = np.asarray(r)
    d = np.nonzero(np.diff(r) == 0)[0]
    return None if not len(d) else int(d[0])


def _reach_at(r, rho):
    i = np.nonzero(np.asarray(r) >= rho)[0]
    return None if not len(i) else int(i[0])


def lie_certificate(segs, seeds, n_leaves, n_states, z_slots, p, cap, map_fn=None, threads=1,
                    start=None):
    """Certified Lie order per block: None where the generic rank is not reached. The
    search of block b begins at order `start[b]` (at most 16), then 16, 32, ..."""
    none = np.zeros((0, 0), dtype=np.int64)

    def one(b):
        seed = np.asarray(seeds[b], dtype=np.int64)
        sb = segs[b]
        n_seg = len(sb)
        base = _core.sym_segment_ranks(sb, n_leaves, n_states, z_slots, seed, p, 0, none)
        if not base.get("ok"):
            return None
        Q = np.asarray(base["start"], dtype=np.int64).reshape(n_seg, n_states)
        draw = {"n": 0}

        def rand_in(q, V):
            y = np.array(q, dtype=np.int64)
            if V.shape[0] == 0:
                return y
            draw["n"] += 1
            r = random_point(V.shape[0], 7919 * (b + 1) + draw["n"])
            for i in range(V.shape[0]):
                y = (y + mulmod(V[i], r[i], p)) % p
            return y
        st_g = np.zeros((n_seg, n_states), dtype=np.int64)
        V = np.zeros((0, n_states), dtype=np.int64)
        for j in range(n_seg):
            sg = sb[j]
            if j > 0 and sg.get("ev_var_idx") is not None and len(np.atleast_1d(sg["ev_var_idx"])):
                evi = np.atleast_1d(sg["ev_var_idx"]).astype(np.int64)
                evm = np.atleast_1d(sg["ev_method"]).astype(np.int64)
                if V.shape[0]:
                    V[:, evi[evm == 0]] = 0
                mult = evi[evm == 2]
                if len(mult):
                    V = np.vstack([V, np.eye(n_states, dtype=np.int64)[mult]])
            while True:
                y = rand_in(Q[j], V)
                Y = Q[:j + 1].copy()
                Y[j] = y
                fz = _core.sym_segment_ranks(sb[:j + 1], n_leaves, n_states, z_slots, seed, p, 0, Y)
                if not fz.get("ok"):
                    return None
                field = np.asarray(fz["field"], dtype=np.int64).reshape(j + 1, n_states)
                rr = _core.sym_rref_mod(np.vstack([V, field[j]]).astype(np.float64), p)
                if int(rr["rank"]) == V.shape[0]:
                    break
                V = np.asarray(rr["R"], dtype=np.int64).reshape(int(rr["rank"]), n_states)
            st_g[j] = y
        Nt = 16 if start is None or start[b] is None else int(min(16, max(2, start[b])))
        while True:
            g = _core.sym_segment_ranks(sb, n_leaves, n_states, z_slots, seed, p, Nt, st_g, threads)
            q = _core.sym_segment_ranks(sb, n_leaves, n_states, z_slots, seed, p, Nt, none, threads)
            if not g.get("ok") or not q.get("ok"):
                return None
            gr = np.asarray(g["ranks"]).reshape(n_seg, Nt + 1)
            qr = np.asarray(q["ranks"]).reshape(n_seg, Nt + 1)
            g_n = [_flat_at(gr[s]) for s in range(n_seg)]
            ordv = [None if g_n[s] is None else _reach_at(qr[s], gr[s, g_n[s] + 1])
                    for s in range(n_seg)]
            if state.env("LIEDIAG"):
                print(f"[liediag] certificate block {b + 1} at order {Nt}")
            if all(o is not None for o in ordv):
                return max(ordv)
            if Nt >= cap or (all(x is not None for x in g_n) and Nt >= 4 * max(g_n) + 16):
                return None
            Nt = 16 if Nt < 16 else int(min(cap, 2 * Nt))

    idx = list(range(len(segs)))
    return map_fn(one, idx) if map_fn is not None else [one(b) for b in idx]
