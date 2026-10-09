"""The forward route: directions fitted per prime with the resting state solved forward for turnover rates."""

import re
import time

import numpy as np

from .. import _core, state
from ..gfp import invmod, mono_table, mulmod, null_basis_rows, null_residues, phys_rref, random_point, rref_modp
from ..rexpr import sym_sort
from ..sym.linalg import sym_rat_recon_big
from ..sym.restforward import solve_forward_modular
from .closedform import RP8, _strip_name, eval_modq, expired, poly_string, simplify


def _nonzero_rows(B, p):
    keep = np.any(B % p != 0, axis=1)
    return B[keep]


def _reduce_rows(B, S, p):
    if S.shape[0] == 0:
        return B % p
    Sr = rref_modp(S, p)
    B = B.copy()
    for j, pc in enumerate(Sr["piv"]):
        fac = B[:, pc] % p
        for i in range(B.shape[0]):
            if fac[i]:
                B[i] = (B[i] - mulmod(Sr["R"][j], fac[i], p)) % p
    return B


class _Tangents:
    """Scaling tangents at a point (state columns weighted 1)."""

    def __init__(self, scaling, znames, is_state_col):
        self.Wn = [list(s["vector"].keys()) for s in scaling]
        self.Wv = [[str(v) for v in s["vector"].values()] for s in scaling]
        self.znames = znames
        self.zidx = {z: i for i, z in enumerate(znames)}
        self.is_state = is_state_col

    def __call__(self, zvals, p):
        nz = len(self.znames)
        n_s = len(self.Wn)
        env = {z: int(v) for z, v in zip(self.znames, zvals)}
        M = np.zeros((n_s, nz), dtype=np.int64)
        drop = [False] * n_s
        for j in range(n_s):
            for t, nm in enumerate(self.Wn[j]):
                cc = self.zidx.get(nm)
                if cc is None:
                    continue
                try:
                    w = float(self.Wv[j][t])
                    w = int(w)
                except ValueError:
                    wv = eval_modq(self.Wv[j][t], env, p)
                    if wv is None:
                        drop[j] = True
                        break
                    w = wv
                zc = 1 if self.is_state[cc] else int(zvals[cc]) % p
                M[j, cc] = (w % p) * zc % p
        return M[[not d for d in drop]]


def _free_cols(rp, nz):
    piv = set(int(x) for x in np.atleast_1d(rp["pivots"]))
    return [c for c in range(nz) if c not in piv]


def _basis(rp, nz, p):
    fc = _free_cols(rp, nz)
    if not fc:
        return np.zeros((0, nz), dtype=np.int64)
    return null_basis_rows(rp, fc, p)


def _fit_entry(primes, bankU_rows, reli_cols, mons, vals_of, vars_):
    """Rational fit of one entry at every prime and the lift; None without a fit."""
    n_mon = mons.shape[0]
    ref_free = None
    coef_res = np.zeros((2 * n_mon, len(primes)), dtype=np.int64)
    need = bankU_rows.shape[0]
    for jp, pp in enumerate(primes):
        s_u = np.asarray(bankU_rows[:need][:, reli_cols], dtype=np.int64).reshape(need, len(reli_cols))
        rv = np.asarray(vals_of(jp, need), dtype=np.int64)
        fit = _core.sym_fit_rational(s_u, mons, rv, pp)
        if fit.get("status") != "ok":
            return None
        raw = np.asarray(fit["coeffs"], dtype=np.int64)
        if ref_free is None:
            ref_free = int(fit["free_col"])
        dn = int(raw[ref_free]) % pp
        if dn == 0:
            return None
        coef_res[:, jp] = mulmod(raw, invmod(dn, pp), pp)
    try:
        rec = sym_rat_recon_big(coef_res, [int(p) for p in primes])
    except Exception:
        return None
    if rec is None or any(d == "0" for d in rec["den"]):
        return None
    num_str = poly_string(rec["num"][:n_mon], rec["den"][:n_mon], mons, vars_)
    den_str = poly_string(rec["num"][n_mon:], rec["den"][n_mon:], mons, vars_)
    return num_str if den_str == "1" else f"({num_str})/({den_str})"


def perprime_forward(f, sc, kcall, kcall_fwd, znames, z_slots, leaf_names, nz, scaling,
                     state_col_names, param_names, recast, spy_on, ctrl, phys_cols, models,
                     real_state_names, solve_param_names, solve_held, ac_in=None, list_anchors=False):
    """Forward-sampling reconstruction of one residual direction over solved resting states."""
    n_prime_fwd = max(2, min(3, len(RP8) - 1))
    primes = RP8[:n_prime_fwd]
    qv = RP8[len(primes)]
    P1 = primes[0]
    track = {"dir_support": None, "pool_n": sc["pool_next"]}

    def fb(reason):
        supp = track["dir_support"]
        if supp is None:
            v = null_residues(sc["ref"], f, P1)
            supp = sym_sort([znames[i] for i in np.nonzero(v)[0]])
        return {"support": supp, "type": "general", "closed_form": False, "reason": reason}

    if kcall_fwd is None or expired(ctrl):
        return [] if list_anchors else fb("forward path unavailable")
    rdiag = state.env("ROBUSTDIAG")
    npt = len(sc["point0"])

    def draw(n):
        u = sc["pool"](track["pool_n"] + np.arange(n))
        track["pool_n"] += n
        return u

    is_state = np.zeros(nz, dtype=bool)
    for i, z in enumerate(znames):
        if z in state_col_names:
            is_state[i] = True
    phys_set = list(range(nz)) if phys_cols is None else [int(c) for c in phys_cols]
    tangents_at = _Tangents(scaling, znames, is_state)
    zs = np.asarray(z_slots, dtype=np.int64)

    def extract(rp, zvals, p, ac):
        B = _basis(rp, nz, p)
        Bred = _nonzero_rows(_reduce_rows(B, tangents_at(zvals, p), p), p)
        if Bred.shape[0] == 0:
            return None
        rr = phys_rref(Bred, phys_set, nz, p)
        piv = list(rr["piv"])
        if ac in piv:
            return rr["R"][piv.index(ac)] % p
        if ac >= nz:
            return None
        v = np.zeros(nz, dtype=np.int64)
        v[ac] = 1
        for i, pc in enumerate(piv):
            v[pc] = (p - rr["R"][i, ac]) % p
        return v

    def kx_f(pt, p, ac, sr):
        rp = kcall_fwd(pt, p, sc["nt_used"], sr)
        if not rp.get("ok") or rp["rank"] != sc["rank"]:
            return None
        return extract(rp, np.asarray(pt)[zs], p, ac)

    ptb_b = None
    for att in range(1, 201):
        if att == 1 and len(sc["point0"]) >= npt:
            cand = np.asarray(sc["point0"], dtype=np.int64)[:npt]
        else:
            cand = draw(npt)
        r = kcall(cand, P1, sc["nt_used"])
        if r.get("ok") and r["rank"] == sc["rank"]:
            ptb_b = cand
            break
    if ptb_b is None:
        return [] if list_anchors else fb("forward: no backward base for support")
    rp0 = kcall(ptb_b, P1, sc["nt_used"])
    B0 = _basis(rp0, nz, P1)
    Bred0 = _nonzero_rows(_reduce_rows(B0, tangents_at(ptb_b[zs], P1), P1), P1)
    if Bred0.shape[0] == 0:
        return [] if list_anchors else fb("forward: empty residual")
    rr0 = phys_rref(Bred0, phys_set, nz, P1)
    piv0 = [int(x) for x in rr0["piv"]]
    anchors = [c for c in piv0 if c in phys_set]
    if list_anchors:
        return anchors
    if not anchors:
        return fb("forward: no physical residual anchor")
    ac = int(ac_in) if ac_in is not None else anchors[0]
    if ac not in piv0:
        return fb("forward: requested anchor is not a residual pivot")
    arow = piv0.index(ac)
    v0 = mulmod(rr0["R"][arow], invmod(rr0["R"][arow, ac] % P1, P1), P1)
    nzv = [int(i) for i in np.nonzero(v0 % P1)[0]]
    track["dir_support"] = sym_sort(list(dict.fromkeys(_strip_name(znames[i]) for i in nzv)))
    supp_names = list(dict.fromkeys(_strip_name(znames[i]) for i in [ac] + [c for c in nzv if c != ac]))

    keep_free = list(dict.fromkeys(supp_names + [l for l in leaf_names if re.match(r"^_E_|^_L_", l)]))
    tsv = {s: ((i + 1) * 7919 + 11) % (P1 - 1) + 1 for i, s in enumerate(real_state_names)}
    tpv = {s: ((i + 1) * 104729 + 3) % (P1 - 1) + 1 for i, s in enumerate(solve_param_names)}
    try:
        ts = solve_forward_modular(models[0], list(real_state_names), list(solve_param_names), tsv,
                                    tpv, P1, forcings=list(solve_held) if solve_held else None,
                                    keep_free=keep_free)
    except Exception:
        ts = None
    if ts is None or not ts.get("ok"):
        return fb("forward: no turnover transversal; " + ("solve error" if ts is None else str(ts.get("why"))))
    solve_rates = SolveRates(ts["solve_rates"], ts.get("state_solve"))

    base0 = None
    for _att in range(80):
        cand = draw(npt)
        if all(kx_f(cand, pp, ac, solve_rates) is not None for pp in primes):
            base0 = cand
            break
    if base0 is None:
        return fb("forward: no base valid at all primes")

    li = {nm: i for i, nm in enumerate(leaf_names)}
    solved = {li[r] for r in solve_rates if r in li}
    cand = [i for i in range(len(leaf_names)) if i not in solved]
    v0f = kx_f(base0, P1, ac, solve_rates)
    phys_supp = [c for c in np.nonzero(v0f % P1)[0] if int(c) in phys_set and int(c) != ac]
    phys_supp = [int(c) for c in phys_supp]
    rel_by = [[] for _ in phys_supp]

    def moved(lis, seed):
        sh = random_point(len(lis), seed)
        pt = np.array(base0, dtype=np.int64)
        pt[lis] = (pt[lis] + sh) % P1
        v = kx_f(pt, P1, ac, solve_rates)
        if v is None:
            return None
        return bool(np.any((v[phys_supp] - v0f[phys_supp]) % P1 != 0))

    gsz = 8
    groups = [cand[i:i + gsz] for i in range(0, len(cand), gsz)]
    single = []
    for gi, g in enumerate(groups, start=1):
        if expired(ctrl):
            return fb("forward: timeout in relevance")
        m = (moved(g, 7919 * gi + 1), moved(g, 7919 * gi + 2))
        if m[0] is not False or m[1] is not False:
            single.extend(g)
    for l in single:
        if expired(ctrl):
            return fb("forward: timeout in relevance")
        ch = np.zeros(len(phys_supp), dtype=bool)
        for dv in (3, 7, 11):
            pt = np.array(base0, dtype=np.int64)
            pt[l] = pt[l] + dv
            v = kx_f(pt, P1, ac, solve_rates)
            if v is None:
                continue
            ch |= (v[phys_supp] - v0f[phys_supp]) % P1 != 0
        for i in np.nonzero(ch)[0]:
            rel_by[i].append(l)
    union_rel = sorted({x for r in rel_by for x in r})
    max_rel = max((len(r) for r in rel_by), default=0)
    if rdiag:
        print(f"[fwd] phys_supp={len(phys_supp)} union_rel={len(union_rel)} max_rel={max_rel}")
    fwd_cap_dir = max(int(ctrl["relevance_cap_dir"]), 48)
    if len(union_rel) > fwd_cap_dir or max_rel > ctrl["relevance_cap_sparse"]:
        return fb(f"forward: couples {len(union_rel)} leaves (entry up to {max_rel})")

    d_cap = max(1, min(3, int(ctrl["degree_cap"])))
    bank = {"U": np.zeros((0, len(union_rel)), dtype=np.int64),
            "V": [np.zeros((0, nz), dtype=np.int64) for _ in primes]}

    def need_of(k, d):
        return 2 * mono_table(k, d).shape[0] + 20

    def fill_bank_to(n):
        tries = 0
        while bank["V"][0].shape[0] < n and tries < 12 * n:
            if expired(ctrl):
                break
            tries += 1
            pt = np.array(base0, dtype=np.int64)
            pt[union_rel] = draw(len(union_rel))
            vs = [kx_f(pt, pp, ac, solve_rates) for pp in primes]
            if any(v is None for v in vs):
                continue
            bank["U"] = np.vstack([bank["U"], pt[union_rel].reshape(1, -1)])
            for jp in range(len(primes)):
                bank["V"][jp] = np.vstack([bank["V"][jp], vs[jp].reshape(1, -1)])
        return bank["V"][0].shape[0] >= n

    def rec_entry(reli, c9):
        reli_cols = [union_rel.index(r) for r in reli]
        vars_ = [leaf_names[r] for r in reli]
        for d in range(d_cap + 1):
            mons = mono_table(len(reli), d)
            if not fill_bank_to(need_of(len(reli), d)):
                continue
            expr = _fit_entry(primes, bank["U"], reli_cols, mons,
                              lambda jp, need: bank["V"][jp][:need, c9], vars_)
            if expr is not None:
                return {"expr": expr, "reli": vars_}
        return None

    entries = {znames[ac]: "1"}
    entry_reli = {znames[ac]: []}
    for i, c9 in enumerate(phys_supp):
        if expired(ctrl):
            return fb("forward: timeout in fitting")
        nm = znames[c9]
        reli = rel_by[i]
        if not reli:
            if not fill_bank_to(1):
                return fb("forward: empty bank")
            resc = [int(bank["V"][jp][0, c9]) for jp in range(len(primes))]
            try:
                rc = sym_rat_recon_big(np.array([resc], dtype=np.int64), [int(p) for p in primes])
            except Exception:
                rc = None
            if rc is None or rc["den"][0] == "0":
                return fb("forward: a constant entry could not be lifted")
            entries[nm] = rc["num"][0] if rc["den"][0] == "1" else f"{rc['num'][0]}/{rc['den'][0]}"
            entry_reli[nm] = []
            continue
        rr = rec_entry(reli, c9)
        if rr is None:
            return fb(f"forward: entry {nm} not a bounded rational (deg>{d_cap})")
        entries[nm] = simplify(rr["expr"]) if spy_on else rr["expr"]
        entry_reli[nm] = rr["reli"]

    ptv = None
    for _ in range(400):
        pt = draw(npt)
        if kx_f(pt, qv, ac, solve_rates) is not None:
            ptv = pt
            break
    if ptv is None:
        return fb("forward: no fresh verification point")
    vver = kx_f(ptv, qv, ac, solve_rates)
    for nm, ex in entries.items():
        col = znames.index(nm)
        rl = entry_reli[nm]
        env = {r: int(ptv[li[r]]) for r in rl}
        pred = eval_modq(ex, env, qv)
        if pred is None or (pred - int(vver[col])) % qv != 0:
            return fb("forward: reconstructed direction failed fresh-point verification")
    return {"support": sym_sort(list(entries)), "vector": entries, "type": "general",
            "closed_form": True, "solved_rates": list(solve_rates)}


class SolveRates(list):
    """Names of the rates solved on the forward route, with the states solved alongside."""

    def __init__(self, rates, state_solve=None):
        super().__init__(str(r) for r in (rates or []))
        ss = state_solve or []
        if isinstance(ss, str):
            ss = [ss]
        self.state_solve = [str(s) for s in ss]

    def key(self):
        """Hashable key of the solved rates and states."""
        return tuple(self), tuple(self.state_solve)


def perprime_forward_shared(sc, kcall, kcall_fwd, znames, z_slots, leaf_names, nz, scaling,
                            state_col_names, recast, spy_on, ctrl, phys_cols, models,
                            real_state_names, solve_param_names, solve_held, kcall_fwd_sh=None,
                            select_fn=None, fwd_state=None, map_fn=None):
    """Every residual direction on the forward route: one transversal, one bank."""
    n_prime_fwd = max(2, min(3, len(RP8) - 1))
    P1 = RP8[0]
    if kcall_fwd is None or expired(ctrl):
        return None
    rdiag = state.env("ROBUSTDIAG")
    t_f = time.monotonic()

    def el():
        return f"{time.monotonic() - t_f:.0f}s"

    npt = len(sc["point0"])
    track = {"pool_n": sc["pool_next"], "n_k": 0, "t_k": 0.0}

    def draw(n):
        u = sc["pool"](track["pool_n"] + np.arange(n))
        track["pool_n"] += n
        return u

    is_state = np.array([z in state_col_names for z in znames], dtype=bool)
    phys_set = list(range(nz)) if phys_cols is None else [int(c) for c in phys_cols]
    phys_set_s = set(phys_set)
    tangents_at = _Tangents(scaling, znames, is_state)
    zs = np.asarray(z_slots, dtype=np.int64)

    def extract_all(rp, zvals, p, acs):
        B = _basis(rp, nz, p)
        Bred = _nonzero_rows(_reduce_rows(B, tangents_at(zvals, p), p), p)
        if Bred.shape[0] == 0:
            return None
        rr = phys_rref(Bred, phys_set, nz, p)
        piv = [int(x) for x in rr["piv"]]
        out = []
        for ac in acs:
            if ac in piv:
                out.append(rr["R"][piv.index(ac)] % p)
                continue
            v = np.zeros(nz, dtype=np.int64)
            v[ac] = 1
            for i, pc in enumerate(piv):
                v[pc] = (p - rr["R"][i, ac]) % p
            out.append(v)
        return out

    rank_fwd = {"v": sc["rank"]}

    def kx_all(pt, p, acs, sr):
        t0k = time.monotonic()
        if kcall_fwd_sh is None:
            rp = kcall_fwd(pt, p, sc["nt_used"], sr)
        else:
            rp = kcall_fwd_sh(pt, p, sc["nt_used"], sr)
        track["n_k"] += 1
        track["t_k"] += time.monotonic() - t0k
        if not rp.get("ok") or rp["rank"] != rank_fwd["v"]:
            return None
        return extract_all(rp, np.asarray(pt)[zs], p, acs)

    ptb_b = None
    for att in range(1, 51):
        if att == 1 and len(sc["point0"]) >= npt:
            cand = np.asarray(sc["point0"], dtype=np.int64)[:npt]
        else:
            cand = draw(npt)
        r = kcall(cand, P1, sc["nt_used"])
        if r.get("ok") and r["rank"] == sc["rank"]:
            ptb_b = cand
            break
    if ptb_b is None:
        return None
    rp0 = kcall(ptb_b, P1, sc["nt_used"])
    B0 = _basis(rp0, nz, P1)
    Bred0 = _nonzero_rows(_reduce_rows(B0, tangents_at(ptb_b[zs], P1), P1), P1)
    if Bred0.shape[0] == 0:
        return None
    rr0 = phys_rref(Bred0, phys_set, nz, P1)
    piv0 = [int(x) for x in rr0["piv"]]
    if kcall_fwd_sh is not None:
        sub = (B0[:, phys_set] % P1).astype(np.float64)
        rk = 0 if sub.size == 0 else int(_core.sym_rref_mod(sub, P1)["rank"])
        rank_fwd["v"] = len(phys_set) - rk
    acs = [c for c in piv0 if c in phys_set_s]
    if rdiag:
        print(f"[fwds] {Bred0.shape[0]} residual rows, {len(acs)} physical anchors")
    if not acs:
        return None
    supp_of = []
    for ac in acs:
        v0 = rr0["R"][piv0.index(ac)] % P1
        nzv = [int(i) for i in np.nonzero(v0)[0]]
        supp_of.append(list(dict.fromkeys(_strip_name(znames[i])
                                          for i in [ac] + [c for c in nzv if c != ac])))
    li = {nm: i for i, nm in enumerate(leaf_names)}

    def run_group(acs, supp_of, ts, n_p=1):
        primes = RP8[:n_p]
        qv = RP8[-1]
        solve_rates = SolveRates(ts["solve_rates"], ts.get("state_solve"))
        if rdiag:
            print(f"[fwds] {len(acs)} anchors, transversal of {len(solve_rates)} rates ({el()})")
        base0 = None
        vb = None
        for _ in range(80):
            cand = draw(npt)
            vs = [kx_all(cand, pp, acs, solve_rates) for pp in primes]
            if not any(v is None for v in vs):
                base0, vb = cand, vs
                break
        if base0 is None:
            return [{"support": sym_sort(sp), "type": "general", "closed_form": False,
                     "reason": "forward: no base"} for sp in supp_of]
        if select_fn is not None and kcall_fwd_sh is not None:
            select_fn(base0, P1, sc["nt_used"], solve_rates)
        v0f = vb[0]
        phys_supp_of = [[int(c) for c in np.nonzero(v0f[a] % P1)[0] if int(c) in phys_set_s and int(c) != acs[a]]
                        for a in range(len(acs))]
        solved = {li[r] for r in solve_rates if r in li}
        cand = [i for i in range(len(leaf_names)) if i not in solved]

        def changed(pt):
            v = kx_all(pt, P1, acs, solve_rates)
            if v is None:
                return None
            return [((v[a][ps] - v0f[a][ps]) % P1 != 0) for a, ps in enumerate(phys_supp_of)]

        rel_by = [[[] for _ in ps] for ps in phys_supp_of]
        n_calls = {"n": 0}

        def shift_test(lis, seed):
            pt = np.array(base0, dtype=np.int64)
            pt[lis] = (pt[lis] + random_point(len(lis), seed)) % P1
            n_calls["n"] += 1
            return changed(pt)

        def bisect(lis, seed):
            """Leaves that move each entry, by halving the groups that move one, a level at a time."""
            level = [(lis, seed)]
            while level:
                if expired(ctrl):
                    return False
                if map_fn is None:
                    chs = [shift_test(g, sd) for g, sd in level]
                else:
                    n_calls["n"] += len(level)
                    chs = map_fn(lambda gs: changed_at(*gs), level)
                nxt = []
                for (g, sd), ch in zip(level, chs):
                    if ch is None:
                        ch = [np.ones(len(ps), dtype=bool) for ps in phys_supp_of]
                    if not any(np.any(c) for c in ch):
                        continue
                    if len(g) == 1:
                        for a in range(len(acs)):
                            for i in np.nonzero(ch[a])[0]:
                                rel_by[a][i].append(g[0])
                        continue
                    h = len(g) // 2
                    nxt += [(g[:h], 2 * sd + 1), (g[h:], 2 * sd + 2)]
                level = nxt
            for r in rel_by:
                for x in r:
                    x.sort()
            return True

        def changed_at(lis, seed):
            pt = np.array(base0, dtype=np.int64)
            pt[lis] = (pt[lis] + random_point(len(lis), seed)) % P1
            return changed(pt)

        if not bisect(cand, 7919):
            return None
        if rdiag:
            print(f"[fwds] relevance in {n_calls['n']} kernel calls ({el()})")
        fwd_cap_dir = max(int(ctrl["relevance_cap_dir"]), 48)
        ok_a = []
        for a in range(len(acs)):
            u = {x for r in rel_by[a] for x in r}
            m = max((len(r) for r in rel_by[a]), default=0)
            ok_a.append(len(u) <= fwd_cap_dir and m <= ctrl["relevance_cap_sparse"])
        union_rel = sorted({x for a in range(len(acs)) if ok_a[a] for r in rel_by[a] for x in r})

        d_cap = max(1, min(3, int(ctrl["degree_cap"])))
        bank = {"U": np.zeros((0, len(union_rel)), dtype=np.int64), "V": [[] for _ in primes]}

        def need_of(k, d):
            return 2 * mono_table(k, d).shape[0] + 6

        def sample(pt):
            return [kx_all(pt, pp, acs, solve_rates) for pp in primes]

        def fill_bank_to(n):
            tries = 0
            while len(bank["V"][0]) < n and tries < 12 * n:
                if expired(ctrl):
                    break
                k = 1 if map_fn is None else n - len(bank["V"][0])
                tries += k
                pts = []
                for _ in range(k):
                    pt = np.array(base0, dtype=np.int64)
                    pt[union_rel] = draw(len(union_rel))
                    pts.append(pt)
                for pt, vs in zip(pts, [sample(pts[0])] if map_fn is None else map_fn(sample, pts)):
                    if vs is None or any(v is None for v in vs):
                        continue
                    bank["U"] = np.vstack([bank["U"], pt[union_rel].reshape(1, -1)])
                    for jp in range(len(primes)):
                        bank["V"][jp].append(vs[jp])
            return len(bank["V"][0]) >= n

        def val_of(jp, a, c9, n):
            return [int(bank["V"][jp][r][a][c9]) for r in range(n)]

        def rec_entry(a, reli, c9):
            reli_cols = [union_rel.index(r) for r in reli]
            vars_ = [leaf_names[r] for r in reli]
            for d in range(d_cap + 1):
                mons = mono_table(len(reli), d)
                if not fill_bank_to(need_of(len(reli), d)):
                    continue
                expr = _fit_entry(primes, bank["U"], reli_cols, mons,
                                  lambda jp, need: val_of(jp, a, c9, need), vars_)
                if expr is not None:
                    return {"expr": expr, "reli": vars_}
            return None

        ptv = vver = None
        for _ in range(400):
            pt = draw(npt)
            vv = kx_all(pt, qv, acs, solve_rates)
            if vv is not None:
                ptv, vver = pt, vv
                break
        out = []
        for a, ac in enumerate(acs):
            def opn(reason, a=a):
                return {"support": sym_sort(supp_of[a]), "type": "general", "closed_form": False,
                        "reason": reason}
            if not ok_a[a]:
                out.append(opn("forward: couples too many leaves"))
                continue
            if ptv is None:
                out.append(opn("forward: no fresh verification point"))
                continue
            entries = {znames[ac]: "1"}
            entry_reli = {znames[ac]: []}
            res = None
            for i, c9 in enumerate(phys_supp_of[a]):
                if expired(ctrl):
                    res = opn("forward: timeout")
                    break
                nm = znames[c9]
                reli = rel_by[a][i]
                if not reli:
                    if not fill_bank_to(1):
                        res = opn("forward: empty bank")
                        break
                    resc = [val_of(jp, a, c9, 1)[0] for jp in range(len(primes))]
                    try:
                        rc = sym_rat_recon_big(np.array([resc], dtype=np.int64), [int(p) for p in primes])
                    except Exception:
                        rc = None
                    if rc is None or rc["den"][0] == "0":
                        res = opn("forward: constant not lifted")
                        break
                    entries[nm] = rc["num"][0] if rc["den"][0] == "1" else f"{rc['num'][0]}/{rc['den'][0]}"
                    entry_reli[nm] = []
                    continue
                rr = rec_entry(a, reli, c9)
                if rr is None:
                    res = opn(f"forward: entry {nm} not a bounded rational")
                    break
                entries[nm] = simplify(rr["expr"]) if spy_on else rr["expr"]
                entry_reli[nm] = rr["reli"]
            if res is not None:
                out.append(res)
                continue
            bad = False
            for nm, ex in entries.items():
                col = znames.index(nm)
                rl = entry_reli[nm]
                pred = eval_modq(ex, {r: int(ptv[li[r]]) for r in rl}, qv)
                if pred is None or (pred - int(vver[a][col])) % qv != 0:
                    bad = True
                    break
            if bad:
                out.append(opn("forward: failed fresh-point verification"))
                continue
            out.append({"support": sym_sort(list(entries)), "vector": entries, "type": "general",
                        "closed_form": True, "solved_rates": list(solve_rates)})
        return out

    tsv = {s: ((i + 1) * 7919 + 11) % (P1 - 1) + 1 for i, s in enumerate(real_state_names)}
    tpv = {s: ((i + 1) * 104729 + 3) % (P1 - 1) + 1 for i, s in enumerate(solve_param_names)}
    recast_leaves = [l for l in leaf_names if re.match(r"^_E_|^_L_", l)]

    def trans_for(sp):
        kf = list(dict.fromkeys([x for s in sp for x in s] + recast_leaves))
        try:
            ts = solve_forward_modular(models[0], list(real_state_names), list(solve_param_names),
                                        tsv, tpv, P1,
                                        forcings=list(solve_held) if solve_held else None,
                                        keep_free=kf)
        except Exception:
            return None
        return ts if ts is not None and ts.get("ok") else None

    def groups_for(idx):
        groups = []
        cur, cur_ts = [], None
        for a in idx:
            ts = trans_for([supp_of[i] for i in cur + [a]])
            if ts is not None:
                cur, cur_ts = cur + [a], ts
                continue
            if cur:
                groups.append({"idx": cur, "ts": cur_ts})
            ts = trans_for([supp_of[a]])
            if ts is None:
                cur, cur_ts = [], None
                groups.append({"idx": [a], "ts": None})
            else:
                cur, cur_ts = [a], ts
        if cur:
            groups.append({"idx": cur, "ts": cur_ts})
        lone = [i for g in groups if g["ts"] is None for i in g["idx"]]
        if lone:
            groups = [g for g in groups if g["ts"] is not None]
            groups.append({"idx": lone, "ts": trans_for([])})
        return groups

    ts1 = None if state.env("FWDGROUPS") else trans_for([])
    all_idx = list(range(len(acs)))
    groups = groups_for(all_idx) if ts1 is None else [{"idx": all_idx, "ts": ts1}]
    out = [None] * len(acs)
    for g in groups:
        gidx = g["idx"]
        if g["ts"] is None:
            res = [{"support": sym_sort(supp_of[i]), "type": "general", "closed_form": False,
                    "reason": "forward: no transversal"} for i in gidx]
        else:
            res = run_group([acs[i] for i in gidx], [supp_of[i] for i in gidx], g["ts"])
        if res is None:
            res = [{"support": sym_sort(supp_of[i]), "type": "general", "closed_form": False,
                    "reason": "forward: timeout"} for i in gidx]
        again = [k for k, e in enumerate(res) if not e.get("closed_form")]
        if again and g["ts"] is not None:
            res2 = run_group([acs[gidx[k]] for k in again], [supp_of[gidx[k]] for k in again],
                            g["ts"], n_p=n_prime_fwd)
            if res2 is not None:
                for j, k in enumerate(again):
                    res[k] = res2[j]
        for k, i in enumerate(gidx):
            out[i] = res[k]
    if ts1 is not None:
        again = [i for i, e in enumerate(out) if not e.get("closed_form")]
        if again:
            for g in groups_for(again):
                if g["ts"] is None:
                    continue
                res = run_group([acs[i] for i in g["idx"]], [supp_of[i] for i in g["idx"]], g["ts"],
                               n_p=n_prime_fwd)
                if res is not None:
                    for k, i in enumerate(g["idx"]):
                        if res[k].get("closed_form"):
                            out[i] = res[k]
    if rdiag:
        n_k = track["n_k"]
        print(f"[fwds] closed {sum(1 for e in out if e.get('closed_form'))} of {len(out)} ({el()}; "
              f"{n_k} kernel calls, {track['t_k'] / n_k if n_k else 0:.1f} s each)")
    return out


def fwd_adopt(fwd, n_res):
    """Forward directions to adopt for n_res residual directions, or None."""
    cl = [e for e in fwd if e.get("closed_form")]
    if len(cl) > n_res:
        cl = [e for e in cl if any(re.search(r"[A-Za-z]", str(v)) for v in e["vector"].values())]
    return cl if len(cl) == n_res and n_res > 0 else None


def perprime_forward_multi(residual_free, sc, kcall, kcall_fwd, znames, z_slots, leaf_names, nz,
                           scaling, state_col_names, param_names, recast, spy_on, ctrl, phys_cols,
                           models, real_state_names, solve_param_names, solve_held):
    """Forward results of the first residual direction, one per anchor; [] without residual directions,
    None without anchors."""
    if not residual_free:
        return []

    def fwd1(ac_in, list_anchors):
        return perprime_forward(residual_free[0], sc, kcall, kcall_fwd, znames, z_slots, leaf_names,
                                nz, scaling, state_col_names, param_names, recast, spy_on, ctrl,
                                phys_cols, models, real_state_names, solve_param_names, solve_held,
                                ac_in=ac_in, list_anchors=list_anchors)
    anchors = fwd1(None, True)
    if not anchors:
        return None
    return [fwd1(a, False) for a in anchors]
