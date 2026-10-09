"""The joint system under a steady state: per-condition resting states as coordinates,
the tangency rows df_rest = [Jx | Jt] stacked with the observability rows."""

import re

import numpy as np

from ... import _core, state
from ...gfp import invmod, mulmod
from ...sym.continuation import continue_resting_state
from ..fields import series_const, series_embed, series_stack
from .context import ints, ok, strs
from .rest import RestingStates

_IDENT = re.compile(r"[A-Za-z._][A-Za-z0-9._]*")


class JointSystem(RestingStates):
    """Resting states per condition, their solves and the wide kernel over
    shared parameter columns and per-condition state columns."""

    def __init__(self, run):
        self.run = run
        multi = run.multi
        self.state_names = strs(multi.get("state_names"))
        self.param_names = strs(multi.get("param_names"))
        self.forcings = strs(multi.get("forcings"))
        self.solve_held = list(dict.fromkeys(self.forcings + run.equil_zero_states))
        self.held_state_params = multi.get("held_state_params") or {}
        self.models = [strs(t.get("constraint_model")) for t in multi["tapes"]]
        is_equil = [len(strs(t.get("constraint_model"))) > 0 and run.first_of_chain[i]
                    for i, t in enumerate(multi["tapes"])]
        self.recast = list(multi.get("power_recast") or [])
        self.real_state_names = strs(multi.get("real_state_names")) if self.recast else self.state_names
        gen_names = [str(r["base"]) if r.get("inverted") else str(r["E"]) for r in self.recast]
        self.l_names = list(dict.fromkeys(str(r["L"]) for r in self.recast))
        held_solve = {str(v) for v in self.held_state_params.values()}
        self.solve_param_names = [n for n in self.param_names + gen_names if n not in held_solve]
        self.held_param_of = {str(k): str(v) for k, v in self.held_state_params.items()}
        self.held_names = [n for n in self.held_param_of
                           if n in self.real_state_names and n not in self.solve_held]
        z_state = set(strs(multi.get("z_state_names")))
        znames = run.znames
        nz = len(znames)
        self.per_cond_cols = [i for i, z in enumerate(znames) if z in z_state]
        self.log_cols = [i for i, z in enumerate(znames) if z in z_state - set(self.l_names)]
        self.joint_state_slot = run.z_slots[self.log_cols]
        self.equil_conds = [i for i, e in enumerate(is_equil) if e]
        self.kc = len(self.equil_conds)
        per = set(self.per_cond_cols)
        self.shared_idx = [i for i in range(nz) if i not in per]
        self.n_shared = len(self.shared_idx)
        self.n_st = len(self.per_cond_cols)
        self.nz_wide = self.n_shared + self.kc * self.n_st
        self.wide_cols = []
        for mi in range(self.kc):
            wc = np.zeros(nz, dtype=np.int64)
            wc[self.shared_idx] = np.arange(self.n_shared)
            wc[self.per_cond_cols] = self.n_shared + mi * self.n_st + np.arange(self.n_st)
            self.wide_cols.append(wc)
        self.state_base = [znames[i] for i in self.per_cond_cols]
        self.znames_wide = [znames[i] for i in self.shared_idx] + \
            [f"{s}|c{mi + 1}" for mi in range(self.kc) for s in self.state_base]
        self.z_slots_wide = np.concatenate([run.z_slots[self.shared_idx]] +
                                           [run.z_slots[self.per_cond_cols]] * self.kc) \
            if nz else run.z_slots
        self.z_state_names_wide = self.znames_wide[self.n_shared:self.nz_wide] if self.n_st else []
        self.nz_local = nz
        self.znames_local = list(znames)
        self.z_slots_local = run.z_slots.copy()
        self.z_index_local = {z: i for i, z in enumerate(znames)}
        zi = self.z_index_local
        self.recast_rel = [{
            "base_col": zi.get(str(rc["base"])), "ECol": zi.get(str(rc["E"])),
            "LCol": zi.get(str(rc["L"])), "exp_col": zi.get(str(rc["exp"])),
            "exp_slot": run.slot(str(rc["exp"])), "LSlot": run.slot(str(rc["L"])),
            "ESlot": run.slot(str(rc["E"])), "base_slot": run.slot(str(rc["base"])),
            "base_param": str(rc["base"]) in self.param_names} for rc in self.recast]
        self.df_droppable = set(self.forcings) | set(run.equil_zero_states) | \
            {s for z in run.cond_zero_states for s in (z or [])} | (set(run.leaf_names) - set(znames))
        self.solve_cache = {}
        self.model_key = ["\n".join(self.models[ci] + self.zero_of(ci)) for ci in range(len(self.models))]
        if run.timing:
            print(f"[sym] {self.kc} equilibrate condition(s), "
                  f"{len({self.model_key[c] for c in self.equil_conds})} distinct steady state(s)")
        sp = set(self.solve_param_names)
        self.model_params = []
        for m in self.models:
            found = [tok for line in m for tok in _IDENT.findall(line) if tok in sp]
            self.model_params.append([n for n in self.solve_param_names if n in set(found)])
        held_vals = set(self.held_param_of.values())
        self.private_of = []
        for ci in range(len(self.models)):
            if ci not in self.equil_conds:
                self.private_of.append([])
                continue
            other = {n for cj in self.equil_conds if self.model_key[cj] != self.model_key[ci]
                     for n in self.model_params[cj]}
            cand = [n for n in self.model_params[ci]
                    if n not in other and n not in self.l_names and n not in held_vals]
            self.private_of.append([n for n in cand if run.slot(n) is not None])
        self.fwd_solve_cache = {}
        self.fwd_cond = self.equil_conds[0] if self.equil_conds else None
        self.fwd_rates = None
        self.fwd_banned = []
        self.fwd_on = (not self.held_names and not any(r.get("inverted") for r in self.recast)
                       and not state.env("NOFWDSEED"))
        self.cont_t = max(1, int(run.ctrl["gap_order_cap"]))
        n_eq = (max(self.equil_conds) + 1) if self.equil_conds else 0
        self.cont_off = [False] * n_eq
        self.cont_seen = [False] * n_eq
        self.redraws = 0
        self.chains = [run.chain_tapes(run.chain_groups[run.chain_idx[self.equil_conds[mi]]])
                       for mi in range(self.kc)] if run.has_gaps else None
        self.fwd_sel = list(range(self.kc))
        self.warm_off = False

    # ---- conditions -----------------------------------------------------------------





    # ---- steady-state solves --------------------------------------------------------







    # ---- continuation from the forward-seeded resting state --------------------------








    # ---- rows of one condition ------------------------------------------------------

    def df_rows(self, sol, p):
        """df constraint rows of one condition over the local columns."""
        Jx, Jt = sol["df_jx"], sol["df_jt"]
        nz_l = self.nz_local
        if not len(Jx):
            return np.zeros((0, nz_l), dtype=np.int64)
        scn, pcn = strs(sol["df_state_cols"]), strs(sol["df_param_cols"])
        n = len(Jx)
        M = np.zeros((n, nz_l), dtype=np.int64)
        if state.env("JOINTDIAG"):
            for nm in scn:
                if nm not in self.z_index_local and nm not in self.df_droppable:
                    print(f"[jointdiag] df drops non-fixed state column: {nm}")
        Jxm = np.array([[int(x) % p for x in row] for row in Jx], dtype=np.int64).reshape(n, len(scn))
        for j, nm in enumerate(scn):
            c = self.z_index_local.get(nm)
            if c is not None:
                M[:, c] = (M[:, c] + Jxm[:, j]) % p
        for th in pcn:
            c = self.z_index_local.get(self.held_param_of.get(th, th))
            if c is not None:
                col = np.array([int(x) % p for x in Jt[th][:n]], dtype=np.int64)
                M[:, c] = (M[:, c] + col) % p
        return M

    def cond_local(self, mi, obs, sc0, p, scale=True):
        """One condition's rows over its local columns: observability, df, pinned zeros and
        recast relations; state columns scaled by the resting values unless continued.
        With a series, the same rows as series. None rejects the point."""
        from ..fields import ser_inv, ser_mul
        if not ok(obs):
            return None
        nz_l = self.nz_local
        cont = bool(sc0.get("cont"))
        o_r = np.asarray(obs["R"], dtype=np.int64).reshape(int(obs["rank"]), nz_l).copy()
        d_r = self.df_rows(self.sol_at(sc0["sol"], 0) if cont else sc0["sol"], p)
        zc = [self.z_index_local[z] for z in self.zero_of(self.equil_conds[mi]) if z in self.z_index_local]
        e_z = np.zeros((len(zc), nz_l), dtype=np.int64)
        for i, c in enumerate(zc):
            e_z[i, c] = 1
        scl = np.ones(nz_l, dtype=np.int64)
        if not cont:
            xvals = np.asarray(sc0["ptc"], dtype=np.int64)[self.joint_state_slot] % p
            zcs = set(zc)
            for m, lc in enumerate(self.log_cols):
                if lc in zcs:
                    xvals[m] = 1
            if np.any(xvals == 0):
                if state.env("FWDDIAG"):
                    print("[fwddiag] zero log-normal coord")
                return None
            if scale:
                scl[self.log_cols] = xvals
        for lc in self.log_cols:
            if scl[lc] != 1:
                o_r[:, lc] = mulmod(o_r[:, lc], scl[lc], p)
                if d_r.shape[0]:
                    d_r[:, lc] = mulmod(d_r[:, lc], scl[lc], p)
        rel = []
        if not cont:
            for rc in self.recast_rel:
                expv = int(sc0["ptc"][rc["exp_slot"]]) % p
                Lv = int(sc0["ptc"][rc["LSlot"]]) % p
                b_scale = 1
                if rc["base_param"]:
                    bv = int(sc0["ptc"][rc["base_slot"]]) % p
                    if bv == 0:
                        return None
                    b_scale = invmod(bv, p)
                r1 = np.zeros(nz_l, dtype=np.int64)
                r2 = np.zeros(nz_l, dtype=np.int64)
                if rc["ECol"] is not None:
                    r1[rc["ECol"]] = 1
                if rc["base_col"] is not None:
                    r1[rc["base_col"]] = (p - expv * b_scale % p) % p
                if rc["exp_col"] is not None:
                    r1[rc["exp_col"]] = (-Lv) % p
                if rc["LCol"] is not None:
                    r2[rc["LCol"]] = 1
                if rc["base_col"] is not None:
                    r2[rc["base_col"]] = (p - b_scale) % p
                rel.append(np.vstack([r1, r2]))
        rel_s = []
        if cont and self.recast_rel:
            Ns = 1 if obs.get("N") is None else int(obs["N"])

            def series_of(sl):
                v = [0] * Ns
                if sl is None:
                    return v
                v[0] = int(sc0["ptc"][sl]) % p
                for j in range(min(Ns - 1, sc0["ser"].shape[1])):
                    v[1 + j] = int(sc0["ser"][sl, j]) % p
                return v
            for rc in self.recast_rel:
                if rc["base_slot"] is None or rc["ESlot"] is None:
                    return None
                Es, bs = series_of(rc["ESlot"]), series_of(rc["base_slot"])
                if bs[0] == 0:
                    return None
                ib = ser_inv(bs, p)
                c_b1 = [(p - x) % p for x in ser_mul(ser_mul(series_of(rc["exp_slot"]), Es, p), ib, p)]
                c_x1 = [(p - x) % p for x in ser_mul(Es, series_of(rc["LSlot"]), p)]
                c_b2 = [(p - x) % p for x in ib]
                R1 = np.zeros((2, nz_l * Ns), dtype=np.int64)
                one = [1] + [0] * (Ns - 1)
                for i, col, v in ((0, rc["ECol"], one), (0, rc["base_col"], c_b1), (0, rc["exp_col"], c_x1),
                                  (1, rc["LCol"], one), (1, rc["base_col"], c_b2)):
                    if col is not None:
                        R1[i, col * Ns:(col + 1) * Ns] = v
                rel_s.append(R1)
                rel.append(R1[:, np.arange(nz_l) * Ns])
        parts = [d_r, e_z] + rel
        cst = np.vstack(parts) if sum(x.shape[0] for x in parts) else np.zeros((0, nz_l), dtype=np.int64)
        out = {"M": np.vstack([o_r, cst]), "S": None, "N": obs.get("N")}
        if obs.get("S") is None:
            return out
        N = int(obs["N"])
        S = np.asarray(obs["S"], dtype=np.int64).reshape(-1, nz_l * N)
        if np.any(scl != 1):
            S = series_embed(S, N, list(range(nz_l)), nz_l, p, scl)
        if not cont:
            out["S"] = np.vstack([S, series_const(cst, N)])
            return out
        d_s = np.zeros((d_r.shape[0], nz_l * N), dtype=np.int64)
        for k in range(N):
            dk = self.df_rows(self.sol_at(sc0["sol"], k), p)
            for c in range(nz_l):
                if np.any(dk[:, c] != 0):
                    d_s[:, c * N + k] = dk[:, c]
        out["S"] = np.vstack([S, d_s, series_const(e_z, N)] + rel_s)
        return out

    def wide_blocks(self, mi, obs, sc0, p):
        """The local rows embedded in the wide space."""
        loc = self.cond_local(mi, obs, sc0, p)
        if loc is None:
            return None
        wc = self.wide_cols[mi]
        W = np.zeros((loc["M"].shape[0], self.nz_wide), dtype=np.int64)
        W[:, wc] = loc["M"]
        ser = None if loc["S"] is None else series_embed(loc["S"], int(loc["N"]), list(wc), self.nz_wide, p)
        return {"W": W, "series": ser}

    def _project(self, locs, p, cores, sel=None):
        """Each condition eliminates its state columns first; the rows free of them on
        the shared columns, with their series projections."""
        st_l, sh_l = self.per_cond_cols, self.shared_idx
        n_first = len(st_l)
        r_p, Q, r_ps, QS, prec, series = 0, [], 0, [], [], True
        for loc in locs:
            M = loc["M"][:, st_l + sh_l]
            rr = _core.sym_rref_mod(M.astype(np.float64), p) if M.shape[0] else \
                {"piv": [], "R": np.zeros((0, M.shape[1]))}
            piv = np.atleast_1d(rr["piv"]).astype(np.int64)
            is_p = piv < n_first
            r_p += int(is_p.sum())
            if np.any(~is_p):
                R = np.asarray(rr["R"], dtype=np.int64).reshape(len(piv), M.shape[1])
                Q.append(R[~is_p][:, n_first:n_first + len(sh_l)])
            if loc["S"] is None:
                series = False
                continue
            sp = _core.sym_series_project(np.asarray(loc["S"], dtype=np.int64), int(self.nz_local),
                                          int(loc["N"]), p, list(st_l), int(cores))
            r_ps += int(sp["r1"])
            QS.append(np.asarray(sp["Q"], dtype=np.int64).reshape(-1, len(sh_l) * int(loc["N"])))
            prec.extend(ints(sp["prec"]))
        return r_p, Q, r_ps, QS, prec, series

    def joint_rank(self, locs, p, cores):
        """The joint rank from the local rows."""
        r_p, Q, r_ps, QS, prec, series = self._project(locs, p, cores)
        QQ = np.vstack(Q) if Q else None
        r_q = int(_core.sym_rref_mod(QQ.astype(np.float64), p)["rank"]) if QQ is not None and QQ.shape[0] else 0
        res = {"ok": True, "rank": int(r_p + r_q), "dim": self.nz_wide, "block_rank": True,
               "pivots": np.zeros(0, dtype=np.int64)}
        if not series or not locs:
            return res
        N = int(locs[0]["N"])
        nsh = len(self.shared_idx)
        QSm = np.vstack(QS) if QS else np.zeros((0, nsh * N), dtype=np.int64)
        r_qs = int(_core.sym_series_rank(QSm, nsh, N, p, [], cores=int(cores), row_prec=prec)["rank"]) \
            if QSm.shape[0] else 0
        res["rank_s"] = int(r_ps + r_qs)
        res["N"] = N
        return res

    # ---- kernels --------------------------------------------------------------------

    def cond_obs(self, ci, ptc, p, Nt, Mtot, ser=None):
        """One condition's observability kernel at its seeded resting state."""
        run = self.run
        pt = np.asarray(ptc, dtype=np.int64) % p
        if run.has_gaps:
            return _core.sym_obs_null_chain([run.chain_tapes(run.chain_groups[run.chain_idx[ci]])],
                                            run.n_leaves, run.n_states, self.z_slots_local, pt, p,
                                            int(Nt), int(Mtot), 1, [],
                                            None if ser is None else np.asarray(ser) % p)
        return _core.sym_obs_null_multi([run.tapes[ci]], run.n_leaves, run.n_states,
                                        self.z_slots_local, pt, p, int(Nt), 1)

    def cond_obs_sel(self, scs, sel, p, Nt, Mtot):
        """cond_obs for the conditions `sel` (others None), one batch on chains."""
        run = self.run
        out = [None] * self.kc
        if not run.has_gaps or len(sel) < 2:
            for mi in sel:
                out[mi] = self.cond_obs(self.equil_conds[mi], scs[mi]["ptc"], p, Nt, Mtot,
                                        scs[mi]["ser"] if scs[mi].get("cont") else None)
            return out
        seeds = np.vstack([np.asarray(scs[mi]["ptc"], dtype=np.int64) % p for mi in sel])
        sers = [np.asarray(scs[mi]["ser"]) % p if scs[mi].get("cont") else None for mi in sel]
        res = _core.sym_obs_null_chain_seed_batch(
            self.chains, list(sel), seeds, [p] * len(sel), run.n_leaves, run.n_states,
            self.z_slots_local, int(Nt), int(Mtot), int(min(run.cores_call, len(sel))),
            [run.nt_cap[mi] for mi in sel] if len(run.nt_cap) == self.kc else [],
            sers if any(s is not None for s in sers) else None)
        for k, mi in enumerate(sel):
            out[mi] = res[k]
        return out

    def cond_obs_all(self, scs, p, Nt, Mtot):
        """cond_obs for every resting condition, one batch on chains when there are gaps."""
        if not self.run.has_gaps or self.kc < 2:
            return [self.cond_obs(self.equil_conds[mi], scs[mi]["ptc"], p, Nt, Mtot)
                    for mi in range(self.kc)]
        return self.cond_obs_sel(scs, list(range(self.kc)), p, Nt, Mtot)

    def kcall(self, point, p, Nt, Mtot=0, solve_fn=None, rank_only=False):
        """The wide kernel at a point: every condition solved, rows stacked and reduced."""
        run = self.run
        if solve_fn is None:
            solve_fn = self.solve_cond
            if self.kc > 1 and self.solve_cond(point, p, self.equil_conds[0]) is not None:
                self.warm_solves([point], [p])
        scs = []
        for mi in range(self.kc):
            sc0 = solve_fn(point, p, self.equil_conds[mi])
            if sc0 is None:
                run.ss_why = "joint solve failed"
                return {"ok": False}
            scs.append(sc0)
        obs = self.cond_obs_all(scs, p, Nt, Mtot)
        if rank_only:
            locs = []
            for mi in range(self.kc):
                loc = self.cond_local(mi, obs[mi], scs[mi], p, scale=len(self.recast_rel) > 0)
                if loc is None:
                    return {"ok": False}
                locs.append(loc)
            return self.joint_rank(locs, p, run.cores_call)
        blocks, sblocks = [], []
        for mi in range(self.kc):
            b = self.wide_blocks(mi, obs[mi], scs[mi], p)
            if b is None:
                return {"ok": False}
            sblocks.append(b["series"])
            blocks.append(b["W"])
        rr = _core.sym_rref_mod(np.vstack(blocks).astype(np.float64), p)
        res = {"ok": True, "R": np.asarray(rr["R"], dtype=np.int64).reshape(int(rr["rank"]), self.nz_wide),
               "pivots": np.atleast_1d(rr["piv"]).astype(np.int64), "rank": int(rr["rank"]),
               "dim": self.nz_wide}
        if run.has_gaps:
            res = series_stack(res, sblocks, self.nz_wide, Mtot + 1, p, run.cores_call)
        return res

    def kcall_fwd(self, point, p, Nt, solve_rates):
        """The wide kernel with the forward solve for the turnover rates."""
        return self.kcall(point, p, Nt, self.run.mtot,
                          lambda pt, pp, cc: self.solve_cond_fwd(pt, pp, cc, solve_rates))

    def kcall_fwd_shared(self, point, p, Nt, solve_rates, sel=None):
        """The forward kernel on the shared columns only, on the conditions `sel`."""
        run = self.run
        if sel is None:
            sel = self.fwd_sel
        scs = [None] * self.kc
        for mi in sel:
            sc0 = self.solve_cond_fwd(point, p, self.equil_conds[mi], solve_rates)
            if sc0 is None:
                return {"ok": False}
            scs[mi] = sc0
        obs = self.cond_obs_all(scs, p, Nt, run.mtot) if len(sel) == self.kc else \
            self.cond_obs_sel(scs, sel, p, Nt, run.mtot)
        locs = []
        for mi in sel:
            loc = self.cond_local(mi, obs[mi], scs[mi], p)
            if loc is None:
                return {"ok": False}
            locs.append(loc)
        _, Q, _, QS, prec, series = self._project(locs, p, run.cores_call)
        nsh = len(self.shared_idx)
        if Q:
            r0 = _core.sym_rref_mod(np.vstack(Q).astype(np.float64), p)
            rk = int(r0["rank"])
            R = np.asarray(r0["R"], dtype=np.int64).reshape(rk, nsh)
            piv = np.atleast_1d(r0["piv"]).astype(np.int64)
        else:
            rk, R, piv = 0, np.zeros((0, nsh), dtype=np.int64), np.zeros(0, dtype=np.int64)
        if series and QS:
            QSm = np.vstack(QS)
            if QSm.shape[0]:
                sr = _core.sym_series_rank(QSm, nsh, run.mtot + 1, p, [], at_one_below=rk,
                                           cores=int(run.cores_call), row_prec=prec)
                if sr.get("R") is not None:
                    piv = np.atleast_1d(sr["pivots"]).astype(np.int64)
                    R = np.asarray(sr["R"], dtype=np.int64).reshape(len(piv), nsh)
        Rw = np.zeros((R.shape[0], self.nz_wide), dtype=np.int64)
        Rw[:, :nsh] = R
        return {"ok": True, "R": Rw, "pivots": piv, "rank": len(piv), "dim": self.nz_wide}

    def fwd_select(self, point, p, Nt, solve_rates):
        """The fewest conditions whose forward kernel has the full rank at the point."""
        self.fwd_sel = list(range(self.kc))
        full = self.kcall_fwd_shared(point, p, Nt, solve_rates)
        if not ok(full) or self.kc < 3:
            return full
        sel = []
        for mi in range(self.kc):
            sel.append(mi)
            r = self.kcall_fwd_shared(point, p, Nt, solve_rates, sel=list(sel))
            if ok(r) and r["rank"] == full["rank"]:
                self.fwd_sel = list(sel)
                break
        return full

    def block_calls(self):
        """One condition's observability rows for the saturation."""
        def mk(mi):
            def f(point, p, Nt, Mtot=0):
                sc0 = self.solve_cond(point, p, self.equil_conds[mi])
                if sc0 is None:
                    self.run.ss_why = "joint solve failed"
                    return {"ok": False}
                return self.cond_obs(self.equil_conds[mi], sc0["ptc"], p, Nt, Mtot, sc0.get("ser"))
            return f
        return [mk(mi) for mi in range(self.kc)]

    def block_profiles(self):
        """The rank per Lie order of a condition's chain from one jet (gap paths)."""
        run = self.run
        if not run.has_gaps:
            return None

        def mk(mi):
            def f(point, p, Nt):
                sc0 = self.solve_cond(point, p, self.equil_conds[mi])
                if sc0 is None:
                    return {"ok": False}
                return _core.sym_obs_chain_rank_profile(
                    run.chain_tapes(run.chain_groups[run.chain_idx[self.equil_conds[mi]]]),
                    run.n_leaves, run.n_states, self.z_slots_local,
                    np.asarray(sc0["ptc"], dtype=np.int64) % p, p, int(Nt), 1)
            return f
        return [mk(mi) for mi in range(self.kc)]

    def chunk(self, point_list, prime_vec, Nt, cols=None, conds=None, touch=False):
        """Coupled gap path: every (point, condition) seed solved, all chain kernels in one
        batch, then assembled and reduced per point. `cols` restricts to wide columns,
        `conds` to conditions, `touch` reports which condition has a row on `cols`."""
        run = self.run
        nz_l = self.nz_local
        mis = list(range(self.kc)) if conds is None else [int(c) for c in conds]
        self.warm_solves(point_list, prime_vec, conds=[self.equil_conds[mi] for mi in mis])
        if cols is None:
            loc_sel = list(range(nz_l))
        else:
            cs = {int(c) for c in cols}
            loc_sel = sorted({j for mi in mis for j, w in enumerate(self.wide_cols[mi]) if int(w) in cs})

        def embed(o):
            if cols is None or not ok(o):
                return o
            o = dict(o)
            R = np.zeros((int(o["rank"]), nz_l), dtype=np.int64)
            R[:, loc_sel] = np.asarray(o["R"], dtype=np.int64).reshape(int(o["rank"]), len(loc_sel))
            o["R"] = R
            if o.get("S") is not None:
                Nn = int(o["N"])
                o["S"] = series_embed(np.asarray(o["S"], dtype=np.int64).reshape(-1, len(loc_sel) * Nn),
                                      Nn, loc_sel, nz_l, 1)
            return o
        n_p = len(point_list)
        per_cond = [None] * n_p
        seed_rows, ser_rows, ev_chain, ev_prime = [], [], [], []
        for i in range(n_p):
            pt, pp = point_list[i], prime_vec[i]
            per = [None] * self.kc
            ok_all = True
            for mi in mis:
                sc0 = self.solve_cond(pt, pp, self.equil_conds[mi])
                if sc0 is None:
                    ok_all = False
                    break
                per[mi] = sc0
            if not ok_all:
                continue
            per_cond[i] = per
            for mi in mis:
                seed_rows.append(np.asarray(per[mi]["ptc"], dtype=np.int64) % pp)
                ser_rows.append(np.asarray(per[mi]["ser"]) % pp if per[mi].get("cont") else None)
                ev_chain.append(mi)
                ev_prime.append(pp)
        kr = _core.sym_obs_null_chain_seed_batch(
            self.chains, ev_chain, np.vstack(seed_rows), ev_prime, run.n_leaves, run.n_states,
            self.z_slots_local[loc_sel], int(Nt), int(run.mtot), int(min(run.cores, len(ev_chain))),
            [run.nt_cap[e] for e in ev_chain] if len(run.nt_cap) == self.kc else [],
            ser_rows if any(s is not None for s in ser_rows) else None) if seed_rows else []
        out = [None] * n_p
        e = 0
        for i in range(n_p):
            if per_cond[i] is None:
                out[i] = {"ok": False}
                continue
            blocks, sblocks, bad, hit = [], [], False, []
            for mi in mis:
                b = self.wide_blocks(mi, embed(kr[e]), per_cond[i][mi], prime_vec[i])
                e += 1
                if b is None:
                    bad = True
                    continue
                blocks.append(b["W"])
                sblocks.append(b["series"])
                if touch:
                    hit.append(bool(np.any(b["W"][:, list(cols)] % prime_vec[i] != 0)))
            if bad:
                out[i] = {"ok": False}
            elif touch:
                out[i] = {"ok": True, "touch": hit}
            elif cols is not None:
                cl = list(cols)
                rr = _core.sym_rref_mod(np.vstack(blocks)[:, cl].astype(np.float64), prime_vec[i])
                Nb = run.mtot + 1
                sk = [c * Nb + j for c in cl for j in range(Nb)]
                out[i] = series_stack(
                    {"ok": True, "R": np.asarray(rr["R"], dtype=np.int64).reshape(int(rr["rank"]), len(cl)),
                     "pivots": np.atleast_1d(rr["piv"]).astype(np.int64), "rank": int(rr["rank"]),
                     "dim": len(cl)},
                    [None if S is None else S[:, sk] for S in sblocks], len(cl), Nb, prime_vec[i],
                    run.cores)
            else:
                rr = _core.sym_rref_mod(np.vstack(blocks).astype(np.float64), prime_vec[i])
                out[i] = series_stack(
                    {"ok": True, "R": np.asarray(rr["R"], dtype=np.int64).reshape(int(rr["rank"]), self.nz_wide),
                     "pivots": np.atleast_1d(rr["piv"]).astype(np.int64), "rank": int(rr["rank"]),
                     "dim": self.nz_wide},
                    sblocks, self.nz_wide, run.mtot + 1, prime_vec[i], run.cores)
        return out

    def hom_line(self, point, d, T, p, Nt):
        """The wide series rows along theta0 + t d (models without later events)."""
        run = self.run
        if run.has_gaps:
            return None
        sp = set(self.solve_param_names)
        blocks = []
        self.hom_states = []
        for mi in range(self.kc):
            ci = self.equil_conds[mi]
            ref = self.solve_cond(point, p, ci)
            if ref is None:
                return None
            pv = self.joint_pv(point, p)
            dv = {k: v for k, v in d.items() if k in sp}
            try:
                sol = continue_resting_state(self.models[ci], list(self.real_state_names),
                                              list(self.solve_param_names), pv["param_vals"], dv,
                                              ref["sol"]["val_by"], p, int(T),
                                              forcings=self.held_of(ci) or None)
            except Exception:
                sol = None
            if sol is None or not sol.get("ok"):
                return None
            ptc = np.array(ref["ptc"], dtype=np.int64)
            ser = np.zeros((run.n_leaves, T), dtype=np.int64)
            self.hom_states.append({nm: [int(x) % p for x in vals[:T + 1]] for nm, vals in sol["val_by"].items()})
            for nm, vals in sol["val_by"].items():
                sl = run.slot(nm)
                if sl is None:
                    continue
                v = [int(x) % p for x in vals]
                ptc[sl] = v[0]
                ser[sl, :] = v[1:1 + T]
            for nm, v in dv.items():
                sl = run.slot(nm)
                if sl is not None:
                    ser[sl, 0] = int(v) % p
            for nm in self.zero_of(ci):
                sl = run.slot(nm)
                if sl is not None:
                    ptc[sl] = 0
                    ser[sl, :] = 0
            try:
                obs = _core.sym_obs_null_chain([[run.tapes[ci]]], run.n_leaves, run.n_states,
                                               self.z_slots_local, ptc % p, p, int(Nt), int(T), 1, [],
                                               ser % p)
            except Exception:
                obs = None
            if not ok(obs):
                return None
            loc = self.cond_local(mi, obs, {"sol": sol, "ptc": ptc, "ser": ser, "cont": True}, p)
            if loc is None or loc["S"] is None:
                return None
            blocks.append(series_embed(loc["S"], int(loc["N"]), list(self.wide_cols[mi]),
                                       self.nz_wide, p))
        return np.vstack(blocks)

    def pinned(self):
        """Per-condition state columns a condition holds at zero."""
        base = set(self.state_base)
        return [f"{z}|c{mi + 1}" for mi in range(self.kc) for z in self.zero_of(self.equil_conds[mi])
                if z in base]
