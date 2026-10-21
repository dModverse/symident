"""Kernel without a resting state: all conditions stacked, with relation rows of recast atoms where present."""

import numpy as np
import sympy as spy

from ... import _core
from ...errors import SYMPY_FAILS, swallow
from ...gfp import PRIMES, exp_rows, invmod, tape_fn
from ...sym.algebraic import radical_relation
from ...sym.parse import model_sympify
from ...sym.taylor import NoSeriesError, series_of
from ..fields import series_const, series_stack
from .context import ok, strs


class PlainSystem:
    """Kernels over the shared coordinates of every condition."""

    def __init__(self, run):
        self.run = run
        multi = run.multi
        self.transient = bool(multi.get("power_recast") or multi.get("exp_atoms"))
        self.recast = [dict(r, inverted=False) for r in (multi.get("power_recast") or [])] \
            if self.transient else []
        self.atom_names = (strs(multi.get("recast_atom_names")) + strs(multi.get("exp_atoms"))) \
            if self.transient else []
        self.exp_rel = multi.get("exp_relation")
        self.exp_rel_fn = tape_fn(self.exp_rel, run.n_leaves) if self.transient and multi.get("exp_atoms") else None
        zi = {z: i for i, z in enumerate(run.znames)}
        self.recast_rel = [{
            "Ecol": zi.get(str(rc["E"])), "base_col": zi.get(str(rc["base"])),
            "exp_col": zi.get(str(rc["exp"])), "Lcol": zi.get(str(rc["L"])),
            "Eslot": run.slot(str(rc["E"])), "base_slot": run.slot(str(rc["base"])),
            "exp_slot": run.slot(str(rc["exp"])), "Lslot": run.slot(str(rc["L"]))}
            for rc in self.recast]

    def relation_rows(self, point, p):
        """The linearized recast relations at the point; None where a base vanishes."""
        run = self.run
        nz = len(run.znames)
        rel, seen_l = [], set()
        for rc in self.recast_rel:
            if None in (rc["base_slot"], rc["Eslot"], rc["exp_slot"], rc["Lslot"]):
                # E and L follow from a given initial value of the base, not from a relation of leaves
                continue
            base0 = int(point[rc["base_slot"]]) % p
            if base0 == 0:
                return None
            invb = invmod(base0, p)
            E0 = int(point[rc["Eslot"]]) % p
            exp0 = int(point[rc["exp_slot"]]) % p
            L0 = int(point[rc["Lslot"]]) % p
            r1 = np.zeros(nz, dtype=np.int64)
            if rc["Ecol"] is not None:
                r1[rc["Ecol"]] = 1
            if rc["base_col"] is not None:
                r1[rc["base_col"]] = (p - exp0 * (E0 * invb % p) % p) % p
            if rc["exp_col"] is not None:
                r1[rc["exp_col"]] = (p - E0 * L0 % p) % p
            rel.append(r1)
            if rc["Lcol"] not in seen_l:
                seen_l.add(rc["Lcol"])
                r2 = np.zeros(nz, dtype=np.int64)
                if rc["Lcol"] is not None:
                    r2[rc["Lcol"]] = 1
                if rc["base_col"] is not None:
                    r2[rc["base_col"]] = (p - invb) % p
                rel.append(r2)
        if self.exp_rel_fn is not None:
            e_m = exp_rows(self.exp_rel, self.exp_rel_fn, run.znames,
                           np.asarray(point, dtype=np.int64), p)
            if e_m is None:
                return None
            rel.extend(list(e_m))
        return np.vstack(rel) if rel else np.zeros((0, nz), dtype=np.int64)

    def with_relations(self, o, point, p):
        """Observability rows stacked with the relation rows at the point."""
        run = self.run
        nz = len(run.znames)
        if not ok(o):
            return {"ok": False}
        o_r = np.asarray(o["R"], dtype=np.int64).reshape(int(o["rank"]), nz)
        rel_m = self.relation_rows(point, p)
        if rel_m is None:
            return {"ok": False}
        rr = _core.sym_rref_mod(np.vstack([o_r, rel_m]).astype(np.float64), p)
        res = {"ok": True, "R": np.asarray(rr["R"], dtype=np.int64).reshape(int(rr["rank"]), nz),
               "pivots": np.atleast_1d(rr["piv"]).astype(np.int64), "rank": int(rr["rank"]), "dim": nz}
        if run.has_gaps:
            Nn = int(o["N"])
            res = series_stack(res, [np.asarray(o["S"], dtype=np.int64).reshape(-1, nz * Nn),
                                     series_const(rel_m, Nn)], nz, Nn, p, run.cores_call)
        return res

    def kcall(self, point, p, Nt, Mtot=0, solve_fn=None, rank_only=False):
        """Observability kernel over all tapes or gap chains at `point` mod `p`, with relation rows when transient."""
        run = self.run
        pt = np.asarray(point, dtype=np.int64)
        if run.has_gaps:
            o = _core.sym_obs_null_chain([run.chain_tapes(idx) for idx in run.chain_groups], run.n_leaves,
                                         run.n_states, run.z_slots, pt, p, run.exact_nt(Nt, Mtot), int(Mtot),
                                         run.cores_call, [] if self.transient else list(run.nt_cap))
        else:
            o = _core.sym_obs_null_multi(run.tapes, run.n_leaves, run.n_states, run.z_slots, pt, p,
                                         run.exact_nt(Nt), run.cores_call)
        return run.modulo_nuisance(self.with_relations(o, point, p) if self.transient else o, p)

    # ---- homotopy along parameter lines ---------------------------------------------------

    hom_kind = "initial values"
    hom_tries = 32
    solve_held = ()

    def hom_prepare(self, point):
        """Moves every leaf; fits the radical initial values, each with the factor of its relation zero at `point`."""
        run = self.run
        self.hom_params = list(run.leaf_names)
        self.nz_wide = self.n_shared = len(run.znames)
        self.n_st = 0
        self.equil_conds, self.models, self.ic_expr = [], [], []
        names = []
        p = PRIMES[0]
        env = {spy.Symbol(nm): int(point[i]) % p for i, nm in enumerate(run.leaf_names)}
        for ci, t in enumerate(run.multi["tapes"]):
            ics, rel = {}, []
            for line in strs(t.get("jet_ic")):
                x, e = (u.strip() for u in line.split("=", 1))
                try:
                    ex = model_sympify(e)
                    facs = radical_relation(x, ex)
                    if facs is None:
                        continue
                    at = dict(env)
                    at[spy.Symbol(x)] = series_of(ex, env, 1, p)[0]
                    hold = [f for f in facs if series_of(f, at, 1, p)[0] == 0]
                except (NoSeriesError, *SYMPY_FAILS):
                    continue
                if len(hold) == 1:
                    ics[x] = ex
                    rel.append(f"{x} = {hold[0]}")
            self.models.append(rel)
            self.ic_expr.append(ics)
            if ics:
                self.equil_conds.append(ci)
                names += [x for x in ics if x not in names]
        self.real_state_names = names
        self.kc = len(self.equil_conds)
        self.model_key = ["\n".join(m) for m in self.models]
        self.hom_states = []

    def zero_of(self, ci):
        return []

    def hom_line(self, point, d, T, p, Nt):
        """Series rows of every chain along theta0 + t d; the radical initial values as series in hom_states."""
        run = self.run
        if self.transient:
            return None
        ser = np.zeros((run.n_leaves, T), dtype=np.int64)
        line = {}
        for i, nm in enumerate(run.leaf_names):
            v = int(d.get(nm, 0)) % p
            ser[i, 0] = v
            line[spy.Symbol(nm)] = [int(point[i]) % p, v]
        ptc = np.asarray(point[:run.n_leaves], dtype=np.int64) % p
        blocks = []
        chains = [run.chain_tapes(idx) for idx in run.chain_groups] if run.has_gaps else [[tp] for tp in run.tapes]
        caps = list(run.nt_cap) if run.has_gaps and len(run.nt_cap) == len(chains) else [None] * len(chains)
        for chain, cap in zip(chains, caps):
            try:
                o = _core.sym_obs_null_chain([chain], run.n_leaves, run.n_states, run.z_slots, ptc, p, int(Nt),
                                             int(run.mtot) if run.has_gaps else 0, 1, [] if cap is None else [cap],
                                             ser, int(T))
            except Exception as exc:
                swallow(exc, "homotopy series")
                return None
            if not ok(o):
                return None
            blocks.append(np.asarray(o["S"], dtype=np.int64).reshape(int(o["rank_s"]), -1))
        try:
            self.hom_states = [{x: series_of(e, line, T + 1, p) for x, e in self.ic_expr[ci].items()}
                               for ci in self.equil_conds]
        except NoSeriesError:
            return None
        return np.vstack(blocks)

    def hom_env(self, point, q, mi_plain=None):
        """Leaves at `point`, radical initial values as X_c<k> (X in `mi_plain`); None where one has no root mod q."""
        run = self.run
        env = {nm: int(point[i]) % q for i, nm in enumerate(run.leaf_names)}
        vals = {spy.Symbol(k): v for k, v in env.items()}
        try:
            for mi, ci in enumerate(self.equil_conds):
                for x, e in self.ic_expr[ci].items():
                    v = series_of(e, vals, 1, q)[0]
                    env[f"{x}_c{mi + 1}"] = v
                    if mi == mi_plain:
                        env[x] = v
        except NoSeriesError:
            return None
        return env

    def block_calls(self):
        """The kernel per tape or chain, where the saturation decides the Lie order."""
        return self.run.obs_block_calls()

    def block_profiles(self):
        """The rank per Lie order of a block from one jet."""
        return self.run.obs_block_profiles()
