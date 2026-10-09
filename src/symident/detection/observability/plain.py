"""The kernel without a resting state: all conditions stacked, with the relation rows of
recast atoms (E = base^exp, L = log base, exponential leaves) where present."""

import numpy as np

from ... import _core
from ...gfp import exp_rows, invmod, tape_fn
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
        """The linearised recast relations at the point; None where a base vanishes."""
        run = self.run
        nz = len(run.znames)
        rel, seen_l = [], set()
        for rc in self.recast_rel:
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
        """The observability kernel over all tapes or gap chains at `point` mod `p`, with the relation rows
        when transient."""
        run = self.run
        pt = np.asarray(point, dtype=np.int64)
        if run.has_gaps:
            o = _core.sym_obs_null_chain([run.chain_tapes(idx) for idx in run.chain_groups], run.n_leaves,
                                         run.n_states, run.z_slots, pt, p, int(Nt), int(Mtot),
                                         run.cores_call, [] if self.transient else list(run.nt_cap))
        else:
            o = _core.sym_obs_null_multi(run.tapes, run.n_leaves, run.n_states, run.z_slots, pt, p,
                                         int(Nt), run.cores_call)
        return self.with_relations(o, point, p) if self.transient else o

    def block_calls(self):
        """The kernel per tape or chain, where the saturation decides the Lie order."""
        return self.run.obs_block_calls()

    def block_profiles(self):
        """The rank per Lie order of a block from one jet."""
        return self.run.obs_block_profiles()
