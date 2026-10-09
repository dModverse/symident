"""Observability over GF(p) for all conditions: the saturated kernel, its certified
Lie order, the exact scalings peeled off and the residual directions reconstructed."""

import re
import warnings

import numpy as np

from ... import _core, state
from ...gfp import PRIMES, in_span, null_residues
from ...par import thread_map
from ...rexpr import sym_sort
from ...sym.msolve import msolve_timeout
from ..classify import joint_expand_scal, peel_scalings, report_physical
from ..fields import (
    exact_scaling_set,
    field_check,
    field_proof,
    field_setup,
    lie_certificate,
    lie_reach_max,
    own_fields,
    rank_of,
    rank_score,
    regime_keys,
    static_fields,
)
from ..rank import nullspace_basis, saturate_certify
from .context import Run, ok, strs
from .guard import saturation_guard
from .joint import JointSystem
from .plain import PlainSystem

_CSUFFIX = re.compile(r"\|c[0-9]+$")


def _plateau():
    return max(1, int(state.env_value("LIEPLATEAU", "3")))


def _rational_forms(exprs):
    """Whether every expression is rational in its symbols, model names read as symbols."""
    from ...sym.parse import _is_rational_expr, _make_local_parse
    try:
        _, parse = _make_local_parse([str(x) for x in exprs])
        return all(_is_rational_expr(parse(str(x).replace("^", "**"))) for x in exprs)
    except Exception:
        return False


class Analysis:
    """One observability analysis of the compiled tapes of all conditions.

    In joint mode the coordinates are wide: the shared columns followed by one block of
    state columns per resting condition.
    """

    def __init__(self, multi, closed_form=False, cores=1, equil_zero_states=(),
                 cond_zero_states=(), t0events=(), n_conditions=None, chain_of=None, n_gaps=0,
                 implicit_steady_state=False, control=None, verify=False, codim_spec=None):
        from ...options import reconst_control
        self.ctrl = dict(control or reconst_control())
        self.closed_form = bool(closed_form)
        self.verify = bool(verify)
        self.codim_spec = codim_spec
        self.n_conditions = n_conditions
        self.run = run = Run(multi, cores, self.ctrl, chain_of, n_gaps, t0events, cond_zero_states,
                             equil_zero_states)
        self.joint = bool(multi.get("joint_steady_state")) and bool(implicit_steady_state)
        self.n_aug = run.n_leaves
        self.leaf_names_aug = list(run.leaf_names)
        self.transient = False
        self.atom_names = []
        self.aux_leaves = []
        if self.joint:
            self.sys = sys = JointSystem(run)
            self.nz = sys.nz_wide
            self.znames = sys.znames_wide
            self.z_slots = sys.z_slots_wide
            self.multi = dict(multi, z_state_names=sys.z_state_names_wide)
            pn = set(sys.param_names)
            self.aux_leaves = [i for i, nm in enumerate(self.leaf_names_aug)
                               if nm in set(run.leaf_names) - pn]
            self.ctrl["relevance_cap"] = self.ctrl["relevance_cap"] + sys.n_st
        else:
            self.sys = sys = PlainSystem(run)
            self.nz = len(run.znames)
            self.znames = run.znames
            self.z_slots = run.z_slots
            self.multi = multi
            self.transient = sys.transient
            self.atom_names = sys.atom_names
            if self.transient:
                self.aux_leaves = [i for i, nm in enumerate(self.leaf_names_aug) if nm in set(self.atom_names)]
        self.recast = sys.recast
        self.zsn = set(strs(self.multi.get("z_state_names")))
        self.can_batch = not run.ss_constraint and not run.has_gaps and not self.transient
        self.scal_memo = {}
        self.rank_cap = self.nz
        self.scal_ws = None
        self.stacked_first = self.stacked_quick = False
        self.static_rows = None

    # ---- kernels ----------------------------------------------------------------------

    def kcall4(self, point, p, Nt, Mtot=0):
        """The observability kernel at `point` mod `p`, Lie order `Nt` and gap order `Mtot`."""
        return self.sys.kcall(point, p, Nt, Mtot)

    def kcall_sat(self, point, p, Nt, Mtot=0):
        """The kernel for the saturation, rank only in joint mode unless WIDESAT is set."""
        if self.joint and not state.env("WIDESAT"):
            return self.sys.kcall(point, p, Nt, Mtot, rank_only=True)
        return self.sys.kcall(point, p, Nt, Mtot)

    def kcall(self, point, p, Nt):
        """The kernel at the saturated gap order."""
        return self.sys.kcall(point, p, Nt, self.run.mtot)

    @property
    def chunked(self):
        """Whether the coupled per-chunk joint kernel applies."""
        return self.joint and self.run.has_gaps

    def kbatch(self, point_list, prime_vec, Nt):
        """The kernel over many (point, prime) pairs, batched where the path allows it."""
        run, n = self.run, len(point_list)

        def leaves():
            return np.vstack([np.asarray(pp[:run.n_leaves], dtype=np.int64) for pp in point_list])
        if self.can_batch and n > 0:
            return _core.sym_obs_null_batch(run.tapes, run.n_leaves, run.n_states, run.z_slots, leaves(),
                                            list(prime_vec), int(Nt), run.cores)
        if run.has_gaps and not run.ss_constraint and not self.joint and not self.transient and \
                n > 0 and not state.env("NOPOINTBATCH"):
            return _core.sym_obs_null_chain_point_batch([run.chain_tapes(idx) for idx in run.chain_groups],
                                                        run.n_leaves, run.n_states, run.z_slots, leaves(),
                                                        list(prime_vec), int(Nt), int(run.mtot), run.cores,
                                                        list(run.nt_cap))
        if self.transient and not run.has_gaps and n > 0 and not state.env("NOPOINTBATCH"):
            os_ = _core.sym_obs_null_batch(run.tapes, run.n_leaves, run.n_states, run.z_slots, leaves(),
                                           list(prime_vec), int(Nt), run.cores)
            return [self.sys.with_relations(os_[i], point_list[i], prime_vec[i]) for i in range(n)]
        if self.chunked and n > 0 and not state.env("NOCHUNK"):
            per = state.get("chunk_points", 2 * max(1, run.cores))
            out = []
            for s in range(0, n, per):
                out.extend(self.sys.chunk(point_list[s:s + per], prime_vec[s:s + per], Nt))
            return out
        if self.joint:
            self.sys.warm_solves(point_list, prime_vec)
        cc_saved = run.cores_call
        warm_saved = self.sys.warm_off if self.joint else None
        if run.cores > 1:
            run.cores_call = 1
            if self.joint:
                self.sys.warm_off = True
        try:
            return run.par_map(range(n), lambda i: self.kcall(point_list[i], prime_vec[i], Nt))
        finally:
            run.cores_call = cc_saved
            if self.joint:
                self.sys.warm_off = warm_saved

    def dir_call(self):
        """Whether a direction annihilates all rows, from jets along it; None without the path."""
        run = self.run
        if run.has_gaps or run.ss_constraint or self.recast:
            return None

        def f(pt, q, Nt, v):
            o = _core.sym_obs_directional(run.tapes, run.n_leaves, run.n_states, run.z_slots,
                                          np.asarray(pt[:run.n_leaves], dtype=np.int64), q, int(Nt),
                                          np.asarray(v, dtype=np.float64), run.cores)
            if not ok(o):
                return None
            if not o.get("zero"):
                return False
            if not self.transient:
                return True
            rel_m = self.sys.relation_rows(pt, q)
            if rel_m is None:
                return None
            return bool(np.all(np.asarray(rel_m % q, dtype=object) @
                               (np.asarray(v, dtype=object) % q) % q == 0))
        return f

    def exact_scalings(self, fixed):
        """The exact scalings with `fixed` held, memoised."""
        key = "k:" + ",".join(sorted(fixed))
        if key not in self.scal_memo:
            self.scal_memo[key] = exact_scaling_set(self.multi, fixed)
        return self.scal_memo[key]

    # ---- saturation and the certified Lie order ---------------------------------------

    def _lower_caps(self, cap0, cap_rank):
        """Lower the deepest chain caps by bisection while the rank score stays; False
        when there is nothing to lower."""
        run = self.run
        ref = cap_rank()
        levels = sorted(set(cap0), reverse=True)
        if ref is None or len(levels) < 2:
            return False
        lo, hi = 0, len(levels) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            run.nt_cap = [min(c, levels[mid]) for c in cap0]
            if cap_rank() == ref:
                lo = mid
            else:
                hi = mid - 1
        run.nt_cap = [min(c, levels[lo]) for c in cap0] if lo > 0 else cap0
        return True

    def _sat_caps(self, orders, point, Mtot, first):
        """Each chain at its own order plus the plateau during the saturation."""
        run = self.run
        nb = self.sys.kc if self.joint else (len(run.chain_groups) if run.chain_groups else 0)
        if not (run.has_gaps and not state.env("NOCAP") and len(orders) == nb
                and (self.joint or not run.ss_constraint)):
            return
        cap0 = [int(o) + _plateau() for o in orders]
        run.nt_cap = cap0
        if first is None or state.env("NOJOINTCAP"):
            return
        Nt = max(cap0)

        def cap_rank():
            try:
                r = self.kcall_sat(point, PRIMES[0], Nt, Mtot)
            except Exception:
                r = None
            return rank_score(r) if ok(r) else None
        if self._lower_caps(cap0, cap_rank) and state.env("LIEDIAG"):
            print(f"[liediag] joint caps in the saturation at gap order {Mtot}: "
                  f"{','.join(map(str, run.nt_cap))}")

    def _block_map(self, block_call, block_profile):
        run = self.run
        if block_profile is not None and len(block_profile) == len(block_call) and len(block_call) > 1:
            return lambda f, xs: [f(x) for x in xs]
        if (not run.ss_constraint or self.joint) and run.cores > 1 and len(block_call) > 1 \
                and not state.env("SERIALBLOCKS"):
            return lambda f, xs: thread_map(f, xs, run.cores)
        return None

    def _warm_probe(self, pts, primes):
        sys = self.sys
        if not sys.fwd_on:
            return sys.warm_solves(pts, primes, conds=[sys.equil_conds[0]])
        other = [i for i, q in enumerate(primes) if q != PRIMES[0]]
        if other:
            sys.warm_solves([pts[i] for i in other], [primes[i] for i in other])
        return None

    def _scaling_cap(self):
        """With continued conditions the exact scalings bound the rank from above."""
        run, sys = self.run, self.sys
        if not (self.joint and any(sys.cont_eligible(ci) for ci in sys.equil_conds)):
            return
        not_fixed = {z for z in self.znames if z not in self.zsn} | set(sys.state_base)
        self.scal_ws = self.exact_scalings([l for l in run.leaf_names if l not in not_fixed])
        shared = self.znames[:sys.n_shared]
        rows = []
        for v in self.scal_ws:
            x = np.zeros(sys.n_shared, dtype=np.int64)
            for k, w in v.items():
                if k in shared:
                    x[shared.index(k)] = int(w) % PRIMES[0]
            rows.append(x)
        if rows:
            self.rank_cap = self.nz - int(_core.sym_rref_mod(np.vstack(rows).astype(np.float64),
                                                             PRIMES[0])["rank"])

    def saturate(self):
        """The saturated kernel at a generic base point, or None."""
        run, sys, multi = self.run, self.sys, self.multi
        block_call = sys.block_calls()
        block_profile = sys.block_profiles()
        budget = None if run.has_gaps or self.codim_spec is None else \
            int(self.codim_spec) + 2 * len(self.recast) + int(multi.get("exp_codim") or 0)
        block_map = self._block_map(block_call, block_profile)
        if self.joint:
            reach = [run.chain_groups[run.chain_idx[ci]] if run.has_gaps else [ci] for ci in sys.equil_conds]
        elif run.has_gaps:
            reach = run.chain_groups
        else:
            reach = [[i] for i in range(len(multi["tapes"]))]
        nt_min_block = [lie_reach_max([multi["tapes"][i] for i in idx]) for idx in reach]
        if len(nt_min_block) != len(block_call):
            nt_min_block = None
        self._scaling_cap()
        self.stacked_first = (((not run.ss_constraint and not run.has_gaps)
                               or (self.joint and run.has_gaps and self.rank_cap < self.nz
                                   and not state.env("NOSTACKEDSS")))
                              and not self.recast and self.nz > state.get("stacked_min", 150)
                              and state.get("stacked_first") is not False and not state.get("inner")
                              and state.get("order_hint") is None)
        self.stacked_quick = self.stacked_first and bool(state.get("stacked_quick"))
        if self.stacked_first:
            block_call, block_profile, block_map = None, None, None
        sc = saturate_certify(self.kcall_sat, self.n_aug, self.nz, run.max_gap,
                              warm=self._warm_probe if self.joint else (lambda pts, primes: None),
                              probe_block=max(1, min(8, run.cores)), block_call=block_call,
                              budget=budget, block_map=block_map, nt_min=lie_reach_max(multi["tapes"]),
                              nt_min_block=nt_min_block, prep=sys.sat_prep if self.joint else None,
                              rank_cap=self.rank_cap, block_profile=block_profile, set_caps=self._sat_caps,
                              checks=not (self.stacked_first and (not self.stacked_quick or run.ss_constraint)))
        if sc is None:
            if run.ss_constraint and run.ss_why is not None:
                warnings.warn("no steady-state point over the finite field after "
                              f"the generic-point retries ({run.ss_why}). The equilibrate constraint "
                              "could not be evaluated.", stacklevel=2)
            return None
        if sc["ref"].get("block_rank"):
            rc = self.kcall4(sc["point0"], PRIMES[0], sc["nt_used"], sc.get("mtot_used") or 0)
            if not ok(rc):
                return None
            sc.update(ref=rc, rank=rc["rank"], rank_s=rank_of(rc), pivots=rc["pivots"])
        run.stage("saturated")
        return sc

    def _cert_segments(self, sc):
        """The certificate inputs: chains of tapes and a seed per chain."""
        run, sys = self.run, self.sys
        if self.joint:
            segs = sys.chains if run.has_gaps else [[run.tapes[ci]] for ci in sys.equil_conds]
            seeds = []
            for ci in sys.equil_conds:
                sc0 = sys.solve_cond(sc["point0"], PRIMES[0], ci)
                seeds.append(None if sc0 is None else sc0["ptc"])
            return segs, seeds
        segs = [run.chain_tapes(idx) for idx in run.chain_groups] if run.has_gaps else \
            [[t] for t in run.tapes]
        return segs, [np.asarray(sc["point0"][:run.n_leaves], dtype=np.int64)] * len(segs)

    def certify(self, sc):
        """The certified Lie order per block, then the per-chain caps lowered."""
        run = self.run
        run.nt_cap = []
        sc["certified"] = self.rank_cap < self.nz and rank_of(sc["ref"]) >= self.rank_cap
        hint = state.get("order_hint")
        if hint is not None and hint.get("certified") is not False and \
                list(sc.get("block_orders") or []) == list(hint.get("block_orders") or []):
            sc["certified"] = True
            sc["cert_orders"] = True
        if not sc["certified"] and not self.stacked_first and not state.env("NOCERT"):
            self._certificate(sc)
        nb = self.sys.kc if self.joint else (len(run.chain_groups) if run.chain_groups else -1)
        if run.has_gaps and (self.joint or not run.ss_constraint) and \
                sc.get("block_orders") is not None and len(sc["block_orders"]) == nb and not state.env("NOCAP"):
            add = 0 if sc.get("cert_orders") else _plateau()
            run.nt_cap = [int(o) + add for o in sc["block_orders"]]
        run.stage("Lie order certified")
        if run.nt_cap and not state.env("NOJOINTCAP"):
            mt = int(sc.get("mtot_used") or 0)

            def cap_rank():
                try:
                    r = self.kcall4(sc["point0"], PRIMES[0], sc["nt_used"], mt)
                except Exception:
                    r = None
                return rank_score(r) if ok(r) else None
            self._lower_caps(list(run.nt_cap), cap_rank)
            if state.env("LIEDIAG"):
                print(f"[liediag] joint block Lie orders: {','.join(map(str, run.nt_cap))}")

    def _cert_start(self, sc, n):
        """Per block the saturated order plus the plateau, where the saturation knows it."""
        bo = sc.get("block_orders")
        if state.env("CERTFROM16") or bo is None or len(bo) != n:
            return None
        return [int(o) + _plateau() + 1 for o in bo]

    def _certificate(self, sc):
        """The Lie order per block past which the rank provably stays."""
        run = self.run
        segs, seeds = self._cert_segments(sc)
        P1 = PRIMES[0]
        own_threads = not run.has_gaps and not self.joint
        threads = run.cores if own_threads else 1
        mapf = (lambda f, xs: thread_map(f, xs, run.cores)) if threads == 1 and run.cores > 1 else None
        orders = None
        if all(s is not None and len(s) > 0 for s in seeds):
            try:
                orders = lie_certificate(segs, seeds, run.n_leaves, run.n_states,
                                         self.sys.z_slots_local if self.joint else run.z_slots, P1,
                                         cap=max(64, 4 * (len(self.z_slots) + run.n_states)),
                                         map_fn=mapf, threads=threads,
                                         start=self._cert_start(sc, len(segs)))
            except Exception as e:
                if state.env("LIEDIAG"):
                    print(f"[liediag] certificate failed: {e}")
                orders = None
        if orders is not None and len(orders) == len(segs) and not all(o is None for o in orders):
            bo = list(sc["block_orders"]) if sc.get("block_orders") is not None and \
                len(sc["block_orders"]) == len(orders) else [int(sc["nt_used"])] * len(orders)
            ordv = [bo[i] + _plateau() if o is None else int(o) for i, o in enumerate(orders)]
            need = max([1, *ordv])
            try:
                rc = self.kcall4(sc["point0"], P1, need, sc.get("mtot_used") or 0)
            except Exception:
                rc = None
            if ok(rc):
                sc.update(ref=rc, nt_used=need, rank=rc["rank"], rank_s=rank_of(rc), pivots=rc["pivots"],
                          block_orders=ordv, block_driver=int(np.argmax(ordv)),
                          certified=all(o is not None for o in orders),
                          uncertified=[i for i, o in enumerate(orders) if o is None], cert_orders=True)
        if state.env("LIEDIAG"):
            print(f"[liediag] certified block orders: {orders}")

    # ---- the analysis -----------------------------------------------------------------

    def _mark_states(self):
        """The base point with every state column at one."""
        self.sc["point0"] = np.array(self.point0_solved, dtype=np.int64)
        if self.joint:
            for col, z in enumerate(self.znames):
                if z in self.zsn:
                    self.sc["point0"][self.z_slots[col]] = 1

    def execute(self):
        """The observability result: rank, saturation details and the non-identifiable directions; None
        when the saturation fails."""
        run = self.run
        sc = self.saturate()
        if sc is None:
            return None
        self.certify(sc)
        self.sc = sc
        self.point0_solved = np.array(sc["point0"], dtype=np.int64)
        self._mark_states()
        run.mtot = int(sc.get("mtot_used") or 0)
        if run.has_gaps and sc.get("saturated_m") is False:
            warnings.warn(f"the gap power-series order hit the cap ({run.max_gap}) "
                          "before the rank stabilised; the reported rank is a sound (conservative) "
                          "lower bound. Raise reconst_control(gap_order_cap=) for the exact "
                          "generic-timing verdict.", stacklevel=2)
        nz, n_seg = self.nz, len(self.multi["tapes"])
        result = {"method": "observability", "engine": "analytic",
                  "conditions": n_seg if self.n_conditions is None else int(self.n_conditions),
                  "coordinates": list(self.znames), "segments": n_seg, "gap_order_used": run.mtot,
                  "identifiable": sc["rank_s"] == nz, "rank": int(sc["rank_s"]), "dim": int(nz),
                  "lie_order_used": int(sc["nt_used"]), "lie_order_driver": sc.get("block_driver"),
                  "lie_uncertified": sc.get("uncertified"), "lie_plateau": sc.get("plateau"),
                  "lie_certified": bool(sc.get("certified")), "lie_block_orders": sc.get("block_orders"),
                  "sat_score": rank_score(sc["ref"]), "non_identifiable": []}
        self.phys_params0 = [z for z in self.znames if z not in self.zsn]
        if self.joint:
            result["dim"] = len(self.phys_params0)
            if sc["rank_s"] == nz:
                result["rank"] = len(self.phys_params0)
        if self.transient:
            coords = [z for z in self.znames if z not in set(self.atom_names)]
            result["dim"] = len(coords)
            result["coordinates"] = coords
            if sc["rank_s"] == nz:
                result["rank"] = len(coords)
        self.gap_dirs = self._gap_dirs()
        if sc["rank_s"] == nz:
            return result
        if self.joint:
            self.scal_fixed = [l for l in run.leaf_names
                               if l not in set(self.phys_params0) | set(self.sys.state_base)]
        else:
            self.scal_fixed = [l for l in run.leaf_names if l not in set(self.znames)]
        if self.joint and any(self.sys.cont_active(ci) for ci in self.sys.equil_conds):
            done = self._continued(result)
            if done is not None:
                return done
        if sc["rank"] == nz:
            result["non_identifiable"] = self.gap_dirs
            return self._physical(result)
        return self._residual(result)

    def _gap_dirs(self):
        """Directions that depend on the time between events, by their support."""
        sc = self.sc
        if sc["rank_s"] >= sc["rank"]:
            return []
        sr = _core.sym_series_rank(np.asarray(sc["ref"]["S"], dtype=np.int64), int(self.nz),
                                   int(sc["ref"]["N"]), PRIMES[0], [], support=True, cores=int(self.run.cores))
        supp = [int(x) for x in np.atleast_1d(sr["support"])]
        return [{"support": sym_sort([self.znames[i] for i in supp]), "type": "general",
                 "closed_form": False, "reason": "depends on the time between events"}
                for _ in range(sc["rank"] - sc["rank_s"])]

    def _physical(self, result):
        """Joint mode reported in parameter space, a transient recast without its atoms."""
        if self.joint:
            result = report_physical(result, self.znames, list(self.zsn), self.sc, self.nz,
                                     PRIMES[0], "state_vector", flag_joint=True)
        if self.transient:
            result = report_physical(result, self.znames, self.atom_names, self.sc, self.nz,
                                     PRIMES[0], "recast_vector")
        return result

    def static_fields(self):
        """Fields of the static subsystems of the resting conditions at the solved point."""
        run, sys = self.run, self.sys
        return static_fields(self.multi, sys.models, sys.equil_conds,
                             [run.chain_groups[run.chain_idx[ci]] if run.has_gaps else [ci]
                              for ci in sys.equil_conds],
                             self.phys_params0, run.leaf_names, self.point0_solved,
                             strs(self.multi.get("forcings")), [sys.zero_of(ci) for ci in sys.equil_conds])

    def _continued(self, result):
        """Continued conditions give the kernel over the series only: the exact scalings and
        static fields name the directions, or the kernel is retaken at a point where all
        conditions solve directly. None to go on with the residual directions."""
        sc = self.sc
        ker_dim = int(self.nz - sc["rank_s"])
        rank0, coords0 = result["rank"], result["coordinates"]
        result["rank"] = int(result["dim"] - ker_dim)
        result["coordinates"] = self.phys_params0
        chosen = self._scalings_and_statics(ker_dim)
        if len(chosen) < ker_dim and self.closed_form and not self.stacked_quick and \
                not self.stacked_first and not state.env("NOCONTDIRECT") and self._direct_kernel(ker_dim):
            result["rank"] = rank0
            result["coordinates"] = coords0
            self.gap_dirs = self._gap_dirs()
            return None
        result["non_identifiable"] = chosen + [
            {"support": [], "type": "general", "closed_form": False,
             "reason": "not reconstructed along the continuation"} for _ in range(ker_dim - len(chosen))]
        if len(chosen) == ker_dim:
            result["rank_proven"] = ker_dim
            result["rank_proof"] = ("exact scalings and static subsystems"
                                    if any(d.get("route") == "static subsystem" for d in chosen)
                                    else "exact scalings")
        if self.stacked_first and result.get("rank_proven") is None and not self.stacked_quick:
            return {"rerun_blocks": True}
        result["stacked"] = bool(self.stacked_first)
        return result

    def _scalings_and_statics(self, ker_dim):
        """Independent exact scalings, then static fields, up to the kernel dimension."""
        P0, pp = PRIMES[0], self.phys_params0
        ws = self.scal_ws if self.scal_ws is not None else self.exact_scalings(self.scal_fixed)
        chosen = []
        rows = np.zeros((0, len(pp)), dtype=np.int64)

        def grows(r2):
            return int(_core.sym_rref_mod(r2.astype(np.float64), P0)["rank"]) > rows.shape[0]
        for v in ws:
            x = np.zeros(len(pp), dtype=np.int64)
            for k, w in v.items():
                if k in pp:
                    x[pp.index(k)] = int(w) % P0
            r2 = np.vstack([rows, x])
            if grows(r2):
                rows = r2
                vv = {k: str(w) for k, w in v.items() if k in pp and str(w) != "0"}
                chosen.append({"support": sym_sort(list(vv)), "vector": vv, "type": "scaling",
                               "closed_form": True})
            if len(chosen) == ker_dim:
                break
        if len(chosen) < ker_dim and not state.env("NOSTATIC"):
            stf = self.static_fields()
            for f in (stf or []):
                r2 = np.vstack([rows, f["lval"]])
                if grows(r2):
                    rows = r2
                    chosen.append({k: v for k, v in f.items() if k not in ("val", "lval")})
                if len(chosen) == ker_dim:
                    break
            if self.run.timing:
                print(f"[sym] static subsystems: {len(stf or [])} field(s), {len(chosen)} of "
                      f"{ker_dim} directions with the scalings")
        return chosen

    def _direct_kernel(self, ker_dim):
        """The kernel at a point where all conditions solve without continuation."""
        sc, sys, nz = self.sc, self.sys, self.nz
        cont_saved = list(sys.cont_off)
        for i in range(len(sys.cont_off)):
            sys.cont_off[i] = True
        with msolve_timeout(float(state.env_value("DIRECTSECS", "120"))):
            try:
                pt_d = sys.sat_prep(self.point0_solved, rounds=2)
            except Exception:
                pt_d = None
            if pt_d is None:
                pt_d = self.point0_solved
            try:
                rc = self.kcall4(pt_d, PRIMES[0], sc["nt_used"], sc.get("mtot_used") or 0)
            except Exception as e:
                rc = {"ok": False, "why": str(e)}
        direct = ok(rc) and nz - rank_of(rc) == ker_dim
        if state.env("LIEDIAG"):
            print("[liediag] direct kernel for the continued conditions: " +
                  (f"no point: {rc.get('why')}" if not ok(rc) else
                   f"dimension {nz - rank_of(rc)} (rank {rc['rank']}, series {rank_of(rc)}), "
                   f"series kernel {ker_dim}"))
        if not direct:
            sys.cont_off[:] = cont_saved
            return False
        sc.update(ref=rc, rank=rc["rank"], rank_s=rank_of(rc), pivots=rc["pivots"])
        self.point0_solved = np.array(pt_d, dtype=np.int64)
        self._mark_states()
        return True

    def _peel(self):
        """The exact scalings of the model as tangents in the nullspace."""
        from ...sym.scaling import scaling_symmetries_multi
        model_lines = [strs(t.get("model_lines")) for t in self.multi["tapes"]]
        obs_lines = [strs(t.get("obs_lines")) for t in self.multi["tapes"]]
        keys = ["\n".join([*m, "\x1f", *o]) for m, o in zip(model_lines, obs_lines)]
        keep = [i for i, k in enumerate(keys) if k not in keys[:i]]
        try:
            res = scaling_symmetries_multi(per_cond_model=[model_lines[i] for i in keep],
                                            per_cond_obs=[obs_lines[i] for i in keep],
                                            inputs=strs(self.multi.get("forcings")) or None,
                                            fixed=self.scal_fixed or None, recast=self.recast or None)
        except Exception:
            res = None
        if res is None:
            return [], np.zeros((self.nz, 0), dtype=np.int64)
        if self.joint:
            sys = self.sys
            res = joint_expand_scal(res, sys.state_base, sys.kc, sys.held_param_of, sys.pinned())
        peel = peel_scalings(res, self.znames, self.nz, np.asarray(self.sc["point0"])[self.z_slots],
                             PRIMES[0], self.N)
        return peel["scaling"], peel["Bmat"]

    def _residual(self, result):
        """Scalings peeled, the residual directions reconstructed, the rank proven."""
        from .reconstruct import Reconstruction
        sc, run, nz, P = self.sc, self.run, self.nz, PRIMES[0]
        piv = {int(x) for x in np.atleast_1d(sc["pivots"])}
        self.free_cols = [c for c in range(nz) if c not in piv]
        self.N = nullspace_basis(sc["ref"], self.free_cols, P)
        run.stage("joint cap and checks")
        self.scaling, B = self._peel()
        scal_cols = B.shape[1]
        run.stage("scalings peeled")
        residual = []
        for fc in self.free_cols:
            bf = null_residues(sc["ref"], fc, P)
            if not in_span(B, bf, nz, P):
                residual.append(fc)
                B = np.concatenate([B, bf.reshape(nz, 1)], axis=1)
        self.scal_tangents = B[:, :scal_cols]
        self.scal_rows = B[:, :scal_cols].T.copy() if scal_cols > 0 else np.zeros((0, nz), dtype=np.int64)
        self.field_chains = run.chain_groups if run.has_gaps else [[i] for i in range(len(self.multi["tapes"]))]
        self.field_joint = {"conds": self.sys.equil_conds, "solve": self.sys.solve_cond} if self.joint else None
        if self.closed_form or (self.stacked_first and not self.stacked_quick):
            dirs = Reconstruction(self, residual).run()
            if dirs is None:
                return {"rerun_blocks": True}
            dirs = self._eliminate_states(self._complete_forward(dirs))
            result["non_identifiable"] = self._independent(dirs) + self.gap_dirs
        else:
            support = []
            for fc in residual:
                v = null_residues(sc["ref"], fc, P)
                support.append({"support": sym_sort([self.znames[i] for i in np.nonzero(v)[0]]),
                                "type": "general", "closed_form": False})
            result["non_identifiable"] = self.scaling + support + self.gap_dirs
        run.stage("reconstruction")
        result = self._physical(result)
        self._prove_by_scalings(result)
        run.stage("exact scalings proof")
        self._prove_by_fields(result)
        run.stage("field proof")
        kernel = {"coords": list(self.znames), "N": self.N, "p": P}
        if self.stacked_first:
            if result.get("rank_proven") is None and not self.stacked_quick:
                return {"rerun_blocks": True}
            result["kernel"] = kernel
            if not self.closed_form:
                result["non_identifiable"] = [d if d.get("type") == "scaling" else
                                             {"support": d.get("support"), "type": d.get("type"),
                                              "closed_form": False} for d in result["non_identifiable"]]
            run.stage("done (stacked)" if result.get("rank_proven") is None else "done (stacked, rank proven)")
            result["stacked"] = bool(run.ss_constraint)
            return result
        if self.verify and not sc.get("certified") and result.get("rank_proven") is None:
            try:
                result["verification"] = saturation_guard(self._guard_call, self.point0_solved,
                                                          sc["nt_used"], sc["rank_s"])
            except Exception as e:
                result["verification"] = {"ok": None, "method": "saturation guard", "reason": str(e)}
        result["kernel"] = kernel
        run.stage("done")
        return result

    def _kernel_point(self):
        """Leaf values of the base point with the resting states the kernel solved there;
        None where one resting value is not well defined (several equilibrated conditions)."""
        env = {nm: int(v) for nm, v in zip(self.leaf_names_aug, self.sc["point0"])}
        if not self.joint:
            return env
        conds = list(self.sys.equil_conds)
        if len(conds) != 1:
            return None
        try:
            sc0 = self.sys.solve_cond(self.sc["point0"], PRIMES[0], conds[0])
        except Exception:
            sc0 = None
        if sc0 is None:
            return None
        return {nm: int(v) for nm, v in zip(self.leaf_names_aug, sc0["ptc"])}

    def _in_kernel(self, vec, env):
        """Whether the closed form {name: expr} lies in the kernel at the base point on the
        physical coordinates; None when it does not evaluate there."""
        from ..closedform import eval_modq
        P = PRIMES[0]
        if env is None or not _rational_forms(list(vec.values())):
            return None
        phys = [i for i, z in enumerate(self.znames) if not _CSUFFIX.search(z)]
        v = np.zeros(len(phys), dtype=np.int64)
        for j, i in enumerate(phys):
            if self.znames[i] in vec:
                x = eval_modq(vec[self.znames[i]], env, P)
                if x is None:
                    return None
                v[j] = x % P
        Nphys = np.asarray(self.N, dtype=np.int64)[phys, :] % P
        return bool(in_span(Nphys, v, len(phys), P)) if np.any(v) else None

    def _complete_forward(self, dirs):
        """Forward closed forms that the kernel refutes, completed by their solved rates: a
        rate R solved from the resting equations moves with the coordinates it depends on,
        eta_R = sum_c dR/dc eta_c."""
        import sympy as spy
        lines = (self.multi.get("tapes") or [{}])[0].get("constraint_model")
        if not self.joint or not lines:
            return dirs
        try:
            eqs = [spy.sympify(ln.split("=", 1)[1].replace("^", "**")) for ln in lines]
        except (spy.SympifyError, TypeError, ValueError):
            return dirs
        out = []
        for d in dirs:
            vec, solved = d.get("vector"), [r for r in d.get("solved_rates") or [] if r]
            missing = [r for r in solved if isinstance(vec, dict) and r not in vec]
            if not d.get("closed_form") or not missing:
                out.append(d)
                continue
            try:
                comp = {k: spy.sympify(str(v).replace("^", "**")) for k, v in vec.items()}
                for r in missing:
                    R = spy.Symbol(r)
                    sol = next((spy.solve(spy.fraction(spy.together(e))[0], R) for e in eqs
                                if R in e.free_symbols), [])
                    if len(sol) != 1:
                        raise ValueError(r)
                    comp[r] = spy.cancel(sum((spy.diff(sol[0], spy.Symbol(c)) * v for c, v in comp.items()),
                                             spy.Integer(0)))
                full = {k: str(v).replace("**", "^") for k, v in comp.items() if v != 0}
                env = self._kernel_point()
                # an absent solved rate either stays or moves with the others: the form as
                # reconstructed stands unless the kernel refutes it
                if self._in_kernel(vec, env) is not False:
                    out.append(d)
                elif self._in_kernel(full, env) is not False:
                    out.append(dict(d, vector=full, support=sym_sort(list(full))))
                else:
                    out.append({"support": sym_sort(list(full)), "type": "general",
                                "closed_form": False, "reason": "no closed form on the kernel"})
            except (spy.SympifyError, TypeError, ValueError, NotImplementedError):
                out.append({"support": sym_sort(list(vec) + missing), "type": "general",
                            "closed_form": False, "reason": "a solved rate of the forward route "
                            "has no closed form"})
        return out

    def _eliminate_states(self, dirs):
        """Closed forms that hold resting states rewritten in the coordinates where the resting
        equations allow it: solved rates are substituted until no state remains."""
        import sympy as spy
        lines = (self.multi.get("tapes") or [{}])[0].get("constraint_model")
        states = set(self.multi.get("real_state_names") or []) - set(self.znames)
        if not self.joint or not lines or not states:
            return dirs
        coords = {z for z in self.znames if not _CSUFFIX.search(z)}
        try:
            eqs = [spy.sympify(ln.split("=", 1)[1].replace("^", "**")) for ln in lines]
        except (spy.SympifyError, TypeError, ValueError):
            return dirs
        out = []
        for d in dirs:
            vec = d.get("vector") if d.get("closed_form") else None
            if not isinstance(vec, dict) or any("|" in str(k) + str(v) for k, v in vec.items()):
                out.append(d)
                continue
            try:
                held = any(states & {str(x) for x in spy.sympify(str(v).replace("^", "**")).free_symbols}
                           for v in vec.values())
                out.append(self._rewrite_in_coords(d, vec, eqs, states, coords) if held else d)
            except (spy.SympifyError, TypeError, ValueError):
                out.append(d)
        return out

    @staticmethod
    def _rewrite_in_coords(d, vec, eqs, states, coords):
        import sympy as spy
        comp0 = {k: spy.sympify(str(v).replace("^", "**")) for k, v in vec.items()}

        def held(c):
            return {str(x) for e in c.values() for x in e.free_symbols} & states

        def normal(c):
            anchor = next((k for k in sorted(c) if c[k] != 0), None)
            if anchor is None:
                return None
            c = {k: spy.cancel(e / c[anchor]) for k, e in c.items()}
            return None if held(c) else c

        def solvable(e, c):
            num, den = spy.fraction(spy.together(e))
            cand = [x for x in num.free_symbols if str(x) in coords and x not in den.free_symbols
                    and any(x in v.free_symbols for v in c.values()) and spy.degree(num, x) == 1]
            return num, sorted(cand, key=str)

        def search(c, left):
            done = normal(c)
            if done is not None or not left:
                return done
            for i in left:
                num, cand = solvable(eqs[i], c)
                for x in cand:
                    sol = spy.solve(num, x)
                    if len(sol) != 1:
                        continue
                    r = search({k: spy.cancel(v.subs(x, sol[0])) for k, v in c.items()},
                               [j for j in left if j != i])
                    if r is not None:
                        return r
            return None
        comp = search(comp0, list(range(len(eqs))))
        if comp is None:
            return d
        den = spy.lcm([spy.fraction(c)[1] for c in comp.values()])
        vec = {k: str(spy.factor(spy.cancel(c * den))).replace("**", "^") for k, c in comp.items() if c != 0}
        return dict(d, vector=vec, support=sorted(vec))

    def _independent(self, dirs):
        """The directions with dependent closed forms and closed forms off the kernel
        dropped, and support-only directions from the kernel added until they span it on the
        physical coordinates."""
        from ..closedform import eval_modq
        from ..finalize import public_symmetry
        P = PRIMES[0]
        phys = [i for i, z in enumerate(self.znames) if not _CSUFFIX.search(z)]
        n = len(phys)
        env = self._kernel_point()
        # a form without resting states evaluates at the leaves of the base point alone
        leaves = {nm: int(v) for nm, v in zip(self.leaf_names_aug, self.sc["point0"])}
        rest = set(self.multi.get("real_state_names") or []) - {self.znames[i] for i in phys}
        tok = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
        span, keep = np.zeros((n, 0), dtype=np.int64), []
        for d in dirs:
            gen = public_symmetry(d)["generator"]
            # the resting-state columns of the joint system have no physical component
            gen = {k: v for k, v in gen.items() if not _CSUFFIX.search(k)} if gen else gen
            at = env if env is not None or not gen else \
                (None if rest & {t for x in gen.values() for t in tok.findall(str(x))} else leaves)
            inside = None if not gen else self._in_kernel(gen, at)
            if inside is None:
                keep.append(d)
                continue
            if not inside:
                continue  # the kernel fill below supplies the direction with its support
            v = np.zeros(n, dtype=np.int64)
            for j, i in enumerate(phys):
                if self.znames[i] in gen:
                    v[j] = eval_modq(gen[self.znames[i]], at, P) % P
            if in_span(span, v, n, P):
                continue
            span = np.concatenate([span, v.reshape(n, 1)], axis=1)
            keep.append(d)
        Nphys = np.asarray(self.N, dtype=np.int64)[phys, :] % P
        basis = np.zeros((n, 0), dtype=np.int64)
        for k in range(Nphys.shape[1]):
            if not in_span(basis, Nphys[:, k], n, P):
                basis = np.concatenate([basis, Nphys[:, [k]]], axis=1)
        # a single coordinate the kernel moves alone first, then the kernel columns
        for j in range(n):
            if len(keep) >= basis.shape[1]:
                break
            e = np.zeros(n, dtype=np.int64)
            e[j] = 1
            if in_span(basis, e, n, P) and not in_span(span, e, n, P):
                span = np.concatenate([span, e.reshape(n, 1)], axis=1)
                nm = self.znames[phys[j]]
                keep.append({"support": [nm], "type": "scaling", "vector": {nm: 1}, "closed_form": True})
        for k in range(Nphys.shape[1]):
            if len(keep) >= basis.shape[1]:
                break
            if in_span(span, Nphys[:, k], n, P):
                continue
            span = np.concatenate([span, Nphys[:, [k]]], axis=1)
            supp = sym_sort([self.znames[phys[j]] for j in np.nonzero(Nphys[:, k])[0]])
            if len(supp) == 1:
                keep.append({"support": supp, "type": "scaling", "vector": {supp[0]: 1}, "closed_form": True})
                continue
            keep.append({"support": supp, "type": "general", "closed_form": False,
                         "reason": "not reconstructed in closed form"})
        return keep

    def _guard_call(self, point, p, Nt):
        """The kernel at a higher Lie order, every chain cap raised with it."""
        run = self.run
        saved = list(run.nt_cap)
        try:
            if run.nt_cap:
                run.nt_cap = [int(c + (Nt - self.sc["nt_used"])) for c in run.nt_cap]
            return self.kcall(point, p, Nt)
        finally:
            run.nt_cap = saved

    def field_fixed(self, N):
        """The parameters no kernel direction moves."""
        pr = [i for i, z in enumerate(self.znames) if not _CSUFFIX.search(z)]
        sub = np.asarray(N, dtype=np.int64)[pr, :] % PRIMES[0]
        return [self.znames[pr[i]] for i in range(len(pr)) if not np.any(sub[i] != 0)]

    def _prove_by_scalings(self, result):
        """Proven when the rank plus the independent exact scalings (and static fields)
        equals the dimension."""
        if result.get("coordinates") is None or result["rank"] >= result["dim"]:
            return
        ws = self.exact_scalings(self.scal_fixed)
        if not ws:
            return
        coords = list(result["coordinates"])
        V = np.zeros((len(ws), len(coords)), dtype=np.int64)
        for i, v in enumerate(ws):
            for k, w in v.items():
                if k in coords:
                    V[i, coords.index(k)] = int(w) % PRIMES[0]
        if self.static_rows is not None and all(c in self.phys_params0 for c in coords):
            V = np.vstack([V, np.asarray(self.static_rows)[:, [self.phys_params0.index(c) for c in coords]]])
        k = int(_core.sym_rref_mod(V.astype(np.float64), PRIMES[0])["rank"]) if V.size else 0
        if result["rank"] + k == result["dim"]:
            result["rank_proven"] = k
            result["rank_proof"] = ("exact scalings and static subsystems"
                                    if self.static_rows is not None else "exact scalings")

    def _prove_by_fields(self, result):
        """Proven when every kernel direction lies in the span of invariant fields."""
        if result.get("rank_proven") is not None or result["rank"] >= result["dim"] or self.recast \
                or state.get("inner") or self.stacked_quick or state.env("NOFIELDS"):
            return
        run = self.run
        own = own_fields(result["non_identifiable"])
        proof = {"ok": False, "why": "no closed form for every direction"}
        if own is not None:
            try:
                proof = field_check(field_setup(self.multi, self.field_chains, run.cores, fields=own),
                                    self.N, self.znames, run.leaf_names, self.point0_solved,
                                    self.field_chains, self.field_joint)
            except Exception as e:
                proof = {"ok": False, "why": str(e)}
        if not proof.get("ok") and len(set(regime_keys(self.multi["tapes"]))) <= 3:
            try:
                proof = field_proof(self.multi, self.N, self.znames, run.leaf_names, self.point0_solved,
                                    self.field_chains, self.field_joint, run.cores, setup=None,
                                    fixed=self.field_fixed(self.N))
            except Exception as e:
                proof = {"ok": False, "why": str(e)}
        if state.env("LIEDIAG"):
            print("[liediag] invariant fields: " + ("proven" if proof.get("ok") else str(proof.get("why"))))
        if proof.get("ok"):
            result["rank_proven"] = int(result["dim"] - result["rank"])
            result["rank_proof"] = "invariant fields"


def observability_analytic_multi(multi, closed_form=False, cores=1, equil_zero_states=(),
                                 cond_zero_states=(), t0events=(), n_conditions=None, chain_of=None,
                                 n_gaps=0, implicit_steady_state=False, control=None, verify=False,
                                 codim_spec=None):
    """The observability analysis of compiled tapes; the raw result or None."""
    return Analysis(multi, closed_form, cores, equil_zero_states, cond_zero_states, t0events, n_conditions,
                    chain_of, n_gaps, implicit_steady_state, control, verify, codim_spec).execute()
