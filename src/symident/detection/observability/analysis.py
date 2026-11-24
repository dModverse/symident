"""Observability over GF(p) for all conditions: saturated kernel, certified Lie order, scalings, residual directions."""

import re
import time
import warnings

import numpy as np

from ... import _core, session, state
from ...errors import SYMPY_FAILS, InapplicableError, RouteError, swallow, sympy_fail
from ...gfp import PRIMES, SpanTest, in_span, null_residues, random_point, uniform_point, verify_prime
from ...par import thread_map
from ...rexpr import sym_sort
from ...sym.msolve import MsolveMissingError
from ...sym.parse import model_sympify
from ...sym.shared import kernel_tapes
from ..classify import joint_expand_scal, peel_scalings, report_physical
from ..closedform import recast_backsub
from ..conditions import scaling_component
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
    scaling_tapes,
    static_fields,
)
from ..rank import nullspace_basis, saturate_certify
from ..verify import verify_in_nullspace
from . import select
from .context import Run, ints, ok, strs
from .guard import saturation_guard
from .joint import JointSystem
from .plain import PlainSystem

# regimes beyond which the exact proof does not run a separate field setup
_CSUFFIX = re.compile(r"\|c[0-9]+$")


def _plateau():
    return max(1, int(state.env_value("LIEPLATEAU", "3")))


def _margin():
    return max(1, int(state.env_value("VERIFY_MARGIN", "6")))


def _rational_forms(exprs):
    """Whether every expression is rational in the symbols it reads, model names read as symbols."""
    from ...sym.parse import _is_rational_expr, _make_local_parse
    try:
        _, parse = _make_local_parse([str(x) for x in exprs])
        return all(_is_rational_expr(parse(str(x).replace("^", "**"))) for x in exprs)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return False


# steps of the search that rewrites a direction from resting states into coordinates
_REWRITE_STEPS = 64


def _int_weight(w, p):
    """An integer weight modulo p; 0 for a weight symbolic in a coordinate, which leaves the rank a lower bound."""
    try:
        return int(str(w)) % p
    except ValueError:
        return 0


def _in_rational_span(w, ws):
    """Whether the integer weights `w` are a rational combination of the weight vectors `ws`."""
    names = sorted({k for v in [w, *ws] for k in v})
    if not ws:
        return False
    try:
        import sympy as spy
        E = spy.Matrix([[spy.Rational(str(v.get(k, 0))) for k in names] for v in ws])
        return E.rank() == E.col_join(spy.Matrix([[spy.Rational(str(w.get(k, 0))) for k in names]])).rank()
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return False


class Analysis:
    """Observability analysis of all conditions; joint mode has shared columns, then state columns per resting one."""

    def __init__(self, multi, closed_form=False, cores=1, equil_zero_states=(),
                 cond_zero_states=(), t0events=(), n_conditions=None, chain_of=None, n_gaps=0,
                 implicit_steady_state=False, control=None, verify=False, codim_spec=None):
        from ...options import reconst_control
        self.ctrl = {**reconst_control(), **dict(control or {})}
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
            # nuisance coordinates are parameters, shared by every condition
            run.nuis_cols = [int(sys.wide_cols[0][i]) for i in run.nuis_cols]
            run.nuis_drop = [int(sys.wide_cols[0][i]) for i in run.nuis_drop]
            pn = set(sys.param_names)
            self.aux_leaves = [i for i, nm in enumerate(self.leaf_names_aug)
                               if nm in set(run.leaf_names) - pn]
            self.ctrl["relevance_cap"] = self.ctrl["relevance_cap"] + sys.n_st
            # exponential leaves outside the resting model leave the report like a transient recast
            if multi.get("exp_atoms"):
                self.transient = True
                self.atom_names = strs(multi.get("exp_atoms"))
                self.aux_leaves += [i for i, nm in enumerate(self.leaf_names_aug) if nm in set(self.atom_names)]
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
        self.can_batch = not run.ss_constraint and not run.has_gaps and not self.transient and \
            not run.nuis_cols
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
        """The kernel for the saturation, rank only in joint mode without nuisance columns."""
        if self.joint and not self.run.nuis_cols:
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
            return _core.sym_obs_null_batch(run.kernel_tapes(), run.n_leaves, run.n_states, run.z_slots, leaves(),
                                            list(prime_vec), int(Nt), run.cores)
        if run.has_gaps and not run.ss_constraint and not self.joint and not self.transient and \
                not run.nuis_cols and n > 0:
            emulated = run.stacked_chain_batch(point_list, prime_vec, int(Nt), int(run.mtot), list(run.nt_cap))
            if emulated is not None:
                return emulated
            return _core.sym_obs_null_chain_point_batch([run.chain_tapes(idx) for idx in run.chain_groups],
                                                        run.n_leaves, run.n_states, run.z_slots, leaves(),
                                                        list(prime_vec), int(Nt), int(run.mtot), run.cores,
                                                        list(run.nt_cap))
        if self.transient and not self.joint and not run.has_gaps and not run.nuis_cols and n > 0:
            os_ = _core.sym_obs_null_batch(run.kernel_tapes(), run.n_leaves, run.n_states, run.z_slots, leaves(),
                                           list(prime_vec), int(Nt), run.cores)
            return [self.sys.with_relations(os_[i], point_list[i], prime_vec[i]) for i in range(n)]
        if self.chunked and n > 0:
            per = 2 * max(1, run.cores)
            out = []
            for s in range(0, n, per):
                out.extend(self.sys.chunk(point_list[s:s + per], prime_vec[s:s + per], Nt))
            return [run.modulo_nuisance(o, q) for o, q in zip(out, prime_vec)]
        if self.joint:
            self.sys.warm_solves(point_list, prime_vec)
        cc_saved = run.cores_call
        warm_saved = self.sys.warm_off if self.joint else None
        if run.cores > 1:
            run.cores_call = 1
            if self.joint:
                self.sys.warm_off = True
        try:
            return run.par_map(range(n), lambda i: self.kcall(point_list[i], prime_vec[i], Nt),
                               self.sys if self.joint else None)
        finally:
            run.cores_call = cc_saved
            if self.joint:
                self.sys.warm_off = warm_saved

    def cover_tapes(self):
        """The tapes of the conditions whose rows reach the rank of all of them at the base point (sym_obs_cover),
        formed once; None where the path has no cover or every condition is needed. Kernels on them propose
        values only: every closed form is checked on all conditions."""
        run, sc = self.run, self.sc
        if "_cover" in self.__dict__:
            return self._cover
        self._cover = None
        if not self.can_batch or len(run.tapes) < 16:
            return None
        r = _core.sym_obs_cover(run.kernel_tapes(), run.n_leaves, run.n_states, run.z_slots,
                                np.asarray(sc["point0"][:run.n_leaves], dtype=np.int64), PRIMES[0],
                                int(sc["nt_used"]), int(sc["rank"]), run.cores)
        if ok(r) and int(r["rank"]) == int(sc["rank"]) and len(ints(r["conditions"])) < len(run.tapes):
            self._cover = kernel_tapes([run.tapes[i] for i in ints(r["conditions"])])
            if run.timing:
                print(f"[sym] cover: {len(ints(r['conditions']))} of {len(run.tapes)} conditions reach rank "
                      f"{int(sc['rank'])}", flush=True)
        return self._cover

    def kbatch_cover(self, point_list, prime_vec, Nt):
        """kbatch() on the cover of the conditions where there is one."""
        tapes = self.cover_tapes()
        if tapes is None or not point_list:
            return self.kbatch(point_list, prime_vec, Nt)
        run = self.run
        M = np.vstack([np.asarray(pp[:run.n_leaves], dtype=np.int64) for pp in point_list])
        return _core.sym_obs_null_batch(tapes, run.n_leaves, run.n_states, run.z_slots, M, list(prime_vec),
                                        int(Nt), run.cores)

    def lanes(self):
        """Whether each column of V annihilates the rows of the cover at each of many points, from jets seeded
        along the columns (sym_obs_annihilates_batch): per point a boolean array, None at a point without
        the rows; None without the path."""
        if not self.can_batch:
            return None
        run = self.run

        def f(points, primes, Nt, V, tapes=None):
            if tapes is None:
                tapes = self.cover_tapes() or run.kernel_tapes()
            M = np.vstack([np.asarray(pp[:run.n_leaves], dtype=np.int64) for pp in points])
            o = _core.sym_obs_annihilates_batch(tapes, run.n_leaves, run.n_states, run.z_slots, M, list(primes),
                                                int(Nt), np.asarray(V, dtype=np.int64) % int(primes[0]), run.cores)
            if o is None:
                return None
            return [np.asarray(x["zero"], dtype=bool) if ok(x) else None for x in o]
        return f

    def dir_batch(self):
        """Whether each column of V annihilates all rows, in one pass over the conditions; None without the path."""
        run = self.run
        if run.has_gaps or run.ss_constraint or self.recast or run.nuis_cols or self.transient:
            return None

        def f(pt, q, Nt, V):
            o = _core.sym_obs_annihilates(run.kernel_tapes(), run.n_leaves, run.n_states, run.z_slots,
                                          np.asarray(pt[:run.n_leaves], dtype=np.int64), q, run.exact_nt(Nt),
                                          np.asarray(V, dtype=np.int64) % q, run.cores)
            return [bool(x) for x in np.atleast_1d(o["zero"])] if ok(o) else None
        return f

    def dir_call(self):
        """Whether a direction annihilates all rows, from jets along it; None without the path."""
        run = self.run
        if run.has_gaps or run.ss_constraint or self.recast or run.nuis_cols:
            return None

        def f(pt, q, Nt, v):
            o = _core.sym_obs_directional(run.kernel_tapes(), run.n_leaves, run.n_states, run.z_slots,
                                          np.asarray(pt[:run.n_leaves], dtype=np.int64), q, run.exact_nt(Nt),
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
        """The exact scalings with `fixed` held, memoized."""
        key = "k:" + ",".join(sorted(fixed))
        if key not in self.scal_memo:
            self.scal_memo[key] = exact_scaling_set(self.multi, fixed)
        return self.scal_memo[key]

    # ---- saturation and the proven Lie order -----------------------------------------

    def _lower_caps(self, cap0, cap_rank):
        """Lower the deepest chain caps by bisection while the rank score stays; False with nothing to lower."""
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
        """Each chain at the order of the chain plus the plateau during the saturation."""
        run = self.run
        nb = self.sys.kc if self.joint else (len(run.chain_groups) if run.chain_groups else 0)
        if not (run.has_gaps and len(orders) == nb
                and (self.joint or not run.ss_constraint)):
            return
        cap0 = [int(o) + _plateau() for o in orders]
        run.nt_cap = cap0
        if first is None:
            return
        Nt = max(cap0)

        def cap_rank():
            try:
                r = self.kcall_sat(point, PRIMES[0], Nt, Mtot)
            except Exception as exc:
                swallow(exc, "joint cap")
                r = None
            return rank_score(r) if ok(r) else None
        if self._lower_caps(cap0, cap_rank) and state.env("LIEDIAG"):
            print(f"[liediag] joint caps in the saturation at gap order {Mtot}: "
                  f"{','.join(map(str, run.nt_cap))}")

    def _block_map(self, block_call, block_profile):
        run = self.run
        if block_profile is not None and len(block_profile) == len(block_call) and len(block_call) > 1:
            if run.has_gaps and not self.joint and run.cores > 1:
                # a chain profile runs on one thread; the chains run in parallel
                return lambda f, xs: thread_map(f, xs, run.cores)
            return lambda f, xs: [f(x) for x in xs]
        if (not run.ss_constraint or self.joint) and run.cores > 1 and len(block_call) > 1:
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
        """Under a steady state the exact scalings bound the rank from above."""
        run, sys = self.run, self.sys
        if not self.joint:
            return
        not_fixed = {z for z in self.znames if z not in self.zsn} | set(sys.state_base)
        self.scal_ws = self.exact_scalings([l for l in run.leaf_names if l not in not_fixed])
        shared = self.znames[:sys.n_shared]
        rows = []
        for v in self.scal_ws:
            x = np.zeros(sys.n_shared, dtype=np.int64)
            for k, w in v.items():
                if k in shared:
                    x[shared.index(k)] = _int_weight(w, PRIMES[0])
            rows.append(x)
        if rows:
            self.rank_cap = self.nz - int(_core.sym_rref_mod(np.vstack(rows).astype(np.float64),
                                                             PRIMES[0])["rank"])

    def saturate(self):
        """The saturated kernel at a generic base point, or None."""
        run, sys, multi = self.run, self.sys, self.multi
        block_call = sys.block_calls()
        block_profile = sys.block_profiles()
        budget = None if run.has_gaps or self.codim_spec is None or run.nuis_cols else \
            int(self.codim_spec) + 2 * len(self.recast) + int(multi.get("exp_codim") or 0)
        block_map = self._block_map(block_call, block_profile)
        if self.joint:
            reach = [run.chain_groups[run.chain_idx[ci]] if run.has_gaps else [ci] for ci in sys.equil_conds]
        elif run.has_gaps:
            reach = run.chain_groups
        else:
            reach = [[i] for i in range(len(multi["tapes"]))]
        floor = int(state.get("nt_floor") or 0)
        nt_min_block = [max(floor, lie_reach_max([multi["tapes"][i] for i in idx])) for idx in reach]
        if len(nt_min_block) != len(block_call):
            nt_min_block = None
        self._scaling_cap()
        self.stacked_first = (((not run.ss_constraint and not run.has_gaps)
                               or (self.joint and run.has_gaps and self.rank_cap < self.nz))
                              and not self.recast and self.nz > state.get("stacked_min", 150)
                              and state.get("stacked_first") is not False and not state.get("inner")
                              and state.get("order_hint") is None)
        self.stacked_quick = self.stacked_first and bool(state.get("stacked_quick"))
        if self.stacked_first:
            block_call, block_profile, block_map = None, None, None
        sc = saturate_certify(self.kcall_sat, self.n_aug, self.nz, run.max_gap,
                              warm=self._warm_probe if self.joint else (lambda pts, primes: None),
                              probe_block=max(1, min(8, run.cores)), block_call=block_call,
                              budget=budget, block_map=block_map, nt_min=max(floor, lie_reach_max(multi["tapes"])),
                              nt_min_block=nt_min_block, prep=sys.sat_prep if self.joint else None,
                              rank_cap=self.rank_cap, block_profile=block_profile, set_caps=self._sat_caps,
                              checks=not (self.stacked_first and (not self.stacked_quick or run.ss_constraint)),
                              longest_first=run.has_gaps and not self.joint,
                              stacked_profile=sys.stacked_profile() if self.stacked_first else None)
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

    def _gap_proof(self, sc):
        """Gap order raised until the truncated rank reaches the gapcert bound; False without a bound or at the cap."""
        from ..gapcert import gap_bound
        run = self.run
        if self.joint or not run.chain_groups or self.recast:
            return False
        b = gap_bound(self.multi["tapes"], run.chain_groups, list(self.znames), int(sc["nt_used"]), verify_prime(),
                      session.current().rng("gap.bound", ",".join(self.znames)))
        self.gap_bound = b
        if b["bound"] is None:
            return False
        if self._gap_reached(rank_of(sc["ref"]), b):
            return True
        for m in range(run.mtot + 1, run.max_gap + 1):
            if b.get("inconsistent"):
                return False
            try:
                r = self.kcall4(sc["point0"], PRIMES[0], sc["nt_used"], m)
            except Exception as exc:
                swallow(exc, "gap order past the plateau")
                return False
            if not ok(r):
                return False
            if self._gap_reached(rank_of(r), b):
                sc.update(ref=r, mtot_used=m, rank=r["rank"], rank_s=rank_of(r), pivots=r["pivots"])
                run.mtot = m
                return True
        return False

    def _gap_reached(self, rank, b):
        """Whether the truncated `rank` reaches the bound of `b`; a rank above it marks the bound inconsistent."""
        if rank > b["bound"]:
            session.note("limits", f"gap bound: the truncated rank {rank} exceeds the bound {b['bound']}, "
                                   "so the bound is not taken")
            b["inconsistent"] = True
            return False
        return rank == b["bound"]

    @staticmethod
    def _order_bound(sc, gap_used):
        """(bound, why): the chance that the certified Lie and gap orders are not sufficient, or why there is none."""
        if not sc.get("certified"):
            return None, None
        ob = sc.get("order_bound")
        if ob is None:
            return None, sc.get("order_bound_why") or "the certificate has no degree bound"
        if gap_used is None:
            return ob, None
        t = gap_used.get("terms")
        return (None, "the gap-order bound has no degree bound") if t is None else (float(ob + t), None)

    def _gap_why(self, sc):
        """None when the rank bound proves the gap order sufficient, else why it stopped."""
        run = self.run
        stop = sc.get("gap_stop") or "cap"
        if rank_of(sc["ref"]) >= self.rank_cap or stop == "full":
            return None
        return {"plateau": f"the rank did not grow from gap order {run.mtot - 1} to {run.mtot}",
                "cap": f"reconst_control(gap_order_cap={run.max_gap}) reached",
                "failed": f"no kernel past gap order {run.mtot}"}[stop]

    def _cert_segments(self, sc, p=None, tries=24):
        """The certificate inputs: chains of tapes, a seed per chain, at rest a start per chain, and the points drawn
        for the seeds. At rest and a prime `p` the seeds are resting states solved modulo `p` at the first of `tries`
        points uniform modulo `p`, random points of the resting variety, and the starts are those of the base
        point."""
        run, sys = self.run, self.sys
        if self.joint:
            segs = sys.chains if run.has_gaps else [[run.tapes[ci]] for ci in sys.equil_conds]
            sols = [sys.solve_cond(sc["point0"], PRIMES[0], ci) for ci in sys.equil_conds]
            base = [None if s0 is None else np.asarray(s0["ptc"], dtype=np.int64) % PRIMES[0] for s0 in sols]
            if p is None:
                return segs, base, None, 1
            n = len(sc["point0"])
            for k in range(1, tries + 1):
                pt = uniform_point(n, p, session.current().seed("certificate point", k))
                sols = [sys.solve_direct(pt, p, ci) for ci in sys.equil_conds]
                if all(s0 is not None for s0 in sols):
                    return segs, [np.asarray(s0["ptc"], dtype=np.int64) % p for s0 in sols], base, k
            return segs, [None] * len(sys.equil_conds), base, tries
        segs = [run.chain_tapes(idx) for idx in run.chain_groups] if run.has_gaps else \
            [[t] for t in run.tapes]
        return segs, [np.asarray(sc["point0"][:run.n_leaves], dtype=np.int64)] * len(segs), None, 1

    def certify(self, sc):
        """The proven Lie order per block, then the per-chain caps lowered."""
        run = self.run
        run.nt_cap = []
        # a full rank, or a rank at the cap set by the exact scalings, cannot rise with the order
        sc["certified"] = rank_of(sc["ref"]) >= self.nz or \
            (self.rank_cap < self.nz and rank_of(sc["ref"]) >= self.rank_cap)
        sc["order_bound"] = 0.0 if sc["certified"] else None
        hint = state.get("order_hint")
        if hint is not None and hint.get("certified") is not False and \
                list(sc.get("block_orders") or []) == list(hint.get("block_orders") or []):
            sc["certified"] = True
            sc["cert_orders"] = True
            sc["order_bound"] = hint.get("order_bound")
            sc["order_bound_why"] = hint.get("order_bound_why")
            sc["order_draws"] = hint.get("order_draws")
        # a power of a state at 0 has no rows of D_k at the start (method: series at a singular start)
        start_pow = any(10 in list(t["op"]) for t in self.multi["tapes"])
        if not sc["certified"] and not self.stacked_first and not run.nuis_cols and not start_pow:
            self._certificate(sc)
        nb = self.sys.kc if self.joint else (len(run.chain_groups) if run.chain_groups else -1)
        if run.has_gaps and (self.joint or not run.ss_constraint) and \
                sc.get("block_orders") is not None and len(sc["block_orders"]) == nb:
            add = 0 if sc.get("cert_orders") else _plateau()
            run.nt_cap = [int(o) + add for o in sc["block_orders"]]
        run.stage("Lie order certified")
        if run.nt_cap:
            mt = int(sc.get("mtot_used") or 0)

            def cap_rank():
                try:
                    r = self.kcall4(sc["point0"], PRIMES[0], sc["nt_used"], mt)
                except Exception as exc:
                    swallow(exc, "joint block caps")
                    r = None
                return rank_score(r) if ok(r) else None
            self._lower_caps(list(run.nt_cap), cap_rank)
            if state.env("LIEDIAG"):
                print(f"[liediag] joint block Lie orders: {','.join(map(str, run.nt_cap))}")

    def _cert_degrees(self, sc):
        """(degrees, why): per certificate block the total degrees of each segment and of the initial values, at rest
        the Bezout number of the resting balances; None where a block has none."""
        from ..bound import _vector_degree, no_degree, total_degrees
        run, tapes = self.run, self.multi["tapes"]
        groups = run.chain_groups if run.has_gaps else [[i] for i in range(len(tapes))]
        if self.joint:
            groups = [run.chain_groups[run.chain_idx[ci]] if run.has_gaps else [ci] for ci in self.sys.equil_conds]
        out, why = [], None
        # segments and conditions repeat the same lines
        memo = {}
        for idx in groups:
            mds = []
            for i in idx:
                t = tapes[i]
                ml = strs(t.get("jet_f") or t.get("model_lines"))
                ol = [f"_y{j} = {e}" for j, e in enumerate(strs(t["jet_g"]))] if t.get("jet_f") else \
                    strs(t.get("obs_lines"))
                key = (tuple(ml), tuple(ol))
                if key not in memo:
                    memo[key] = total_degrees(ml, ol)
                md = memo[key]
                if md is None and why is None:
                    why = no_degree([l.split("=", 1)[1] for l in ml + ol]) or "SymPy found no degree"
                mds.append(md)
            # the degree of the start of each segment: the initial values, or the resting state with the
            # events at the start, then the events; composing a value of degree dv multiplies by at most 1 + dv
            try:
                if self.joint:
                    d0s = [1 + self._event_degree(tapes[idx[0]].get("ver_t0"))]
                else:
                    ic = [l.split("=", 1)[1] for l in strs(tapes[idx[0]].get("jet_ic"))]
                    d0s = [max(1, _vector_degree([model_sympify(e) for e in ic])) if ic else 1]
                for i in idx[1:]:
                    d0s.append(d0s[-1] * (1 + self._event_degree(tapes[i].get("jet_ev") or tapes[i].get("ver_ev"))))
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                d0s = None
            if self.joint and d0s is not None:
                r = self.sys.rest_degree(idx[0], sc["point0"])
                if "why" in r:
                    why = why or r["why"]
                    out.append(None)
                    continue
                out.append({"mds": mds, "d0s": d0s, "bezout": r["bezout"], "extra": r["bezout"]})
                continue
            if d0s is None:
                why = why or "the initial values have no degree"
                out.append(None)
                continue
            out.append({"mds": mds, "d0s": d0s})
        return out, why

    @staticmethod
    def _event_degree(evs):
        """The degree of the event values `evs` in all their symbols, 0 without events."""
        from ..bound import _vector_degree
        vals = [model_sympify(str(e["value"])) for e in (evs or []) if isinstance(e, dict)]
        return _vector_degree(vals) if vals else 0

    def _cert_start(self, sc, n):
        """Per block the saturated order plus the plateau, where the saturation knows it."""
        bo = sc.get("block_orders")
        if bo is None or len(bo) != n:
            return None
        return [int(o) + _plateau() + 1 for o in bo]

    def _certificate(self, sc):
        """The Lie order per block past which the rank provably stays."""
        run = self.run
        P1, q = PRIMES[0], verify_prime()
        degs, why = self._cert_degrees(sc)
        if degs is not None and any(d is None for d in degs):
            degs = None
        # at rest the generic ranks are taken at a random resting point modulo q, where the draws are uniform
        segs, seeds, starts, rest_draws = self._cert_segments(sc, q if self.joint and degs is not None else None)
        if self.joint and degs is not None and any(s is None for s in seeds):
            segs, seeds, starts, rest_draws = self._cert_segments(sc)
            degs, why = None, "no resting state modulo the verification prime at the points tried"
        own_threads = not run.has_gaps and not self.joint
        threads = run.cores if own_threads else 1
        mapf = (lambda f, xs: thread_map(f, xs, run.cores)) if threads == 1 and run.cores > 1 else None
        cap = max(64, 4 * (len(self.z_slots) + run.n_states))
        chain_end = None
        if run.nuis_order is not None:
            # past the exact orders of the input chains a row is that of a polynomial stand-in
            chain_end = int(run.nuis_order) - int(sc.get("mtot_used") or 0)
            cap = min(cap, chain_end)
        sc["order_bound_why"] = why
        orders, terms, draws = None, [], []
        if all(s is not None and len(s) > 0 for s in seeds) and cap >= 1:
            try:
                orders = lie_certificate(segs, seeds, run.n_leaves, run.n_states,
                                         self.sys.z_slots_local if self.joint else run.z_slots,
                                         P1 if degs is None else q, cap=cap,
                                         map_fn=mapf, threads=threads,
                                         start=self._cert_start(sc, len(segs)), degs=degs, terms=terms,
                                         at_rest=self.joint, start_seeds=starts if degs is not None else None,
                                         start_p=P1 if self.joint and degs is not None else None,
                                         rest_draws=rest_draws, draws=draws,
                                         drop_states=run.known_states if not (self.joint or run.has_gaps) else (),
                                         cores=run.cores)
            except Exception as e:
                swallow(e, "Lie order certificate")
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
            except Exception as exc:
                swallow(exc, "certified kernel")
                rc = None
            if ok(rc) and rank_of(rc) < rank_of(sc["ref"]):
                # an order below the one the scan needed cannot be sufficient
                session.note("limits", f"Lie order certificate: the rank at the certified order {need} is below "
                                       f"the rank of the scan at order {int(sc['nt_used'])}; the scanned order "
                                       "is kept")
                sc.update(certified=False, uncertified=list(range(len(orders))))
            elif ok(rc):
                cert = all(o is not None for o in orders)
                sc.update(ref=rc, nt_used=need, rank=rc["rank"], rank_s=rank_of(rc), pivots=rc["pivots"],
                          block_orders=ordv, block_driver=int(np.argmax(ordv)), certified=cert,
                          uncertified=[i for i, o in enumerate(orders) if o is None], cert_orders=True,
                          order_bound=(float(sum(terms)) if cert and len(terms) == len(orders) and
                                       all(t is not None for t in terms) else None))
                if sc["order_bound"] is not None and sc["order_bound"] >= 1:
                    # a chance of one or more bounds nothing: the degrees reach the verification prime
                    sc["order_bound"] = None
                    sc["order_bound_why"] = f"the Schwartz-Zippel bound reaches 1 at the verification prime {q}"
                sc["order_draws"] = max(draws) if sc["order_bound"] is not None and draws and \
                    all(d is not None for d in draws) else None
        if cap == chain_end and (orders is None or any(o is None for o in orders)):
            session.note("limits", f"Lie order certificate: no start reached the generic rank up to order {cap}, "
                                   "the end of the jet chains of the inputs; the order is not proven sufficient")
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
        """Observability result: rank, saturation details, non-identifiable directions; None when saturation fails."""
        run = self.run
        sc = self.saturate()
        if sc is None:
            return None
        self.certify(sc)
        self.sc = sc
        self.point0_solved = np.array(sc["point0"], dtype=np.int64)
        self._mark_states()
        run.mtot = int(sc.get("mtot_used") or 0)
        gap_why = self._gap_why(sc) if run.has_gaps else None
        gap_used = None
        if gap_why is not None and self._gap_proof(sc):
            gap_why, gap_used = None, self.gap_bound
        elif gap_why is not None and getattr(self, "gap_bound", None) is not None:
            b = self.gap_bound
            gap_why += (f" (no rank bound: {b['why']})" if b["bound"] is None else
                        f" (the rank exceeds the bound {b['bound']})" if b.get("inconsistent") else
                        f" (the rank stays below the bound {b['bound']} up to the cap)")
        nz, n_seg = self.nz, len(self.multi["tapes"])
        result = {"method": "observability", "engine": "analytic",
                  "conditions": n_seg if self.n_conditions is None else int(self.n_conditions),
                  "coordinates": list(self.znames), "segments": n_seg,
                  "gap_order_used": 0 if run.series_gaps else run.mtot,
                  "identifiable": sc["rank_s"] == nz, "rank": int(sc["rank_s"]), "dim": int(nz),
                  "lie_order_used": int(sc["nt_used"]), "lie_order_driver": sc.get("block_driver"),
                  "lie_uncertified": sc.get("uncertified"), "lie_plateau": sc.get("plateau"),
                  "lie_certified": bool(sc.get("certified")), "lie_block_orders": sc.get("block_orders"),
                  "gap_certified": None if not run.has_gaps or run.series_gaps else gap_why is None,
                  "gap_certificate": None if run.series_gaps else gap_why,
                  "series_certified": gap_why is None if run.series_gaps else None,
                  "series_certificate": gap_why if run.series_gaps else None,
                  "gap_bound": None if gap_used is None else int(gap_used["bound"]),
                  "order_bound": self._order_bound(sc, gap_used)[0],
                  "order_bound_why": self._order_bound(sc, gap_used)[1],
                  "order_draws": sc.get("order_draws") if self._order_bound(sc, gap_used)[0] is not None else None,
                  "sat_score": rank_score(sc["ref"]), "non_identifiable": []}
        self.phys_params0 = [z for z in self.znames if z not in self.zsn]
        if self.joint:
            result["dim"] = len(self.phys_params0)
            if sc["rank_s"] == nz:
                result["rank"] = len(self.phys_params0)
                result["coordinates"] = list(self.phys_params0)
        if self.transient:
            coords = [z for z in (self.phys_params0 if self.joint else self.znames) if z not in set(self.atom_names)]
            result["dim"] = len(coords)
            result["coordinates"] = coords
            if sc["rank_s"] == nz:
                result["rank"] = len(coords)
        self.gap_dirs = self._gap_dirs()
        if run.nuis_cols:
            result["nuisance"] = {"names": [self.znames[i] for i in run.nuis_cols],
                                  "pinned": int(sc["ref"].get("nuisance_pinned") or 0),
                                  "exact_order": int(run.nuis_order),
                                  "short": sc["rank_s"] != nz and
                                  int(sc["nt_used"]) + (0 if run.series_gaps else run.mtot) +
                                  (_margin() if self.verify else 0) > int(run.nuis_order)}
        if run.nuis_cols:
            result["projection"] = self._projection(sc)
        elif self.joint and not self._jx_regular(sc):
            result["projection"] = {"status": "unchecked",
                                    "of": "the resting states, whose Jacobian is singular at the base point"}
        if run.nuis_order is not None and not run.nuis_cols:
            # the chains of known inputs: exact up to their order, a polynomial stand-in past it
            result["known_short"] = sc["rank_s"] != nz and \
                int(sc["nt_used"]) + (0 if run.series_gaps else run.mtot) + \
                (_margin() if self.verify else 0) > int(run.nuis_order)
        if sc["rank_s"] == nz:
            return self._modulo_nuisance(result)
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
        supp = [int(x) for x in np.atleast_1d(sr["support"]) if int(x) not in set(self.run.nuis_cols)]
        return [{"support": sym_sort([self.znames[i] for i in supp]), "type": "general",
                 "closed_form": False, "reason": "depends on the time between events"}
                for _ in range(sc["rank"] - sc["rank_s"])]

    def _modulo_nuisance(self, result):
        """The report without the nuisance coordinates, their components moved to "input_vector"."""
        if not self.run.nuis_cols:
            return result
        aux = result["nuisance"]["names"] + (list(self.atom_names) if self.transient else []) + \
            (list(self.zsn) if self.joint else [])
        return report_physical(result, self.znames, aux, self.sc, self.nz, PRIMES[0], "input_vector")

    def _physical(self, result):
        """Joint mode reported in parameter space, a transient recast without the recast atoms, no nuisance
        coordinates."""
        if self.joint:
            # the atoms of exponential leaves leave the joint report with the state columns
            result = report_physical(result, self.znames, list(self.zsn) + list(self.atom_names), self.sc,
                                     self.nz, PRIMES[0], "state_vector", flag_joint=True)
        elif self.transient:
            result = report_physical(result, self.znames, self.atom_names, self.sc, self.nz,
                                     PRIMES[0], "recast_vector")
        return self._modulo_nuisance(result)

    def static_fields(self):
        """Fields of the static subsystems of the resting conditions at the solved point."""
        run, sys = self.run, self.sys
        return static_fields(self.multi, sys.models, sys.equil_conds,
                             [run.chain_groups[run.chain_idx[ci]] if run.has_gaps else [ci]
                              for ci in sys.equil_conds],
                             self.phys_params0, run.leaf_names, self.point0_solved,
                             strs(self.multi.get("forcings")), [sys.zero_of(ci) for ci in sys.equil_conds])

    def _continued(self, result):
        """Kernel directions of continued conditions by scalings, static fields or a direct point; None to go on."""
        sc = self.sc
        ker_dim = int(self.nz - sc["rank_s"])
        rank0, coords0 = result["rank"], result["coordinates"]
        result["rank"] = int(result["dim"] - ker_dim)
        result["coordinates"] = self.phys_params0
        chosen = self._scalings_and_statics(ker_dim)
        if len(chosen) < ker_dim and self.closed_form and not self.stacked_quick and \
                not self.stacked_first and self._direct_kernel(ker_dim):
            result["rank"] = rank0
            result["coordinates"] = coords0
            result["series_certified"] = result["series_certificate"] = None
            self.gap_dirs = self._gap_dirs()
            return None
        result["non_identifiable"] = chosen + [
            {"support": [], "type": "general", "closed_form": False,
             "reason": "not reconstructed along the continuation"} for _ in range(ker_dim - len(chosen))]
        run = self.run
        self.field_chains = run.chain_groups if run.has_gaps else [[i] for i in range(len(self.multi["tapes"]))]
        self.field_joint = {"conds": self.sys.equil_conds, "solve": self.sys.solve_cond}
        self._prove_exact(result)
        if len(chosen) == ker_dim:
            result["rank_proven"] = ker_dim
            result["rank_proof"] = "proven symmetries"
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
            if any(k in pp and not re.fullmatch(r"-?[0-9]+", str(w)) for k, w in v.items()):
                continue
            x = np.zeros(len(pp), dtype=np.int64)
            for k, w in v.items():
                if k in pp:
                    x[pp.index(k)] = int(w) % P0
            r2 = np.vstack([rows, x])
            if grows(r2):
                rows = r2
                vv = {k: str(w) for k, w in v.items() if k in pp and str(w) != "0"}
                chosen.append({"support": sym_sort(list(vv)), "vector": vv, "type": "scaling",
                               "closed_form": True, "exact": True})
            if len(chosen) == ker_dim:
                break
        if len(chosen) < ker_dim:
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
        chained = sys.chained
        if chained:
            sys.unchain()
        cont_saved = list(sys.cont_off)
        for i in range(len(sys.cont_off)):
            sys.cont_off[i] = True
        # under the diagonal start the base point need not solve directly, other points may
        bases = [self.point0_solved] + ([random_point(len(self.point0_solved), seed=k)
                                        for k in range(1, 4)] if sys.diag is not None else [])
        pt_d = None
        for base in bases:
            try:
                pt_d = sys.sat_prep(base, rounds=2)
            except Exception as exc:
                swallow(exc, "direct kernel point")
                pt_d = None
            if pt_d is not None:
                break
        if pt_d is None:
            pt_d = self.point0_solved
        try:
            rc = self.kcall4(pt_d, PRIMES[0], sc["nt_used"], sc.get("mtot_used") or 0)
        except Exception as e:
            swallow(e, "direct kernel")
            rc = {"ok": False, "why": str(e)}
        direct = ok(rc) and nz - rank_of(rc) == ker_dim
        if state.env("LIEDIAG"):
            print("[liediag] direct kernel for the continued conditions: " +
                  (f"no point: {rc.get('why')}" if not ok(rc) else
                   f"dimension {nz - rank_of(rc)} (rank {rc['rank']}, series {rank_of(rc)}), "
                   f"series kernel {ker_dim}"))
        if not direct:
            sys.cont_off[:] = cont_saved
            if chained:
                sys.chain()
            return False
        sc.update(ref=rc, rank=rc["rank"], rank_s=rank_of(rc), pivots=rc["pivots"])
        self.point0_solved = np.array(pt_d, dtype=np.int64)
        self._mark_states()
        return True

    def _peel(self):
        """The exact scalings of the model as tangents in the nullspace."""
        from ...sym.scaling import scaling_symmetries_multi
        tapes = self.multi["tapes"] if self.joint else scaling_tapes(self.multi)
        model_lines = [strs(t.get("model_lines")) for t in tapes]
        obs_lines = [strs(t.get("obs_lines")) for t in tapes]
        keys = ["\n".join([*m, "\x1f", *o]) for m, o in zip(model_lines, obs_lines)]
        keep = [i for i, k in enumerate(keys) if k not in keys[:i]]
        try:
            res = scaling_symmetries_multi(per_cond_model=[model_lines[i] for i in keep],
                                            per_cond_obs=[obs_lines[i] for i in keep],
                                            inputs=strs(self.multi.get("forcings")) or None,
                                            fixed=self.scal_fixed or None, recast=self.recast or None)
        except Exception as exc:
            swallow(exc, "scalings of the model")
            res = None
        # a scaling is exact where initial values and events are checked too: those of the whole model come first
        ws = self.exact_scalings(self.scal_fixed)
        states = {l.split("=", 1)[0].strip() for t in tapes for l in strs(t.get("model_lines"))}

        def lift(w):
            # a scaling acts on the states by the scaling weights; with unknown inputs the lift gives their change
            return {x: scaling_component(v, x) for x, v in w.items() if x in states and str(v) != "0"} \
                if self.run.nuis_cols else None
        dirs = [{"vector": dict(w), "exact": True, "state_components": lift(w)} for w in ws] + \
            [dict(d, exact=_in_rational_span(d["vector"], ws)) for d in (res or {}).get("non_identifiable", [])]
        res = dict(res or {}, non_identifiable=dirs)
        if self.joint:
            sys = self.sys
            res = joint_expand_scal(res, sys.state_base, sys.kc, sys.held_param_of, sys.pinned())
            res["non_identifiable"] = list(res["non_identifiable"]) + self._condition_scalings(
                model_lines, obs_lines, keys, keep, [d["vector"] for d in res["non_identifiable"]])
        if not res["non_identifiable"]:
            return [], np.zeros((self.nz, 0), dtype=np.int64)
        peel = peel_scalings(res, self.znames, self.nz, np.asarray(self.sc["point0"])[self.z_slots],
                             PRIMES[0], self.N,
                             modulo=(self.free_cols, self.run.nuis_cols) if self.run.nuis_cols else None,
                             shifts={str(r["L"]) for r in self.recast or [] if not r.get("inverted")})
        return peel["scaling"], peel["Bmat"]

    def _condition_scalings(self, model_lines, obs_lines, keys, keep, known):
        """Exact scalings whose state weights differ between the resting conditions, each checked on every tape with
        the state weights of the condition of the tape; those in the span of `known` are left out."""
        from ...sym.scaling import scaling_symmetries_per_condition, verify_scalings
        sys = self.sys
        tapes = self.multi["tapes"]
        rest = [strs(tapes[i].get("ver_rest")) for i in keep]
        try:
            found = scaling_symmetries_per_condition(
                [model_lines[i] for i in keep], [obs_lines[i] for i in keep],
                inputs=strs(self.multi.get("forcings")) or None, fixed=self.scal_fixed or None, extra_model=rest)
        except Exception as exc:
            swallow(exc, "scalings per condition")
            return []
        pinned = set(sys.pinned())
        held = sys.held_param_of or {}
        state_base = set(sys.state_base)
        out, span = [], [dict(v) for v in known]
        for sc in found:
            if len({tuple(sorted(st.items())) for st in sc["states"]}) < 2:
                continue
            ok = True
            # every tape, also one that shares model and observables with a kept one, with the weights of that one
            for i in range(len(tapes)):
                w = dict(sc["params"], **sc["states"][keep.index(keys.index(keys[i]))])
                try:
                    ok = bool(verify_scalings([w], [tapes[i]])[0])
                except Exception as exc:
                    swallow(exc, "scaling per condition check")
                    ok = False
                if not ok:
                    break
            if not ok:
                continue
            vec = {}
            for nm, w in sc["params"].items():
                vec[held.get(nm, nm)] = str(w)
            for m_, ci in enumerate(sys.equil_conds, start=1):
                st = sc["states"][keep.index(keys.index(keys[ci]))]
                for x, w in st.items():
                    wn = f"{x}|c{m_}"
                    if x in state_base and wn not in pinned:
                        vec[wn] = str(w)
            if _in_rational_span(vec, span):
                continue
            span.append(vec)
            out.append({"vector": vec, "support": list(vec), "type": "scaling", "exact": True})
        return out

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
        # a free column is residual where the null vector of the column is not in the span of the scalings and
        # the null vectors before it: the pivot columns of the reduced form of all of them side by side
        residual = []
        bfs = [null_residues(sc["ref"], fc, P).reshape(nz, 1) for fc in self.free_cols]
        if bfs:
            piv = {int(x) for x in np.atleast_1d(_core.sym_rref_mod(
                np.concatenate([B.reshape(nz, -1)] + bfs, axis=1).astype(np.float64), P)["piv"])}
            keep = [j for j in range(len(bfs)) if scal_cols + j in piv]
            residual = [self.free_cols[j] for j in keep]
            B = np.concatenate([B.reshape(nz, -1)] + [bfs[j] for j in keep], axis=1)
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
        self._prove_exact(result)
        run.stage("exact scalings proof")
        self._prove_by_fields(result)
        run.stage("field proof")
        if not state.get("inner"):
            from .proof import chain_bound
            result["bound"] = chain_bound(self, result)
        kernel = {"coords": list(self.znames), "N": self.N, "p": P}
        if state.get("random_kernel"):
            kernel["random"] = self._random_kernel()
        if self.stacked_first:
            if not _settled(result) and not self.stacked_quick:
                return {"rerun_blocks": True}
            result["kernel"] = kernel
            if not self.closed_form:
                result["non_identifiable"] = [d if d.get("type") == "scaling" else
                                             {"support": d.get("support"), "type": d.get("type"),
                                              "closed_form": False} for d in result["non_identifiable"]]
            run.stage("done (stacked)" if not _settled(result) else "done (stacked, rank settled)")
            result["stacked"] = bool(run.ss_constraint)
            return result
        gap_open = run.has_gaps and result.get("gap_certificate") is not None
        if self.verify and (not sc.get("certified") or gap_open) and not _settled(result):
            checks = []
            if not sc.get("certified"):
                try:
                    checks.append(saturation_guard(self._guard_call, self.point0_solved, sc["nt_used"]))
                except Exception as e:
                    swallow(e, "saturation guard")
                    checks.append({"ok": None, "method": "saturation guard", "reason": str(e)})
            if gap_open:
                checks.append(self._gap_guard(sc))
            result["verification"] = _joint_check(checks)
        result["kernel"] = kernel
        return result

    def _kernel_point(self):
        """Base-point leaf values with the resting states solved there; None where a resting value is ill defined."""
        env = {nm: int(v) for nm, v in zip(self.leaf_names_aug, self.sc["point0"])}
        if not self.joint:
            return env
        conds = list(self.sys.equil_conds)
        if len(conds) != 1:
            return None
        try:
            # at the kernel's point, whose leaves include the recast coordinates
            sc0 = self.sys.solve_cond(self.point0_solved, PRIMES[0], conds[0])
        except Exception as exc:
            swallow(exc, "kernel point")
            sc0 = None
        if sc0 is None:
            return None
        return {nm: int(v) for nm, v in zip(self.leaf_names_aug, sc0["ptc"])}

    def _kept_cols(self):
        """The columns a closed form is checked on: no resting-state copy, no jet projected off."""
        drop = set(self.run.nuis_drop)
        return [i for i, z in enumerate(self.znames) if not _CSUFFIX.search(z) and i not in drop]

    def _phys_vector(self, vec, env):
        """The closed form {name: expr} at `env` on the physical coordinates; None where it does not evaluate."""
        from ..closedform import eval_modq
        P = PRIMES[0]
        if env is None or not _rational_forms(list(vec.values())):
            return None
        phys = self._kept_cols()
        v = np.zeros(len(phys), dtype=np.int64)
        for j, i in enumerate(phys):
            if self.znames[i] in vec:
                x = eval_modq(vec[self.znames[i]], env, P)
                if x is None:
                    return None
                v[j] = x % P
        return v

    def _in_kernel(self, vec, env):
        """Whether the closed form {name: expr} is in the kernel at the base point; None where it does not evaluate."""
        P = PRIMES[0]
        v = self._phys_vector(vec, env)
        if v is None:
            return None
        phys = self._kept_cols()
        Nphys = np.asarray(self.N, dtype=np.int64)[phys, :] % P
        return bool(in_span(Nphys, v, len(phys), P)) if np.any(v) else None

    def _complete_forward(self, dirs):
        """Kernel-refuted forward closed forms completed by their solved rates, eta_R = sum_c dR/dc eta_c."""
        import sympy as spy
        lines = (self.multi.get("tapes") or [{}])[0].get("constraint_model")
        if not self.joint or not lines:
            return dirs
        try:
            eqs = [model_sympify(ln.split("=", 1)[1]) for ln in lines]
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            return dirs
        out = []
        for d in dirs:
            vec, solved = d.get("vector"), [r for r in d.get("solved_rates") or [] if r]
            missing = [r for r in solved if isinstance(vec, dict) and r not in vec]
            if not d.get("closed_form") or not missing:
                out.append(d)
                continue
            # the completion in the recast coordinates of the resting equations
            vec = d.get("checked") or vec
            try:
                comp = {k: model_sympify(v) for k, v in vec.items()}
                moved = dict(comp)
                for r in self.recast:
                    # a recast power whose base the direction fixes: E moves with the exponent
                    E, L, n = (spy.Symbol(str(r[k])) for k in ("E", "L", "exp"))
                    if str(r["base"]) not in vec and str(r["exp"]) in vec:
                        moved[str(E)] = E * L * comp[str(n)]
                for r in missing:
                    R = spy.Symbol(r)
                    sol = next((spy.solve(spy.fraction(spy.together(e))[0], R) for e in eqs
                                if R in e.free_symbols), [])
                    if len(sol) != 1:
                        raise InapplicableError(r)
                    comp[r] = spy.cancel(sum((spy.diff(sol[0], spy.Symbol(c)) * v for c, v in moved.items()),
                                             spy.Integer(0)))
                full = {k: str(v).replace("**", "^") for k, v in comp.items() if v != 0}
                env = self._kernel_point()
                # an absent solved rate either stays or moves with the others: the form as
                # reconstructed stands unless the kernel refutes it
                if self._in_kernel(vec, env) is not False:
                    out.append(d)
                elif self._in_kernel(full, env) is not False:
                    back = recast_backsub(full, self.recast) if self.recast else full
                    out.append(dict(d, vector=back, support=sym_sort(list(full)),
                                    checked=full if self.recast else d.get("checked")))
                else:
                    out.append({"support": sym_sort(list(full)), "type": "general",
                                "closed_form": False, "reason": "no closed form on the kernel"})
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                out.append({"support": sym_sort(list(vec) + missing), "type": "general",
                            "closed_form": False, "reason": "a solved rate of the forward route "
                            "has no closed form"})
        return out

    def _eliminate_states(self, dirs):
        """Closed forms with resting states rewritten in the coordinates by substituting solved rates where possible."""
        lines = (self.multi.get("tapes") or [{}])[0].get("constraint_model")
        states = set(self.multi.get("real_state_names") or []) - set(self.znames)
        if not self.joint or not lines or not states:
            return dirs
        coords = {z for z in self.znames if not _CSUFFIX.search(z)}
        atoms = set(self.atom_names or [])
        try:
            eqs = [model_sympify(ln.split("=", 1)[1]) for ln in lines]
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            return dirs
        out = []
        for d in dirs:
            vec = d.get("vector") if d.get("closed_form") else None
            if not isinstance(vec, dict) or any("|" in str(k) + str(v) for k, v in vec.items()):
                out.append(d)
                continue
            try:
                # entries of the leaves of terms at rest leave the report, resting states in them stay
                held = any(states & {str(x) for x in model_sympify(v).free_symbols}
                           for k, v in vec.items() if k not in atoms)
                out.append(self._rewrite_in_coords(d, vec, eqs, states, coords) if held else d)
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                out.append(d)
        return out

    @staticmethod
    def _rewrite_in_coords(d, vec, eqs, states, coords):
        import sympy as spy
        comp0 = {k: model_sympify(v) for k, v in vec.items()}

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

        # orders of substitution that solve the same equations for the same coordinates reach the same components and
        # are tried once; the search stops after a fixed number of steps, and the direction then stays in the resting
        # states, as where no rewrite exists
        failed, steps = set(), [0]

        def search(c, left, solved=frozenset()):
            key = (frozenset(left), solved)
            if key in failed or steps[0] >= _REWRITE_STEPS:
                return None
            steps[0] += 1
            done = normal(c)
            if done is not None or not left:
                return done
            r = _search_from(c, left, solved)
            if r is None:
                failed.add(key)
            return r

        def _search_from(c, left, solved):
            for i in left:
                num, cand = solvable(eqs[i], c)
                for x in cand:
                    sol = spy.solve(num, x)
                    if len(sol) != 1:
                        continue
                    rest = [j for j in left if j != i]
                    # a failed node is skipped before the substitution, which is the costly part
                    if (frozenset(rest), solved | {x}) in failed:
                        continue
                    r = search({k: spy.cancel(v.subs(x, sol[0])) for k, v in c.items()}, rest, solved | {x})
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
        """Independent closed forms on the kernel, filled with support-only kernel directions until they span it."""
        from ..closedform import eval_modq
        from ..finalize import public_symmetry
        P = PRIMES[0]
        phys = self._kept_cols()
        n = len(phys)
        env = self._kernel_point()
        # a form without resting states evaluates at the leaves of the base point alone
        leaves = {nm: int(v) for nm, v in zip(self.leaf_names_aug, self.sc["point0"])}
        rest = set(self.multi.get("real_state_names") or []) - {self.znames[i] for i in phys}
        tok = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
        # spans kept in reduced form: the tests below run once per direction and per column
        span, keep = SpanTest(np.zeros((n, 0), dtype=np.int64), n, P), []
        for d in dirs:
            gen = public_symmetry(d)["generator"]
            # the resting-state columns of the joint system have no physical component
            gen = {k: v for k, v in gen.items() if not _CSUFFIX.search(k)} if gen else gen
            at = env if env is not None or not gen else \
                (None if rest & {t for x in gen.values() for t in tok.findall(str(x))} else leaves)
            inside = None if not gen else self._in_kernel(gen, at)
            if inside is None and d.get("checked") and env is not None:
                # a form with recast or exponential terms, in the coordinates it was checked in
                gen, at = {k: v for k, v in d["checked"].items() if not _CSUFFIX.search(k)}, env
                inside = self._in_kernel(gen, at)
            if inside is None:
                kv = d.pop("kernel_residue", None)
                if kv is not None:
                    v = np.asarray(kv, dtype=np.int64)[phys] % P
                    if v in span:
                        continue
                    span.add(v)
                keep.append(d)
                continue
            if not inside:
                continue  # the kernel fill below supplies the support of the direction
            v = np.zeros(n, dtype=np.int64)
            for j, i in enumerate(phys):
                if self.znames[i] in gen:
                    v[j] = eval_modq(gen[self.znames[i]], at, P) % P
            if v in span:
                continue
            span.add(v)
            keep.append(d)
        Nphys = np.asarray(self.N, dtype=np.int64)[phys, :] % P
        basis, n_basis = SpanTest(np.zeros((n, 0), dtype=np.int64), n, P), 0
        for k in range(Nphys.shape[1]):
            if Nphys[:, k] not in basis:
                basis.add(Nphys[:, k])
                n_basis += 1
        # a single coordinate the kernel moves alone first, then the kernel columns
        for j in range(n):
            if len(keep) >= n_basis:
                break
            e = np.zeros(n, dtype=np.int64)
            e[j] = 1
            if e in basis and e not in span:
                span.add(e)
                keep.append(self._unit_fill(phys[j]))
        for k in range(Nphys.shape[1]):
            if len(keep) >= n_basis:
                break
            if Nphys[:, k] in span:
                continue
            span.add(Nphys[:, k])
            supp = sym_sort([self.znames[phys[j]] for j in np.nonzero(Nphys[:, k])[0]])
            if len(supp) == 1:
                keep.append(self._unit_fill(self.znames.index(supp[0])))
                continue
            keep.append({"support": supp, "type": "general", "closed_form": False,
                         "reason": "not reconstructed in closed form"})
        return keep

    def _distances(self):
        """Per state of the tapes the fewest steps of the right-hand sides from it to an observable."""
        dist = {}
        for t in self.multi["tapes"]:
            rhs = {}
            for l in strs(t.get("model_lines")):
                lhs, e = l.split("=", 1)
                rhs[lhs.strip()] = set(re.findall(r"[A-Za-z_.][A-Za-z0-9_.]*", e))
            reach = set(re.findall(r"[A-Za-z_.][A-Za-z0-9_.]*", " ".join(l.split("=", 1)[-1]
                                                                          for l in strs(t.get("obs_lines")))))
            k, seen = 0, set()
            while reach - seen:
                for x in reach - seen:
                    dist[x] = min(dist.get(x, k), k)
                seen |= reach
                reach = seen | {y for x in seen if x in rhs for y in rhs[x]}
                k += 1
        return dist

    def _jx_regular(self, sc):
        """Whether the resting Jacobian J_x of every condition solved directly is invertible at the base point."""
        sys, P = self.sys, PRIMES[0]
        for ci in sys.equil_conds:
            try:
                sc0 = sys.solve_cond(sc["point0"], P, ci)
            except MsolveMissingError:
                # a check of the proof, not of the result: without msolve the rank stays unproven
                session.note("limits", "resting Jacobian at the base point: not checked, msolve is missing")
                return False
            except Exception as exc:
                swallow(exc, "resting Jacobian at the base point")
                return False
            if sc0 is None:
                return False
            if sc0.get("cont"):
                # the rows of a continued condition enter as series, not at the point
                continue
            Jx = (sc0.get("sol") or {}).get("df_jx") or []
            if not len(Jx):
                continue
            M = np.array([[int(x) % P for x in row] for row in Jx], dtype=np.int64)
            if M.shape[0] != M.shape[1] or int(_core.sym_rref_mod(M.astype(np.float64), P)["rank"]) < M.shape[0]:
                return False
        return True

    def _entry_orders(self, names, top):
        """Per jet of an unknown input the first Lie order at which it can enter an observable, proven over Q.

        If no L^k h with k < k0 holds a jet of the input u, jet j of u enters from order k0 + j on: the
        derivative along the flow raises the jets of u by one and brings in u only through the right-hand
        sides. k0 is the first order at which the derivative by u of some observable of some condition does
        not cancel to zero. Under events the rows come from several times, so there every jet stays a column."""
        if self.joint or self.run.has_gaps or self.multi.get("nuisance_starts"):
            return {}
        import sympy as spy

        from ...sym.exact import SIZE_CAP, _zero
        try:
            tapes = []
            for t in self.multi["tapes"]:
                rhs = {}
                for line in strs(t.get("model_lines")):
                    lhs, e = line.split("=", 1)
                    rhs[spy.Symbol(lhs.strip())] = model_sympify(e)
                tapes.append((rhs, [model_sympify(line.split("=", 1)[1]) for line in strs(t.get("obs_lines"))]))
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            return {}
        nxt = {str(x): str(e) for rhs, _ in tapes for x, e in rhs.items()
               if str(x) in names and e.is_Symbol and str(e) in names}
        out = {}
        for u in [nm for nm in names if nm not in set(nxt.values())]:
            k0 = top + 1
            for rhs, obs in tapes:
                es, k = list(obs), 0
                try:
                    while k < k0 and all(spy.count_ops(e) <= SIZE_CAP for e in es) \
                            and all(_zero(spy.diff(e, spy.Symbol(u))) is True for e in es):
                        es = [sum((spy.diff(e, x) * f for x, f in rhs.items()), spy.S.Zero) for e in es]
                        k += 1
                except SYMPY_FAILS as exc:
                    sympy_fail(exc)
                k0 = min(k0, k)
            nm, j = u, 0
            while nm in names and nm not in out:
                out[nm] = k0 + j
                nm, j = nxt.get(nm), j + 1
        return out

    def _projection(self, sc):
        """{"status", ...} of the rank modulo the nuisance coordinates, which a point where they drop rank can raise."""
        run = self.run
        top = int(sc["nt_used"]) + int(run.mtot)
        names = [self.znames[c] for c in run.nuis_cols]
        dist, starts = self._distances(), self.multi.get("nuisance_starts") or {}
        entry = self._entry_orders(names, top)
        # a jet that k steps of the dependencies separate from an observable enters the rows from order k on;
        # jet j of an input that the Lie derivatives below order k0 do not hold enters from order k0 + j on
        can = [j for j, nm in enumerate(names)
               if dist.get(starts.get(nm, nm), dist.get(re.sub(r"_0$", "", nm), 0)) <= top
               and entry.get(nm, 0) <= top]

        def full(r):
            B = np.asarray(r.get("nuisance_block"), dtype=np.int64).reshape(-1, len(names))
            return not can or int(_core.sym_rref_mod(B[:, can].astype(np.float64), int(r["p"]))["rank"]) == len(can)
        # full column rank of the columns that can enter a row proves the rank, at the base point or at a uniform
        # point of the same rank
        base = dict(sc["ref"], p=PRIMES[0])
        if base.get("nuisance_block") is not None and full(base):
            return {"status": "proven"}
        if self.joint:
            return {"status": "unchecked"}
        q = verify_prime()
        pt = uniform_point(len(sc["point0"]), q, session.current().seed("projection", ",".join(names)))
        try:
            r = self.kcall4(pt, q, sc["nt_used"], sc.get("mtot_used") or 0)
        except Exception as exc:
            swallow(exc, "the rank modulo the nuisance coordinates at a random point")
            r = None
        if not ok(r) or r.get("nuisance_block") is None:
            return {"status": "unchecked"}
        if rank_of(r) != rank_of(sc["ref"]):
            return {"status": "differs", "base": int(rank_of(sc["ref"])), "random": int(rank_of(r))}
        if full(dict(r, p=q)):
            return {"status": "proven"}
        return {"status": "verified", "columns": len(can), "order": top}

    def _random_kernel(self):
        """{"N", "p"}: the kernel at the orders taken at a point uniform modulo the verification prime, or None."""
        sc, q = self.sc, verify_prime()
        pt = uniform_point(len(sc["point0"]), q, session.current().seed("random kernel"))
        try:
            r = self.kcall(pt, q, sc["nt_used"])
        except Exception as exc:
            swallow(exc, "the kernel at a random point")
            return None
        if not ok(r):
            return None
        piv = {int(x) for x in np.atleast_1d(r["pivots"])}
        return {"N": nullspace_basis(r, [c for c in range(self.nz) if c not in piv], q), "p": q}

    def _unit_fill(self, col):
        """The unit direction of column `col`, closed only where the kernel holds it at the verification points too."""
        nm, sc = self.znames[col], self.sc
        e = {"support": [nm], "type": "scaling", "vector": {nm: 1}, "closed_form": True}
        try:
            held = verify_in_nullspace(e, col, self.znames, self.leaf_names_aug, sc["point0"], sc["nt_used"],
                                       self.kcall, 0, self.nz)
        except Exception as exc:
            swallow(exc, "unit direction at the verification points")
            held = False
        if held:
            return e
        return {"support": [nm], "type": "general", "closed_form": False,
                "reason": "the kernel holds it at the base point only"}

    def _gap_guard(self, sc, margin=2):
        """The rank at `margin` gap orders past the reported one, at the same point and Lie order."""
        mt, nt = self.run.mtot, int(sc["nt_used"])
        try:
            r0 = self.kcall4(sc["point0"], PRIMES[0], nt, mt)
            r1 = self.kcall4(sc["point0"], PRIMES[0], nt, mt + margin)
        except Exception as e:
            swallow(e, "gap guard")
            return {"ok": None, "method": "gap guard", "reason": str(e)}
        if not (ok(r0) and ok(r1)):
            return {"ok": None, "method": "gap guard", "reason": "no kernel past the reported gap order"}
        base, top = int(rank_of(r0)), int(rank_of(r1))
        if top > base:
            return {"ok": False, "method": "gap guard", "kernel_rank": base, "kernel_rank_extended": top,
                    "reason": f"rank grows {base} -> {top} by gap order {mt + margin}"}
        return {"ok": True, "method": "gap guard", "kernel_rank": base,
                "reason": f"rank {base} stable through gap order {mt + margin}"}

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
        pr = self._kept_cols()
        sub = np.asarray(N, dtype=np.int64)[pr, :] % PRIMES[0]
        return [self.znames[pr[i]] for i in range(len(pr)) if not np.any(sub[i] != 0)]

    def _prove_by_scalings(self, result):
        """Proven when the rank plus the independent exact scalings and static fields equals the dimension."""
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
                    V[i, coords.index(k)] = _int_weight(w, PRIMES[0])
        if self.static_rows is not None and all(c in self.phys_params0 for c in coords):
            V = np.vstack([V, np.asarray(self.static_rows)[:, [self.phys_params0.index(c) for c in coords]]])
        k = int(_core.sym_rref_mod(V.astype(np.float64), PRIMES[0])["rank"]) if V.size else 0
        if result["rank"] + k == result["dim"]:
            result["rank_proven"] = k
            result["rank_proof"] = "proven symmetries"

    def _prove_exact(self, result):
        """Closed forms proven over Q (proof.prove_exact)."""
        from .proof import prove_exact
        prove_exact(self, result)

    def _prove_by_fields(self, result):
        """Proven when every kernel direction lies in the span of invariant fields."""
        dirs = result.get("non_identifiable") or []
        if result.get("rank_proven") is None and result["rank"] < result["dim"] and dirs and \
                len(dirs) == result["dim"] - result["rank"] and all(d.get("exact") for d in dirs):
            result["rank_proven"] = len(dirs)
            result["rank_proof"] = "proven symmetries"
            return
        if _settled(result) or result["rank"] >= result["dim"] or self.recast or self.run.nuis_cols \
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
            except RouteError as e:
                proof = {"ok": False, "why": str(e)}
            except Exception as e:
                swallow(e, "field proof")
                proof = {"ok": False, "why": str(e)}
        if not proof.get("ok") and len(set(regime_keys(self.multi["tapes"]))) <= 3:
            try:
                proof = field_proof(self.multi, self.N, self.znames, run.leaf_names, self.point0_solved,
                                    self.field_chains, self.field_joint, run.cores,
                                    fixed=self.field_fixed(self.N))
            except RouteError as e:
                proof = {"ok": False, "why": str(e)}
            except Exception as e:
                swallow(e, "field proof")
                proof = {"ok": False, "why": str(e)}
        if state.env("LIEDIAG"):
            print("[liediag] invariant fields: " + ("proven" if proof.get("ok") else str(proof.get("why"))))
        if proof.get("ok"):
            result["rank_fields"] = int(result["dim"] - result["rank"])
            result["rank_fields_bound"] = proof.get("bound")


def _joint_check(checks):
    """One verification from several guards: failed if one fails, inconclusive if one is."""
    if len(checks) == 1:
        return checks[0]
    oks = [c.get("ok") for c in checks]
    grow = [c["grow_at"] for c in checks if c.get("grow_at") is not None]
    return {"ok": False if False in oks else (None if None in oks else True),
            "grow_at": max(grow) if grow else None,
            "method": " and ".join(c["method"] for c in checks),
            "reason": "; ".join(f"{c['method']}: {c.get('reason')}" for c in checks)}


def _settled(result):
    """Whether the rank is proven, or the invariant fields hold the kernel at every order."""
    return result.get("rank_proven") is not None or result.get("rank_fields") is not None


def observability_analytic_multi(multi, closed_form=False, cores=1, equil_zero_states=(),
                                 cond_zero_states=(), t0events=(), n_conditions=None, chain_of=None,
                                 n_gaps=0, implicit_steady_state=False, control=None, verify=False,
                                 codim_spec=None):
    """The observability analysis of compiled tapes; the raw result or None."""
    K = len(multi.get("tapes") or [])

    def build(idx):
        """The Analysis of the conditions `idx`."""
        def part(x):
            x = list(x or [])
            return [x[i] for i in idx] if len(x) == K else x
        m = multi if len(idx) == K else select.restrict(multi, idx)
        return Analysis(m, closed_form, cores, equil_zero_states, part(cond_zero_states), part(t0events),
                        n_conditions, part(chain_of), n_gaps, implicit_steady_state, control, verify, codim_spec)

    def analyze(idx):
        an = build(idx)
        out = an.execute()
        if isinstance(out, dict):
            an.run.stage("done")
            out["timing"] = list(an.run.stages)
        return out
    if select.eligible(multi, n_gaps, implicit_steady_state):
        out = _on_selection(multi, build, analyze, select.resting(multi, implicit_steady_state), cores)
        if out is not None:
            return out
    return analyze(list(range(K)))


def _rest_on(build, conds, d):
    """_rest_proof() of the direction `d` on the resting conditions `conds`, at a point with the first of them seeded
    forward; the reason where it does not apply."""
    from .proof import _rest_proof
    try:
        an = build(conds)
        if an.sys.chained:
            an.sys.unchain()
        for i in range(len(an.sys.cont_off)):
            an.sys.cont_off[i] = True
        pt = random_point(an.run.n_leaves, seed=session.current().seed("selection rest proof") % (1 << 31))
        pt = an.sys.sat_prep(pt, rounds=2)
        if pt is None:
            return "no resting state at a random point"
        an.sc = {"point0": pt}
        return _rest_proof(an, d)
    except SYMPY_FAILS as exc:
        sympy_fail(exc)
        return f"{type(exc).__name__}"
    except Exception as exc:
        swallow(exc, "selection rest proof")
        return f"{type(exc).__name__}"


def _on_selection(multi, build, analyze, rest, cores):
    """The analysis on the conditions that change the rank at a random point, with each direction proven on the
    others; None where that does not pay or a direction is not shown on a condition left out. With `rest`, the
    conditions start at rest."""
    t0 = time.monotonic()
    K = len(multi["tapes"])
    if rest:
        sel = select.choose_resting(build(list(range(K))), *select.taking_order(multi["tapes"]))
    else:
        sel = select.choose(multi, cores)
    if sel is None:
        return None
    kept = sel["kept"]
    for rnd in range(select._ROUNDS):
        if 2 * len(kept) > K:
            return None
        if state.env("TIMING"):
            print(f"[select] {len(kept)} of {K} conditions ({sel['structures']} structures) at Lie order "
                  f"{sel['lie_order']}, rank {sel['rank']} at the point, {time.monotonic() - t0:.1f}s", flush=True)
        out = analyze(kept)
        if not isinstance(out, dict) or out.get("rerun_blocks"):
            return out
        nt = int(out.get("lie_order_used") or 0)
        if not rest and nt > sel["lie_order"] and out["rank"] < out["dim"]:
            # the conditions that raise the rank of the kept ones at the order the analysis reached
            more = select.choose(multi, cores, nt, kept)
            sel["lie_order"] = nt
            if more is None:
                return None
            if set(more["kept"]) - set(kept):
                sel["rank"] = more["rank"]
                kept = more["kept"]
                continue
        cov = select.cover(multi, out, kept, cores, rest=(lambda conds, d: _rest_on(build, conds, d)) if rest
                           else None)
        if cov is None:
            return None
        if cov["ok"]:
            out["selection"] = {"kept": list(kept), "conditions": K, "structures": sel["structures"],
                                "lie_order": sel["lie_order"], "rank_at_point": sel["rank"], "rounds": rnd + 1,
                                "dropped_by": cov["how"], "seconds": round(time.monotonic() - t0, 3)}
            out["conditions"] = K
            return out
        kept = sorted(set(kept) | set(cov["add"]))
    return None
