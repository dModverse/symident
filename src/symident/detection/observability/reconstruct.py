"""Closed forms for the residual kernel directions: narrow fits, static subsystems, the
forward route, invariant fields, gauged and dense interpolation, each verified at a
fresh prime before it is adopted."""

import time

import numpy as np

from ... import _core, state
from ...gfp import (
    PRIMES,
    canon_gauge,
    in_span,
    logcoord_gauge,
    minsupport_gauge,
    mulmod,
    null_residues,
    random_point,
)
from ...rexpr import sym_sort
from ...sym.msolve import msolve_timeout
from ..closedform import expired, recast_backsub, simplify
from ..fields import field_directions, field_setup
from ..forward import fwd_adopt, perprime_forward_multi, perprime_forward_shared
from ..interp import exp_backsub, interpolate_direction
from ..perprime import interpolate_perprime
from ..relevance import build_shared_bank, dense_need, direction_relevance
from ..verify import verify_direction, verify_in_nullspace, verify_perprime
from .context import ints, ok, strs
from .homotopy import Homotopy
from .narrow import Narrow


class Reconstruction:
    """The residual directions of one analysis, closed in turn by each route."""

    def __init__(self, an, residual):
        self.an = an
        self.sc = an.sc
        self.residual = list(residual)
        self.ctrl = an.ctrl
        self.dir_call = an.dir_call()
        self.t0 = time.monotonic()
        to = self.ctrl.get("timeout")
        self.ctrl["deadline"] = self.t0 + to if to is not None and np.isfinite(to) else None
        self.probe_next = self.sc["pool_next"]
        self.pool_next = self.probe_next + an.n_aug * (self.ctrl["probe_retries"] + 16)
        self.zvals0 = np.asarray(self.sc["point0"])[an.z_slots] if len(an.z_slots) else None
        self.phys_cols = [i for i, z in enumerate(an.znames) if z not in an.zsn] if an.joint else None

    def tlog(self, msg):
        """Print `msg` with the elapsed time when timing is on."""
        if self.an.run.timing:
            print(f"[sym {time.monotonic() - self.t0:6.1f}s] {msg}", flush=True)

    def residues(self, fc):
        """Kernel vector of the saturation reference with a one in the free column `fc`."""
        return null_residues(self.sc["ref"], fc, PRIMES[0])

    def drop_spanned(self, span):
        """Keep the residual directions outside `span`, growing it with each kept one."""
        nz, P = self.an.nz, PRIMES[0]
        keep = []
        for fc in self.residual:
            bf = self.residues(fc)
            if in_span(span, bf, nz, P):
                continue
            keep.append(fc)
            span = np.concatenate([span, bf.reshape(nz, 1)], axis=1)
        self.residual = keep
        return span

    # ---- relevance probes -------------------------------------------------------------

    def probe(self, kb, piv0, first, base=None, leaves=None):
        """The kernel with one leaf perturbed at a time, retried until the pivots match."""
        an, sc, ctrl = self.an, self.sc, self.ctrl
        if base is None:
            base = sc["point0"]
        if leaves is None:
            leaves = list(range(an.n_aug))
        zs = an.z_slots
        rp = [None] * an.n_aug
        pending = list(leaves)
        piv0 = ints(piv0)
        for att in range(ctrl["probe_retries"]):
            if not pending or expired(ctrl):
                break
            perts = []
            for li in pending:
                pert = np.array(base, dtype=np.int64)
                pert[li] = sc["pool"](first + li * ctrl["probe_retries"] + att)[0]
                perts.append(pert)
            cands = kb(perts, [PRIMES[0]] * len(perts), sc["nt_used"])
            resolved = [False] * len(pending)
            for j, li in enumerate(pending):
                pert, cand = perts[j], cands[j]
                rp[li] = {"rp": cand, "zvals": pert[zs] if len(zs) else None, "pertval": pert[li]}
                if ok(cand) and ints(cand["pivots"]) == piv0:
                    resolved[j] = True
            pending = [li for li, r in zip(pending, resolved) if not r]
        return rp

    def probe_groups(self, kb, piv0, first, res_fn, ref, base=None):
        """The probe by bisection over leaf groups: a group whose joint perturbation leaves
        the residues unchanged is irrelevant as a whole."""
        an, sc, ctrl, P = self.an, self.sc, self.ctrl, PRIMES[0]
        if base is None:
            base = sc["point0"]
        bv = res_fn(ref, P)
        if bv is None:
            return self.probe(kb, piv0, first, base)
        piv0 = ints(piv0)
        zs = an.z_slots
        zb = base[zs] if len(zs) else None
        rp = [{"rp": ref, "zvals": zb, "pertval": base[li]} for li in range(an.n_aug)]
        off = first + an.n_aug * ctrl["probe_retries"]
        groups = [list(range(an.n_aug))]
        single = []
        while groups:
            perts = []
            for g in groups:
                pt = np.array(base, dtype=np.int64)
                pt[g] = sc["pool"](off + np.arange(len(g)))
                off += len(g)
                perts.append(pt)
            cands = kb(perts, [P] * len(perts), sc["nt_used"])
            nxt = []
            for gi, g in enumerate(groups):
                cd = cands[gi]
                piv_ok = ok(cd) and ints(cd["pivots"]) == piv0
                v = res_fn(cd, P) if piv_ok else None
                if v is not None and np.array_equal(np.asarray(v), np.asarray(bv)):
                    # the kernel stays, but a residue fn in log coordinates divides by z
                    for li in g:
                        own = [c for c, nm in enumerate(an.znames) if nm == an.leaf_names_aug[li]]
                        if zb is not None and own:
                            zv = np.array(zb, dtype=np.int64)
                            zv[own] = perts[gi][li]
                            rp[li] = {"rp": ref, "zvals": zv, "pertval": perts[gi][li]}
                    continue
                if len(g) > 1:
                    h = len(g) // 2
                    nxt.extend([g[:h], g[h:]])
                elif piv_ok:
                    rp[g[0]] = {"rp": cd, "zvals": perts[gi][zs] if len(zs) else None,
                                "pertval": perts[gi][g[0]]}
                else:
                    single.append(g[0])
            if len(nxt) > len(groups):
                nxt = [[x] for g in nxt for x in g]
            groups = nxt
        if single:
            rs = self.probe(kb, piv0, first, base, leaves=single)
            for li in single:
                rp[li] = rs[li]
        return rp

    # ---- routes -----------------------------------------------------------------------

    def narrow_ok(self):
        """Whether the reconstruction may narrow onto column subsets."""
        an, run = self.an, self.an.run
        return not state.env("NONARROW") and (
            an.can_batch or (run.has_gaps and not run.ss_constraint and not an.joint and not an.transient)
            or (an.transient and not run.has_gaps and not run.ss_constraint and not an.recast)
            or an.chunked)

    def narrowed(self):
        """Directions closed on their own support, minimal-support candidates first."""
        an, nz, P = self.an, self.an.nz, PRIMES[0]
        try:
            ms0 = minsupport_gauge(self.residual, an.scal_rows, P, nz, self.sc, an.free_cols,
                                   cand_cap=self.ctrl["minsupport_cand_cap"])
        except Exception:
            ms0 = {}
        cands = []
        vecs = ms0.get("vectors")
        if vecs is not None and np.asarray(vecs).shape[1]:
            for gi in range(vecs.shape[1]):
                cands.append((vecs[:, gi], ms0["anchors"][gi]))
        for fc in self.residual:
            cands.append((self.residues(fc), fc))
        span = an.scal_rows.T.copy() if an.scal_rows.shape[0] else np.zeros((nz, 0), dtype=np.int64)
        out = []
        for v, anchor in cands:
            if len(out) >= len(self.residual) or expired(self.ctrl):
                break
            if in_span(span, v, nz, P):
                continue
            try:
                e = Narrow(self, v, anchor).run()
            except Exception:
                if state.env("ROBUSTDIAG"):
                    import traceback
                    traceback.print_exc()
                e = None
            if e is not None and e.get("closed_form"):
                out.append(e)
                span = np.concatenate([span, np.asarray(v, dtype=np.int64).reshape(nz, 1)], axis=1)
        self.tlog(f"narrow: {len(out)} of {len(self.residual)} closed")
        self.drop_spanned(span)
        return out

    def statics(self):
        """Static subsystem fields that lie in the kernel, the residual reduced by them."""
        an, nz, P = self.an, self.an.nz, PRIMES[0]
        N, pp = an.N, an.phys_params0
        stf = an.static_fields()
        p_cols = [i for i, z in enumerate(an.znames) if z in set(pp)]
        Np = N[p_cols, :] % P
        span = an.scal_tangents.copy()
        out = []
        for f in (stf or []):
            val = [int(f["val"][pp.index(an.znames[c])]) for c in p_cols]
            cc = _core.sym_solve_mod(Np, val, P)
            if cc is None:
                continue
            vw = np.zeros(N.shape[0], dtype=np.int64)
            for k, ck in enumerate(np.atleast_1d(cc)):
                vw = (vw + mulmod(N[:, k], int(ck), P)) % P
            if in_span(span, vw, nz, P):
                continue
            span = np.concatenate([span, vw.reshape(nz, 1)], axis=1)
            an.static_rows = np.asarray(f["lval"]).reshape(1, -1) if an.static_rows is None else \
                np.vstack([an.static_rows, f["lval"]])
            out.append({k: v for k, v in f.items() if k not in ("val", "lval")})
        if out:
            self.drop_spanned(span)
        self.tlog(f"static subsystems: {len(out)} field(s) closed, {len(self.residual)} residual direction(s) left")
        return out

    def _forward_entries(self, closed):
        out = []
        for e in closed:
            e = dict(e)
            if self.an.recast:
                e["vector"] = recast_backsub(e["vector"], self.an.recast)
            e["route"] = "forward fit per prime"
            out.append(e)
        return out

    def _forward_multi(self):
        an, sys = self.an, self.an.sys
        try:
            return perprime_forward_multi(
                self.residual, self.sc, an.kcall, sys.kcall_fwd, an.znames, an.z_slots, an.leaf_names_aug,
                an.nz, an.scaling, strs(an.multi.get("z_state_names")), strs(an.multi.get("param_names")),
                an.recast, True, self.ctrl, self.phys_cols, sys.models, sys.real_state_names,
                sys.solve_param_names, sys.held_of(0))
        except Exception:
            return None

    def fork_map(self, f, xs):
        """f over xs in forked workers, one kernel thread each (OpenMP does not survive a fork)."""
        run, sys = self.an.run, self.an.sys
        xs = list(xs)
        saved = run.cores_call, sys.warm_off
        run.cores_call, sys.warm_off = (run.cores if len(xs) == 1 else 1), True
        try:
            return run.par_map(xs, f)
        finally:
            run.cores_call, sys.warm_off = saved

    def solve_cost(self):
        """Seconds for one backward resting-state solve, infinite when it fails."""
        an, sys = self.an, self.an.sys
        t0 = time.monotonic()
        try:
            with msolve_timeout(5):
                pr = sys.solve_raw(PRIMES[0], sys.joint_pv(random_point(an.run.n_leaves, 104729), PRIMES[0]),
                                   sys.equil_conds[0])
        except Exception:
            pr = None
        return time.monotonic() - t0 if ok(pr) else float("inf")

    def forward_first(self, stacked_ss):
        """Every residual direction from the forward route when the backward solve is slow."""
        an, sys = self.an, self.an.sys
        if not (an.run.ss_constraint and not stacked_ss and self.residual and an.joint
                and len({sys.model_key[c] for c in sys.equil_conds}) == 1 and not state.env("NOFWDFIRST")):
            return []
        cost = float("inf") if state.env("FWDFIRST") else self.solve_cost()
        if cost <= state.get("slow_solve", 0.5):
            return []
        try:
            fwd = perprime_forward_shared(
                self.sc, an.kcall, sys.kcall_fwd, an.znames, an.z_slots, an.leaf_names_aug, an.nz,
                an.scaling, strs(an.multi.get("z_state_names")), an.recast, True, self.ctrl, self.phys_cols,
                sys.models, sys.real_state_names, sys.solve_param_names, sys.held_of(0),
                kcall_fwd_sh=None if state.env("FWDWIDE") else sys.kcall_fwd_shared,
                select_fn=None if state.env("FWDALLCOND") else sys.fwd_select,
                map_fn=None if state.env("FWDSERIAL") else self.fork_map)
        except Exception as err:
            if state.env("ROBUSTDIAG"):
                print(f"[fwds] error: {err}")
            fwd = None
        if fwd is None:
            fwd = self._forward_multi()
        closed = fwd_adopt(fwd or [], len(self.residual))
        self.tlog(f"forward first (backward solve {cost:.1f} s): " +
                  ("not adopted" if closed is None else "every direction closed"))
        if closed is None:
            return []
        self.residual = []
        return self._forward_entries(closed)

    def homotopy(self, interp):
        """The open directions of `interp` closed along parameter lines through the base point,
        where the route applies: equilibrate conditions, no later events, no recast coordinates."""
        an, run = self.an, self.an.run
        open_ = [i for i, e in enumerate(interp) if not e.get("closed_form")]
        if not (open_ and self.ctrl.get("homotopy", True) and run.ss_constraint and an.joint and not run.has_gaps
                and not an.recast and len(interp) == len(self.residual)) or expired(self.ctrl):
            return interp
        try:
            hr = Homotopy(self, self.sc["point0"], self.sc["nt_used"]).run([self.residual[i] for i in open_])
        except Exception as e:
            self.tlog(f"homotopy: {e}")
            hr = None
        found = {} if hr is None else {fc: d for fc, d in zip(*hr) if d is not None}
        out = [found.get(self.residual[i], e) if i in open_ else e for i, e in enumerate(interp)]
        self.tlog(f"homotopy: {len(found)} of {len(open_)} open direction(s) closed")
        return out

    def fields(self, partial=False):
        """Directions from invariant fields of the unspecialised system."""
        an = self.an
        out = None
        try:
            fs = field_setup(an.multi, an.field_chains, an.run.cores, fixed=an.field_fixed(an.N))
        except Exception:
            fs = None
        if fs is not None:
            try:
                out = field_directions(fs, an.N, len(self.residual), an.scal_rows, an.znames,
                                       an.run.leaf_names, an.point0_solved, an.field_chains,
                                       an.field_joint, partial=partial)
            except Exception:
                out = None
        is_partial = isinstance(out, dict) and out.get("partial")
        self.tlog("global symmetries: " + ("no close" if out is None else
                  f"{len(out['dirs'] if is_partial else out)} direction(s) closed" +
                  (", the rest sampled" if is_partial else "")))
        return out

    # ---- interpolation ----------------------------------------------------------------

    def one(self, fc, rfn, log_coords=False, bank=None, fast_only=False, per_prime=False):
        """One direction interpolated, verified at a fresh prime and back-substituted."""
        an, sc, P = self.an, self.sc, PRIMES[0]

        def open_entry(reason):
            v = rfn(sc["ref"], P, self.zvals0) if rfn is not None else self.residues(fc)
            if v is None:
                v = self.residues(fc)
            return {"support": sym_sort([an.znames[i] for i in np.nonzero(v)[0]]), "type": "general",
                    "closed_form": False, "reason": reason}
        if expired(self.ctrl):
            return open_entry("reconstruction time budget (reconst_control(timeout=)) exceeded")
        args = (an.znames, an.z_slots, an.leaf_names_aug, an.n_aug, sc["point0"], sc["pool"],
                self.pool_next, sc["nt_used"], an.kcall)
        if per_prime:
            d = interpolate_perprime(fc, sc["ref"], sc["pivots"], *args, an.kbatch, self.rel_probe,
                                     self.ctrl, aux_leaves=an.aux_leaves, phys_cols=self.phys_cols)
        else:
            d = interpolate_direction(fc, sc["ref"], sc["pivots"], *args, self.rel_probe, rfn, self.ctrl,
                                      an.kbatch, bank, fast_only, aux_leaves=an.aux_leaves)
        self.pool_next = d["pool_next"]
        e = dict(d["entry"])
        rel_leaves = e.pop("relevant_leaves", None)
        if log_coords and e.get("closed_form"):
            e["vector"] = {nm: simplify(f"({x})*{nm}") for nm, x in e["vector"].items()}
        verified = False
        if e.get("closed_form"):
            vargs = (e, fc, an.znames, an.leaf_names_aug, sc["point0"], sc["nt_used"], an.kcall)
            def in_null():
                return (verify_in_nullspace(*vargs, sc["pool"], self.pool_next, an.nz,
                                                               self.dir_call))
            if per_prime:
                verified = verify_perprime(*vargs, sc["pivots"], sc["pool"], self.pool_next, an.nz,
                                           rel_leaves=rel_leaves)
            elif log_coords:
                verified = in_null()
            else:
                verified = verify_direction(*vargs, rfn) and in_null()
        if e.get("closed_form") and not verified:
            e = open_entry("a closed form was reconstructed but failed verification at a fresh prime")
        if e.get("closed_form"):
            e["route"] = ("fit per prime" if per_prime else "read-off" if fast_only else "fit") + \
                (" in log coordinates" if log_coords else "") + \
                (f" ({e['fit_how']})" if e.get("fit_how") else "")
            e.pop("fit_how", None)
            if an.recast:
                e["vector"] = recast_backsub(e["vector"], an.recast)
            eb = an.multi.get("exp_back") or {}
            if eb.get("names"):
                e["vector"] = exp_backsub(e["vector"], eb, an.multi.get("exp_atoms"))
        return e

    def gauge(self):
        """The residual directions of minimal support."""
        an = self.an
        return minsupport_gauge(self.residual, an.scal_rows, PRIMES[0], an.nz, self.sc, an.free_cols,
                                cand_cap=self.ctrl["minsupport_cand_cap"])

    def full_probe(self):
        """The probe over all leaves, by bisection over leaf groups when there are many."""
        an = self.an
        if an.n_aug > state.get("group_probe_min", 150):
            def all_res(rp, p, zvals=None):
                v = [null_residues(rp, fc, p) for fc in self.residual]
                return np.concatenate(v) if v else np.zeros(0, dtype=np.int64)
            return self.probe_groups(an.kbatch, self.sc["pivots"], self.probe_next, all_res, self.sc["ref"])
        return self.probe(an.kbatch, self.sc["pivots"], self.probe_next)

    def peeled(self):
        """Minimal-support directions read off in log coordinates."""
        nz = self.an.nz
        ms = self.gauge()
        out = []
        if not (ms.get("anchors") and ms.get("vectors") is not None):
            return out
        vec = np.zeros((nz, 0), dtype=np.int64)
        for gi, anchor in enumerate(ms["anchors"]):
            if ms["residue_fns"][gi] is None:
                continue
            e = self.one(anchor, ms["residue_fns"][gi], log_coords=True, fast_only=True)
            if e.get("closed_form"):
                out.append(e)
                vec = np.concatenate([vec, ms["vectors"][:, gi].reshape(nz, 1)], axis=1)
        if out:
            self.drop_spanned(vec)
        return out

    def ms_try(self, ms, fast_only):
        """All directions in the minimal-support gauge, or None unless all close."""
        if not fast_only:
            self.tlog(f"ms gauge: {len(ms['anchors'])} anchor(s) for {len(self.residual)} residual "
                      f"direction(s), {sum(f is not None for f in ms['residue_fns'])} residue fn(s)")
        if len(ms["anchors"]) != len(self.residual) or all(f is None for f in ms["residue_fns"]):
            return None
        rec = [self.one(anchor, ms["residue_fns"][gi], log_coords=True, fast_only=fast_only)
               for gi, anchor in enumerate(ms["anchors"])]
        if not fast_only:
            for e in rec:
                self.tlog(f"ms fit [{','.join(e.get('support', []))}]: " +
                          ("closed" if e.get("closed_form") else str(e.get("reason"))))
        return rec if all(e.get("closed_form") for e in rec) else None

    def dense(self, ms):
        """Each direction from a shared sample bank, per prime where the bank is missing,
        then rescued in the canonical and log-coordinate gauges."""
        an, sc, run = self.an, self.sc, self.an.run
        metas = [direction_relevance(fc, sc["ref"], sc["pivots"], an.z_slots, an.n_aug, sc["point0"],
                                     self.rel_probe, None) for fc in self.residual]
        needs = [dense_need(m["rel_by_entry"], self.ctrl)["max_need"] for m in metas]
        dense_dir = [i for i, nd in enumerate(needs) if nd > 0]
        self.tlog(f"relevance/need: {len(dense_dir)} dense dir(s), needs=[{','.join(map(str, needs))}]")
        bank = None
        if dense_dir:
            union = sorted({x for i in dense_dir for x in metas[i]["relevant"]})
            bk = build_shared_bank(union, max(needs[i] for i in dense_dir), sc["point0"], sc["pool"],
                                   self.pool_next, an.kbatch, sc["pivots"], sc["nt_used"], len(PRIMES), self.ctrl)
            self.pool_next = bk["pool_next"]
            if bk.get("ok"):
                bank = bk
            self.tlog(f"shared bank built ({len(union)} leaves, ok={bool(bk.get('ok'))})")
        bank_missing = run.ss_constraint and len(dense_dir) > 0 and bank is None
        interp = []
        for ii, fc in enumerate(self.residual):
            r = self.one(fc, None, per_prime=True) if bank_missing else self.one(fc, None, bank=bank)
            if not r.get("closed_form") and run.ss_constraint and not bank_missing:
                r = self.one(fc, None, per_prime=True)
            self.tlog(f"dense dir {ii + 1}/{len(self.residual)} done (closed={bool(r.get('closed_form'))})")
            interp.append(r)
        if run.ss_constraint and an.joint and an.recast and not all(e.get("closed_form") for e in interp):
            closed = fwd_adopt(self._forward_multi() or [], len(self.residual))
            if closed is not None:
                interp = self._forward_entries(closed)
                self.tlog(f"forward-multi closed {len(closed)} residual direction(s)")
        if bank_missing:
            return interp

        def rescue(gauge, log_coords=False):
            if len(gauge["anchors"]) != len(self.residual):
                return
            for ii in [i for i, e in enumerate(interp) if not e.get("closed_form")]:
                if gauge["residue_fns"][ii] is None or expired(self.ctrl):
                    continue
                rec = self.one(gauge["anchors"][ii], gauge["residue_fns"][ii], log_coords=log_coords)
                if rec.get("closed_form"):
                    interp[ii] = rec
        P, nz, scal_rows = PRIMES[0], an.nz, an.scal_rows
        if not all(e.get("closed_form") for e in interp):
            rescue(canon_gauge(self.residual, scal_rows, P, nz, sc))
        if not all(e.get("closed_form") for e in interp):
            rescue(logcoord_gauge(self.residual, scal_rows, P, nz, sc, self.zvals0), True)
        if not all(e.get("closed_form") for e in interp):
            self.tlog("rescues exhausted, trying ms sampling")
            msrec = self.ms_try(ms, False)
            if msrec is not None:
                interp = msrec
        return interp

    # ---- the sequence -----------------------------------------------------------------

    def run(self):
        """All directions, scalings first; None when the stacked first step must rerun
        condition by condition."""
        an, run = self.an, self.an.run
        if an.joint and self.residual:
            self.tlog("solve fill: " + ("serial" if run.cores <= 1 else f"fork, {run.cores} cores"))
        self.tlog(f"start: {len(self.residual)} residual direction(s)")
        static_part = self.statics() if an.joint and self.residual and not state.env("NOSTATIC") else []
        stacked_ss = an.stacked_first and run.ss_constraint
        fwd_first = self.forward_first(stacked_ss)
        narrowed = self.narrowed() if self.narrow_ok() and not stacked_ss and self.residual else []
        if stacked_ss and self.residual:
            return None
        field_ok = run.ss_constraint and not an.recast and not state.get("inner") and not state.env("NOFIELDS")
        interp, field_part, n_field_tried = None, [], 0
        if field_ok and self.residual:
            n_field_tried = len(self.residual)
            interp = self.fields(partial=True)
            if isinstance(interp, dict) and interp.get("partial"):
                field_part = interp["dirs"]
                self.drop_spanned(np.concatenate([an.scal_tangents, interp["vectors"]], axis=1))
                interp, n_field_tried = None, 0
        fields_closed = interp is not None
        ms_early = None
        if run.ss_constraint and self.residual and not fields_closed and not state.env("NOMSFIT"):
            ms_early = self.gauge()
            if len(ms_early["anchors"]) != len(self.residual) or any(f is None for f in ms_early["residue_fns"]):
                ms_early = None
        if not self.residual or fields_closed:
            self.rel_probe = [None] * an.n_aug
        elif ms_early is not None:
            def ms_res(rp, p, zvals=None):
                v = [f(rp, p) for f in ms_early["residue_fns"]]
                return None if any(x is None for x in v) else np.concatenate(v)
            self.rel_probe = self.probe_groups(an.kbatch, self.sc["pivots"], self.probe_next, ms_res,
                                               self.sc["ref"])
        else:
            self.rel_probe = self.full_probe()
        self.tlog("relevance probe done" + (" (minimal-support gauge)" if ms_early is not None else ""))
        peeled = self.peeled() if self.residual and not fields_closed else []
        self.tlog(f"peel done: {len(peeled)} peeled, {len(self.residual)} remaining")
        ms = self.gauge() if self.residual else {"anchors": [], "residue_fns": []}
        if interp is None and self.residual and len(self.residual) < n_field_tried:
            interp = self.fields()
            if isinstance(interp, dict):
                interp = None
        if interp is None:
            interp = self.ms_try(ms, True) if self.residual else []
        self.tlog("ms fast_only: no close, going dense" if interp is None else "ms fast_only: all closed")
        if interp is None and run.ss_constraint and not state.env("NOMSFIT"):
            interp = self.ms_try(ms, False)
            self.tlog("ms fit: " + ("no close" if interp is None else "all closed"))
            if interp is None and ms_early is not None:
                self.probe_next = self.pool_next
                self.pool_next = self.probe_next + an.n_aug * (self.ctrl["probe_retries"] + 16)
                self.rel_probe = self.full_probe()
                self.tlog("relevance probe done (free-column gauge)")
        if interp is None:
            interp = self.homotopy(self.dense(ms))
        self.tlog(f"reconstruction done: {sum(bool(e.get('closed_form')) for e in interp)}/{len(interp)} closed")
        return (an.scaling + narrowed + peeled + static_part + list(field_part) + fwd_first
                + list(interp))
