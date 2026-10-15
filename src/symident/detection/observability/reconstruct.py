"""Closed forms for the residual kernel directions: narrow fits, static subsystems, the
forward route, invariant fields, gauged and dense interpolation, each verified at a
fresh prime before it is adopted."""

import re
import time

import numpy as np

from ... import _core, state
from ...errors import SYMPY_FAILS, RouteError, swallow, sympy_fail
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
from ..closedform import expired, recast_backsub, simplify
from ..fields import field_directions, field_setup
from ..forward import fwd_adopt, perprime_forward_multi, perprime_forward_shared
from ..interp import entry_degrees, exp_backsub, interpolate_direction
from ..perprime import interpolate_perprime
from ..relevance import build_shared_bank, dense_need, direction_relevance
from ..verify import verify_direction, verify_in_nullspace, verify_perprime
from .context import ints, ok, strs
from .homotopy import Homotopy
from .narrow import _CSUFFIX, Narrow


def _actionable(reason):
    """Whether a reason names a cap or setting to change."""
    return reason is not None and any(w in str(reason) for w in ("raise", "cap", "timeout", "time limit"))


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
        # one deadline per detection: reruns and nested analyses share it
        ses = state.get("session")
        dl = self.t0 + to if to is not None and np.isfinite(to) else None
        if ses is not None and dl is not None:
            ses.deadline = dl if ses.deadline is None else ses.deadline
            dl = ses.deadline
        self.ctrl["deadline"] = dl
        self.probe_next = self.sc["pool_next"]
        self.pool_next = self.probe_next + an.n_aug * (self.ctrl["probe_retries"] + 16)
        self.zvals0 = np.asarray(self.sc["point0"])[an.z_slots] if len(an.z_slots) else None
        self._rest_kind = None
        self._gauge = None
        self.phys_cols = [i for i, z in enumerate(an.znames) if z not in an.zsn] if an.joint else None
        # (route, reason, support) of every route that left a direction open
        self.attempts = []
        # kernel vectors of the directions closed so far, beyond the scalings
        self.closed = np.zeros((an.nz, 0), dtype=np.int64)

    def tlog(self, msg):
        """Print `msg` with the elapsed time when timing is on."""
        if self.an.run.timing:
            print(f"[sym {time.monotonic() - self.t0:6.1f}s] {msg}", flush=True)

    def attempt(self, route, entry=None, reason=None, support=None):
        """Record why `route` left a direction open: from an open `entry` or a `reason`."""
        if entry is not None:
            if entry.get("closed_form"):
                return
            reason, support = entry.get("reason"), entry.get("support")
        self.attempts.append((route, str(reason), None if support is None else list(support)))
        self.tlog(f"{route}: {reason}")

    def with_attempts(self, entries):
        """The open entries with the attempts on their support; a reason that names no cap is
        replaced by the last attempt that does."""
        out = []
        for e in entries:
            if e.get("closed_form") or not self.attempts:
                out.append(e)
                continue
            supp = set(e.get("support") or [])
            att = list(dict.fromkeys((r, why) for r, why, s in self.attempts
                                     if s is None or not supp or supp & set(s)))
            e = dict(e, attempts=att)
            if not _actionable(e.get("reason")):
                better = [why for _, why in att if _actionable(why)]
                if better:
                    e["reason"] = better[-1]
            out.append(e)
        return out

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
        return (
            an.can_batch or (run.has_gaps and not run.ss_constraint and not an.joint and not an.transient)
            or (an.transient and not run.has_gaps and not run.ss_constraint and not an.recast)
            or an.chunked)

    def narrowed(self):
        """Directions closed on their own support, minimal-support candidates first."""
        an, nz, P = self.an, self.an.nz, PRIMES[0]
        try:
            ms0 = self.gauge()
        except Exception as exc:
            swallow(exc, "minimal-support gauge")
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
            except Exception as exc:
                swallow(exc, "narrow route")
                if state.env("ROBUSTDIAG"):
                    import traceback
                    traceback.print_exc()
                e = None
            if e is None:
                self.attempt("narrow", reason="no result (see info['errors'])", support=[an.znames[anchor]])
            else:
                self.attempt("narrow", e)
            if e is not None and e.get("closed_form"):
                out.append(e)
                span = np.concatenate([span, np.asarray(v, dtype=np.int64).reshape(nz, 1)], axis=1)
        self.tlog(f"narrow: {len(out)} of {len(self.residual)} closed")
        self.closed = np.concatenate([self.closed, span[:, an.scal_rows.shape[0]:]], axis=1)
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
            self.closed = np.concatenate([self.closed, vw.reshape(nz, 1)], axis=1)
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
                e.setdefault("checked", dict(e["vector"]))
                e["vector"] = recast_backsub(e["vector"], self.an.recast)
            e["route"] = "forward fit per prime"
            out.append(e)
        return out

    def _forward_checked(self, entries):
        """Forward forms completed by their solved rates; one zero or in the span of the directions found stays open."""
        an, P = self.an, PRIMES[0]
        env = an._kernel_point()
        phys = [i for i, z in enumerate(an.znames) if not _CSUFFIX.search(z)]
        span = np.concatenate([an.scal_tangents, self.closed], axis=1)[phys, :] % P
        kept, n_open = [], 0
        for e in an._complete_forward(entries):
            v = an._phys_vector(e.get("checked") or e["vector"], env) if e.get("closed_form") else None
            if v is not None and (not np.any(v) or in_span(span, v, len(phys), P)):
                n_open += 1
                self.attempt("forward", reason="the forward form lies in the span of the directions found",
                             support=e.get("support"))
                continue
            if v is not None:
                span = np.concatenate([span, v.reshape(-1, 1)], axis=1)
            kept.append(e)
        # the open ones take residual columns outside the span of the kept forms
        free = []
        for fc in self.residual:
            r = self.residues(fc)[phys] % P
            if len(free) < n_open and not in_span(span, r, len(phys), P):
                free.append(fc)
                span = np.concatenate([span, r.reshape(-1, 1)], axis=1)
        if len(free) < n_open:
            free = self.residual[len(self.residual) - n_open:]
        out = []
        for fc in self.residual:
            if fc in free:
                res = self.residues(fc)
                out.append({"support": sym_sort([an.znames[i] for i in phys if res[i] % P]),
                            "type": "general", "closed_form": False,
                            "reason": "the forward form lies in the span of the directions found"})
            else:
                out.append(kept.pop(0))
        return out

    def _forward_multi(self):
        an, sys = self.an, self.an.sys
        try:
            return perprime_forward_multi(
                self.residual, self.sc, an.kcall, sys.kcall_fwd, an.znames, an.z_slots, an.leaf_names_aug,
                an.nz, an.scaling, strs(an.multi.get("z_state_names")), strs(an.multi.get("param_names")),
                an.recast, True, self.ctrl, self.phys_cols, sys.models, sys.real_state_names,
                sys.solve_param_names, sys.held_of(0))
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            return None
        except Exception as exc:
            swallow(exc, "forward route")
            return None

    def fork_map(self, f, xs):
        """f over xs in forked workers, one kernel thread each (OpenMP does not survive a fork)."""
        run, sys = self.an.run, self.an.sys
        xs = list(xs)
        saved = run.cores_call, sys.warm_off
        run.cores_call, sys.warm_off = (run.cores if len(xs) == 1 else 1), True
        try:
            return run.par_map(xs, f, sys)
        finally:
            run.cores_call, sys.warm_off = saved

    def rest_kind(self):
        """How the resting state depends on the parameters (sym.steady.resting_kind), read
        from the structure of the first resting system."""
        if self._rest_kind is None:
            an, sys = self.an, self.an.sys
            pv = sys.joint_pv(random_point(an.run.n_leaves, 104729), PRIMES[0])
            self._rest_kind = sys.rest_kind(PRIMES[0], pv, sys.equil_conds[0])
        return self._rest_kind

    def forward_first(self, stacked_ss, late=False):
        """Every residual direction from the forward route, ahead of the backward routes when
        the resting state needs msolve, after the fast ones when a resultant gives it."""
        an, sys = self.an, self.an.sys
        if not (an.run.ss_constraint and not stacked_ss and self.residual and an.joint
                and len({sys.model_key[c] for c in sys.equil_conds}) == 1):
            return []
        kind = self.rest_kind()
        if kind != ("core" if late else "msolve"):
            return []
        try:
            fwd = perprime_forward_shared(
                self.sc, an.kcall, sys.kcall_fwd, an.znames, an.z_slots, an.leaf_names_aug, an.nz,
                an.scaling, strs(an.multi.get("z_state_names")), an.recast, True, self.ctrl, self.phys_cols,
                sys.models, sys.real_state_names, sys.solve_param_names, sys.held_of(0),
                kcall_fwd_sh=sys.kcall_fwd_shared,
                select_fn=sys.fwd_select,
                map_fn=self.fork_map)
        except SYMPY_FAILS as err:
            sympy_fail(err)
            if state.env("ROBUSTDIAG"):
                print(f"[fwds] error: {err}")
            fwd = None
        except Exception as err:
            swallow(err, "forward route")
            if state.env("ROBUSTDIAG"):
                print(f"[fwds] error: {err}")
            fwd = None
        if fwd is None:
            fwd = self._forward_multi()
        closed = fwd_adopt(fwd or [], len(self.residual))
        self.tlog(f"forward {'after the fast routes' if late else 'first'} (resting state: {kind}): " +
                  ("not adopted" if closed is None else "every direction closed"))
        if closed is None:
            for d in fwd or []:
                self.attempt("forward", d)
            if not fwd:
                self.attempt("forward", reason="no result")
            return []
        self.residual = []
        return self._forward_entries(closed)

    def _moving_times(self):
        """Whether an event time of some segment holds a coordinate."""
        names = set(self.an.znames) | set(self.an.run.leaf_names)
        for t in self.an.multi["tapes"]:
            tm = str(t.get("ver_time") or "")
            if tm and set(re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", tm)) & names:
                return True
        return False

    def homotopy(self, interp):
        """`interp` with its open directions closed along parameter lines through the base point."""
        an, run = self.an, self.an.run
        open_ = [i for i, e in enumerate(interp) if not e.get("closed_form")]
        why = ("switched off (reconst_control(homotopy=False))" if not self.ctrl.get("homotopy", True) else
               "needs the joint resting state" if run.ss_constraint and not an.joint else
               "not run where the kernel depends on the time between events"
               if run.has_gaps and (self.sc.get("ref") or {}).get("at_one") else
               "not run where an event time depends on the parameters" if run.has_gaps and self._moving_times() else
               "not run in transient recast coordinates" if getattr(an, "transient", False) else
               "timeout" if expired(self.ctrl) else None)
        if open_ and why is not None:
            self.attempt("homotopy", reason=why)
        if not (open_ and why is None and len(interp) == len(self.residual)):
            return interp
        try:
            hr = Homotopy(self, self.sc["point0"], self.sc["nt_used"]).run([self.residual[i] for i in open_])
        except Exception as e:
            swallow(e, "homotopy")
            self.tlog(f"homotopy: {e}")
            hr = None
        found = {} if hr is None else {fc: d for fc, d in zip(*hr) if d is not None}
        for d in found.values():
            if an.recast:
                d["checked"] = dict(d["vector"])
                d["vector"] = recast_backsub(d["vector"], an.recast)
        for i in open_:
            if self.residual[i] not in found:
                self.attempt("homotopy", reason="no closed form along parameter lines",
                             support=interp[i].get("support"))
        out = [found.get(self.residual[i], e) if i in open_ else e for i, e in enumerate(interp)]
        self.tlog(f"homotopy: {len(found)} of {len(open_)} open direction(s) closed")
        return out

    def fields(self, partial=False):
        """Directions from invariant fields of the unspecialised system."""
        an = self.an
        out = None
        try:
            fs = field_setup(an.multi, an.field_chains, an.run.cores, fixed=an.field_fixed(an.N))
        except Exception as exc:
            swallow(exc, "field setup")
            fs = None
        an.field_setup_done = fs
        if fs is not None:
            try:
                # a field is taken only outside the scalings and the directions closed before
                out = field_directions(fs, an.N, len(self.residual),
                                       np.vstack([an.scal_rows, self.closed.T]) if self.closed.shape[1]
                                       else an.scal_rows, an.znames,
                                       an.run.leaf_names, an.point0_solved, an.field_chains,
                                       an.field_joint, partial=partial)
            except RouteError as err:
                self.attempt("invariant fields", reason=str(err))
                out = None
            except Exception as exc:
                swallow(exc, "field directions")
                out = None
        is_partial = isinstance(out, dict) and out.get("partial")
        self.tlog("global symmetries: " + ("no close" if out is None else
                  f"{len(out['dirs'] if is_partial else out)} direction(s) closed" +
                  (", the rest sampled" if is_partial else "")))
        return out

    # ---- interpolation ----------------------------------------------------------------

    def one(self, fc, rfn, log_coords=False, bank=None, fast_only=False, per_prime=False, degrees=None):
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
                                      an.kbatch, bank, fast_only, aux_leaves=an.aux_leaves, degrees=degrees)
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
            eb = an.multi.get("exp_back") or {}
            if an.recast or eb.get("names"):
                e.setdefault("checked", dict(e["vector"]))
            if an.recast:
                e["vector"] = recast_backsub(e["vector"], an.recast)
            if eb.get("names"):
                e["vector"] = exp_backsub(e["vector"], eb, an.multi.get("exp_atoms"))
        return e

    def gauge(self):
        """The residual directions of minimal support, computed once per residual."""
        an, key = self.an, tuple(self.residual)
        if self._gauge is None or self._gauge[0] != key:
            self._gauge = (key, minsupport_gauge(self.residual, an.scal_rows, PRIMES[0], an.nz, self.sc,
                                                 an.free_cols, cand_cap=self.ctrl["minsupport_cand_cap"]))
        return self._gauge[1]

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
            self.attempt("read-off in log coordinates", e)
            if e.get("closed_form"):
                out.append(e)
                vec = np.concatenate([vec, ms["vectors"][:, gi].reshape(nz, 1)], axis=1)
        if out:
            self.closed = np.concatenate([self.closed, vec], axis=1)
            self.drop_spanned(vec)
        return out

    def ms_try(self, ms, fast_only):
        """All directions in the minimal-support gauge, or None unless all close."""
        if not fast_only:
            self.tlog(f"ms gauge: {len(ms['anchors'])} anchor(s) for {len(self.residual)} residual "
                      f"direction(s), {sum(f is not None for f in ms['residue_fns'])} residue fn(s)")
        if len(ms["anchors"]) != len(self.residual) or all(f is None for f in ms["residue_fns"]):
            if not fast_only:
                n_ms = 0 if ms.get("vectors") is None else np.asarray(ms["vectors"]).shape[1]
                self.attempt("minimal-support fit", reason=f"{n_ms} direction(s) of minimal support found for "
                             f"{len(self.residual)} within minsupport_cand_cap; raise it")
            return None
        rec = [self.one(anchor, ms["residue_fns"][gi], log_coords=True, fast_only=fast_only)
               for gi, anchor in enumerate(ms["anchors"])]
        for e in rec:
            self.attempt("minimal-support " + ("read-off" if fast_only else "fit"), e)
        return rec if all(e.get("closed_form") for e in rec) else None

    def dense(self, ms):
        """Each direction from a shared sample bank, per prime where the bank is missing,
        then rescued in the canonical and log-coordinate gauges."""
        an, sc, run = self.an, self.sc, self.an.run
        metas = [direction_relevance(fc, sc["ref"], sc["pivots"], an.z_slots, an.n_aug, sc["point0"],
                                     self.rel_probe, None) for fc in self.residual]

        def zvals_of(pt):
            return np.asarray(pt)[an.z_slots] if len(an.z_slots) else None
        needs, degrees = [], []
        for fc, m in zip(self.residual, metas):
            degs = entry_degrees(fc, sc["pivots"], sc["point0"], lambda rp, p, z, fc=fc: null_residues(rp, fc, p),
                                 zvals_of, m, an.kbatch, sc["nt_used"], self.ctrl)
            degrees.append(degs)
            # a direction with an entry past degree_cap cannot use the bank
            needs.append(0 if "over" in degs else
                         dense_need(m["rel_by_entry"], self.ctrl,
                                    [d if isinstance(d, int) else None for d in degs])["max_need"])
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
            r = self.one(fc, None, per_prime=True) if bank_missing else \
                self.one(fc, None, bank=bank, degrees=degrees[ii])
            if not r.get("closed_form") and run.ss_constraint and not bank_missing:
                self.attempt("dense fit", r)
                r = self.one(fc, None, per_prime=True)
            self.tlog(f"dense dir {ii + 1}/{len(self.residual)} done (closed={bool(r.get('closed_form'))})")
            interp.append(r)
        if run.ss_constraint and an.joint and an.recast and not all(e.get("closed_form") for e in interp):
            closed = fwd_adopt(self._forward_multi() or [], len(self.residual))
            if closed is not None:
                interp = self._forward_checked(self._forward_entries(closed))
                self.tlog(f"forward-multi closed {sum(bool(e.get('closed_form')) for e in interp)} "
                          "residual direction(s)")
        if bank_missing:
            return interp

        def rescue(gauge, log_coords=False):
            if len(gauge["anchors"]) != len(self.residual):
                return
            for ii in [i for i, e in enumerate(interp) if not e.get("closed_form")]:
                if gauge["residue_fns"][ii] is None or expired(self.ctrl):
                    continue
                rec = self.one(gauge["anchors"][ii], gauge["residue_fns"][ii], log_coords=log_coords)
                self.attempt("log-coordinate gauge" if log_coords else "canonical gauge", rec)
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
        static_part = self.statics() if an.joint and self.residual else []
        stacked_ss = an.stacked_first and run.ss_constraint
        fwd_first = self.forward_first(stacked_ss)
        if stacked_ss and self.residual:
            return None
        # the invariant fields come from the regimes alone, which costs less than narrowing
        # each direction over every condition
        field_ok = run.ss_constraint and not an.recast and not state.get("inner") and not state.env("NOFIELDS")
        interp, field_part, n_field_tried = None, [], 0
        if field_ok and self.residual:
            n_field_tried = len(self.residual)
            interp = self.fields(partial=True)
            if isinstance(interp, dict) and interp.get("partial"):
                field_part = interp["dirs"]
                self.closed = np.concatenate([self.closed, interp["vectors"]], axis=1)
                self.drop_spanned(np.concatenate([an.scal_tangents, interp["vectors"]], axis=1))
                interp, n_field_tried = None, 0
        fields_closed = interp is not None
        narrowed = self.narrowed() if self.narrow_ok() and not fields_closed and self.residual else []
        ms_early = None
        if run.ss_constraint and self.residual and not fields_closed:
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
        if interp is None:
            fwd_late = self.forward_first(stacked_ss, late=True)
            if fwd_late:
                fwd_first, interp = fwd_first + fwd_late, []
        self.tlog("ms fast_only: no close, going dense" if interp is None else "ms fast_only: all closed")
        if interp is None and run.ss_constraint:
            interp = self.ms_try(ms, False)
            self.tlog("ms fit: " + ("no close" if interp is None else "all closed"))
            if interp is None and ms_early is not None:
                self.probe_next = self.pool_next
                self.pool_next = self.probe_next + an.n_aug * (self.ctrl["probe_retries"] + 16)
                self.rel_probe = self.full_probe()
                self.tlog("relevance probe done (free-column gauge)")
        if interp is None:
            interp = self.homotopy(self.dense(ms))
            # an open direction keeps its kernel vector, so that the kernel fill does not repeat it
            interp = [e if e.get("closed_form") else dict(e, kernel_residue=self.residues(fc).tolist())
                      for e, fc in zip(interp, self.residual)]
        self.tlog(f"reconstruction done: {sum(bool(e.get('closed_form')) for e in interp)}/{len(interp)} closed")
        return (an.scaling + narrowed + peeled + static_part + list(field_part) + fwd_first
                + self.with_attempts(list(interp)))
