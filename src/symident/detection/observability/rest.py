"""Resting states over GF(p): backward and forward solves, continuation and warm-up across worker processes."""

import copy
import os
import time

import numpy as np

from ... import session, state
from ...errors import SYMPY_FAILS, swallow, sympy_fail
from ...gfp import PRIMES, random_point
from ...par import process_map
from ...sym.continuation import continue_resting_state
from ...sym.msolve import MsolveMissingError, msolve_threads
from ...sym.restforward import solve_forward_modular
from ...sym.steady import resting_kind, solve_steady_state_modular
from ..forward import SolveRates


class RestingStates:
    """Resting-state solves of a joint system, mixed into JointSystem, whose constructor sets their attributes."""

    def zero_of(self, ci):
        """States condition ci forces to zero at rest."""
        if ci < len(self.run.cond_zero_states):
            z = self.run.cond_zero_states[ci] or []
            return [s for s in z if s in self.state_names and s not in self.held_state_params]
        return []

    def held_of(self, ci):
        """The held states of condition `ci`: the solve-held ones and those it forces to zero."""
        return list(dict.fromkeys(self.solve_held + self.zero_of(ci)))

    def joint_pv(self, point, p):
        """Parameter, log and held-pivot values of a point for the steady-state solve."""
        def val(nm):
            s = self.run.slot(nm)
            return 0 if s is None else int(point[s]) % p
        return {"param_vals": {nm: val(nm) for nm in self.solve_param_names},
                "l_vals": {nm: val(nm) for nm in self.l_names} if self.l_names else None,
                "held_vals": {nm: val(self.held_param_of[nm]) for nm in self.held_names}
                if self.held_names else None}

    def solve_key(self, p, pv, ci):
        """Cache key of the steady state solve of condition `ci` at the values `pv` mod `p`."""
        return (self.model_key[ci], p, tuple(pv["param_vals"][n] for n in self.model_params[ci]),
                tuple((pv["l_vals"] or {}).values()), tuple((pv["held_vals"] or {}).values()))

    def solve_raw(self, p, pv, ci):
        """The modular steady state of condition `ci` at the values `pv`, or None on failure."""
        run = self.run
        ev_c = run.t0events[ci] if ci < len(run.t0events) and run.t0events[ci] else None
        t0 = time.monotonic()
        try:
            out = solve_steady_state_modular(
                model=self.models[ci], state_names=list(self.real_state_names),
                param_names=list(self.solve_param_names), param_vals=pv["param_vals"], prime=p,
                forcings=self.held_of(ci) or None, t0events=ev_c, recast=self.recast or None,
                l_vals=pv["l_vals"], joint_mode=True,
                held_states=pv["held_vals"] if self.held_names else None)
        except MsolveMissingError:
            raise
        except SYMPY_FAILS as exc:
            sympy_fail(exc)
            out = None
        except Exception as exc:
            swallow(exc, "resting-state solve")
            out = None
        self.solve_secs[0] += time.monotonic() - t0
        self.solve_secs[1] += 1
        if state.env("SOLVEDIAG"):
            print(f"[sym]   solve cond {ci + 1}, pid {os.getpid()}: {time.monotonic() - t0:.1f}s")
        return out

    _ROUTE = ("cont_off", "fwd_on", "fwd_rates", "fwd_banned", "fwd_sel")

    def route_state(self):
        """The settings that solves change and later solves read."""
        return {k: copy.copy(getattr(self, k)) for k in self._ROUTE}

    def set_route_state(self, st):
        """Restore a route_state()."""
        for k, v in st.items():
            setattr(self, k, copy.copy(v))

    def merge_route_states(self, base, states):
        """Route state after items that each started from `base`, merged in item order."""
        out = {k: copy.copy(v) for k, v in base.items()}
        for st in states:
            out["cont_off"] = [a or b for a, b in zip(out["cont_off"], st["cont_off"])]
            out["fwd_on"] = out["fwd_on"] and st["fwd_on"]
            out["fwd_banned"] = list(dict.fromkeys(out["fwd_banned"] + st["fwd_banned"]))
            if out["fwd_rates"] is None:
                out["fwd_rates"] = st["fwd_rates"]
            if out["fwd_sel"] == base["fwd_sel"]:
                out["fwd_sel"] = st["fwd_sel"]
        self.set_route_state(out)

    def rest_kind(self, p, pv, ci):
        """resting_kind() of condition `ci` at the values `pv`."""
        return resting_kind(self.models[ci], list(self.real_state_names), list(self.solve_param_names),
                            pv["param_vals"], p, forcings=self.held_of(ci) or None,
                            held_states=pv["held_vals"] if self.held_names else None)

    def warm_solves(self, pts, primes, conds=None):
        """The solve cache for a batch of points, in forked workers."""
        run = self.run
        if self.warm_off or run.cores <= 1:
            return
        # the mean solve time decides only where the solves run, not their values
        if self.solve_secs[1] >= 2 and self.solve_secs[0] < 0.02 * self.solve_secs[1]:
            return
        if conds is None:
            conds = self.equil_conds
        jobs, seen = [], set()
        for pt, p in zip(pts, primes):
            pv = self.joint_pv(pt, p)
            for ci in conds:
                if self.cont_active(ci):
                    continue
                key = self.solve_key(p, pv, ci)
                if key in self.solve_cache or key in seen:
                    continue
                seen.add(key)
                jobs.append({"key": key, "ci": ci, "p": p, "pv": pv})
        if len(jobs) < 2:
            return
        t0 = time.monotonic()
        res = process_map(lambda j: self.solve_raw(j["p"], j["pv"], j["ci"]), jobs,
                          min(run.cores, len(jobs)), on_error=lambda j: None)
        if run.timing:
            print(f"[sym] warm fill: {len(jobs)} solve(s), fork x{min(run.cores, len(jobs))}, "
                  f"{time.monotonic() - t0:.1f}s")
        for j, r in zip(jobs, res):
            if r is not None:
                self.solve_cache[j["key"]] = r if r.get("ok") else {"ok": False}

    def _seed(self, point, p, sol, ci, rates=None):
        """The point with the solved resting states (and rates) in the leaves."""
        run = self.run
        ptc = np.array(point[:run.n_leaves], dtype=np.int64)
        for r, v in (rates or {}).items():
            s = run.slot(r)
            if s is not None:
                ptc[s] = int(v) % p
        for nm in self.real_state_names:
            s = run.slot(nm)
            v = sol["val_by"].get(nm)
            if s is not None and v is not None:
                ptc[s] = int(v) % p
        for nm in self.zero_of(ci):
            s = run.slot(nm)
            if s is not None:
                ptc[s] = 0
        return {"sol": sol, "ptc": ptc}

    def solve_cond(self, point, p, ci):
        """Condition ci solved at the point (continued where that applies), cached."""
        if self.cont_active(ci) and (self.diag is None or self.base_key(point, p) in self.diag_ref):
            cr = self.cont_cond(point, p, ci)
            self.cont_trace(ci, cr is not None)
            if cr is not None:
                return cr
            self.cont_off[ci] = True
        return self.solve_direct(point, p, ci)

    def solve_direct(self, point, p, ci):
        """Condition ci solved at the point itself, cached."""
        pv = self.joint_pv(point, p)
        key = self.solve_key(p, pv, ci)
        sol = self.solve_cache.get(key)
        if sol is None:
            with msolve_threads(self.run.cores_call):
                sol = self.solve_raw(p, pv, ci)
            self.solve_cache[key] = {"ok": False} if sol is None or not sol.get("ok") else sol
        if sol is None or not sol.get("ok"):
            return None
        return self._seed(point, p, sol, ci)

    def solve_cond_fwd(self, point, p, ci, solve_rates):
        """States from the point and the turnover rates solved linearly (every prime)."""
        run = self.run

        def rd(nm):
            s = run.slot(nm)
            return 0 if s is None else int(point[s]) % p
        sv = {nm: rd(nm) for nm in self.real_state_names}
        pv = {nm: rd(nm) for nm in self.solve_param_names}
        key = (self.model_key[ci], p, tuple(sv.values()), tuple(pv.values()),
               solve_rates.key() if hasattr(solve_rates, "key") else tuple(solve_rates),
               tuple(self.held_of(ci)))
        sol = self.fwd_solve_cache.get(key)
        if sol is None:
            try:
                sol = solve_forward_modular(self.models[ci], list(self.real_state_names),
                                             list(self.solve_param_names), sv, pv, p,
                                             forcings=self.held_of(ci) or None,
                                             solve_rates=list(solve_rates),
                                             state_solve=list(getattr(solve_rates, "state_solve", [])),
                                             held=self.held_names or None)
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                sol = None
            except Exception as exc:
                swallow(exc, "forward seed")
                sol = None
            if len(self.fwd_solve_cache) > 2000:
                self.fwd_solve_cache.clear()
            self.fwd_solve_cache[key] = {"ok": False} if sol is None else sol
        if sol is None or not sol.get("ok"):
            return None
        return self._seed(point, p, sol, ci, rates=sol.get("rates"))

    def base_key(self, point, p):
        """Key of the diagonal base of a point: its values off Z."""
        jp = self.joint_pv(point, p)
        z = set(self.diag.Z)
        return (p, tuple(v for n, v in jp["param_vals"].items() if n not in z),
                tuple((jp["held_vals"] or {}).values()))

    def fwd_seed(self, point, p):
        """The first condition solved forward, at Z = 0 under the diagonal start; rates written into the point."""
        run = self.run
        diag = self.diag
        sv = {nm: (0 if run.slot(nm) is None else int(point[run.slot(nm)]) % p)
              for nm in self.real_state_names}
        pv = self.joint_pv(point, p)["param_vals"]
        forc = list(self.held_of(self.fwd_cond))
        keep = list(self.fwd_banned)
        if diag is not None:
            pv = dict(pv, **{z: 0 for z in diag.Z})
            sv.update({s: 0 for s in diag.S})
            forc = list(dict.fromkeys(forc + diag.S))
            keep = list(dict.fromkeys(diag.Z + diag.dead + keep))

        def solve():
            fr = self.fwd_rates
            try:
                return solve_forward_modular(
                    self.models[self.fwd_cond], list(self.real_state_names),
                    list(self.solve_param_names), sv, pv, p, forcings=forc or None,
                    solve_rates=list(fr) if fr is not None else None,
                    state_solve=list(fr.state_solve) if fr is not None else None,
                    keep_free=keep or None, held=self.held_names or None)
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                return None
            except Exception as exc:
                swallow(exc, "forward seed")
                return None
        sol = solve()
        if self.fwd_rates is None:
            for _ in range(5):
                if sol is None or not sol.get("ok"):
                    break
                z = [k for k, v in sol["rates"].items() if int(v) % p == 0]
                if not z:
                    break
                self.fwd_banned = list(dict.fromkeys(self.fwd_banned + z))
                keep = list(dict.fromkeys(keep + z))
                sol = solve()
        if sol is None or not sol.get("ok"):
            if state.env("FWDDIAG"):
                print("[fwddiag] forward seed failed:", "error" if sol is None else sol.get("why"))
            if self.fwd_rates is None:
                self.fwd_on = False
                if run.timing:
                    print("[sym] forward seed off:", "solve error" if sol is None else sol.get("why"))
            return None
        if self.fwd_rates is None:
            fr = SolveRates(sol["solve_rates"], sol.get("state_solve"))
            self.fwd_rates = fr
            if any(run.slot(r) is None for r in fr):
                self.fwd_on = False
                return None
        rv = [int(sol["rates"][r]) % p for r in self.fwd_rates]
        if any(v == 0 for v in rv):
            return None
        point = np.array(point, dtype=np.int64)
        for r, v in zip(self.fwd_rates, rv):
            point[run.slot(r)] = v
        if diag is None:
            self.solve_cache[self.solve_key(p, self.joint_pv(point, p), self.fwd_cond)] = sol
            return point
        x0 = {s: 0 for s in self.real_state_names}
        x0.update({str(k): int(v) % p for k, v in sol["val_by"].items()})
        x0.update({s: 0 for s in diag.S})
        if diag.pole({**self.joint_pv(point, p)["param_vals"], **{z: 0 for z in diag.Z}, **x0}, p):
            return None
        self.diag_ref[self.base_key(point, p)] = x0
        return point

    def cont_eligible(self, ci):
        """Whether condition `ci` may be continued from the forward condition."""
        if getattr(self, "exp_rel_fn", None) is not None:
            return False
        if self.diag is not None:
            return self.fwd_on and ci in self.equil_conds
        return (self.fwd_on and self.run.has_gaps and ci != self.fwd_cond
                and self.model_key[ci] != self.model_key[self.fwd_cond]
                and not self.held_names)

    def cont_active(self, ci):
        """Whether condition `ci` is eligible for continuation and its continuation is not switched off."""
        return self.cont_eligible(ci) and not self.cont_off[ci]

    def cont_trace(self, ci, success):
        """Print once per condition whether its continuation succeeded, when timing is on."""
        if self.cont_seen[ci] or not self.run.timing:
            return
        self.cont_seen[ci] = True
        print(f"[sym] condition {ci + 1}: " +
              ("continued" if success else "continuation failed, solved directly"))

    def cont_params(self, ci):
        """The parameters condition `ci` is continued along."""
        if self.diag is not None:
            return [z for z in self.diag.Z if z in set(self.model_params[ci])]
        base = set(self.model_params[self.fwd_cond])
        return [n for n in self.model_params[ci] if n not in base]

    def cont_cond(self, point, p, ci):
        """Condition ci continued along its own parameters from the first condition."""
        run = self.run
        pv = self.joint_pv(point, p)
        key = ("cont", self.solve_key(p, pv, ci), self.cont_t)
        sol = self.solve_cache.get(key)
        if sol is None:
            if self.diag is not None:
                rest = self.diag_ref.get(self.base_key(point, p))
            else:
                ref = self.solve_cond(point, p, self.fwd_cond)
                rest = None if ref is None else ref["sol"]["val_by"]
            if rest is None:
                return None
            priv = self.cont_params(ci)
            pv0 = dict(pv["param_vals"])
            for n in priv:
                pv0[n] = 0
            try:
                sol = continue_resting_state(self.models[ci], list(self.real_state_names),
                                              list(self.solve_param_names), pv0,
                                              {n: pv["param_vals"][n] for n in priv},
                                              rest, p, self.cont_t,
                                              forcings=self.held_of(ci) or None,
                                              held={n: 0 for n in self.held_names} or None)
            except SYMPY_FAILS as exc:
                sympy_fail(exc)
                sol = None
            except Exception as exc:
                swallow(exc, "continuation")
                sol = None
            success = sol is not None and sol.get("ok") and self.in_torus(sol, ci)
            self.solve_cache[key] = sol if success else {"ok": False}
            if not success:
                return None
        elif not sol.get("ok"):
            return None
        ptc = np.array(point[:run.n_leaves], dtype=np.int64)
        ser = np.zeros((run.n_leaves, self.cont_t), dtype=np.int64)
        for nm, vals in sol["val_by"].items():
            sl = run.slot(nm)
            if sl is None:
                continue
            v = [int(x) % p for x in vals]
            ptc[sl] = v[0]
            ser[sl, :] = v[1:1 + self.cont_t]
        for nm in self.cont_params(ci):
            sl = run.slot(nm)
            if sl is None:
                continue
            ser[sl, 0] = ptc[sl] % p
            ptc[sl] = 0
        for nm in self.zero_of(ci):
            sl = run.slot(nm)
            if sl is not None:
                ptc[sl] = 0
        return {"sol": sol, "ptc": ptc, "ser": ser, "cont": True}

    def in_torus(self, sol, ci):
        """Whether a continued rest leaves the zero set of the diagonal start."""
        if self.diag is None:
            return True
        return all(any(int(v) for v in sol["val_by"][s])
                   for s in self.diag.S if s in sol["val_by"] and s not in self.held_of(ci))

    @staticmethod
    def sol_at(sol, k):
        """The df payload of a continued solve at eps^k."""
        return {"df_jx": [[int(x[k]) for x in r] for r in sol["df_jx"]],
                "df_jt": {th: [int(x[k]) for x in col] for th, col in sol["df_jt"].items()},
                "df_state_cols": sol["df_state_cols"], "df_param_cols": sol["df_param_cols"]}

    def sat_prep(self, point, P1=None, rounds=8):
        """A point with every resting state solved at P1; failed conditions redraw their private parameters."""
        run = self.run
        P1 = PRIMES[0] if P1 is None else P1
        point = np.array(point, dtype=np.int64)
        for rnd in range(rounds + 1):
            if self.fwd_on:
                fp = self.fwd_seed(point, P1)
                if fp is not None:
                    point = fp
            failed = [self.equil_conds[0]] if self.solve_cond(point, P1, self.equil_conds[0]) is None else []
            if not failed:
                self.warm_solves([point], [P1])
                failed = [ci for ci in self.equil_conds[1:] if self.solve_cond(point, P1, ci) is None]
                if not failed:
                    return point
            run.ss_why = ("joint solve failed for condition " +
                          ", ".join(str(c + 1) for c in dict.fromkeys(failed)))
            redraw = list(dict.fromkeys(n for ci in failed for n in self.private_of[ci]))
            if any(len(self.private_of[ci]) == 0 for ci in failed):
                return None
            vals = random_point(len(redraw), seed=session.current().seed("redraw", point.tobytes(), rnd))
            for k, nm in enumerate(redraw):
                point[run.slot(nm)] = vals[k]
            if run.timing:
                print(f"[sym] redraw {','.join(redraw)} for condition(s) "
                      f"{','.join(str(c + 1) for c in dict.fromkeys(failed))}")
        return None
