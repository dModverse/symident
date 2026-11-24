"""A residual direction fitted from the kernel restricted to the support columns of the direction."""

import re

import numpy as np

from ... import _core, session, state
from ...errors import SYMPY_FAILS, swallow, sympy_fail
from ...gfp import PRIMES, invmod, mulmod
from ...rexpr import sym_sort
from ...sym.shared import kernel_tapes
from ..groups import shift_groups, translation_groups
from ..interp import exp_backsub, interpolate_direction
from ..perprime import _lift_const
from ..relevance import direction_relevance
from ..verify import verify_in_nullspace
from .context import ints, ok, strs

_CSUFFIX = re.compile(r"\|c[0-9]+$")


class Narrow:
    """Kernels on a column subset and the fit or jet of one direction there."""

    def __init__(self, rec, bf, anchor):
        an = rec.an
        self.rec, self.an = rec, an
        self.bf = np.asarray(bf, dtype=np.int64)
        self.anchor = anchor
        self.cols = [int(c) for c in np.nonzero(bf)[0]]
        self.m = len(self.cols)
        self.zs = an.z_slots[self.cols]
        self.nt = an.sc["nt_used"]
        self.conds = None
        self.fpos = None
        self.sub = None
        self.wide = False

        def res(rp, p, zvals=None):
            return self.residue(rp, p)

        res.narrow = True
        self.res = res

    def kb(self, point_list, prime_vec, Nt):
        """The kernel on the support columns over many points."""
        an, run = self.an, self.an.run
        if not point_list:
            return []
        Nt = min(Nt, self.nt)
        if an.joint:
            per = 2 * max(1, run.cores)
            out = []
            for s in range(0, len(point_list), per):
                out.extend(an.sys.chunk(point_list[s : s + per], prime_vec[s : s + per], Nt, self.cols, self.conds))
            return [run.modulo_nuisance(o, q, self.cols) for o, q in zip(out, prime_vec)] if run.nuis_cols else out
        M = np.vstack([np.asarray(pp[: run.n_leaves], dtype=np.int64) for pp in point_list])
        nc = self.cols if run.nuis_cols else None
        if run.has_gaps:
            os_ = run.stacked_chain_batch(
                point_list, prime_vec, run.exact_nt(Nt, run.mtot), int(run.mtot), list(run.nt_cap), self.zs
            )
            if os_ is not None:
                return [run.modulo_nuisance(o, q, nc) for o, q in zip(os_, prime_vec)] if nc else os_
            os_ = _core.sym_obs_null_chain_point_batch(
                [run.chain_tapes(idx) for idx in run.chain_groups],
                run.n_leaves,
                run.n_states,
                self.zs,
                M,
                list(prime_vec),
                run.exact_nt(Nt, run.mtot),
                int(run.mtot),
                run.cores,
                list(run.nt_cap),
            )
            return [run.modulo_nuisance(o, q, nc) for o, q in zip(os_, prime_vec)] if nc else os_
        tapes = run.kernel_tapes() if self.sub is None else self.sub[1]
        os_ = _core.sym_obs_null_batch(
            tapes, run.n_leaves, run.n_states, self.zs, M, list(prime_vec), run.exact_nt(Nt), run.cores
        )
        if not an.transient:
            return [run.modulo_nuisance(o, q, nc) for o, q in zip(os_, prime_vec)] if nc else os_
        out = []
        for i, o in enumerate(os_):
            p = prime_vec[i]
            if not ok(o):
                out.append(o)
                continue
            rel_m = an.sys.relation_rows(point_list[i], p)
            if rel_m is None:
                out.append({"ok": False})
                continue
            R0 = np.asarray(o["R"], dtype=np.int64).reshape(int(o["rank"]), self.m)
            rr = _core.sym_rref_mod(np.vstack([R0, rel_m[:, self.cols]]).astype(np.float64), p)
            out.append(
                {
                    "ok": True,
                    "R": np.asarray(rr["R"], dtype=np.int64).reshape(int(rr["rank"]), self.m),
                    "pivots": np.atleast_1d(rr["piv"]).astype(np.int64),
                    "rank": int(rr["rank"]),
                    "dim": self.m,
                }
            )
        return [run.modulo_nuisance(o, q, nc) for o, q in zip(out, prime_vec)] if nc else out

    def _cover(self):
        """The conditions whose rows on the support reach the rank of all of them at the base point (sym_obs_cover):
        their kernel on the support is the kernel of all conditions there, so the fits read them alone; the
        closed form is checked on every condition afterwards."""
        an, run = self.an, self.an.run
        if an.joint or run.has_gaps or an.transient or run.nuis_cols or len(run.tapes) < 16:
            return
        r = _core.sym_obs_cover(
            run.kernel_tapes(),
            run.n_leaves,
            run.n_states,
            self.zs,
            np.asarray(an.sc["point0"][: run.n_leaves], dtype=np.int64),
            PRIMES[0],
            run.exact_nt(self.nt),
            self.m - 1,
            run.cores,
        )
        if not ok(r) or int(r["rank"]) != self.m - 1:
            return
        idx = ints(r["conditions"])
        self.sub = (idx, kernel_tapes([run.tapes[i] for i in idx]))

    def lanes(self):
        """Analysis.lanes on the conditions the kernels of the support read, at the Lie order of the support."""
        f = self.an.lanes()
        if f is None or self.an.joint:
            return None
        tapes = self.an.run.kernel_tapes() if self.sub is None else self.sub[1]
        return lambda points, primes, Nt, V: f(points, primes, min(Nt, self.nt), V, tapes)

    def kc(self, point, p, Nt):
        """The kernel on the column subset at `point` mod `p`, Lie order `Nt`."""
        return self.kb([point], [p], Nt)[0]

    def residue(self, rp, p):
        """The one-dimensional kernel on the support, scaled to one at the anchor."""
        m = self.m
        if rp is None or not ok(rp) or rp["rank"] != m - 1:
            return None
        piv = ints(rp["pivots"])
        pset = set(piv)
        free = [j for j in range(m) if j not in pset]
        R = np.asarray(rp["R"], dtype=np.int64).reshape(len(piv), m)
        v = np.zeros(m, dtype=np.int64)
        v[free] = 1
        for ri, pc in enumerate(piv):
            v[pc] = (p - int(R[ri, free[0]])) % p if len(free) == 1 else (p - int(np.sum(R[ri, free]))) % p
        if v[self.fpos] == 0:
            return None
        out = np.zeros(self.an.nz, dtype=np.int64)
        out[self.cols] = mulmod(v, invmod(int(v[self.fpos]), p), p)
        return out

    def _touched(self):
        """The conditions with a row on the support; False when there is none."""
        an, rec = self.an, self.rec
        try:
            tc = an.sys.chunk([an.sc["point0"]], [PRIMES[0]], self.nt, cols=self.cols, touch=True)
        except Exception as e:
            swallow(e, "narrow support check")
            tc = [{"ok": False, "why": str(e)}]
        if not ok(tc[0]):
            rec.tlog(f"narrow: no rows on the support at the base point {tc[0].get('why')} {an.run.ss_why}")
            return False
        self.conds = [i for i, t in enumerate(tc[0]["touch"]) if t]
        if not self.conds:
            rec.tlog("narrow: no condition has a row on the support")
            return False
        return True

    def _jet(self, limit):
        """A closed form from Taylor jets along the support; None if not found."""
        an, rec, run = self.an, self.rec, self.an.run
        if state.env("NOJET") or self.m > 6:
            return None
        from ...sym.jets import jet_generator

        chains = (
            run.chain_groups
            if run.has_gaps
            else [[i] for i in (range(len(an.multi["tapes"])) if self.sub is None else self.sub[0])]
        )
        conds = []
        for idx in chains:
            segs = []
            for ti in idx:
                t = an.multi["tapes"][ti]
                if t.get("jet_f") is None:
                    break
                segs.append(
                    {
                        "f": strs(t["jet_f"]),
                        "g": strs(t["jet_g"]),
                        "ic": strs(t.get("jet_ic")),
                        "ev": t.get("jet_ev") or [],
                        "time_fixed": t.get("jet_time_fixed") is not False,
                    }
                )
            conds.append({"segments": segs} if segs else None)
        if not conds or any(c is None for c in conds):
            return None
        reach = []
        for j in self.cols:
            r = []
            for t in an.multi["tapes"]:
                v = ints(t.get("lie_reach"))
                if len(v) > j and v[j] >= 0:
                    r.append(v[j])
            reach.append(min(r) if r else 0)
        max_ord = int(min(24, max(8, max(reach) + 2 * self.m)))
        try:
            jg = jet_generator(
                conds, [an.znames[c] for c in self.cols], an.znames[self.anchor], max_order=max_ord, work_limit=limit
            )
        except SYMPY_FAILS as err:
            sympy_fail(err)
            rec.tlog(f"narrow: jet error: {err}")
            jg = None
        except Exception as err:
            swallow(err, "jets")
            rec.tlog(f"narrow: jet error: {err}")
            jg = None
        rec.tlog(
            "narrow: jet closed form "
            + (
                f"found at order {int(jg['order'])}"
                if jg is not None and jg.get("ok")
                else f"not found ({'error' if jg is None else jg.get('reason')})"
            )
        )
        if jg is not None and jg.get("reason") == "work limit":
            session.note("limits", f"jets: stopped after {limit} characters of jets")
        if jg is None or not jg.get("ok"):
            return None
        return {
            "support": sym_sort(list(jg["vector"])),
            "vector": {k: str(x).replace("**", "^") for k, x in jg["vector"].items()},
            "type": "general",
            "closed_form": True,
            "route": f"jets (order {int(jg['order'])})",
        }

    def _entries_fit(self, ref, rel):
        """Whether, past the direction cap, every entry stays within the cap of the sparse fit, so that no
        translation group is needed."""
        if not self.wide:
            return False
        an = self.an
        r = direction_relevance(self.anchor, ref, ref["pivots"], an.z_slots, an.n_aug, an.sc["point0"], rel, self.res)
        aux = set(an.aux_leaves)
        return (
            max([0] + [len([x for x in e if x not in aux]) for e in r["rel_by_entry"]])
            <= an.ctrl["relevance_cap_sparse"]
        )

    def _fit(self, ref, base0, rel):
        an, rec = self.an, self.rec
        ctrl = dict(an.ctrl, relevance_cap_dir=an.n_aug) if self.wide else an.ctrl
        d = interpolate_direction(
            self.anchor,
            ref,
            ref["pivots"],
            an.znames,
            an.z_slots,
            an.leaf_names_aug,
            an.n_aug,
            base0,
            an.sc["pool"],
            rec.pool_next,
            an.sc["nt_used"],
            self.kc,
            rel,
            self.res,
            ctrl,
            self.kb,
            aux_leaves=an.aux_leaves,
        )
        rec.pool_next = d["pool_next"]
        r = dict(d["entry"])
        r.pop("relevant_leaves", None)
        if r.get("closed_form"):
            r["route"] = "narrow fit" + (f" ({r['fit_how']})" if r.get("fit_how") else "")
        r.pop("fit_how", None)
        return r

    def _read_off(self, want):
        """The direction where the entries are constants, or constants times the coordinate (a scaling on the
        support, the direction times the anchor coordinate): read at the base point, held at two fresh points and
        lifted from the primes; None otherwise."""
        an, sc, P = self.an, self.an.sc, PRIMES[0]
        zs = an.z_slots
        if not len(zs):
            return None
        a = self.anchor

        def logw(v, pt, p):
            z = np.asarray(pt, dtype=np.int64)[zs] % p
            if not np.all(z[self.cols]):
                return None
            return np.array([mulmod(v[c], mulmod(z[a], invmod(int(z[c]), p), p), p) for c in self.cols], dtype=np.int64)

        base0 = np.asarray(sc["point0"], dtype=np.int64)
        w0 = logw(want, base0, P)
        kinds = {"constant": want[self.cols] % P, "scaling": w0}
        for _ in range(2):
            pt = np.asarray(sc["pool"](self.rec.pool_next + np.arange(len(base0))), dtype=np.int64)
            self.rec.pool_next += len(base0)
            v = self.res(self.kc(pt, P, self.nt), P)
            if v is None:
                return None
            got = {"constant": v[self.cols] % P, "scaling": logw(v, pt, P)}
            kinds = {
                kd: x for kd, x in kinds.items() if x is not None and got[kd] is not None and np.array_equal(got[kd], x)
            }
            if not kinds:
                return None
        kd = "constant" if "constant" in kinds else "scaling"
        per = []
        for p in PRIMES:
            v = self.res(self.kc(base0, p, self.nt), p)
            if v is None:
                continue
            per.append((v[self.cols] % p if kd == "constant" else logw(v, base0, p), p))
        if not per or any(x is None for x, _ in per):
            return None
        vec = {}
        for j, c in enumerate(self.cols):
            w = _lift_const([int(x[j]) for x, _ in per], [p for _, p in per])
            if w is None:
                return None
            nm = an.znames[c]
            if w != "0":
                vec[nm] = w if kd == "constant" else (nm if w == "1" else f"({w})*{nm}")
        return {
            "support": sym_sort(list(vec)),
            "vector": vec,
            "type": "general",
            "closed_form": True,
            "route": f"narrow read-off ({kd})",
        }

    def _shorter_order(self, ref, want):
        """The lowest Lie order among the powers of two from 4 below the order of the kernel that still gives the
        direction."""
        k = 4
        while k < self.nt:
            rk = self.kc(self.an.sc["point0"], PRIMES[0], k)
            bk = self.res(rk, PRIMES[0])
            if bk is not None and np.array_equal(bk, want):
                self.nt = k
                return rk
            k *= 2
        return ref

    def run(self):
        """The direction as an entry, or None when the support does not contain it."""
        an, rec, sc, P = self.an, self.rec, self.an.sc, PRIMES[0]
        run, ctrl = an.run, an.ctrl
        if not self.m:
            return None
        if an.joint and any(_CSUFFIX.search(an.znames[c]) for c in self.cols):
            return None
        if self.m == 1:
            # a column without rows at the base point: the direction of the coordinate itself
            nm = an.znames[self.anchor]
            rec.tlog(f"narrow dir (anchor {nm}): a column without rows")
            return self._checked(
                {
                    "support": [nm],
                    "vector": {nm: "1"},
                    "type": "general",
                    "closed_form": True,
                    "route": "narrow (a column without rows)",
                }
            )
        self.fpos = self.cols.index(self.anchor)
        if an.joint and not self._touched():
            return None
        res = self.res
        self._cover()
        ref = self.kc(sc["point0"], P, sc["nt_used"])
        base = res(ref, P)
        want = mulmod(self.bf % P, invmod(int(self.bf[self.anchor]) % P, P), P)
        if base is None or not np.array_equal(base, want):
            rec.tlog(
                f"narrow (anchor {an.znames[self.anchor]}): the restricted rows do not give the "
                f"direction (rank {ref.get('rank') if ref else None} of {self.m})"
            )
            return None
        if not run.has_gaps:
            ref = self._shorter_order(ref, want)
        rec.tlog(
            f"narrow dir (anchor {an.znames[self.anchor]}): support {self.m} of {an.nz} columns, "
            f"Lie order {self.nt}" + (f", {len(self.conds)} of {an.sys.kc} conditions" if an.joint else "")
        )
        # a support past the jets is fitted only within the relevance cap, so the probe stops past it; where the
        # probe runs along the direction (Narrow.lanes) a sample costs one support kernel, and each entry is
        # bounded by the caps of an entry alone
        lanes = self.lanes()
        cap = ctrl["relevance_cap_dir"] if self.m > 6 or state.env("NOJET") else None
        if cap is not None:
            e = self._read_off(want)
            if e is not None:
                return self._checked(e)
            if lanes is not None:
                cap, self.wide = an.n_aug, True
        rel = rec.probe_groups(
            self.kb, ref["pivots"], rec.pool_next, res, ref, stop_above=cap, vecs=want.reshape(-1, 1), lanes=lanes
        )
        rec.pool_next += an.n_aug * (ctrl["probe_retries"] + 16)
        if rel is None:
            return {
                "support": sym_sort([an.znames[c] for c in self.cols]),
                "type": "general",
                "closed_form": False,
                "reason": f"direction couples more than {cap} parameters, above relevance_cap_dir={cap}; "
                f"raise the relevant cap",
            }
        groups = (
            []
            if state.env("NOGROUP") or self._entries_fit(ref, rel)
            else translation_groups(rel, ref, res, self.kb, sc, P, run.n_leaves)
        )
        rec.pool_next += 4 * run.n_leaves
        base0 = np.array(sc["point0"], dtype=np.int64)
        mul = [v for g in groups if g.get("type") == "mul" for v in g["members"]]
        if mul:
            pt1 = base0.copy()
            pt1[mul] = 1
            ref1 = self.kc(pt1, P, sc["nt_used"])
            if res(ref1, P) is not None:
                base0, ref = pt1, ref1
                rel = rec.probe_groups(
                    self.kb,
                    ref["pivots"],
                    rec.pool_next,
                    res,
                    ref,
                    base0,
                    stop_above=cap,
                    vecs=res(ref, P).reshape(-1, 1),
                    lanes=lanes,
                )
                if rel is None:
                    return {
                        "support": sym_sort([an.znames[c] for c in self.cols]),
                        "type": "general",
                        "closed_form": False,
                        "reason": f"direction couples more than {cap} parameters, above relevance_cap_dir={cap}; "
                        f"raise the relevant cap",
                    }
                rec.pool_next += an.n_aug * (ctrl["probe_retries"] + 16)
            else:
                groups = [g for g in groups if g.get("type") != "mul"]
        for gr in groups:
            for v in gr["members"]:
                rel[v] = dict(rel[v], rp=ref)
        base_v = res(ref, P)
        n_rel = 0
        for li in range(an.n_aug):
            rp = rel[li]["rp"] if rel[li] is not None else None
            v = res(rp, P) if ok(rp) else None
            if v is not None and base_v is not None and np.any(v != base_v):
                n_rel += 1
        jet_work = int(state.env_value("JETWORK", "1000000"))
        jet_first = n_rel > ctrl["relevance_cap"]
        e = self._jet(jet_work) if jet_first else None
        if e is not None:
            groups = []
        else:
            e = self._fit(ref, base0, rel)
            if not e.get("closed_form") and not jet_first:
                ej = self._jet(jet_work)
                if ej is not None:
                    e, groups = ej, []
        if not e.get("closed_form"):
            return e
        if groups:
            e["vector"] = shift_groups(e["vector"], groups, an.leaf_names_aug, base0)
            if e["vector"] is None:
                return {
                    "support": e["support"],
                    "type": "general",
                    "closed_form": False,
                    "reason": "translation-group back-substitution failed",
                }
            e["route"] = f"{e['route']}, {len(groups)} exchange group" + ("s" if len(groups) > 1 else "")
        return self._checked(e)

    def _checked(self, e):
        """The closed form `e` checked at fresh points, or marked for the joint check of the narrow route."""
        an, rec, sc = self.an, self.rec, self.an.sc
        if rec.dir_batch is not None and not an.joint:
            e["pending_check"] = self.anchor
            return e
        if not verify_in_nullspace(
            e,
            self.anchor,
            an.znames,
            an.leaf_names_aug,
            sc["point0"],
            sc["nt_used"],
            an.kcall,
            rec.pool_next,
            an.nz,
            rec.dir_call,
            env_fn=an.sys.rest_env if an.joint else None,
        ):
            return {
                "support": e["support"],
                "type": "general",
                "closed_form": False,
                "reason": "a closed form was reconstructed but failed verification at a fresh prime",
            }
        return back_substituted(an, e)


def back_substituted(an, e):
    """The closed form `e` in the coordinates of the model, the checked form kept."""
    eb = an.multi.get("exp_back") or {}
    if eb.get("names"):
        e.setdefault("checked", dict(e["vector"]))
        e["vector"] = exp_backsub(e["vector"], eb, an.multi.get("exp_atoms"))
    return e
