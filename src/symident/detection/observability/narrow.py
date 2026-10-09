"""A residual direction fitted from the kernel restricted to its own support columns."""

import re

import numpy as np

from ... import _core, state
from ...gfp import PRIMES, invmod, mulmod
from ...rexpr import sym_sort
from ..groups import shift_groups, translation_groups
from ..interp import exp_backsub, interpolate_direction
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
            per = state.get("chunk_points", 2 * max(1, run.cores))
            out = []
            for s in range(0, len(point_list), per):
                out.extend(an.sys.chunk(point_list[s:s + per], prime_vec[s:s + per], Nt, self.cols, self.conds))
            return out
        M = np.vstack([np.asarray(pp[:run.n_leaves], dtype=np.int64) for pp in point_list])
        if run.has_gaps:
            return _core.sym_obs_null_chain_point_batch([run.chain_tapes(idx) for idx in run.chain_groups],
                                                        run.n_leaves, run.n_states, self.zs, M, list(prime_vec),
                                                        int(Nt), int(run.mtot), run.cores, list(run.nt_cap))
        os_ = _core.sym_obs_null_batch(run.tapes, run.n_leaves, run.n_states, self.zs, M, list(prime_vec),
                                       int(Nt), run.cores)
        if not an.transient:
            return os_
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
            out.append({"ok": True, "R": np.asarray(rr["R"], dtype=np.int64).reshape(int(rr["rank"]), self.m),
                        "pivots": np.atleast_1d(rr["piv"]).astype(np.int64),
                        "rank": int(rr["rank"]), "dim": self.m})
        return out

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
            v[pc] = (p - int(R[ri, free[0]])) % p if len(free) == 1 else \
                (p - int(np.sum(R[ri, free]))) % p
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
        chains = run.chain_groups if run.has_gaps else [[i] for i in range(len(an.multi["tapes"]))]
        conds = []
        for idx in chains:
            segs = []
            for ti in idx:
                t = an.multi["tapes"][ti]
                if t.get("jet_f") is None:
                    break
                segs.append({"f": strs(t["jet_f"]), "g": strs(t["jet_g"]), "ic": strs(t.get("jet_ic")),
                             "ev": t.get("jet_ev") or [], "time_fixed": t.get("jet_time_fixed") is not False})
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
            # a closed form much longer than this is no usable result and costs minutes to verify
            jg = jet_generator(conds, [an.znames[c] for c in self.cols], an.znames[self.anchor],
                                 max_order=max_ord, max_chars=20000, time_limit=limit)
        except Exception as err:
            rec.tlog(f"narrow: jet error: {err}")
            jg = None
        rec.tlog("narrow: jet closed form " + (f"found at order {int(jg['order'])}"
                                                if jg is not None and jg.get("ok") else
                                                f"not found ({'error' if jg is None else jg.get('reason')})"))
        if jg is None or not jg.get("ok"):
            return None
        return {"support": sym_sort(list(jg["vector"])),
                "vector": {k: str(x).replace("**", "^") for k, x in jg["vector"].items()},
                "type": "general", "closed_form": True, "route": f"jets (order {int(jg['order'])})"}

    def _fit(self, ref, base0, rel):
        an, rec = self.an, self.rec
        d = interpolate_direction(self.anchor, ref, ref["pivots"], an.znames, an.z_slots, an.leaf_names_aug,
                                  an.n_aug, base0, an.sc["pool"], rec.pool_next, an.sc["nt_used"], self.kc,
                                  rel, self.res, an.ctrl, self.kb, aux_leaves=an.aux_leaves)
        rec.pool_next = d["pool_next"]
        r = dict(d["entry"])
        r.pop("relevant_leaves", None)
        if r.get("closed_form"):
            r["route"] = "narrow fit" + (f" ({r['fit_how']})" if r.get("fit_how") else "")
        r.pop("fit_how", None)
        return r

    def _shorter_order(self, ref, want):
        """The lowest Lie order among 4, 8, 16, 32 that still gives the direction."""
        for k in (4, 8, 16, 32):
            if k >= self.nt:
                break
            rk = self.kc(self.an.sc["point0"], PRIMES[0], k)
            bk = self.res(rk, PRIMES[0])
            if bk is not None and np.array_equal(bk, want):
                self.nt = k
                return rk
        return ref

    def run(self):
        """The direction as an entry, or None when the support does not contain it."""
        an, rec, sc, P = self.an, self.rec, self.an.sc, PRIMES[0]
        run, ctrl = an.run, an.ctrl
        if self.m < 2:
            return None
        if an.joint and any(_CSUFFIX.search(an.znames[c]) for c in self.cols):
            return None
        self.fpos = self.cols.index(self.anchor)
        if an.joint and not self._touched():
            return None
        res = self.res
        ref = self.kc(sc["point0"], P, sc["nt_used"])
        base = res(ref, P)
        want = mulmod(self.bf % P, invmod(int(self.bf[self.anchor]) % P, P), P)
        if base is None or not np.array_equal(base, want):
            rec.tlog(f"narrow (anchor {an.znames[self.anchor]}): the restricted rows do not give the "
                     f"direction (rank {ref.get('rank') if ref else None} of {self.m})")
            return None
        if not run.has_gaps:
            ref = self._shorter_order(ref, want)
        rec.tlog(f"narrow dir (anchor {an.znames[self.anchor]}): support {self.m} of {an.nz} columns, "
                 f"Lie order {self.nt}" + (f", {len(self.conds)} of {an.sys.kc} conditions" if an.joint else ""))
        rel = rec.probe_groups(self.kb, ref["pivots"], rec.pool_next, res, ref)
        rec.pool_next += an.n_aug * (ctrl["probe_retries"] + 16)
        groups = [] if state.env("NOGROUP") else translation_groups(rel, ref, res, self.kb, sc, P, run.n_leaves)
        rec.pool_next += 4 * run.n_leaves
        base0 = np.array(sc["point0"], dtype=np.int64)
        mul = [v for g in groups if g.get("type") == "mul" for v in g["members"]]
        if mul:
            pt1 = base0.copy()
            pt1[mul] = 1
            ref1 = self.kc(pt1, P, sc["nt_used"])
            if res(ref1, P) is not None:
                base0, ref = pt1, ref1
                rel = rec.probe_groups(self.kb, ref["pivots"], rec.pool_next, res, ref, base0)
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
        jet_time = float(state.env_value("JETTIME", "20"))
        jet_first = n_rel > ctrl["relevance_cap"] or state.env("JETFIRST")
        e = self._jet(jet_time) if jet_first else None
        if e is not None:
            groups = []
        else:
            e = self._fit(ref, base0, rel)
            if not e.get("closed_form") and not jet_first:
                ej = self._jet(jet_time)
                if ej is not None:
                    e, groups = ej, []
        if not e.get("closed_form"):
            return e
        if groups:
            e["vector"] = shift_groups(e["vector"], groups, an.leaf_names_aug, base0)
            if e["vector"] is None:
                return {"support": e["support"], "type": "general", "closed_form": False,
                        "reason": "translation-group back-substitution failed"}
            e["route"] = f"{e['route']}, {len(groups)} exchange group" + ("s" if len(groups) > 1 else "")
        if not verify_in_nullspace(e, self.anchor, an.znames, an.leaf_names_aug, sc["point0"], sc["nt_used"],
                                   an.kcall, sc["pool"], rec.pool_next, an.nz, rec.dir_call):
            return {"support": e["support"], "type": "general", "closed_form": False,
                    "reason": "a closed form was reconstructed but failed verification at a fresh prime"}
        eb = an.multi.get("exp_back") or {}
        if eb.get("names"):
            e["vector"] = exp_backsub(e["vector"], eb, an.multi.get("exp_atoms"))
        return e
