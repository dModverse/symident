"""The shared state of one observability analysis."""

import time

import numpy as np

from ... import _core, state
from ...par import process_map


def ints(x):
    """`x` as a list of ints, [] for None."""
    if x is None:
        return []
    return [int(v) for v in np.atleast_1d(x)]


def strs(x):
    """`x` as a list of strings, [] for None."""
    if x is None:
        return []
    if isinstance(x, str):
        return [x]
    return [str(v) for v in x]


def ok(r):
    """Whether the kernel result `r` exists and reports success."""
    return r is not None and bool(r.get("ok"))


def tape_fields(t):
    """The kernel's view of one compiled tape."""
    out = {"op": ints(t.get("op")), "a": ints(t.get("a")), "b": ints(t.get("b")),
           "cnum": strs(t.get("cnum")), "cden": strs(t.get("cden")),
           "state_slots": ints(t.get("state_slots")), "f_out": ints(t.get("f_out")),
           "g_out": ints(t.get("g_out")), "ic_leaf": ints(t.get("ic_leaf")),
           "ic_num": strs(t.get("ic_num")), "ic_den": strs(t.get("ic_den"))}
    if t.get("ic_op") is not None:
        out.update({"ic_op": ints(t["ic_op"]), "ic_a": ints(t["ic_a"]), "ic_b": ints(t["ic_b"]),
                    "ic_cnum": strs(t["ic_cnum"]), "ic_cden": strs(t["ic_cden"]),
                    "ic_out": ints(t["ic_out"])})
    if t.get("ev_var_idx") is not None:
        out.update({"ev_op": ints(t["ev_op"]), "ev_a": ints(t["ev_a"]), "ev_b": ints(t["ev_b"]),
                    "ev_cnum": strs(t["ev_cnum"]), "ev_cden": strs(t["ev_cden"]),
                    "ev_out": ints(t["ev_out"]), "ev_var_idx": ints(t["ev_var_idx"]),
                    "ev_method": ints(t["ev_method"]), "ev_state_slots": int(t.get("ev_state_slots", 0)),
                    "ev_leaf_only": [bool(x) for x in t.get("ev_leaf_only", [True] * len(t["ev_var_idx"]))]})
    if t.get("tm_op") is not None:
        out.update({"tm_op": ints(t["tm_op"]), "tm_a": ints(t["tm_a"]), "tm_b": ints(t["tm_b"]),
                    "tm_cnum": strs(t["tm_cnum"]), "tm_cden": strs(t["tm_cden"]),
                    "tm_out": ints(t["tm_out"])})
    return out


class Run:
    """Coordinates, tapes, chains and parallel budget of one analysis; nt_cap, mtot and cores_call change during it."""

    def __init__(self, multi, cores, ctrl, chain_of, n_gaps, t0events, cond_zero_states,
                 equil_zero_states):
        self.multi = multi
        self.ctrl = ctrl
        self.timing = state.env("TIMING")
        self.t0 = time.monotonic()
        self.stages = []
        self.cores = int(max(1, cores))
        self.n_leaves = int(multi["n_leaves"])
        self.n_states = int(multi["n_states"])
        self.z_slots = np.asarray(ints(multi["z_slots"]), dtype=np.int64)
        self.znames = strs(multi["znames"])
        self.leaf_names = strs(multi["leaf_names"])
        self.leaf_index = {nm: i for i, nm in enumerate(self.leaf_names)}
        self.ss_constraint = bool(multi.get("equilibrate"))
        self.has_gaps = n_gaps is not None and n_gaps > 0 and chain_of is not None
        self.chain_of = list(chain_of) if chain_of is not None else None
        if self.has_gaps:
            keys = list(dict.fromkeys(self.chain_of))
            self.chain_groups = [[i for i, c in enumerate(self.chain_of) if c == k] for k in keys]
            self.chain_idx = {ci: keys.index(c) for ci, c in enumerate(self.chain_of)}
            seen = set()
            self.first_of_chain = []
            for c in self.chain_of:
                self.first_of_chain.append(c not in seen)
                seen.add(c)
        else:
            self.chain_groups = None
            self.chain_idx = {}
            self.first_of_chain = [True] * len(multi["tapes"])
        self.max_gap = int(ctrl["gap_order_cap"]) if self.has_gaps else 0
        units = len(self.chain_groups) if self.has_gaps else len(multi["tapes"])
        self.cores_call = int(max(1, min(self.cores, units)))
        self.tapes = [tape_fields(t) for t in multi["tapes"]]
        self.t0events = list(t0events or [])
        self.cond_zero_states = list(cond_zero_states or [])
        self.equil_zero_states = list(equil_zero_states or [])
        self.nt_cap = []
        self.mtot = 0
        self.ss_why = None
        self.series_gaps = False
        nuis = set(strs(multi.get("nuisance")))
        kept = set(strs(multi.get("nuisance_kept")))
        self.nuis_cols = [i for i, z in enumerate(self.znames) if z in nuis]
        self.nuis_drop = [i for i in self.nuis_cols if self.znames[i] not in kept]
        self.nuis_order = multi.get("nuisance_order")

    def exact_nt(self, Nt, Mtot=0):
        """The Lie order `Nt` held to the orders at which the jet chains are exact after gap order `Mtot`."""
        return int(Nt) if self.nuis_order is None else max(0, min(int(Nt), int(self.nuis_order) - int(Mtot)))

    def modulo_nuisance(self, o, p, cols=None):
        """The kernel result `o` modulo the nuisance directions, off the dropped jets; `cols` maps a column subset."""
        if not self.nuis_cols:
            return o
        from .nuisance import quotient
        if cols is None:
            return quotient(o, self.nuis_cols, p, self.nuis_drop)
        pos = {c: j for j, c in enumerate(cols)}
        return quotient(o, [pos[c] for c in self.nuis_cols if c in pos], p,
                        [pos[c] for c in self.nuis_drop if c in pos])

    _CHAIN_FIELDS = ("has_gaps", "chain_of", "chain_groups", "chain_idx", "first_of_chain", "max_gap",
                     "cores_call", "mtot", "nt_cap", "series_gaps")

    def as_chains(self):
        """Every tape a chain of one segment, whose gap order is then the series order."""
        self._unchained = {k: getattr(self, k) for k in self._CHAIN_FIELDS}
        n = len(self.multi["tapes"])
        self.has_gaps = True
        self.chain_of = list(range(n))
        self.chain_groups = [[i] for i in range(n)]
        self.chain_idx = {i: i for i in range(n)}
        self.first_of_chain = [True] * n
        self.max_gap = int(self.ctrl["gap_order_cap"])
        self.cores_call = int(max(1, min(self.cores, n)))
        self.series_gaps = True

    def as_segments(self):
        """Undo as_chains()."""
        for k, v in self._unchained.items():
            setattr(self, k, v)

    def stage(self, what):
        """Record the elapsed time at `what`, printed when timing is on."""
        t = time.monotonic() - self.t0
        self.stages.append({"stage": what, "seconds": round(t, 3)})
        if self.timing:
            print(f"[stage {t:7.1f}s] {what}", flush=True)

    def slot(self, name):
        """The leaf slot of `name`, or None."""
        return self.leaf_index.get(name)

    def par_map(self, xs, f, sys=None):
        """f over xs in forked workers, None for an item that raises; route states merge in item order."""
        xs = list(xs)
        if sys is None or not hasattr(sys, "route_state"):
            return process_map(f, xs, self.cores, on_error=lambda x: None)
        base = sys.route_state()

        def g(x):
            sys.set_route_state(base)
            return f(x), sys.route_state()
        res = process_map(g, xs, self.cores, on_error=lambda x: None)
        sys.merge_route_states(base, [r[1] for r in res if r is not None])
        return [None if r is None else r[0] for r in res]

    def block_threads(self):
        """Threads per block call: one with gaps, else all cores."""
        return 1 if self.has_gaps else self.cores

    def chain_tapes(self, idx):
        """The tapes at the indices `idx`."""
        return [self.tapes[i] for i in idx]

    def obs_block_calls(self):
        """The kernel per tape or chain, where the saturation decides the Lie order."""
        zs, nl, ns = self.z_slots, self.n_leaves, self.n_states
        if self.has_gaps:
            def mk(idx):
                def f(point, p, Nt, Mtot=0):
                    return self.modulo_nuisance(_core.sym_obs_null_chain(
                        [self.chain_tapes(idx)], nl, ns, zs, np.asarray(point, dtype=np.int64), p,
                        self.exact_nt(Nt, Mtot), int(Mtot), 1), p)
                return f
            return [mk(idx) for idx in self.chain_groups]

        def mk1(i):
            def f(point, p, Nt, Mtot=0):
                return self.modulo_nuisance(_core.sym_obs_null_multi(
                    [self.tapes[i]], nl, ns, zs, np.asarray(point, dtype=np.int64), p,
                    self.exact_nt(Nt), self.block_threads()), p)
            return f
        return [mk1(i) for i in range(len(self.tapes))]

    def obs_block_profiles(self):
        """The rank per Lie order of a block from one jet; None with nuisance columns."""
        if self.nuis_cols:
            return None
        zs, nl, ns = self.z_slots, self.n_leaves, self.n_states
        if self.has_gaps:
            def mk(bi):
                def f(point, p, Nt):
                    return _core.sym_obs_chain_rank_profile(
                        self.chain_tapes(self.chain_groups[bi]), nl, ns, zs,
                        np.asarray(point, dtype=np.int64), p, int(Nt), self.block_threads())
                return f
            return [mk(bi) for bi in range(len(self.chain_groups))]

        def mk1(i):
            def f(point, p, Nt):
                return _core.sym_obs_rank_profile([self.tapes[i]], nl, ns, zs,
                                                  np.asarray(point, dtype=np.int64), p, int(Nt),
                                                  self.block_threads())
            return f
        return [mk1(i) for i in range(len(self.tapes))]
