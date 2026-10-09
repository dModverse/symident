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
                    "ev_method": ints(t["ev_method"])})
    if t.get("tm_op") is not None:
        out.update({"tm_op": ints(t["tm_op"]), "tm_a": ints(t["tm_a"]), "tm_b": ints(t["tm_b"]),
                    "tm_cnum": strs(t["tm_cnum"]), "tm_cden": strs(t["tm_cden"]),
                    "tm_out": ints(t["tm_out"])})
    return out


class Run:
    """Coordinates, tapes, chains and the parallel budget of one analysis.

    `nt_cap` (per-chain Lie order caps), `mtot` (the gap order of the samples) and
    `cores_call` (kernel threads, 1 inside forked workers) change during the analysis
    and are read by every kernel call.
    """

    def __init__(self, multi, cores, ctrl, chain_of, n_gaps, t0events, cond_zero_states,
                 equil_zero_states):
        self.multi = multi
        self.ctrl = ctrl
        self.timing = state.env("TIMING")
        self.t0 = time.monotonic()
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

    def stage(self, what):
        """Print the elapsed time and `what` when timing is on."""
        if self.timing:
            print(f"[stage {time.monotonic() - self.t0:7.1f}s] {what}", flush=True)

    def slot(self, name):
        """The leaf slot of `name`, or None."""
        return self.leaf_index.get(name)

    def par_map(self, xs, f):
        """Per-point kernel calls in forked workers, serial with one core."""
        xs = list(xs)
        if self.cores > 1:
            return process_map(f, xs, self.cores, on_error=lambda x: None)
        return [f(x) for x in xs]

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
                    return _core.sym_obs_null_chain([self.chain_tapes(idx)], nl, ns, zs,
                                                    np.asarray(point, dtype=np.int64), p, int(Nt),
                                                    int(Mtot), 1)
                return f
            return [mk(idx) for idx in self.chain_groups]

        def mk1(i):
            def f(point, p, Nt, Mtot=0):
                return _core.sym_obs_null_multi([self.tapes[i]], nl, ns, zs,
                                                np.asarray(point, dtype=np.int64), p, int(Nt),
                                                self.block_threads())
            return f
        return [mk1(i) for i in range(len(self.tapes))]

    def obs_block_profiles(self):
        """The rank per Lie order of a block from one jet."""
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
