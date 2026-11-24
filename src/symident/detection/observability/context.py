"""The shared state of one observability analysis."""

import hashlib
import os
import threading
import time

import numpy as np

from ... import _core, state
from ...par import process_map, thread_map
from ...sym.shared import kernel_tapes
from ...sym.tape import _OP_SPOW


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


def tape_fields(t, memo=None):
    """The kernel's view of one compiled tape; with `memo`, arrays that tapes share are converted once and stay
    shared."""
    memo = {} if memo is None else memo

    def conv(x, f):
        if x is None:
            return f(x)
        k = (id(x), f)
        if k not in memo:
            memo[k] = (x, f(x))
        return memo[k][1]

    def iv(k):
        return conv(t.get(k), ints)

    def sv(k):
        return conv(t.get(k), strs)
    out = {"op": iv("op"), "a": iv("a"), "b": iv("b"), "cnum": sv("cnum"), "cden": sv("cden"),
           "state_slots": iv("state_slots"), "f_out": iv("f_out"), "g_out": iv("g_out"), "ic_leaf": iv("ic_leaf"),
           "ic_num": sv("ic_num"), "ic_den": sv("ic_den")}
    if t.get("ic_op") is not None:
        out.update({"ic_op": iv("ic_op"), "ic_a": iv("ic_a"), "ic_b": iv("ic_b"), "ic_cnum": sv("ic_cnum"),
                    "ic_cden": sv("ic_cden"), "ic_out": iv("ic_out")})
    if t.get("leaf_slot") is not None:
        out.update({"leaf_slot": ints(t["leaf_slot"]), "leaf_src": ints(t["leaf_src"]),
                    "leaf_num": strs(t["leaf_num"]), "leaf_den": strs(t["leaf_den"])})
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


# tape fields that hold slot indices; a slot below n_leaves is a leaf
_SLOT_FIELDS = ("a", "b", "f_out", "g_out", "ic_leaf", "ic_a", "ic_b", "ic_out", "ev_a", "ev_b", "ev_out",
                "tm_a", "tm_b", "tm_out", "leaf_src")


def read_columns(tapes, n_leaves, z_slots):
    """The columns of the coordinates that the tapes read, ascending, or None for all columns.

    The jets of a chain have zero derivative in a coordinate that no tape of the chain reads, so the
    kernel over the read columns has the same rows there. None where a tape holds a seed of the initial
    values, whose columns are set per coordinate."""
    used = np.zeros(int(n_leaves), dtype=bool)
    for t in tapes:
        if t.get("ic_seed") is not None:
            return None
        for key in _SLOT_FIELDS:
            v = t.get(key)
            if v is None:
                continue
            v = np.asarray(v, dtype=np.int64).ravel()
            used[v[(v >= 0) & (v < n_leaves)]] = True
    cols = np.nonzero(used[np.asarray(z_slots, dtype=np.int64)])[0]
    if len(cols) == len(z_slots):
        return None
    # the kernel takes at least one column
    return cols if len(cols) else np.zeros(1, dtype=np.int64)


def embed_columns(o, cols, nz):
    """The kernel result `o` over the columns `cols` as a result over all `nz` columns."""
    if cols is None or not ok(o):
        return o
    o = dict(o)
    m = len(cols)
    R = np.asarray(o["R"], dtype=np.int64).reshape(-1, m)
    full = np.zeros((R.shape[0], nz), dtype=np.int64)
    full[:, cols] = R
    o["R"] = full
    o["pivots"] = cols[np.atleast_1d(np.asarray(o["pivots"], dtype=np.int64))]
    o["dim"] = nz
    if o.get("S") is not None:
        N = int(o.get("N") or 1)
        S = np.asarray(o["S"], dtype=np.int64).reshape(-1, m * N)
        fullS = np.zeros((S.shape[0], nz * N), dtype=np.int64)
        fullS[:, (cols[:, None] * N + np.arange(N)).ravel()] = S
        o["S"] = fullS
    return o


def _with_gap_order(e, Mtot):
    """The rows `e` of a chain of one segment at gap order 0 as the rows of the chain at gap order `Mtot`: the row
    of the constant gap monomial with zero rows of the other monomials, and each series row at eps^0."""
    m, nz = Mtot + 1, e["nz"]
    rows = np.zeros((e["rows"].shape[0] * m, nz), dtype=np.int32)
    rows[::m] = e["rows"]
    # the series rows of gap order 0 are the rows themselves
    srows = np.zeros((e["srows"].shape[0], nz * m), dtype=np.int32)
    srows[:, ::m] = e["srows"]
    return {"Nt": e["Nt"], "plain": e["plain"], "N": m, "nz": nz, "rows": rows, "tag": np.repeat(e["tag"], m),
            "srows": srows, "stag": e["stag"].copy(), "ranks": None,
            "bytes": rows.nbytes + srows.nbytes}


def _block_bytes():
    """Sets the Taylor values per column block of a kernel call to an eighth of the physical memory over the cores,
    between 256 MB and 1 GB; SYMIDENT_BLOCKMB overrides it."""
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return
    _core.sym_set_block_bytes(float(min(1 << 30, max(256 << 20, mem // (8 * (os.cpu_count() or 1))))))


_block_bytes()

# bytes of unreduced chain rows a Run keeps
_ROWS_BYTES = 4 << 30
_ROWS_LOCK = threading.Lock()


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
        memo = {}
        self.tapes = [tape_fields(t, memo) for t in multi["tapes"]]
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
        # the states in the order of the tape, which the model lines follow
        lines = strs(multi["tapes"][0].get("model_lines")) if multi.get("tapes") else []
        names = [ln.split("=", 1)[0].strip() for ln in lines]
        names = names if len(names) == self.n_states else []
        self.known_states = [names.index(x) for x in strs(multi.get("known_jet_states")) if x in names]

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

    def chain_columns(self, idx, zs=None):
        """The positions in `zs` (the coordinate slots by default) of the coordinates the chain of the tapes
        `idx` reads (read_columns), memoized."""
        zs = self.z_slots if zs is None else np.asarray(zs, dtype=np.int64)
        memo = self.__dict__.setdefault("_chain_columns", {})
        key = (tuple(idx), zs.tobytes())
        if key not in memo:
            memo[key] = read_columns(self.chain_tapes(idx), self.n_leaves, zs)
        return memo[key]

    def chain_rows(self, idx, point, p, Nt, Mtot, threads, zs, cols):
        """The unreduced rows of the chain of the tapes `idx` over the columns `cols` of the slots `zs` to Lie
        order `Nt` or above (sym_obs_chain_rows), kept per point, prime and gap order.

        The coefficients at a Lie order do not depend on the order the jets are taken to, so where no segment
        boundary moves (`plain`) the rows of a lower order are a selection of those kept."""
        memo = self.__dict__.setdefault("_chain_rows", {})
        key = (tuple(idx), point.tobytes(), int(p), int(Mtot), zs.tobytes())
        e = memo.get(key)
        if e is not None and (e["Nt"] == int(Nt) or (e["plain"] and e["Nt"] > int(Nt))):
            return e
        z = zs if cols is None else zs[cols]
        if int(Mtot) > 0 and len(idx) == 1 and _OP_SPOW not in self.tapes[idx[0]]["op"]:
            # one segment has no gap: the coefficients of a gap monomial are zero, so the rows are those of gap
            # order 0 with zero rows of the other monomials (not for a power of a state at 0, whose rows are
            # ordered otherwise)
            e0 = self.chain_rows(idx, point, p, Nt, 0, threads, zs, cols)
            if e0 is None:
                return None
            return self._keep_rows(key, _with_gap_order(e0, int(Mtot)))
        # the jets of the chain continue those of a call at a lower order
        jets = hashlib.sha256(repr((self.chain_content(idx), point.tobytes(), int(p), int(Mtot),
                                    np.asarray(z, dtype=np.int64).tobytes())).encode()).hexdigest()
        r = _core.sym_obs_chain_rows(self.chain_tapes(idx), self.n_leaves, self.n_states, z, point, p, int(Nt),
                                     int(Mtot), int(threads), int(Mtot) == 0, jets)
        if not ok(r):
            return None
        e = {"Nt": int(r["Nt"]), "plain": bool(r["plain"]), "N": int(r["N"]), "nz": len(z),
             "rows": np.asarray(r["rows"], dtype=np.int32).reshape(-1, len(z)),
             "tag": np.atleast_1d(np.asarray(r["tag"], dtype=np.int32)),
             "srows": np.asarray(r["srows"], dtype=np.int32).reshape(-1, len(z) * int(r["N"])),
             "stag": np.atleast_1d(np.asarray(r["stag"], dtype=np.int32)),
             "ranks": None if r.get("ranks") is None else np.atleast_1d(np.asarray(r["ranks"], dtype=np.int64))}
        return self._keep_rows(key, e)

    def chain_content(self, idx):
        """A digest of what the kernel reads of the tapes `idx`, memoized."""
        memo = self.__dict__.setdefault("_chain_content", {})
        key = tuple(idx)
        if key not in memo:
            memo[key] = hashlib.sha256(repr([sorted((k, v.tolist() if hasattr(v, "tolist") else v)
                                                    for k, v in self.tapes[i].items())
                                             for i in idx]).encode()).hexdigest()
        return memo[key]

    def _keep_rows(self, key, e):
        """Keep the rows `e` of chain_rows under `key`."""
        memo = self.__dict__.setdefault("_chain_rows", {})
        e["bytes"] = e["rows"].nbytes + e["srows"].nbytes
        # the rows of the chains at the points of the saturation and of the checks; the oldest are dropped first
        with _ROWS_LOCK:
            held = self.__dict__.setdefault("_chain_rows_bytes", [0])
            old = memo.pop(key, None)
            if old is not None:
                held[0] -= old["bytes"]
            while memo and held[0] + e["bytes"] > _ROWS_BYTES:
                held[0] -= memo.pop(next(iter(memo)))["bytes"]
            memo[key] = e
            held[0] += e["bytes"]
        return e

    def chain_kernel(self, idx, point, p, Nt, Mtot, threads, zs=None):
        """sym_obs_null_chain of the chain of the tapes `idx` alone over the slots `zs`, built over the columns
        the chain reads, from the rows kept by chain_rows."""
        zs = self.z_slots if zs is None else np.asarray(zs, dtype=np.int64)
        cols = self.chain_columns(idx, zs)
        point = np.asarray(point, dtype=np.int64)
        # the scans of the saturation and the stacked kernels after them take the same chains at the same
        # orders; the results are kept over the columns read
        memo = self.__dict__.setdefault("_chain_kernels", {})
        key = (tuple(idx), point.tobytes(), int(p), int(Nt), int(Mtot), zs.tobytes())
        # a kept result is only read: embed_columns copies it
        o = memo.get(key)
        if o is None:
            e = self.chain_rows(idx, point, p, Nt, Mtot, threads, zs, cols)
            if e is None:
                o = {"ok": False}
            else:
                sel, ssel = (slice(None), slice(None)) if e["Nt"] == int(Nt) else \
                    (e["tag"] <= int(Nt), e["stag"] <= int(Nt))
                o = _core.sym_chain_reduce(e["rows"][sel], e["srows"][ssel], e["nz"], e["N"], p)
            if len(memo) >= 8 * max(1, len(self.chain_groups or ())):
                memo.clear()
            memo[key] = o
        return embed_columns(o, cols, len(zs))

    def stacked_chain_kernel(self, point, p, Nt, Mtot, caps, zs=None):
        """sym_obs_null_chain over every chain and the slots `zs`, chain `t` to Lie order min(Nt, caps[t]), from
        the kernels of the chains alone over the columns each reads; None where that saves little or the rows
        of a chain change at t = 1.

        The rows of the stacked kernel are those of the chains, so their reduced form is the reduced form of
        the stacked reduced rows of the chains, and the series rows span the same module."""
        zs = self.z_slots if zs is None else np.asarray(zs, dtype=np.int64)
        T = len(self.chain_groups)
        cols = [self.chain_columns(idx, zs) for idx in self.chain_groups]
        nz = len(zs)
        if T < 2 or sum(nz if c is None else len(c) for c in cols) > T * nz // 2:
            return None
        nts = [min(int(Nt), int(caps[t])) if len(caps) == T else int(Nt) for t in range(T)]
        # the costly chains first, so that none starts last: the cost grows with the columns, the segments and
        # the square of the order
        order = sorted(range(T), key=lambda t: -(nz if cols[t] is None else len(cols[t])) *
                       len(self.chain_groups[t]) * (nts[t] + 1) ** 2)
        got = thread_map(lambda t: self.chain_kernel(self.chain_groups[t], point, p, nts[t], Mtot, 1, zs),
                         order, self.cores_call)
        per = [None] * T
        for t, o in zip(order, got):
            per[t] = o
        if not all(ok(o) for o in per):
            return {"ok": False}
        if any(o.get("at_one") for o in per):
            return None
        rows = np.vstack([np.asarray(o["R"], dtype=np.int64).reshape(-1, nz) for o in per])
        N = int(Mtot) + 1
        sers = [np.asarray(o["S"], dtype=np.int64).reshape(-1, nz * N) for o in per]
        if not rows.shape[0] or not sum(S.shape[0] for S in sers):
            return None
        rr = _core.sym_rref_mod(rows.astype(np.float64), p)
        res = {"ok": True, "R": np.asarray(rr["R"], dtype=np.int64).reshape(int(rr["rank"]), nz),
               "pivots": np.atleast_1d(rr["piv"]).astype(np.int64), "rank": int(rr["rank"]), "dim": nz}
        from ..fields import series_stack
        res = series_stack(res, sers, nz, N, p, self.cores_call)
        return None if res["at_one"] else res

    def stacked_chain_batch(self, points, primes, Nt, Mtot, caps, zs=None):
        """stacked_chain_kernel at each point and prime, as sym_obs_null_chain_point_batch; None where it does
        not apply."""
        out = []
        for pt, q in zip(points, primes):
            o = self.stacked_chain_kernel(np.asarray(pt, dtype=np.int64)[:self.n_leaves], q, Nt, Mtot, caps, zs)
            if o is None:
                return None
            out.append(o)
        return out

    def block_threads(self):
        """Threads per block call: one with gaps, else all cores."""
        return 1 if self.has_gaps else self.cores

    def kernel_tapes(self):
        """The tapes for a kernel call over every tape (kernel_tapes), formed once."""
        kt = self.__dict__.get("_kernel_tapes")
        if kt is None or kt[0] is not self.tapes:
            kt = self._kernel_tapes = (self.tapes, kernel_tapes(self.tapes))
        return kt[1]

    def chain_tapes(self, idx):
        """The tapes at the indices `idx`."""
        return [self.tapes[i] for i in idx]

    def obs_block_calls(self):
        """The kernel per tape or chain, where the saturation decides the Lie order."""
        zs, nl, ns = self.z_slots, self.n_leaves, self.n_states
        if self.has_gaps:
            def mk(idx):
                def f(point, p, Nt, Mtot=0):
                    return self.modulo_nuisance(self.chain_kernel(idx, point, p, self.exact_nt(Nt, Mtot),
                                                                  Mtot, 1), p)
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
                idx = self.chain_groups[bi]

                def f(point, p, Nt):
                    # the rank does not see the columns the chain does not read; the rows stay for the kernel at
                    # the order the scan stops at
                    cols = self.chain_columns(idx)
                    e = self.chain_rows(idx, np.asarray(point, dtype=np.int64), p, Nt, 0, self.block_threads(),
                                        zs, cols)
                    if e is None or e["ranks"] is None:
                        return {"ok": False}
                    return {"ok": True, "ranks": e["ranks"][:int(Nt) + 1], "cols": e["nz"]}
                return f
            return [mk(bi) for bi in range(len(self.chain_groups))]

        def mk1(i):
            def f(point, p, Nt):
                return _core.sym_obs_rank_profile([self.tapes[i]], nl, ns, zs,
                                                  np.asarray(point, dtype=np.int64), p, int(Nt),
                                                  self.block_threads())
            return f
        return [mk1(i) for i in range(len(self.tapes))]
