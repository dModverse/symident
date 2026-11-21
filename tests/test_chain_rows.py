"""The kernels of a chain from the kept rows equal the kernels built at their order."""

import numpy as np
import pytest

import symident as si
from symident import _core
from symident.detection.observability import context as cx


def _same(o, d):
    for k in ("ok", "rank", "rank_s", "at_one", "N", "dim"):
        assert o.get(k) == d.get(k), k
    if d.get("ok"):
        for k in ("R", "pivots", "S"):
            np.testing.assert_array_equal(np.asarray(o[k]), np.asarray(d[k]), err_msg=k)


@pytest.fixture
def checked(monkeypatch):
    """Every chain kernel and rank profile of a detection compared with the kernel built at the same order."""
    seen = {"kernels": 0, "profiles": 0, "orders": set()}
    kernel, rows = cx.Run.chain_kernel, cx.Run.chain_rows

    def chain_kernel(self, idx, point, p, Nt, Mtot, threads, zs=None):
        o = kernel(self, idx, point, p, Nt, Mtot, threads, zs)
        zs = self.z_slots if zs is None else np.asarray(zs, dtype=np.int64)
        cols = self.chain_columns(idx, zs)
        d = _core.sym_obs_null_chain([self.chain_tapes(idx)], self.n_leaves, self.n_states,
                                     zs if cols is None else zs[cols], np.asarray(point, dtype=np.int64), p,
                                     int(Nt), int(Mtot), 1)
        _same(o, cx.embed_columns(d, cols, len(zs)))
        seen["kernels"] += 1
        seen["orders"].add((len(idx), int(Mtot)))
        return o

    def chain_rows(self, idx, point, p, Nt, Mtot, threads, zs, cols):
        e = rows(self, idx, point, p, Nt, Mtot, threads, zs, cols)
        if e is not None and e["ranks"] is not None:
            r = _core.sym_obs_chain_rank_profile(self.chain_tapes(idx), self.n_leaves, self.n_states,
                                                 zs if cols is None else zs[cols], point, p, e["Nt"], 1)
            np.testing.assert_array_equal(np.asarray(r["ranks"]), e["ranks"])
            seen["profiles"] += 1
        return e

    monkeypatch.setattr(cx.Run, "chain_kernel", chain_kernel)
    monkeypatch.setattr(cx.Run, "chain_rows", chain_rows)
    return seen


def test_kept_rows_give_the_kernels_of_events_and_single_segments(checked):
    # two conditions with a dose, one without: chains of two segments and of one segment, a scaling and a
    # gap order past 0
    model = si.Model({"x": "-k1*x + u", "y": "k2*x - k3*y", "u": "-k4*u"}, {"obs": "s*y"},
                     trafo=[{"x": "0", "y": "0", "u": "u0"}, {"x": "0", "y": "0", "u": "u0"},
                            {"x": "x0", "y": "0", "u": "0"}],
                     conditions={"rows": ["a", "b", "c"], "cols": {"dose": [1, 2, 0]}},
                     events=[{"var": "x", "time": 1, "value": "dose", "method": "add", "cond": [0, 1]}])
    r = si.detect(model)
    assert checked["kernels"] > 0 and checked["profiles"] > 0
    assert {n for n, _ in checked["orders"]} == {1, 2}
    assert r.rank < r.dim


def test_rows_of_a_lower_order_are_a_selection():
    model = si.Model({"x": "-k1*x*y", "y": "k1*x*y - k2*y"}, {"obs": "y"}, trafo={"x": "x0", "y": "y0"},
                     events=[{"var": "x", "time": 2, "value": "1", "method": "add"}])
    calls = {}
    orig = _core.sym_obs_null_chain

    def spy_null_chain(chains, *a, **k):
        calls["args"] = (chains, a)
        return orig(chains, *a, **k)
    try:
        _core.sym_obs_null_chain = spy_null_chain
        si.detect(model)
    finally:
        _core.sym_obs_null_chain = orig
    chains, (nl, ns, zs, point, p, *_rest) = calls["args"]
    chain = chains[0]
    for M in (0, 1):
        e = _core.sym_obs_chain_rows(chain, nl, ns, zs, point, p, 12, M, 1, M == 0)
        assert e["plain"]
        for nt in (3, 7, 12):
            sel, ssel = np.asarray(e["tag"]) <= nt, np.asarray(e["stag"]) <= nt
            o = _core.sym_chain_reduce(np.asarray(e["rows"])[sel], np.asarray(e["srows"])[ssel], len(zs),
                                       int(e["N"]), p)
            _same(o, _core.sym_obs_null_chain([chain], nl, ns, zs, point, p, nt, M, 1))
