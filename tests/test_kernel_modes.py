"""The kernel builds the same rows in forward and in reverse mode, from one shared tape as from a tape per
condition, and with continued jets as from fresh ones."""

import numpy as np
import pytest

import symident as si
from symident import _core
from symident.gfp import null_residues

FUNS = ["sym_obs_null_multi", "sym_obs_null_batch", "sym_obs_rank_profile", "sym_obs_null_chain",
        "sym_obs_chain_rank_profile", "sym_obs_chain_rows", "sym_obs_null_chain_seed_batch",
        "sym_obs_null_chain_point_batch", "sym_segment_ranks"]


def ev(var, time, value, method):
    return {"var": var, "time": time, "value": value, "method": method}


MODELS = {
    "rational": lambda: si.Model({"x": "v*x/(K + x) - d*x"}, {"y": "s*x"}),
    "start in closed form": lambda: si.Model(
        {"x": "v*u*x/(K + x) + kb - d*x"}, {"y": "s*x"},
        trafo={"x": "(v*u + kb - d*K + sqrt((v*u + kb - d*K)^2 + 4*d*kb*K))/(2*d)"},
        conditions={"rows": ["a", "b"], "cols": {"u": [1, 2]}}),
    "unknown input": lambda: si.Model({"L": "-(c*T + L)/tau"}, {"y": "L"}, unknown_inputs=["T"]),
    "dose at an unknown time": lambda: si.Model(
        {"D": "-ka*D", "A": "ka*D - k*A"}, {"y": "A"}, events=[ev("D", "tau", "d", "add")],
        trafo={"D": "0", "A": "0"}),
    "switch with a kink": lambda: si.Model(
        {"x": "-k1*x + u*z", "z": "-(1-u)*k2*z", "u": "0"}, {"y": "x"},
        events=[ev("u", 0, "0", "replace"), ev("u", "tau", "1", "replace")]),
    "scaled and reset state": lambda: si.Model(
        {"A": "-k*A/(1 + A)"}, {"y": "s*A"}, events=[ev("A", 2, "f", "multiply"), ev("A", 4, "r", "replace")]),
    "switch at a given time": lambda: si.Model(
        {"x": "piecewise(-a*x + c*z, time > 1, -b*x*z)", "z": "-d*z/(1 + z)"}, {"y": "s*x"}),
    "conditions in values": lambda: si.Model(
        {"x": "k1*u - k2*x*z", "z": "k3*x - k4*z"}, {"y": "s*z"},
        conditions={"rows": ["a", "b", "c"], "cols": {"u": [1, 2, 0], "k4": ["k4", "k4", "k5"]}}),
}


def _calls(monkeypatch, model, **kw):
    seen = []

    def spy(name):
        f = getattr(_core, name)

        def call(*a, **k):
            seen.append((name, a, k))
            return f(*a, **k)
        return call
    for name in FUNS:
        monkeypatch.setattr(_core, name, spy(name))
    si.detect(model, **kw)
    monkeypatch.undo()
    return seen


def _both(monkeypatch, name, a, k):
    out = []
    for mode in ("forward", "reverse"):
        monkeypatch.setenv("SYMIDENT_KERNEL_MODE", mode)
        out.append(getattr(_core, name)(*a, **k))
    monkeypatch.delenv("SYMIDENT_KERNEL_MODE")
    return out


def _equal(x, y):
    if isinstance(x, list):
        assert len(x) == len(y)
        for a, b in zip(x, y):
            _equal(a, b)
        return
    if not isinstance(x, dict):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        return
    assert x.keys() == y.keys()
    for key in x:
        _equal(x[key], y[key])


@pytest.mark.parametrize("name", MODELS)
def test_both_modes_give_the_same_rows(monkeypatch, name):
    calls = _calls(monkeypatch, MODELS[name](), reconstruct=True)
    assert calls
    for fun in FUNS:
        for _, a, k in [c for c in calls if c[0] == fun][:4]:
            _equal(*_both(monkeypatch, fun, a, k))


def test_the_cases_reach_every_part_of_the_kernel(monkeypatch):
    calls = [c for name in MODELS for c in _calls(monkeypatch, MODELS[name]())]
    tapes = [t for n, a, _ in calls if n == "sym_obs_null_multi" for t in a[0]]
    segs = [s for n, a, _ in calls if n == "sym_obs_null_chain" for ch in a[0] for s in ch]
    assert any(3 in list(t.get("op", [])) for t in tapes + segs)
    assert any(t.get("ic_op") is not None for t in tapes + segs)
    assert {int(m) for s in segs if "ev_method" in s for m in s["ev_method"]} == {0, 1, 2}
    assert any("tm_op" in s for s in segs)
    assert any("base" in t for t in tapes) and any("leaf_slot" in t for t in tapes)


def test_a_vanishing_reciprocal_fails_in_both_modes(monkeypatch):
    _, (tapes, nl, ns, zs, pt, p, nt, *_), _ = [c for c in _calls(monkeypatch, MODELS["rational"]())
                                                if c[0] == "sym_obs_null_multi"][-1]
    a, b = _both(monkeypatch, "sym_obs_null_multi", (tapes, nl, ns, zs, np.zeros_like(pt), p, nt), {})
    assert a["ok"] is False and b["ok"] is False


def test_conditions_in_values_share_one_tape(monkeypatch):
    shared = si.detect(MODELS["conditions in values"](), reconstruct=True)
    monkeypatch.setenv("SYMIDENT_NOSHARED", "1")
    per = si.detect(MODELS["conditions in values"](), reconstruct=True)
    assert shared.rank == per.rank and shared.dim == per.dim
    assert sorted(map(str, shared.symmetries)) == sorted(map(str, per.symmetries))


def test_continued_jets_equal_fresh_ones(monkeypatch):
    calls = _calls(monkeypatch, MODELS["switch at a given time"](), reconstruct=True)
    rows = [(a, k) for n, a, k in calls if n == "sym_obs_chain_rows"]
    assert rows
    a, k = rows[0]
    chain, nl, ns, z, pt, p = a[:6]
    for Nt in (4, 9, 17):
        fresh = _core.sym_obs_chain_rows(chain, nl, ns, z, pt, p, Nt, 0, 1, True)
        kept = _core.sym_obs_chain_rows(chain, nl, ns, z, pt, p, Nt, 0, 1, True, "test-chain")
        _equal(fresh, kept)
    seg = [s for s in chain if s.get("ic_op") is not None][:1]
    st = np.arange(1, ns + 1, dtype=np.int64).reshape(1, ns) * 7 + 3
    for Nt in (3, 8, 20):
        fresh = _core.sym_segment_ranks(seg, nl, ns, z, pt, p, Nt, st, 1, None, True)
        kept = _core.sym_segment_ranks(seg, nl, ns, z, pt, p, Nt, st, 1, None, True, "test-seg")
        _equal(fresh, kept)


def test_lanes_along_directions_equal_the_rows(monkeypatch):
    calls = _calls(monkeypatch, MODELS["conditions in values"](), reconstruct=True)
    _, (tapes, nl, ns, zs, pt, p, nt, *_), _ = [c for c in calls if c[0] == "sym_obs_null_multi"][-1]
    nz = len(zs)
    ker = _core.sym_obs_null_multi(tapes, nl, ns, zs, pt, p, nt)
    free = [c for c in range(nz) if c not in set(np.atleast_1d(ker["pivots"]).tolist())]
    assert free
    vecs = [null_residues(ker, c, p) for c in free]
    off = np.zeros(nz, dtype=np.int64)
    off[np.atleast_1d(ker["pivots"])[0]] = 1
    V = np.stack(vecs + [off], axis=1) % p
    other = (np.asarray(pt, dtype=np.int64) * 3 + 7) % p
    got = _core.sym_obs_annihilates_batch(tapes, nl, ns, zs, np.vstack([pt, other]), [p, p], nt, V)
    assert list(got[0]["zero"]) == [True] * len(free) + [False]
    # past nz columns the call reads the rows over every coordinate
    pad = np.concatenate([V, np.zeros((nz, nz), dtype=np.int64)], axis=1)
    for k, point in enumerate((pt, other)):
        rows = _core.sym_obs_annihilates(tapes, nl, ns, zs, point, p, nt, pad)
        assert list(rows["zero"])[:V.shape[1]] == list(got[k]["zero"])
