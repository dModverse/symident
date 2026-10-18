"""The compiled kernel reproduces recorded calls of the R reference bit for bit."""

import glob
import os

import numpy as np
import pytest
from spec_util import snake
from tjson import tload

from symident import _core

HERE = os.path.dirname(__file__)
FILES = sorted(glob.glob(os.path.join(HERE, "data", "kernel", "*.json.gz")))
extra = os.environ.get("SYMIDENT_KCAP")
if extra:
    FILES += sorted(glob.glob(os.path.join(extra, "*.json.gz")))


def same(a, b, path="result"):
    if isinstance(b, dict):
        assert isinstance(a, dict), path
        for k in b:
            assert snake(k) in a, f"{path}.{k} missing"
            same(a[snake(k)], b[k], f"{path}.{k}")
        return
    if isinstance(b, list) and not all(isinstance(x, str) for x in b):
        assert isinstance(a, list) and len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f"{path}[{i}]")
        return
    if b is None:
        assert a is None or (hasattr(a, "__len__") and len(a) == 0), path
        return
    if isinstance(b, str) or (isinstance(b, list)):
        bb = [b] if isinstance(b, str) else b
        aa = [a] if isinstance(a, str) else list(a)
        assert aa == bb, path
        return
    aa = np.asarray(a)
    bb = np.asarray(b)
    if aa.size == 0 and bb.size == 0:
        return
    assert aa.size == bb.size, f"{path}: size {aa.size} != {bb.size}"
    np.testing.assert_array_equal(aa.reshape(bb.shape) if aa.shape != bb.shape else aa, bb,
                                  err_msg=path)


def snake_keys(x):
    if isinstance(x, dict):
        return {snake(k): snake_keys(v) for k, v in x.items()}
    if isinstance(x, list):
        return [snake_keys(v) for v in x]
    return x


@pytest.mark.parametrize("path", FILES, ids=[os.path.basename(f) for f in FILES])
def test_kernel_call(path):
    rec = tload(path)
    fun = snake(rec["fun"])
    if fun == "sym_rat_recon" and not getattr(_core, "has_int128", True):
        pytest.skip("sym_rat_recon is left out without 128-bit integers")
    fn = getattr(_core, fun)
    args = {snake(k): snake_keys(v) for k, v in rec["args"].items() if v is not None}
    out = fn(**args)
    same(out, rec["result"])


def _multi_call():
    """A recorded call with its tape repeated, so that the conditions run in parallel, and
    the slot of its first constant."""
    path = next(f for f in FILES if "symObsNullMulti" in f)
    args = {snake(k): snake_keys(v) for k, v in tload(path)["args"].items() if v is not None}
    tape = args["tapes"][0]
    args["tapes"] = [{k: list(v) if k in ("cnum", "cden") else v for k, v in tape.items()} for _ in range(4)]
    args["cores"] = 4
    return args, list(tape["op"]).index(0)


def test_an_error_inside_a_parallel_region_reaches_python():
    args, i = _multi_call()
    args["tapes"][2]["cden"][i] = str(int(args["p_in"]))
    with pytest.raises(ValueError, match="no residue modulo"):
        _core.sym_obs_null_multi(**args)


def test_a_constant_that_is_no_integer_is_refused():
    args, i = _multi_call()
    args["tapes"][1]["cnum"][i] = "1.5"
    with pytest.raises(ValueError, match="not an integer"):
        _core.sym_obs_null_multi(**args)


def _chain_call(kind):
    path = next(f for f in FILES if "symObsNullChain_" in f and kind in f)
    return {snake(k): snake_keys(v) for k, v in tload(path)["args"].items() if v is not None}


def test_the_line_mode_of_one_segment_matches_the_shared_series_axis():
    args = _chain_call("Exponential_symObsNullChain_1762442_03")
    T, rng = 3, np.random.default_rng(5)
    ser = rng.integers(1, 1000, size=(args["n_leaves"], T)).astype(np.int64)
    shared = _core.sym_obs_null_chain(**dict(args, Mtot=T, point_ser=ser))
    line = _core.sym_obs_null_chain(**dict(args, Mtot=0, point_ser=ser, ser_order=T))
    plain = _core.sym_obs_null_chain(**dict(args, Mtot=0))
    assert line["rank_s"] == shared["rank_s"] and line["N"] == shared["N"] == T + 1
    # the same series row space: the same kernel along the line
    from symident.detection.series import series_null
    p, nz = int(args["p_in"]), len(args["z_slots"])
    ks = [series_null(np.asarray(o["S"], dtype=np.int64).reshape(int(o["rank_s"]), -1), nz, T + 1, p)
          for o in (line, shared)]
    assert ks[0]["free"] == ks[1]["free"]
    for a, b in zip(ks[0]["V"], ks[1]["V"]):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # the rows at t^0 are the kernel at the point
    assert line["rank"] == plain["rank"] and not line["at_one"]
    np.testing.assert_array_equal(np.asarray(line["R"]), np.asarray(plain["R"]))


def test_the_line_mode_keeps_the_gaps_formal_and_truncates_consistently():
    from symident.detection.series import series_null
    args = _chain_call("Detection_symObsNullChain_1762441_01")
    plain = _core.sym_obs_null_chain(**dict(args, Mtot=2))
    zero = np.zeros((args["n_leaves"], 3), dtype=np.int64)
    line = _core.sym_obs_null_chain(**dict(args, Mtot=2, point_ser=zero, ser_order=3))
    assert line["rank_s"] == line["rank"] and line["N"] == 4
    if not plain["at_one"]:
        np.testing.assert_array_equal(np.asarray(line["R"]), np.asarray(plain["R"]))
    rng = np.random.default_rng(3)
    ser = rng.integers(1, 1000, size=(args["n_leaves"], 5)).astype(np.int64)
    p, nz = int(args["p_in"]), len(args["z_slots"])
    out = {}
    for T in (3, 5):
        o = _core.sym_obs_null_chain(**dict(args, Mtot=2, point_ser=ser[:, :T], ser_order=T))
        out[T] = series_null(np.asarray(o["S"], dtype=np.int64).reshape(int(o["rank_s"]), -1), nz, T + 1, p)
    assert out[3]["free"] == out[5]["free"]
    for v3, v5 in zip(out[3]["V"], out[5]["V"]):
        np.testing.assert_array_equal(np.asarray(v3)[:, :4], np.asarray(v5)[:, :4])


def test_the_line_mode_refuses_an_event_time_that_moves_along_the_line():
    args = _chain_call("Detection_symObsNullChain_1762441_01")
    seg = dict(args["chains"][0][1], tm_op=[1], tm_a=[0], tm_b=[0], tm_cnum=["0"], tm_cden=["1"],
               tm_out=[args["n_leaves"]])
    args = dict(args, chains=[[args["chains"][0][0], seg]])
    ser = np.ones((args["n_leaves"], 2), dtype=np.int64)
    assert not _core.sym_obs_null_chain(**dict(args, Mtot=1, point_ser=ser, ser_order=2))["ok"]
    with pytest.raises(ValueError):
        _core.sym_obs_null_chain(**dict(args, Mtot=1, point_ser=ser[:2], ser_order=2))
