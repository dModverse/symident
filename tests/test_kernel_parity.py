"""The compiled kernel reproduces recorded calls of the R reference bit for bit."""

import glob
import os

import numpy as np
import pytest
from golden_util import snake
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
