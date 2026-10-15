"""Reference runs of the R implementation: same rank, directions and charts."""

import glob
import os

import pytest
from golden_util import run_detection, run_reduction

from symident.install import find_msolve

HERE = os.path.dirname(__file__)
DET = sorted(glob.glob(os.path.join(HERE, "data", "golden", "detect", "*.json")))
RED = sorted(glob.glob(os.path.join(HERE, "data", "golden", "reduce", "*.json")))
MSOLVE = find_msolve()[0] is not None
NEEDS_MSOLVE = {"test-symmetryDetection_151_symmetryDetection.json"}


def cases(paths):
    names = [os.path.basename(p)[:-5] for p in paths]
    return [pytest.param(p, id=n) for p, n in zip(paths, names)]


@pytest.mark.parametrize("path", cases(DET))
def test_detection(path):
    if os.path.basename(path) in NEEDS_MSOLVE and not MSOLVE:
        pytest.skip("needs msolve")
    assert run_detection(path) == []


@pytest.mark.parametrize("path", cases(RED))
def test_reduction(path):
    assert run_reduction(path) == []


def test_a_direction_closed_or_a_rank_proven_beyond_the_reference_is_better():
    from golden_util import compare
    open_ = {"type": "general", "support": ["k"], "explicit": False}
    ref = {"rank": 1, "dim": 2, "identifiable": False, "symmetries": [open_]}
    closed = dict(open_, explicit=True, generator={"k": "k"})
    py = {"rank": 1, "dim": 2, "identifiable": False, "symmetries": [closed], "info": {"rank_proven": 1}}
    assert compare(py, ref) == ["BETTER: closed by symident only", "BETTER: rank proven by symident only"]
    assert compare(dict(py, rank=2), ref)[0] == "rank: 2 != 1"
