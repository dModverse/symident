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
# cases of ten seconds and more, run only without -m "not slow"
SLOW = {"test-symmetryReduction_087_symmetryDetection", "test-symmetryReduction_033_symmetryReduction",
        "test-symmetryReduction_035_symmetryReduction", "test-symmetryReduction_036_symmetryReduction"}


def cases(paths):
    names = [os.path.basename(p)[:-5] for p in paths]
    return [pytest.param(p, id=n, marks=[pytest.mark.slow] if n in SLOW else []) for p, n in zip(paths, names)]


@pytest.mark.parametrize("path", cases(DET))
def test_detection(path):
    if os.path.basename(path) in NEEDS_MSOLVE and not MSOLVE:
        pytest.skip("needs msolve")
    assert run_detection(path) == []


@pytest.mark.parametrize("path", cases(RED))
def test_reduction(path):
    assert run_reduction(path) == []
