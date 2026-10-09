import os
import sys

import pytest

from symident.par import process_map, thread_map


def test_thread_map_keeps_the_order():
    assert thread_map(lambda x: x * x, range(10), 4) == [x * x for x in range(10)]
    assert thread_map(lambda x: x + 1, [], 4) == []


@pytest.mark.skipif(sys.platform == "win32", reason="no fork")
def test_process_map_runs_a_closure_in_forked_workers():
    offset = 7
    assert process_map(lambda x: x + offset, range(6), 3) == [x + 7 for x in range(6)]


@pytest.mark.skipif(sys.platform == "win32", reason="no fork")
def test_process_map_recomputes_a_failed_item():
    parent = os.getpid()

    def fn(x):
        if x == 2 and os.getpid() != parent:
            raise ValueError("in a worker")
        return 10 * x

    assert process_map(fn, range(4), 2) == [0, 10, 20, 30]
    assert process_map(fn, range(4), 2, on_error=lambda x: -x) == [0, 10, -2, 30]
