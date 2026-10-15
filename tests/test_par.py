import os
import sys

import pytest

from symident import state
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
    with state.switches(DEBUG="0"):
        assert process_map(fn, range(4), 2, on_error=lambda x: -x) == [0, 10, -2, 30]


def test_process_map_gives_on_error_on_the_serial_path_too():
    def fn(x):
        if x == 1:
            raise ValueError("bad item")
        return x
    with state.switches(DEBUG="0"):
        assert process_map(fn, range(3), 1, on_error=lambda x: None) == [0, None, 2]
    with pytest.raises(ValueError):
        process_map(fn, range(3), 1)
    with state.switches(DEBUG="1"), pytest.raises(ValueError):
        process_map(fn, range(3), 1, on_error=lambda x: None)


@pytest.mark.skipif(sys.platform == "win32", reason="no fork")
def test_process_map_raises_when_a_worker_dies():
    def fn(x):
        if x == 1:
            os._exit(1)
        return x
    with pytest.raises(RuntimeError, match="died"):
        process_map(fn, range(4), 2)
