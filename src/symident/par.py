"""Parallel maps: threads for kernel calls (the kernel releases the GIL), forked
processes for Python-heavy tasks such as symbolic steady-state solves."""

import multiprocessing as mp
import sys
from concurrent.futures import ThreadPoolExecutor

from . import state


def thread_map(fn, xs, n):
    """`fn` mapped over `xs` on up to `n` threads, in order."""
    xs = list(xs)
    n = max(1, min(int(n), len(xs)))
    if n == 1:
        return [fn(x) for x in xs]
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(fn, xs))


_FN = None


def _call(i):
    try:
        return ("ok", _FN(_XS[i]))
    except Exception as e:
        return ("err", repr(e))


_XS = None


def process_map(fn, xs, n, on_error=None):
    """fn over xs in forked workers (fn may be a closure); serial with n = 1 or off Unix.

    An item whose worker raised is recomputed serially."""
    global _FN, _XS
    xs = list(xs)
    n = max(1, min(int(n), len(xs)))
    if n == 1 or sys.platform == "win32" or state.env("NOFORK"):
        return [fn(x) for x in xs]
    _FN, _XS = fn, xs
    try:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=n) as pool:
            res = pool.map(_call, range(len(xs)), chunksize=1)
    finally:
        _FN, _XS = None, None
    out = []
    for x, (st, v) in zip(xs, res):
        out.append(v if st == "ok" else (on_error(x) if on_error else fn(x)))
    return out
