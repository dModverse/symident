"""Parallel maps: threads for kernel calls, which release the GIL, forked processes for Python-heavy tasks."""

import itertools
import multiprocessing as mp
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from . import session, state
from .errors import swallow


def thread_map(fn, xs, n):
    """`fn` mapped over `xs` on up to `n` threads, in order."""
    xs = list(xs)
    n = max(1, min(int(n), len(xs)))
    if n == 1:
        return [fn(x) for x in xs]
    with ThreadPoolExecutor(max_workers=n) as ex:
        return list(ex.map(fn, xs))


_JOBS = {}
_TOKENS = itertools.count()


def _call(token, i):
    fn, xs = _JOBS[token]
    m = session.marks()
    try:
        return ("ok", fn(xs[i]), session.notes_since(m))
    except Exception as e:
        return ("err", repr(e), session.notes_since(m))


def process_map(fn, xs, n, on_error=None):
    """fn over xs in forked workers, serial with n = 1, off Unix or under NOFORK; errors as on the serial path."""
    xs = list(xs)
    n = max(1, min(int(n), len(xs)))
    if n == 1 or sys.platform == "win32":
        return [_serial(fn, x, on_error) for x in xs]
    token = next(_TOKENS)
    _JOBS[token] = (fn, xs)
    try:
        with ProcessPoolExecutor(max_workers=n, mp_context=mp.get_context("fork")) as ex:
            res = list(ex.map(_call, [token] * len(xs), range(len(xs))))
    except BrokenProcessPool as e:
        raise RuntimeError("a worker process died, possibly out of memory") from e
    finally:
        _JOBS.pop(token, None)
    out = []
    for x, (st, v, notes) in zip(xs, res):
        session.merge_notes(notes)
        if st == "ok":
            out.append(v)
        elif on_error is None or state.env("DEBUG"):
            out.append(fn(x))
        else:
            session.note("errors", f"worker: {v[:200]}")
            out.append(on_error(x))
    return out


def _serial(fn, x, on_error):
    if on_error is None:
        return fn(x)
    try:
        return fn(x)
    except Exception as exc:
        swallow(exc, "worker")
        return on_error(x)
