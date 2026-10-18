"""The state one detection shares across nested analyses: seed, reconstruction deadline, limits that fired."""

import contextlib
import hashlib
import json
import random

from . import state


class Session:
    """Seeds of one detection, derived from the control seed and the model."""

    def __init__(self, seed=0, model=""):
        self.root = hashlib.blake2b(f"{int(seed)}\0{model}".encode(), digest_size=16).digest()
        self.deadline = None
        self.notes = {"limits": [], "errors": [], "branches": []}
        self.control = None
        self.primes = None
        self.depth = 0
        self.vprimes = {}

    def seed(self, name, *key):
        """A 63-bit seed for the stage `name` and the key, the same in every process."""
        h = hashlib.blake2b(self.root, digest_size=8)
        h.update(repr((name,) + tuple(key)).encode())
        return int.from_bytes(h.digest(), "little") >> 1

    def rng(self, name, *key):
        """A generator for the stage `name` and the key."""
        return random.Random(self.seed(name, *key))


def note(kind, what):
    """Record `what` under `kind` ("limits", "errors") once in the running session."""
    s = state.get("session")
    if s is not None and what not in s.notes[kind]:
        s.notes[kind].append(what)


def marks():
    """Lengths of the session's records, for notes_since()."""
    s = state.get("session")
    return None if s is None else {k: len(v) for k, v in s.notes.items()}


def notes_since(m):
    """The records added after marks() returned `m`."""
    s = state.get("session")
    return None if s is None or m is None else {k: v[m[k]:] for k, v in s.notes.items()}


def merge_notes(new):
    """Add records that a worker process made."""
    for k, vs in (new or {}).items():
        for v in vs:
            note(k, v)


def current():
    """The session of the running detection; outside one, the session of seed 0."""
    return state.get("session") or Session()


def _model_text(f, g):
    def strs(d):
        return {str(k): str(v) for k, v in dict(d).items()}
    gs = [strs(x) for x in g] if isinstance(g, (list, tuple)) else strs(g or {})
    return json.dumps([strs(f or {}), gs], sort_keys=True)


@contextlib.contextmanager
def opened(kw):
    """A session for the detection with keyword arguments `kw` unless one is running."""
    if state.get("session") is not None:
        yield state.get("session")
        return
    from .options import reconst_control
    control = {**reconst_control(), **dict(kw.get("control") or {})}
    s = Session(control["seed"], _model_text(kw.get("f"), kw.get("g")))
    s.control = control
    with state.options(session=s):
        yield s
