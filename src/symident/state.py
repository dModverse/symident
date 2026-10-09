"""Run-wide options and diagnostic switches shared between nested analyses."""

import contextlib
import os

_OPTIONS = {}
_SWITCHES = {}
_MISSING = object()


@contextlib.contextmanager
def _scoped(store, kw):
    old = {k: store.get(k, _MISSING) for k in kw}
    for k, v in kw.items():
        if v is None:
            store.pop(k, None)
        else:
            store[k] = v
    try:
        yield
    finally:
        for k, v in old.items():
            if v is _MISSING:
                store.pop(k, None)
            else:
                store[k] = v


def get(name, default=None):
    """The option `name`, else `default`."""
    return _OPTIONS.get(name, default)


def options(**kw):
    """Set options for the duration of the block; None removes an option."""
    return _scoped(_OPTIONS, kw)


def switches(**kw):
    """Set switches for the duration of the block; None removes a switch."""
    return _scoped(_SWITCHES, kw)


def env_value(name, default=None):
    """The switch `name`: set by `switches()`, else the environment variable
    SYMIDENT_<name>."""
    if name in _SWITCHES:
        v = _SWITCHES[name]
        return v if isinstance(v, str) else str(int(v)) if isinstance(v, bool) else str(v)
    return os.environ.get("SYMIDENT_" + name, default)


def env(name):
    """Whether the switch `name` is on."""
    v = env_value(name)
    return bool(v) and v != "0"
