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
    """Set switches for the duration of a ``with`` block.

    Parameters
    ----------
    **kw
        Switch names (see the user guide) with their values; a value sets the switch,
        ``False`` or ``None`` clears it. A switch set here wins over the
        environment variable ``SYMIDENT_<NAME>``.

    Examples
    --------
    >>> import symident as si
    >>> with si.switches(DEBUG=True, TIMING=True):  # doctest: +SKIP
    ...     si.detect(model)
    """
    return _scoped(_SWITCHES, kw)


# every switch, set as SYMIDENT_<NAME> or by switches(<NAME>=...): kind and meaning
SWITCHES = {
    "DEBUG": ("debug", "exceptions a route would record in info['errors'] are raised, except failures of SymPy"),
    "DIAG": ("debug", "every diagnostic switch on"),
    "TIMING": ("diag", "stage times"),
    "LIEDIAG": ("diag", "saturation, Lie order caps and sufficiency proofs"),
    "ROUTEDIAG": ("diag", "errors inside the narrow and forward routes"),
    "FWDDIAG": ("diag", "the forward seed of the resting states"),
    "SOLVEDIAG": ("diag", "each resting-state solve"),
    "HOMDIAG": ("diag", "the homotopy route"),
    "BANKDIAG": ("diag", "the sample bank of the dense fit"),
    "STATICDIAG": ("diag", "the static subsystem fields"),
    "REDDIAG": ("diag", "the reduction solve"),
    "JOINTDIAG": ("diag", "the joint resting-state rows"),
    "NOFIELDS": ("route", "no invariant-field routes (global symmetries, field proof)"),
    "NOJET": ("route", "no jet route inside the narrow route"),
    "NOGROUP": ("route", "no translation groups in the narrow route"),
    "NOFWDSEED": ("route", "no forward seed of the resting states"),
    "NODIAG": ("route", "no diagonal start: resting conditions continued from the forward one only"),
    "LIEPLATEAU": ("number", "orders the rank stays flat before the saturation ends (3)"),
    "LIEPLATEAU_BLOCK": ("number", "the plateau per condition block; 0 for one plateau"),
    "VERIFY_MARGIN": ("number", "orders the saturation guard checks past the reported one (6)"),
    "JETWORK": ("number", "characters of jets the jet route may form (1000000)"),
    "DIRECTSECS": ("number", "seconds msolve may take for the direct kernel (120)"),
    "BLOCKMB": ("number", "megabytes of Taylor values per column block of the kernel (256); environment only"),
    "MSOLVE": ("install", "path of the msolve executable"),
    "CACHE": ("install", "directory of the symident cache"),
    "MSOLVE_VERSION": ("install", "msolve version install_msolve() builds"),
    "GMP_VERSION": ("install", "GMP version install_msolve() builds"),
    "MPFR_VERSION": ("install", "MPFR version install_msolve() builds"),
    "FLINT_VERSION": ("install", "FLINT version install_msolve() builds"),
    "JOBS": ("install", "parallel jobs of install_msolve()"),
    "OMP_STUB_H": ("install", "guard of the OpenMP stub header of the msolve build"),
}


def env_value(name, default=None):
    """The switch `name`: set by `switches()`, else the environment variable SYMIDENT_<name>."""
    if name in _SWITCHES:
        v = _SWITCHES[name]
        return v if isinstance(v, str) else str(int(v)) if isinstance(v, bool) else str(v)
    return os.environ.get("SYMIDENT_" + name, default)


def env(name):
    """Whether the switch `name` is on; DIAG turns every diagnostic switch on."""
    v = env_value(name)
    if bool(v) and v != "0":
        return True
    return SWITCHES.get(name, ("",))[0] == "diag" and name != "DIAG" and env("DIAG")


def active():
    """The switches that are set, by name."""
    names = {k[len("SYMIDENT_"):] for k in os.environ if k.startswith("SYMIDENT_")} | set(_SWITCHES)
    return {n: env_value(n) for n in sorted(names) if env_value(n) not in (None, "")}


def check_names():
    """Warn about SYMIDENT_* variables and switches that name no switch."""
    import warnings
    unknown = [n for n in active() if n not in SWITCHES]
    if unknown:
        warnings.warn("unknown switch " + ", ".join("SYMIDENT_" + n for n in unknown) +
                      "; see symident.state.SWITCHES", stacklevel=4)
