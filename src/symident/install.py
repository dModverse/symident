"""External programs: msolve built from source into a per-user cache, and its lookup."""

import functools
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from importlib import resources

MSOLVE_VERSION = "0.10.1"
_TESTED = "0.10."


def cache_dir():
    """The per-user cache of symident: `SYMIDENT_CACHE`, else the platform's cache directory."""
    if os.environ.get("SYMIDENT_CACHE"):
        return os.environ["SYMIDENT_CACHE"]
    if sys.platform == "darwin":
        return os.path.expanduser("~/Library/Caches/symident")
    if sys.platform == "win32":
        return os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "symident", "cache")
    return os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "symident")


@functools.cache
def msolve_check(path):
    """The version of the msolve at `path` if it solves a test system correctly, else None."""
    if not path or not os.path.isfile(path):
        return None
    with tempfile.TemporaryDirectory() as td:
        fi, fo = os.path.join(td, "in.ms"), os.path.join(td, "out.ms")
        with open(fi, "w") as fh:
            fh.write("x, y\n65521\nx^2+y-7,\nx*y-6\n")
        try:
            r = subprocess.run([path, "-f", fi, "-o", fo, "-P", "1", "-t", "1"],
                               capture_output=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return None
        if r.returncode != 0 or not os.path.exists(fo):
            return None
        with open(fo) as fh:
            out = fh.read().replace(" ", "").replace("\n", "")
    if not out.startswith("[0,[65521,2,3,") or "[36,0,65514,1]" not in out:
        return None
    try:
        v = subprocess.run([path, "-V"], capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        v = ""
    return v or "unknown"


def _cache_bins():
    dirs = [d for d in glob.glob(os.path.join(cache_dir(), "deps-msolve-*"))
            if os.path.exists(os.path.join(d, ".msolve-complete"))]

    def key(d):
        return tuple(int(x) if x.isdigit() else 0 for x in os.path.basename(d)[12:].split("."))
    return [os.path.join(d, "bin", "msolve") for d in sorted(dirs, key=key, reverse=True)]


def find_msolve():
    """The msolve to use and where it was found, (path, source), or (None, None).

    Searched in order: `SYMIDENT_MSOLVE`, the builds of
    :func:`install_msolve` in the cache, then `msolve` on the `PATH` if it is a
    tested version. Each candidate must solve a test system."""
    p = os.environ.get("SYMIDENT_MSOLVE", "")
    if p and msolve_check(p):
        return p, "SYMIDENT_MSOLVE"
    for p in _cache_bins():
        if msolve_check(p):
            return p, "cache"
    p = shutil.which("msolve")
    if p:
        v = msolve_check(p)
        if v and v.startswith(_TESTED):
            return p, "PATH"
    return None, None


def install_msolve(dir=None, version=MSOLVE_VERSION, jobs=None, quiet=False, ask=None):
    """Build msolve from source into a per-user cache.

    msolve solves the coupled resting-state systems of ``detect(..., equilibrate=True)``.
    It is built with static GMP, MPFR and FLINT; the build needs a C compiler, ``make``,
    ``m4`` and ``curl`` or ``wget``, network access and a few minutes. Not available on
    Windows. symident finds the build without further settings.

    Parameters
    ----------
    dir : str, optional
        Cache directory; defaults to :func:`cache_dir`.
    version : str
        msolve release.
    jobs : int, optional
        Parallel ``make`` jobs; defaults to the available cores, at most 8.
    quiet : bool
        Suppress the build output.
    ask : bool, optional
        Ask before downloading; defaults to whether the session is interactive.

    Returns
    -------
    str or None
        Path of the msolve executable, None if the user declined.
    """
    if jobs is None:
        jobs = max(1, min(8, len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity")
                          else (os.cpu_count() or 2)))
    if sys.platform == "win32":
        raise RuntimeError("install_msolve() is not supported on Windows; install msolve "
                           "(e.g. through MSYS2) and set SYMIDENT_MSOLVE to its path.")
    if not (shutil.which("curl") or shutil.which("wget")):
        raise RuntimeError("install_msolve() needs curl or wget to download the sources.")
    missing = [t for t in ("make", "m4") if not shutil.which(t)]
    if missing:
        raise RuntimeError(f"install_msolve() needs {' and '.join(missing)} for the build.")
    dir = os.path.abspath(dir or cache_dir())
    if ask is None:
        ask = sys.stdin is not None and sys.stdin.isatty()
    if ask:
        reply = input(f"This downloads and builds msolve {version} with GMP, MPFR and FLINT\n"
                      f"into: {dir}\nIt requires network access and takes a few minutes. "
                      "Proceed? [Y/n] ").strip().lower()
        if reply not in ("", "y", "yes"):
            print("Aborted; nothing was downloaded or written.")
            return None
    os.makedirs(dir, exist_ok=True)
    script = resources.files("symident").joinpath("tools", "build-msolve.sh")
    with resources.as_file(script) as sp:
        env = dict(os.environ, SYMIDENT_LIBS_CACHE=dir, SYMIDENT_MSOLVE_VERSION=str(version),
                   SYMIDENT_JOBS=str(int(jobs)), CC=os.environ.get("CC", "cc"))
        r = subprocess.run(["sh", str(sp)], env=env, stdout=subprocess.PIPE, text=True,
                           stderr=subprocess.DEVNULL if quiet else None)
    if r.returncode != 0:
        raise RuntimeError("the msolve build failed; see the messages above.")
    lines = [line for line in r.stdout.splitlines() if line.strip()]
    path = os.path.join(lines[-1], "bin", "msolve") if lines else ""
    msolve_check.cache_clear()
    if not msolve_check(path):
        raise RuntimeError("the build reported success but produced no working msolve.")
    if not quiet:
        print(f"msolve installed in {path}")
    return path
