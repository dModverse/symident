"""Work limits: every search and count stops at a cap on a number of steps, points, terms or unknowns, so that a
result is the same on every machine. Nothing in the package stops on a clock."""

import contextlib
import math
import os
import re
import subprocess
import threading

# the address space of one call of msolve: past it the call fails instead of the process that waits for it
MSOLVE_MEMORY = 8 << 30


class Meter:
    """Units of work spent against a cap; `what` names the stage, `unit` the units. A meter made by share() spends
    on the parent as well and is over where either is."""

    def __init__(self, what, cap, unit, parent=None):
        self.what, self.cap, self.unit, self.parent = what, cap, unit, parent
        self.spent = 0

    def spend(self, n=1):
        """Adds n units; whether the meter is over the cap afterwards."""
        self.spent += n
        if self.parent is not None:
            self.parent.spend(n)
        return self.over

    @property
    def over(self):
        return self.spent > self.cap or (self.parent is not None and self.parent.over)

    def share(self, what, cap, unit=None):
        """A meter of at most `cap` units of this one."""
        return Meter(what, cap, unit or self.unit, self)

    def reason(self):
        """The cap that ended the stage, as text."""
        m = self
        while m.parent is not None and not m.spent > m.cap:
            m = m.parent
        return f"{m.what} reached the cap of {_num(m.cap)} {m.unit}"


def unlimited(what="", unit="units"):
    return Meter(what, math.inf, unit)


def _num(x):
    return "infinitely many" if x == math.inf else f"{int(x):,}" if float(x).is_integer() else f"{x:g}"


@contextlib.contextmanager
def msolve_memory(gib):
    """The address space of one call of msolve within the block, in GiB; None takes the switch MSOLVE_GB, else keeps
    the current one."""
    global MSOLVE_MEMORY
    old = MSOLVE_MEMORY
    if gib is None:
        from .. import state

        gib = state.env_value("MSOLVE_GB") or None
    if gib is not None:
        MSOLVE_MEMORY = int(float(gib) * (1 << 30))
    try:
        yield
    finally:
        MSOLVE_MEMORY = old


def _limiter(memory):
    """A function that sets the address space `memory` in a child process before it runs the command, at most the
    hard limit of this process; None where the platform has none."""
    try:
        import resource
    except ImportError:
        return None
    if not memory or not hasattr(resource, "RLIMIT_AS"):
        return None

    def apply():
        try:
            top = resource.getrlimit(resource.RLIMIT_AS)[1]
            lim = int(memory) if top == resource.RLIM_INFINITY else min(int(memory), top)
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        except (OSError, ValueError):
            pass

    return apply


def run(cmd, input=None, text=False, memory=None):
    """subprocess.run(cmd, capture_output=True), with at most `memory` bytes of address space for the process."""
    return subprocess.run(cmd, input=input, capture_output=True, text=text, preexec_fn=_limiter(memory))


# ---- the caps ---------------------------------------------------------------------------------

# the work caps of the global analysis; discrete() sets the public ones for one call (caps())
CAPS = {
    # nonzero entries of the matrices of the F4 rounds of one call of msolve, and the unknowns and terms of the input
    "msolve_entries": 4 * 10**8,
    "msolve_unknowns": 400,
    "msolve_terms": 100000,
    # the same for the short counts: the first count of discrete(), the size of a square subsystem, the check of a
    # resting state
    "quick_entries": 6 * 10**7,
    "quick_unknowns": 400,
    "quick_terms": 100000,
    # entries of the matrices over GF(p) that the exact search reduces, and the verified functions past it
    "search_entries": 5 * 10**8,
    "verified_entries": 2 * 10**8,
    # terms of a polynomial of the elimination of the states
    "elimination_terms": 3 * 10**6,
    # products of terms that the Taylor series of the outputs may take, (k + 1)^2 per order k and term
    "series_products": 10**10,
    # permutations of coordinates tested as candidates
    "candidates": 20000,
    # points of the fiber that the reconstruction of the maps solves
    "map_points": 200,
    # products of terms in the Taylor series that the steps of the paths of the monodromy evaluate
    "monodromy_products": 10**9,
    # the terms of the expansion of the equations of a map of the states solved from the Lie derivatives of the outputs
    "state_map_terms": 10**5,
    # the Bezout number of the resting balances that SymPy is given, and the states of their largest coupled block
    "rest_bezout": 4,
    "rest_block": 8,
}


@contextlib.contextmanager
def caps(**kw):
    """The caps `kw` (keys of CAPS) within the block; None keeps a cap."""
    bad = sorted(set(kw) - set(CAPS))
    if bad:
        raise KeyError(bad[0])
    old = dict(CAPS)
    CAPS.update({k: v for k, v in kw.items() if v is not None})
    try:
        yield
    finally:
        CAPS.clear()
        CAPS.update(old)


_LOCAL = threading.local()


def cap(name):
    """The cap `name` in force in this thread: an override of local_caps(), else CAPS."""
    return getattr(_LOCAL, "caps", {}).get(name, CAPS[name])


@contextlib.contextmanager
def local_caps(**kw):
    """The caps `kw` within the block in this thread only, as threads of one count share CAPS."""
    old = dict(getattr(_LOCAL, "caps", {}))
    _LOCAL.caps = {**old, **kw}
    try:
        yield
    finally:
        _LOCAL.caps = old


@contextlib.contextmanager
def quick_msolve():
    """The short caps of msolve (quick_entries, quick_unknowns, quick_terms) within the block, at most those in
    force."""
    with local_caps(
        **{k: min(cap(k), cap("quick" + k[6:])) for k in ("msolve_entries", "msolve_unknowns", "msolve_terms")}
    ):
        yield


def msolve_size(names, eqs):
    """(unknowns, terms) of the msolve input `eqs` (texts) in `names`."""
    terms = 0
    for e in eqs:
        s = str(e).replace(" ", "")
        terms += 1 + sum(1 for i, ch in enumerate(s) if ch in "+-" and i and s[i - 1] not in "^*eE")
    return len(names), terms


def msolve_too_large(names, eqs):
    """Why the system `eqs` in `names` is past the caps of msolve on the size of the input, None where it is within
    them."""
    n, t = msolve_size(names, eqs)
    nu, nt = cap("msolve_unknowns"), cap("msolve_terms")
    if n > nu:
        return f"a system of {n} unknowns, past the cap of {nu} unknowns of msolve"
    if t > nt:
        return f"a system of {t:,} terms, past the cap of {nt:,} terms of msolve"
    return None


# the matrix of a round of F4 in the output of msolve -v 2, rows x columns and density, written before the linear
# algebra of the round
_MATRIX = re.compile(rb"(\d+) x (\d+)\s+([\d.]+)%")


def run_msolve(cmd, text=True):
    """work.run() of an msolve command, with the address space MSOLVE_MEMORY. The nonzero entries of the matrices of
    the F4 rounds (rows times columns times density, from the output of msolve) count against the cap
    msolve_entries: the process ends at the first matrix past it, before the linear algebra of that matrix, and the
    result has returncode 1 and `capped` the reason. The count of a finished call decides as well, so the outcome
    does not depend on when the output is read."""
    most = cap("msolve_entries")
    out = []
    total, buf, done = 0, b"", 0
    # leaving the block closes both pipes and waits for the process
    with subprocess.Popen(
        list(cmd) + ["-v", "2"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=_limiter(MSOLVE_MEMORY)
    ) as proc:
        reader = threading.Thread(target=lambda: out.append(proc.stdout.read()))
        reader.start()
        try:
            fd = proc.stderr.fileno()
            while True:
                chunk = os.read(fd, 1 << 16)
                if not chunk:
                    break
                buf += chunk
                for m in _MATRIX.finditer(buf, done):
                    total += int(m.group(1)) * int(m.group(2)) * float(m.group(3)) / 100
                    done = m.end()
                if total > most:
                    proc.kill()
                    break
        except BaseException:
            proc.kill()
            raise
        finally:
            reader.join()
    stdout, stderr = out[0] if out else b"", buf
    if text:
        stdout, stderr = stdout.decode(errors="replace"), stderr.decode(errors="replace")
    if total > most:
        r = subprocess.CompletedProcess(cmd, 1, stdout, stderr)
        r.capped = f"msolve reached the cap msolve_entries of {int(most):,} nonzero entries of the matrices of F4"
    else:
        r = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        r.capped = None
    r.entries = total
    return r


# ---- the work counter of the searches over GF(p) -----------------------------------------------

# entries of the matrices whose kernels or echelon forms the searches compute, since the start of the process
_TICKS = [0]


def ticks():
    """The entries of the matrices over GF(p) reduced so far in this process, the clock of the searches."""
    return _TICKS[0]


def tick(shape):
    """Counts a matrix of the given shape, or a number of entries."""
    _TICKS[0] += int(shape[0]) * int(shape[1]) if isinstance(shape, tuple) else int(shape)
