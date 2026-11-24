Installation
============

.. code-block:: sh

   pip install symident

Wheels exist for Linux (x86_64, aarch64, glibc 2.28 or newer; tested on Debian and
Fedora), macOS (arm64) and Windows (x86_64), Python 3.10 to 3.15. They depend on NumPy, SciPy,
SymPy and python-flint, which pip installs with them. Elsewhere, such as on
Intel Macs, pip builds from source with a C++17 compiler and CMake; with OpenMP the kernel
runs on several threads. Development version:
``pip install git+https://github.com/dModverse/symident``.

PEtab support
-------------

:func:`symident.read_petab` needs python-libsbml and PyYAML:

.. code-block:: sh

   pip install "symident[petab]"

The msolve solver
-----------------

`msolve <https://msolve.lip6.fr>`_ serves the global analysis (``detect(mode="global")``,
:meth:`~symident.Symmetries.discrete`), :meth:`~symident.Symmetries.solution_count` and
steady-state analyses (``equilibrate=True``) whose resting states need a coupled nonlinear
polynomial solve. Without it these stop with :class:`~symident.sym.msolve.MsolveMissingError`;
all other analyses run without it. On Linux and macOS

.. code-block:: python

   import symident
   symident.install_msolve()

downloads msolve, GMP, MPFR and FLINT and builds them, in a few minutes, into the per-user
cache: ``~/.cache/symident`` on Linux (or ``$XDG_CACHE_HOME/symident``),
``~/Library/Caches/symident`` on macOS, ``%LOCALAPPDATA%\symident\cache`` on Windows, or
``SYMIDENT_CACHE``. The build needs a C compiler, ``make``, ``m4``, ``curl`` or ``wget``
and network access.

symident takes the first msolve that solves a test system, in this order:
``SYMIDENT_MSOLVE``, the builds in the cache, msolve 0.10 on the ``PATH``. On Windows
:func:`symident.install_msolve` does not build: set ``SYMIDENT_MSOLVE`` to an
``msolve.exe``, or place it in the cache as ``deps-msolve-<version>\bin\msolve.exe`` next
to an empty file ``deps-msolve-<version>\.msolve-complete``.

Use from R
----------

`dMod2 <https://dmodverse.github.io/dMod2/>`_ calls symident for
`symmetryDetection() <https://dmodverse.github.io/dMod2/reference/symmetryDetection.html>`_ and
`symmetryReduction() <https://dmodverse.github.io/dMod2/reference/symmetryReduction.html>`_ through
``reticulate`` and the JSON entry points of ``symident.rjson``. On first use, reticulate installs
symident from PyPI into the Python environment of the R session. When an analysis needs msolve,
dMod2 offers to build it into the cache, on Windows with Rtools. Every environment variable
``SYMIDENT_<NAME>`` of the R session, also one set with ``Sys.setenv()``, reaches symident as the
switch of that name. The
`guide chapter on symmetries <https://dmodverse.github.io/dMod2/articles/guide-symmetries.html>`_
of dMod2 shows the analysis from R.
