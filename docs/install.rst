Installation
============

.. code-block:: sh

   pip install symident

Wheels are built for Linux (x86_64, aarch64, glibc 2.28 or newer; tested on Debian
and Fedora), macOS (x86_64, arm64) and Windows (x86_64), for Python 3.10 to 3.14. Elsewhere
pip builds from source, which needs a C++17 compiler and CMake; with OpenMP the kernel runs
on several threads. The development version installs with
``pip install git+https://github.com/dModverse/symident``.

PEtab
-----

Reading PEtab problems (:func:`symident.read_petab`) needs python-libsbml and PyYAML:

.. code-block:: sh

   pip install "symident[petab]"

msolve
------

Steady-state analyses (``equilibrate=True``) whose resting states need a coupled nonlinear
polynomial solve, and :meth:`~symident.Symmetries.solution_count`, use
`msolve <https://msolve.lip6.fr>`_. Without it they stop with
:class:`~symident.sym.msolve.MsolveMissingError`; all other analyses run without it.

On Linux and macOS

.. code-block:: python

   import symident
   symident.install_msolve()

downloads msolve, GMP, MPFR and FLINT and builds them into the per-user cache:
``~/.cache/symident`` on Linux (or ``$XDG_CACHE_HOME/symident``),
``~/Library/Caches/symident`` on macOS, ``%LOCALAPPDATA%\symident\cache`` on Windows, or
``SYMIDENT_CACHE``. The build needs a C compiler, ``make``, ``m4``, ``curl`` or ``wget``
and network access, and takes a few minutes.

symident uses the first msolve that solves a test system, in this order:
``SYMIDENT_MSOLVE``, the builds in the cache, msolve 0.10 on the ``PATH``. On Windows
:func:`symident.install_msolve` does not build: set ``SYMIDENT_MSOLVE`` to an
``msolve.exe``, or place it in the cache as ``deps-msolve-<version>\bin\msolve.exe`` next
to an empty file ``deps-msolve-<version>\.msolve-complete``.

From R
------

The R package dMod2 calls symident for ``symmetryDetection()`` and
``symmetryReduction()`` through ``reticulate``, by the JSON entry points of
``symident.rjson``; reticulate installs symident from PyPI into its Python environment on
first use. When an analysis needs msolve, dMod2 offers to build it into the cache, on
Windows with Rtools. Every environment variable ``SYMIDENT_<NAME>`` and every R option
``dMod.sym.<name>`` reaches symident as the switch of that name.
