Installation
============

.. code-block:: sh

   pip install symident

Wheels are built for Linux (x86_64, aarch64, glibc 2.28 or newer; tested on Ubuntu, Debian
and Fedora), macOS (x86_64, arm64) and Windows (x86_64), for Python 3.10 to 3.14. Elsewhere
pip builds from source, which needs a C++17 compiler and CMake; with OpenMP the kernel runs
on several threads. The development version installs with
``pip install git+https://github.com/dModverse/symident``.

msolve
------

Steady-state analyses (``equilibrate=True``) whose resting state is a coupled nonlinear
polynomial system use `msolve <https://msolve.lip6.fr>`_. On Linux and macOS

.. code-block:: python

   import symident
   symident.install_msolve()

downloads and builds msolve with GMP, MPFR and FLINT into a per-user cache
(``~/.cache/symident`` on Linux, ``~/Library/Caches/symident`` on macOS, or
``SYMIDENT_CACHE``). The build needs a C compiler, ``make``, ``m4`` and ``curl`` or
``wget``, and takes a few minutes. symident then finds the build by itself.

msolve is looked up in this order: the environment variable ``SYMIDENT_MSOLVE``, the
builds of :func:`symident.install_msolve` in the cache, and ``msolve`` on the ``PATH`` if
it is a tested version (0.10). Each candidate has to solve a test system. On Windows,
install msolve by other means (e.g. MSYS2) and set ``SYMIDENT_MSOLVE``.

Without msolve, models that need it stop with
:class:`~symident.sym.msolve.MsolveMissingError`; all other analyses run without it.

From R
------

The R package dMod2 calls symident for ``symmetryDetection()`` and
``symmetryReduction()`` through ``reticulate`` and installs symident from this repository
into reticulate's Python environment on first use.
