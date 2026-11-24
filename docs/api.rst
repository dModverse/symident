API reference
=============

.. currentmodule:: symident

Model and analyses
------------------

.. autoclass:: Model
   :members: from_petab

.. autofunction:: read_petab

.. autoexception:: PEtabError

.. autofunction:: detect

.. autofunction:: reduce

Results
-------

.. autoclass:: Symmetries
   :members: summary, unproven, discrete, solution_count, reduce, to_dict

.. autoclass:: Symmetry
   :members: sympy

.. autoclass:: Reduction
   :members: summary, to_dict

Settings and programs
---------------------

.. autofunction:: reconst_control

.. autofunction:: switches

.. autofunction:: install_msolve

.. autofunction:: symident.install.find_msolve

.. autofunction:: symident.install.cache_dir

.. autoexception:: symident.sym.msolve.MsolveMissingError

Switches
--------

Each switch is an environment variable ``SYMIDENT_<NAME>`` or a keyword of
:func:`switches`.

Debugging:

``DEBUG``
   exceptions a route would record in ``info["errors"]`` are raised, except failures of SymPy.
``DIAG``
   every diagnostic switch on.
``KERNEL_MODE``
   ``forward`` or ``reverse`` fixes the mode of the kernel rows, ``check`` builds every call in both and raises
   where the rows differ; environment only.

Diagnostic output:

``TIMING``
   stage times.
``LIEDIAG``
   saturation, Lie order caps and sufficiency proofs.
``ROUTEDIAG``
   errors inside the narrow and forward routes.
``FWDDIAG``
   the forward seed of the resting states.
``SOLVEDIAG``
   each resting-state solve.
``HOMDIAG``
   the homotopy route.
``BANKDIAG``
   the sample bank of the dense fit.
``STATICDIAG``
   the static subsystem fields.
``REDDIAG``
   the reduction solve.
``JOINTDIAG``
   the joint resting-state rows.

Routes:

``NOFIELDS``
   no invariant-field routes (global symmetries, field proof).
``NOJET``
   no jet route inside the narrow route.
``NOGROUP``
   no translation groups in the narrow route.
``NOFWDSEED``
   no forward seed of the resting states.
``NODIAG``
   no diagonal start: resting conditions continued from the forward one only.
``NOSHARED``
   a tape per condition also where the conditions differ only in the values of symbols.
``NOSELECT``
   every condition analyzed, also where many conditions add nothing to the rank.

Numbers:

``LIEPLATEAU``
   orders the rank stays flat before the saturation ends (3).
``LIEPLATEAU_BLOCK``
   the plateau per condition block; 0 for one plateau.
``STACKEDPLATEAU``
   in the first pass over the stacked rows of many conditions, the plateau as a fraction of the order the
   rank last rose at, consecutive orders without change (0.5); it decides no claim: that pass settles a rank
   only by proven directions or invariant fields.
``VERIFY_MARGIN``
   orders the saturation guard checks past the reported one (6).
``JETWORK``
   characters of jets the jet route may form (1000000).
``BLOCKMB``
   megabytes of Taylor values per column block of the kernel (an eighth of the physical memory over the
   cores, 256 to 1024); environment only.
``JETMB``
   megabytes of the jets the kernel keeps for calls at higher orders (eight column blocks); environment only.
``MSOLVE_GB``
   gigabytes of address space of one call of msolve in discrete() where ``memory`` is not given (8).
``FIBER_TERMS``
   terms of the equations of the fiber that one count of discrete() takes (400000).

Installation of msolve:

``MSOLVE``
   path of the msolve executable.
``CACHE``
   directory of the symident cache.
``MSOLVE_VERSION``
   msolve version install_msolve() builds.
``GMP_VERSION``
   GMP version install_msolve() builds.
``MPFR_VERSION``
   MPFR version install_msolve() builds.
``FLINT_VERSION``
   FLINT version install_msolve() builds.
``JOBS``
   parallel jobs of install_msolve().
