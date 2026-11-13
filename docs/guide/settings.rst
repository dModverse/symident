Settings and diagnostics
========================

Reconstruction settings
-----------------------

:func:`symident.reconst_control` builds the ``control`` argument of :func:`~symident.detect`,
which governs the *saturation*, i.e., the choice of the Lie order at which the rank stops
growing, and the reconstruction of closed forms. The most used
settings, with their defaults in parentheses, are:

``degree_cap``, ``degree_cap_max``
   the degree bound of the dense fit (4), and the degree up to which a wider entry is probed
   and fitted (8).
``gap_order_cap``
   the highest gap order, the number of power-series coefficients kept for the time between
   events after the start (8).
``timeout``
   the time limit of the reconstruction in seconds, once per detection (no limit).
``seed``
   the seed of every random choice (0).

A cap that ``info["limits"]`` names can be raised; :func:`~symident.reconst_control` lists
the others. Model, settings and seed fix the result for any number of ``cores``.

Diagnostic output and switches
------------------------------

``detect(trace=True)`` prints the time of each stage, and ``trace="full"`` also the
diagnostics of the Lie order. Further diagnostics and route choices are *switches*. A switch is
set as an environment variable ``SYMIDENT_<NAME>`` or, taking precedence, as a keyword of
:func:`symident.switches` in a ``with`` block:

.. code-block:: python

   import symident as si

   with si.switches(DEBUG=True):
       ...

The most used switches are

``DEBUG``
   raise the exceptions that a route would record in ``info["errors"]``.
``DIAG``
   every diagnostic output on.

A keyword set to ``None`` lets the environment variable decide again. Switches change how a
result is reached, not what it proves. ``info["settings"]["switches"]`` records the active
ones, and an unknown name warns. :doc:`../api` lists all switches, and :doc:`../install`
covers ``SYMIDENT_MSOLVE``, ``SYMIDENT_CACHE`` and the use from R.
