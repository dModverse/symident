Reduction
=========

A non-identifiable model can be reduced to identifiable parameters. :func:`symident.reduce`,
or ``res.reduce()``, describes the model in the invariants of the directions of a detection
result, i.e., the combinations of the coordinates that are constant along them, which become
new parameters ``q_1, q_2, ...``. The invariants are coordinates on the quotient of the
parameter space by the orbits, a *chart*. A scaling (``Symmetry.kind``, :doc:`detection`) is
removed by fixing one of its coordinates to one, a translation by its linear invariants with a
coordinate fixed to zero on a face of the positive orthant, and a curved direction by its
invariants. For the result of the running example in :doc:`detection`:

.. code-block:: python

   red = res.reduce()
   print(red)

.. code-block:: text

   Reduced 2 of 2 directions.

   Trafo (non-identity entries)
     k1  = q_1
     kb  = q_2
     kin = q_3
     kl  = 0
     s   = 1

   Invariants:
     q_1 = k1 + kl
     q_2 = k1*kb/(k1 + kl)
     q_3 = k1*kin/(k1 + kl)

   Sections:
     {X₁}  s = 1
     {X₂}  kl = 0
     summary(detailed=True) explains each section.

``q_1`` is the outflow rate of ``x1``, and ``q_2`` and ``q_3`` are the basal and the
stimulated flux into ``x2`` in units of the signal; ``k2`` stays. Each line under
``Sections`` is one *block* of jointly reduced directions. Along a direction, the coordinates
move on a curve of equal prediction, its *orbit*; a *section* picks one point per orbit, here by
fixing ``s = 1`` and ``kl = 0``, and the trafo is this section in the coordinates of the chart.
``red.summary(detailed=True)`` says why each section meets every orbit once. A chart is
returned only where it is valid on the declared domain, by default the positive orthant. ``kl = 0`` lies on the boundary of the parameter
space, which the orbits reach only in a limit; the model is continuous there, so the limit
gives the output of the orbit.

A direction that shifts some coordinates and rescales positive ones at rates that depend only on
coordinates it does not move, such as an unknown offset of a measurement that also shifts other
coordinates, is a *linear flow* (:doc:`../method`, "Linear flows"). It is removed by fixing as
many moved coordinates as the block has directions, a coordinate that enters only the observables
first: a linear one to 0, a positive one to 1. The other moved coordinates keep their names and
stand for the invariants listed under ``Invariants``.

``red.trafo`` is the transformation as a dict of strings:

.. code-block:: python

   red.trafo
   # {'k1': 'q_1', 'k2': 'k2', 'kb': 'q_2', 'kin': 'q_3', 'kl': '0', 's': '1'}

An entry for a symbol of the condition grid goes into the grid cells. Under ``equilibrate``
an entry named like a state pins the resting value, and a numeric one goes into ``fixed``.
``check=True`` applies the trafo this way and runs the detection again on the reduced model.

Levels and unproven directions
------------------------------

``red.level`` is ``exact`` when every claim is proven, ``verified`` or ``checked`` when a
direction rests on a closed form of that level, and ``open`` when a claim of a section is not
proven (named in the printout). Unproven directions are handled as follows:

- ``open``: a warning; the direction stays in place as an unresolved block;
- ``checked``, or a saturation guard that saw the rank still grow: a warning; the reduction uses
  the closed form;
- ``verified``: a warning.

``allow_open=False`` and ``allow_checked=False`` make the first two an error.

Options
-------

``fixed``
   coordinates to keep fixed, by default the ``gauge`` of the detection; a scaling that moves a
   fixed coordinate is removed by it.
``positive``
   the domain of the sections, by default that of the detection.
``d_poly``, ``d_darboux``, ``d_exp``, ``separable``
   the degree caps of the polynomial, Darboux and exponential invariants (3, 2, 2), and the
   search for invariants by separable quadrature; ``d_exp=0`` skips the exponential factors
   and integrating factors.
``alternatives``
   also returns, in ``red.alternatives``, the charts of the other valid choices of the
   coordinates to fix, and returns the best; the search for them runs anyway when the first
   chart has a square root or a fractional power outside blocks with sign conditions.
``zero_limits``
   checks for each block whether coordinates can be set to zero in a limit, printed under
   ``Zero limits`` and returned in ``to_dict()["zero_compatibility"]``.
``timeout``, ``work_cap``
   the wall time in seconds (600 by default, ``math.inf`` for none) and the work per block of the
   search for a section; reaching either is named in ``red.limits``.
``check``
   runs the detection again on the reduced model (``red.checks["loop"]``).
``branch``
   with a discrete symmetry of a detection with ``mode="global"`` (:doc:`detection`), the side of
   its fixed set, such as ``"k01 > k12"``, whose chart becomes the trafo: free values in the positive
   orthant map one to one onto that side, e.g. ``k01 = delta + k12`` with ``delta > 0``, chained with
   the section of the continuous symmetries. Without it every branch is listed in
   ``red.alternatives`` and the trafo is the section alone.

The result
----------

A :class:`~symident.Reduction` holds the transformation and what is known about it:

``trafo``
   the transformation, the identity on every coordinate it leaves.
``remaining``, ``partial``
   the directions left unreduced, and the removed directions whose chart covers the domain only
   in part, on the orbits where the ``sign_conditions``, inequalities in the new parameters,
   hold.
``blocks``
   one entry per block of jointly reduced directions, with ``status`` (``reduced``,
   ``fixed``, ``invariant_only``, ``unresolved``), ``level``, the section and its
   certificates: ``reach`` (every set of equal invariants is one orbit), ``injective`` (every
   orbit meets the section at most once), ``coverage`` (``total`` when every orbit meets it) and
   ``unproven`` (the claims not proven).
``checks``
   the check of the transformation over Q (``pins``, ``invariants``) and, with
   ``check=True``, the detection on the reduced model (``loop``), whose ``ok`` is None where it
   could not run.
``discrete``
   the discrete symmetries of the detection, the fundamental domain and, under ``branches``, per
   side of the fixed set its name, the ``trafo``, the coordinate it ``solved``, the new parameter,
   the ``boundary``, ``injective``, ``reach`` and ``unproven``; ``branch`` is the chosen one.
``alternatives``, ``limits``, ``assumptions``
   the charts of the other valid choices of the fixed coordinates, what stopped part of the
   search, and the assumptions of the detection with the domain of the reduction.
