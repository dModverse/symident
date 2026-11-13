Reduction
=========

A non-identifiable model can be reduced to identifiable parameters. :func:`symident.reduce`,
or ``res.reduce()``, removes the directions of a detection result: a scaling by fixing one of
its coordinates, a general direction by its invariants, i.e., the combinations that are
constant along it, which become new parameters ``q_1, q_2, ...``. For the result of the
running example in :doc:`detection`:

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
fixing ``s = 1`` and ``kl = 0``, and ``red.summary(detailed=True)`` says why each section meets
every orbit once. ``kl = 0`` lies on the boundary of the parameter
space, which the orbits reach only in a limit; the model is continuous there, so the limit
gives the output of the orbit.

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

- ``open``: an error unless ``allow_open=True``, which leaves the direction in place as an
  unresolved block;
- ``checked``, or a saturation guard that saw the rank still grow: an error unless
  ``allow_checked=True``;
- ``verified``: a warning.

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
   also builds alternative transformations and returns the best; this happens anyway when the
   first transformation has a square root or a fractional power outside blocks with sign
   conditions.
``zero_limits``
   checks for each block whether coordinates can be set to zero in a limit, printed under
   ``Zero limits`` and returned in ``to_dict()["zero_compatibility"]``.
``timeout``, ``work_cap``
   the time and the work per block of the search for a section; reaching either is named in
   ``red.limits``.
``check``
   runs the detection again on the reduced model (``red.checks["loop"]``).

The result
----------

A :class:`~symident.Reduction` holds the transformation and what is known about it:

``trafo``
   the transformation, the identity on every coordinate it leaves.
``remaining``, ``partial``
   the blocks left unreduced, and the reduced blocks whose section covers part of the orbits
   only, those on which the ``sign_conditions``, inequalities in the new parameters, hold.
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
``alternatives``, ``limits``, ``assumptions``
   the alternative transformations, what stopped part of the search, and the assumptions of
   the detection with the domain of the reduction.
