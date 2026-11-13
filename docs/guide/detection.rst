Detection
=========

:func:`symident.detect` computes the rank of the observability-identifiability matrix (OIM,
:doc:`../method`), whose kernel holds the non-identifiable directions, and returns them as
:class:`~symident.Symmetries`. For the running
example of :doc:`models`, ``equilibrate=True`` starts every condition at a steady state with
the inputs at zero, and ``reconstruct=True`` asks for the directions in closed form:

.. code-block:: python

   import symident as si

   m = si.Model({"x1": "kb + kin*u - (k1 + kl)*x1", "x2": "k1*x1 - k2*x2", "u": "0"},
                {"y": "s*x2"},
                inputs=["u"],
                events=[{"var": "u", "time": 0, "value": "dose", "method": "replace"}],
                conditions={"rows": ["low", "high"], "cols": {"dose": [1, 10]}})
   res = si.detect(m, equilibrate=True, reconstruct=True)
   print(res)

.. code-block:: text

   Result:  rank 4 / 6  |  2 non-identifiable directions

   Generators  X = Σᵢ ξ(i) ∂ᵢ

   Scalings:
     X₁  (exact)
         ξ(kb)  :  + kb
         ξ(kin) :  + kin
         ξ(s)   :  - s

   General:
     X₂  (exact)
         ξ(k1)  :  + k1
         ξ(kb)  :  - kb
         ξ(kin) :  - kin
         ξ(kl)  :  - k1

   exact = proven over Q.

The first line gives the rank of the OIM and the number of coordinates (:doc:`models`); their
difference is the number of non-identifiable directions. Each direction is printed as a
generator :math:`X = \sum_i \xi_i \partial_{\theta_i}`, where ``ξ(θ)`` is the rate of change
of the coordinate ``θ`` along the direction; coordinates that do not change are omitted. A
*scaling* has components ``ξ(θ) = w θ`` with integer weights ``w``, so it multiplies the
coordinates by powers of one factor; every other direction is *general*. Along every direction
the observables stay the same:

- :math:`X_1` scales ``kb`` and ``kin`` up and ``s`` down, so that only ``kb*s`` and
  ``kin*s`` are identifiable;
- along :math:`X_2` the outflow ``k1 + kl`` and the flux into ``x2``, ``k1*kb/(k1 + kl)``
  at rest and ``k1*kin/(k1 + kl)`` per dose, stay fixed, which is all that ``x2`` sees.

Levels and rank status
----------------------

A direction is proven, checked with an error bound, or checked without one. Its ``level`` is

``exact``
   proven over Q;
``verified``
   checked at random points, with the error probability in ``info["bound"]``;
``checked``
   the same test without an error bound, or on part of the entries only;
``open``
   no closed form.

``res.unproven()`` returns the verified and checked directions. The rank is ``proven``,
``verified`` (with a probability) or a ``lower bound``, in which case the model may have fewer
non-identifiable directions. ``info["rank_status"]`` gives the status and the reason, the
printout flags an unproven rank, and :doc:`../method` states when each applies.

``res.symmetries`` lists :class:`~symident.Symmetry` objects, with the generator, its
``level``, ``type`` (scaling or general), ``support``, the coordinates it moves, and ``route``,
the way the closed form was obtained (:doc:`../method`). ``res.summary()`` adds the Lie order,
i.e., the highest order of time derivatives of the observables used, the proof of the rank, a
suggested gauge (the option below) and what stopped part of the detection, and ``to_dict()``
returns the raw result.
``res.info`` holds the details, with the keys listed under :class:`~symident.Symmetries`. A
nonempty ``info["limits"]`` means that a direction without closed form may have one, and
``info["assumptions"]`` holds the declared domain, the switch starts and the error probability
of a verified result.

Options
-------

The options of :func:`~symident.detect` change what is computed and how; the docstring lists
all of them.

``reconstruct``
   general directions as exact rational functions; otherwise only by their support.
``equilibrate``
   starts every condition at a steady state, with the inputs at zero. Each conserved total
   adds a free resting value, held by one species of the total, the pivot.
``reduce_cq``
   eliminates one species per conserved total, the pivot, which becomes
   ``(total - rest)/coefficient``; the total becomes a parameter. A species with a given
   initial value is avoided as pivot; where an eliminated species has one,
   ``reduce_cq`` is switched off with a warning.
``free_initial``
   under ``equilibrate`` without ``reduce_cq``, the pivots whose resting value is free. When it
   is not given and directions stay open, :func:`~symident.detect` tries up to six choices of
   pivots and keeps the one with the fewest open directions; ``info["limits"]`` then points to
   ``reduce_cq`` for those left.
``positive``
   coordinates known to be positive (default all); logarithms, roots and switch conditions
   are decided on this domain. An entry may be an expression, such as ``"1-x"`` for a fraction
   ``x`` below one; :func:`~symident.reduce` then prefers sections (:doc:`reduction`) inside
   it.
``gauge``
   ``False`` reconstructs every direction. ``None`` fixes one coordinate per scaling, chosen
   to keep the general directions narrow, and reconstructs the general directions with these
   coordinates fixed; a list ranks the coordinates to fix. Without a gauge,
   ``info["gauge_suggestion"]`` names the coordinates such a choice would fix.
``verify``
   checks the rank where the Lie order is not proven sufficient: the rank is taken again a few
   orders further (``SYMIDENT_VERIFY_MARGIN``). Where it still grows, the detection is repeated
   from the order of growth, and if it grows again, the rank is a lower bound and
   ``info["verification"]`` says where. :func:`~symident.reduce` then refuses, since a
   direction may disappear at a higher order. ``False`` skips the check.
``scalings_only``
   only the scaling symmetries, from an exact integer kernel.
``engine``
   ``"modular"`` (default), the compiled computation over finite fields, or ``"symbolic"``,
   exact rational functions for small models and functions the modular engine does not read.
``cores``
   threads and worker processes.

Discrete symmetries
-------------------

A discrete symmetry, such as two rate constants entering only through their sum and product,
survives local identifiability. ``res.solution_count()`` counts, over the algebraic closure,
the coordinate values that give the output of a random point modulo a prime, also outside
the declared domain (``-k`` where only ``k^2`` enters). A count above one is a candidate at
that point, not a proof. The count needs msolve (:doc:`../install`), an identifiable result,
and one condition without events, switches or unknown inputs, detected without
``equilibrate`` and ``reduce_cq``. It returns None for a non-rational model, and where msolve
finds no finite count.
