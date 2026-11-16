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
survives local identifiability: another point of the domain gives the same output.
``detect(model, discrete=True)``, or ``res.discrete()`` on a result, counts these points in the
domain and reconstructs the maps between them where they are rational. The two-compartment
model of [Godfrey1983]_ with an input :math:`u` into the first compartment, observed in the
second, is the classical case:

.. code-block:: python

   import symident as si

   m = si.Model({"x1": "-(k21 + k01)*x1 + k12*x2 + u", "x2": "k21*x1 - k12*x2"},
                {"y": "x2"}, known_inputs=["u"])
   r = si.detect(m, discrete=True)
   s, = r.discrete_symmetries
   s["map"], s["level"], s["fixed_set"]
   r.info["discrete"]["fundamental_domain"]

.. code-block:: text

   ({'k01': 'k12', 'x1': '(k01*x2 - k12*x2 + k21*x1)/k21', 'k12': 'k01'}, 'exact', ['k01 - k12'])
   ['k01 < k12']

The map exchanges the elimination rate :math:`k_{01}` and the return rate :math:`k_{12}` and
moves the initial value of :math:`x_1`, so the model is locally but not globally identifiable.
It is ``exact``: proven by a map of the states. Its image lies in the domain where
``domain_conditions`` hold, here where the image of :math:`x_1(0)` is positive, so
``info["discrete"]["counts"]`` gives at each sample point the solutions over the complex
numbers, over the reals and in the domain, which can differ between regions of the domain. A
map that never meets the domain, such as :math:`\theta \mapsto -\theta` where only
:math:`\theta^2` enters, has ``preserves_domain`` False and is not a symmetry of the model on the
domain. The fundamental domain keeps one point of each orbit; ``reduce()`` lists it in
``Reduction.discrete``.

A model with directions is reduced first and analyzed in the coordinates of the reduced model
(``info["discrete"]["reduced"]``). The analysis needs msolve (:doc:`../install`) and one
condition without events, switches or unknown inputs, detected without ``equilibrate`` and
``reduce_cq``. The counts hold at the sample points, and the order of the Taylor coefficients is
raised until the count is stable; ``count_final`` is True when exact maps account for every
solution. Where the maps are not found within ``budget`` seconds, the counts are reported
alone. ``res.solution_count()`` gives the count over the algebraic closure at one random point.
