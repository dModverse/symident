Global analysis
===============

A locally identifiable model can still have several parameter values with the same output, such
as two rate constants that enter only through their sum and product. The global analysis counts
these values at a generic point, the *fiber*, and returns the maps between them, the *discrete
symmetries*. :doc:`../method/global` states the mathematics.

Calling the global analysis
---------------------------

``detect(model, mode="global")`` runs the local analysis, reduces the directions it finds
(:doc:`reduction`) and then analyzes the reduced model globally; ``res.discrete()`` does the
second step on a result of a local detection. ``global_options`` passes the arguments of
:meth:`~symident.Symmetries.discrete`: ``order``, the highest Taylor coefficient of the
observables, ``points``, the number of random points of the count in the domain, ``memory``, the
address space of one call of msolve, ``experiments``, the count of several experiments, and the
work limits ``msolve_entries`` (nonzero entries of the F4 matrices of one msolve call, 4e8),
``msolve_terms`` (terms of one system given to msolve, 1e5), ``search_entries`` (entries of the
matrices over :math:`\mathbb{F}_p` the exact search reduces, 5e8) and ``elimination_terms`` (terms
of a polynomial of the elimination, 3e6). There are no time limits: a stage that reaches a work
limit ends with a result that names it. The analysis needs msolve (:doc:`../install`).

Linear compartment models are the classical case of structural identifiability
:cite:p:`Bellman1970`. A two-compartment model with an input :math:`u` into the first compartment,
observed in the second:

.. code-block:: python

   import symident as si

   m = si.Model({"x1": "-(k21 + k01)*x1 + k12*x2 + u", "x2": "k21*x1 - k12*x2"},
                {"y": "x2"}, known_inputs=["u"])
   res = si.detect(m, mode="global")
   print(res)

.. code-block:: text

   Result:  structurally locally identifiable (rank 5 / 5)

   Global: values with the output of a generic one, counted at random points
     2 over the algebraic closure, 1 or 2 in the domain (order 5, stable)
     exactly 2 values at a generic point, proven
     sigma1 (exact): k01 -> k12, x1 -> (k01*x2 - k12*x2 + k21*x1)/k21, k12 ->
     k01; in the domain where (k01*x2 - k12*x2 + k21*x1)/k21 > 0
     fundamental domain: k01 < k12

The map exchanges the elimination rate :math:`k_{01}` and the return rate :math:`k_{12}` and
moves the initial value of :math:`x_1`, so the model is locally but not globally identifiable.
The count of two is proven: the fiber at a generic point is the orbit of the exact map. The image
lies in the domain where the image of :math:`x_1(0)` is positive, so a random point of the
domain has one or two values in it. The fundamental domain keeps one point of each orbit;
``reduce()`` lists it in ``Reduction.discrete``.

.. code-block:: python

   s, = res.discrete_symmetries
   s["map"], s["level"], s["fixed_set"]
   res.info["discrete"]["certificate"]["complete"]

.. code-block:: text

   ({'k01': 'k12', 'x1': '(k01*x2 - k12*x2 + k21*x1)/k21', 'k12': 'k01'}, 'exact', ['k01 - k12'])
   'exact'

The result
----------

``res.discrete_symmetries`` lists the maps :math:`\sigma`, each a dict with

``map``, ``changes``
   the images of the coordinates that :math:`\sigma` moves;
``level``
   ``exact``, proven by a map of the states (``state_maps``) over :math:`\mathbb{Q}`, or
   ``checked``, the output coefficients agree at random points;
``found_by``
   ``candidates`` (permutations of coordinates with the same dependence of the output) or
   ``fiber`` (reconstructed from the points of the fiber);
``preserves_domain``, ``domain_conditions``
   True, False, or None with the conditions under which :math:`\sigma(\theta)` lies in the
   domain; a map that never meets the domain, such as :math:`\theta \mapsto -\theta` where only
   :math:`\theta^2` enters, is no symmetry of the model on the domain;
``fixed_set``
   the factors on which :math:`\sigma` fixes :math:`\theta`.

``res.info["discrete"]`` holds the count:

``count``, ``count_bound``
   the points of the fiber over the algebraic closure at a random point modulo a prime, and the
   probability that a generic point has more (None where the count is exact or has no bound);
``counts``, ``count_domain``
   per random point of the domain the points over the complex numbers, over the reals and in the
   domain; the number in the domain can differ between regions. ``count_domain`` is the number at
   a generic point of the domain where the count is final and each map keeps the domain or leaves
   it entirely;
``count_final``
   every point of the count is the image of a proven map;
``fundamental_domain``
   inequalities that keep one point of each orbit;
``certificate``
   what the count rests on, summed up in ``complete``;
``reduced``, ``rewrites``, ``relaxed``, ``start_powers``, ``experiments``
   the reduction the analysis ran on, and the rewrites into rational form;
``eliminated``
   for a result detected with ``reduce_cq``, the species solved through the conserved totals: the
   count is on the level sets of the totals, which are coordinates or set by the ``trafo``;
``limits``
   the work limits and premises that stopped or qualify part of the analysis.

The certificate
---------------

``complete`` takes one of five values:

``"exact"``
   the fiber at a generic point is the orbit of the exact maps, proven over :math:`\mathbb{Q}`
   without random points; ``certificate["exact"]`` lists the functions of the coordinates shown to
   be rational in the output coefficients.
``"verified"``
   a count over :math:`\mathbb{Q}` at a random integer point equals the orbit; ``count_bound``
   is the probability that a generic point has more points, ``count_bits`` and ``bezout`` the
   numbers it rests on.
``"axiom"``
   the count holds under the named axiom that the count modulo a prime at a random point is the
   count at a generic point, or under a premise in ``certificate["premise"]``;
   ``count_bound_why`` says why no bound applies.
``"no"``
   the count exceeds the orbit of the exact maps: there are further values whose maps are not
   proven.
``"open"``
   no count or no orbit was reached.

Unsupported models and open results
-----------------------------------

A model that the global analysis does not handle gets the reason in
``info["discrete"]["unsupported"]`` and a warning; the local result stays valid. Typical reasons:

- a direction that the reduction does not remove, so the reduced model keeps a continuous
  symmetry;
- unknown inputs together with events, switches or more than one condition;
- switches together with events, under ``equilibrate`` or on the parameters alone, and root
  events;
- under ``equilibrate``, a resting state without one rational closed form in the coordinates, or
  resting balances of a Bezout number past 4 or with more than 8 states coupled;
- a term the rewrites do not bring into rational form, such as a root of a state.

A result is ``open`` or ``axiom`` where the count over the algebraic closure or over
:math:`\mathbb{Q}` exceeds the work limits of msolve, where the maps are not rational functions
of the coordinates, or where the search of the exact count does not reach every coordinate.
``limits`` names the step. The counts in the domain hold at the sample points only, since the
number of points in the domain changes across the hypersurfaces where two points meet or a point
leaves the domain. ``res.solution_count()`` gives the count over the algebraic closure at one
random point, without the maps.
