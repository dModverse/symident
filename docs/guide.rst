User guide
==========

The guide follows one model through detection, reduction and a variant with a switch: a
precursor ``x1`` is made at a basal rate and on a stimulus ``u``, and is either converted
into the observed product ``x2`` or degraded. The system is at steady state before the
stimulus, which is switched on at :math:`t = 0` with two doses.

The model
---------

As ODEs with the observable, the model reads

.. math::

   \dot x_1 = k_b + k_{in}\, u - (k_1 + k_l)\, x_1, \qquad
   \dot x_2 = k_1 x_1 - k_2 x_2, \qquad \dot u = 0, \qquad y = s\, x_2 .

:class:`symident.Model` takes it either as a reaction network or as ODEs; both give the
same model. A reaction network lists the species, the stoichiometry (one row per
reaction, one column per species) and the rates:

.. code-block:: python

   import symident as si

   m = si.Model(reactions={"species": ["x1", "x2", "u"],
                           "stoichiometry": [[1, 0, 0], [1, 0, 0], [-1, 1, 0],
                                             [-1, 0, 0], [0, -1, 0]],
                           "rates": ["kb", "kin*u", "k1*x1", "kl*x1", "k2*x2"]},
                observables={"y": "s*x2"},
                inputs=["u"],
                events=[{"var": "u", "time": 0, "value": "dose", "method": "replace"}],
                conditions={"rows": ["low", "high"], "cols": {"dose": [1, 10]}})

The five reactions are basal production of ``x1`` (``kb``), production on the stimulus
(``kin*u``), conversion into ``x2`` (``k1``), degradation of ``x1`` (``kl``) and
degradation of ``x2`` (``k2``). Only ``x2`` is measured, up to the unknown scale ``s``.
``u`` takes part in no reaction: it is an input, zero at rest, and the event sets it to
``dose`` at :math:`t = 0`. The two conditions differ in the dose.

As ODEs, the right-hand sides are a dict from state to expression (``^`` or ``**`` for
powers), followed by the observables. Detection and reduction below give the same result
for both forms:

.. code-block:: python

   m = si.Model({"x1": "kb + kin*u - (k1 + kl)*x1", "x2": "k1*x1 - k2*x2", "u": "0"},
                {"y": "s*x2"},
                inputs=["u"],
                events=[{"var": "u", "time": 0, "value": "dose", "method": "replace"}],
                conditions={"rows": ["low", "high"], "cols": {"dose": [1, 10]}})

Arguments of :class:`~symident.Model`:

``inputs``
   input states, zero at rest.
``events``
   a list of dicts with ``var``, ``time``, ``value`` and ``method`` (``replace``, ``add``
   or ``multiply``); ``root`` in place of ``time`` fires where an expression changes sign
   (see Switches).
``conditions``
   one row per experimental condition; a numeric cell fixes a symbol in that condition,
   a string renames it. A dict ``{"rows": [...], "cols": {name: [values]}}`` or a pandas
   data frame.
``trafo``
   a parameter transformation; an entry named like a state is its initial value.
``fixed``
   known symbols.
``parameters``
   further symbols treated as parameters.

Expressions
~~~~~~~~~~~

Right-hand sides, observables, initial values and event values are built from ``+``,
``-``, ``*``, ``/``, powers (``^`` or ``**``) and these functions:

==============================  ===========================================================
Exponential and logarithm       ``exp``, ``exp2``, ``exp10``, ``b^x`` with a positive
                                rational ``b``, ``log``, ``ln``, ``log2``, ``log10``,
                                ``log(x, b)``
Roots and powers                ``sqrt``, ``cbrt``, ``root(x, n)``, ``pow(x, y)``, rational
                                powers such as ``x^(1/2)``, free exponents ``x^n`` with a
                                parameter ``n``
Trigonometric                   ``sin``, ``cos``, ``tan``, ``cot``, ``sec``, ``csc``,
                                ``asin``, ``acos``, ``atan``, ``acot``, ``asec``, ``acsc``,
                                ``atan2``
Hyperbolic                      ``sinh``, ``cosh``, ``tanh``, ``coth``, ``sech``, ``csch``,
                                ``asinh``, ``acosh``, ``atanh``, ``acoth``, ``asech``,
                                ``acsch``
Switches                        ``piecewise``, ``ifelse``, ``Heaviside``, ``sign``,
                                ``abs``, ``max``, ``min``, comparisons and logical
                                operators (see Switches)
==============================  ===========================================================

These are the expressions of cppDE (``?cppDE::expressions``) with its definitions:
``acot(x)`` is ``atan(1/x)``, and likewise for the other inverse reciprocal functions.
``erf`` and ``erfc`` need ``engine="symbolic"``; ``floor``, ``ceiling`` and ``round`` are
refused, their steps are written with ``piecewise()``. ``pi`` is the number π and cannot
be used as a name; ``E`` is a name and ``exp(1)`` Euler's number.

A logarithm, a root or a fractional power needs an argument that is positive on the
coordinates declared ``positive``. A power is rational only when written as a fraction:
``x^(1/2)``, not ``x^0.5``. ``time`` is the clock and may appear anywhere. Any other
function is rejected by the default engine with a message that names the term;
``engine="symbolic"`` accepts every SymPy function but is meant for small models.

A function not covered here can be requested from the maintainer, Simon Beyer
(simon.beyer@fdm.uni-freiburg.de), or as an issue at
https://github.com/dModverse/symident/issues.

Detection
---------

:func:`symident.detect` returns the non-identifiable directions as a
:class:`~symident.Symmetries` object. With ``equilibrate=True`` every condition starts at
its steady state with the inputs at zero, here :math:`x_1^\ast = k_b/(k_1 + k_l)` and
:math:`x_2^\ast = k_1 x_1^\ast / k_2`:

.. code-block:: python

   res = si.detect(m, equilibrate=True, reduce_cq=True, reconstruct=True)
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

Of the six parameters only four combinations are identifiable. Each non-identifiable
direction is given by its *generator*, a vector field on the coordinates
:math:`\theta_i` (parameters and free initial values), written in the notation of
differential geometry as

.. math::

   X = \sum_i \xi_i\,\partial_{\theta_i} .

The component :math:`\xi_i`, a function of the coordinates, is the rate at which
:math:`\theta_i` changes when the coordinates move along the direction, and
:math:`\partial_{\theta_i}` marks the coordinate it belongs to. The notation reads
:math:`X` as the derivative along the direction: applied to a function :math:`I` of the
coordinates it gives :math:`X(I) = \sum_i \xi_i\, \partial I / \partial \theta_i`, the
rate of change of :math:`I` along it (see `Vector field
<https://en.wikipedia.org/wiki/Vector_field#Vector_fields_on_subsets_of_Euclidean_space>`_
on Wikipedia). The printed result lists :math:`\xi_i` as ``ξ(θ_i)`` and leaves out the
coordinates with :math:`\xi_i = 0`. Along each direction the observables do not
change:

- :math:`X_1` raises both production rates and lowers the scale ``s`` by the same
  factor. ``x2`` grows in proportion, and ``s*x2`` stays the same: the absolute
  production rates are not identifiable, only ``kb*s`` and ``kin*s``.
- :math:`X_2` shifts flux between the two fates of ``x1``: it raises ``k1``, lowers
  ``kl`` by the same amount and lowers the production rates. The total outflow
  ``k1 + kl`` is unchanged, so ``x1`` relaxes at the same rate, and so is the flux into
  ``x2``, ``k1*kb/(k1 + kl)`` at rest and ``k1*kin/(k1 + kl)`` per dose. ``x2`` sees only
  these combinations, never the split itself.

Each direction with a closed form has a ``level``. ``exact`` means proven over the
rationals. ``verified`` means checked at random points modulo a large prime, where a wrong
closed form passes with a small probability. ``info["bound"]`` gives this probability
where the Lie and gap orders of the check are proven sufficient, ``info["proof"]`` says
why the exact proof does not apply, and ``res.unproven()`` returns these directions. A
direction without closed form is ``open``.

Where the rank, and with it the number of directions, is not proven, the printed result
says so. ``info["rank_status"]`` holds the ``status``, the reason (``why``) and, for a
verified rank, the ``probability`` from ``info["bound"]``. The status is ``"proven"``,
``"verified"`` when the Lie order and, under events after the start, the gap order
(``info["gap_certified"]``) are proven sufficient and every direction has a closed form,
or when invariant fields hold the kernel at every order (``info["rank_fields"]``, checked
at random points with the probability bound ``info["rank_fields_bound"]``), and
``"lower bound"`` otherwise: the model may then have fewer non-identifiable directions,
and ``identifiable = False`` may not hold. ``engine="symbolic"`` proves its Lie order
sufficient where the rows at the initial values reach the generic rank;
``info["lie_certificate"]`` says why it does not.

Options of :func:`~symident.detect`:

``reconstruct``
   return general directions as exact rational functions (otherwise by their support).
``equilibrate``
   start every condition at its steady state, with the inputs at zero.
``reduce_cq``
   report moiety freedoms on conserved totals instead of single species.
``gauge``
   ``False`` reconstructs every direction; ``None`` fixes one coordinate per scaling,
   chosen to keep the general directions narrow; a list ranks the coordinates to fix.
``cores``
   threads and worker processes.

``summary()`` adds how the result was obtained: the Lie order, whether it is proven
sufficient, the gauge and the entries of ``info``:

``info["limits"]``
   the time limits, work budgets and orders that stopped part of the detection; while it
   is not empty, a direction reported without closed form may have one.
``info["attempts"]``
   per direction without closed form, the routes tried and why each leaves it open.
``info["errors"]``
   exceptions that a route turned into a failure.
``info["branches"]``
   resting states with several interior roots modulo a prime, of which the first is
   taken.
``info["settings"]``
   the control list, the active switches, the primes and the symident version.

The raw result is available with ``to_dict()``, each direction as a
:class:`~symident.Symmetry`.

Reduction
---------

:func:`symident.reduce` (or ``res.reduce()``) returns a reparametrization that removes the
directions. A scaling is removed by fixing one of its coordinates; a general direction by
its invariants, the combinations that stay constant along it, and each invariant becomes a
new parameter ``q_1, q_2, ...``:

.. code-block:: python

   red = res.reduce()
   print(red)

.. code-block:: text

   Reduced 2 of 2 directions.
   {X₂} assumes: every level set of the invariants is one orbit; the section kl = 0 lies
   on a face and is met only in a limit.

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

The reduced model fixes the scale at ``s = 1`` and moves all outflow of ``x1`` into the
conversion, ``kl = 0``. Its parameters are the identifiable combinations found above:
``q_1`` is the total outflow rate of ``x1``, ``q_2`` and ``q_3`` are the basal and the
stimulated flux into ``x2``, each in units of the measured signal. ``k2`` is identifiable
and stays as it is.

Each line under ``Sections`` is the equation of a cross-section, which every orbit of
the listed directions meets exactly once; together they form the global section of the
reduction. ``red.summary(detailed=True)`` explains each of them. ``red.level`` says how
far this is established: ``exact`` when every claim is proven, ``verified`` when a
direction rests on a verified closed form, ``assumed`` when a claim rests on an
assumption that the printout names. Here the section ``kl = 0`` lies on the boundary of
the parameter space, which the orbits reach only in a limit. ``red.checks`` holds the
check of the trafo over Q and the detection on the reduced model (``check=True``).

A direction of level ``open`` is an error unless ``allow_open=True``, which leaves it in
place; a ``verified`` one gives a warning. The section search of a block stops after
``work_cap`` candidate sections (4000), which ``red.limits`` then lists.

``red.trafo`` is the transformation as a dict of strings, ready to be composed with
the model's parameters:

.. code-block:: python

   red.trafo
   # {'k1': 'q_1', 'k2': 'k2', 'kb': 'q_2', 'kin': 'q_3', 'kl': '0', 's': '1'}

Switches
--------

Right-hand sides and observables may switch between regimes:
``piecewise(v1, c1, v2, c2, ..., otherwise)`` in SBML order (``0`` where no condition
holds and no ``otherwise`` is given), ``ifelse(c, a, b)``, the comparisons ``<``, ``>``,
``<=``, ``>=``, the operators ``&&``, ``||`` and ``!``, ``Heaviside()``, ``sign()``,
``abs()``, ``max()`` and ``min()``. A condition that the coordinates declared
``positive`` decide is resolved, so ``abs(x)`` is ``x`` and ``x > 0`` is true for a
positive ``x``. Any other condition makes the model hybrid.

In the guide's model, let the degradation of ``x1`` act only above a threshold ``thr``:

.. code-block:: python

   odes = {"x1": "kb + kin*u - k1*x1 - piecewise(kl*x1, x1 > thr, 0)",
           "x2": "k1*x1 - k2*x2", "u": "0"}
   ms = si.Model(odes, {"y": "s*x2"},
                 inputs=["u"],
                 events=[{"var": "u", "time": 0, "value": "dose", "method": "replace"}],
                 conditions={"rows": ["low", "high"], "cols": {"dose": [1, 10]}})
   print(si.detect(ms, equilibrate=True, reconstruct=True))

.. code-block:: text

   Result:  rank 6 / 7  |  1 non-identifiable direction

   Generators  X = Σᵢ ξ(i) ∂ᵢ

   Scalings:
     X₁  (exact)
         ξ(kb)  :  + kb
         ξ(kin) :  + kin
         ξ(s)   :  - s
         ξ(thr) :  + thr

   exact = proven over Q.

The flux split :math:`X_2` of the plain model is gone: below the threshold only ``k1``
drains ``x1``, above it ``k1 + kl`` does, and the observable registers the change, so
``k1`` and ``kl`` are determined separately. The scaling remains, with the threshold
moving along with ``x1``. With a known threshold, ``fixed=["thr"]``, the switch breaks
the scaling as well.

A direction is reported only when the observables of every regime stay unchanged along
it, every switching surface whose switch shows in the observables is mapped to itself,
and every event and the given initial values are kept. Every sign pattern of the
switching functions counts as a regime, whether or not a trajectory reaches it; only a
pattern that the declared signs contradict is dropped, such as ``x > c`` together with
``x < c - d`` for positive ``d``. A parameter that acts only in a regime the trajectory
never enters is therefore reported identifiable, while a reported direction always
exists. :doc:`method` states the rule and how it is decided.

A switch is *visible* when the observables differ between the regimes on its two sides
on the surface, as here (``info["switching"]["conditions"][0]["visibility"]`` reads
``{'thr - x1 = 0': 'visible'}``). A switch that acts only on states that never reach the
observables is invisible, and its surface may move. A second, smaller model shows this: a
state ``z`` that decays at a rate switching at ``c`` and never reaches the observable:

.. code-block:: python

   m2 = si.Model({"x": "-k*x", "z": "piecewise(-a*z, z > c, -b*z)"}, {"y": "s*x"})
   r2 = si.detect(m2, reconstruct=True)
   r2.info["switching"]["conditions"][0]["visibility"]
   # {'c - z = 0': 'invisible'}

``a``, ``b``, ``c`` and the initial value of ``z`` are free, and ``s`` and the initial
value of ``x`` are determined only as the product ``s*x``.

``regimes`` lists the regimes an experiment visits, by condition: each entry is a
comparison or a list of them, the first the regime at the start. Then only these regimes
and the surfaces between them count, and a parameter of a regime outside them is
reported non-identifiable. An experiment that stays below the threshold never sees
``kl``:

.. code-block:: python

   ms.regimes = ["x1 < thr"]
   print(si.detect(ms, equilibrate=True, reconstruct=True))

.. code-block:: text

   Result:  rank 4 / 7  |  3 non-identifiable directions

   Generators  X = Σᵢ ξ(i) ∂ᵢ

   Scalings:
     X₁  (exact)
         ξ(kb)  :  + kb
         ξ(kin) :  + kin
         ξ(s)   :  - s

   General:
     X₂  (exact)
         ξ(kl) :  + 1

     X₃  (exact)
         ξ(thr) :  + 1

   exact = proven over Q.

``kl`` and ``thr`` are free, each on its own. That the start lies below the threshold is
an assumption here, since the initial values do not decide it; ``info["switching"]``
lists it. A start that the initial values contradict is an error. Without ``regimes`` the
result holds for every initial state; with it, for the listed experiment.

A switch on time alone, such as ``time > tau``, is a surface like the others: a
symmetry keeps its switching time when the switch is visible. Events may also fire at a
surface: an event with ``root`` in place of ``time`` resets its state each time the root
expression changes sign.

``info["switching"]`` lists the surfaces, the regimes counted, the sign patterns
excluded and, per condition, the regimes, the visibility of each surface and the
assumptions.

The rank is bounded from both sides. Necessary conditions, the observability rows of
every regime from the start of every condition, from generic points of the visible
surfaces and after a switch between two regimes, bound it from below; directions checked
as symmetries of every regime bound it from above. When the two bounds do not meet, the
directions of the stacked rows that the rule does not prove are listed as ``verified``,
``info["switching"]["uncertified"]`` says why and names the rejected candidates, and the
rank is a lower bound. ``detect()`` raises an error for a switch on a switched
expression, for a switch on parameters alone whose side ``regimes`` does not give, and
for ``gauge``, ``reduce_cq`` and ``free_initial`` together with switches. With
``equilibrate``, every resting state of every regime must have a closed form. ``x == c`` counts as false and
``x != c`` as true unless both sides agree identically.

Settings
--------

:func:`symident.reconst_control` collects the settings of the saturation and the
reconstruction; pass the result as ``control`` to :func:`~symident.detect`. Among them,
with their defaults:

``degree_cap``, ``degree_cap_max``
   the degree bound of the dense fit (4) and the degree up to which an entry past it is
   probed and fitted (8).
``gap_order_cap``
   the highest gap order under events after the start (8).
``homotopy``, ``homotopy_degree``
   directions that stay open are last fitted along parameter lines, up to this degree
   (4), and may then contain the resting states (under ``equilibrate``) or the initial
   values given with radicals; ``homotopy=False`` skips that route.
``exact_proof``, ``exact_work_cap``
   whether closed forms are proven over Q through the fields of each regime, and the
   work budget of that proof (500000).
``seed``
   the seed of every random choice (0).
``timeout``
   the time limit of the reconstruction in seconds, once per detection (none).

The same model, settings and seed give the same result, whatever ``cores``.


Diagnostics and route switches
------------------------------

A switch is an environment variable ``SYMIDENT_<NAME>`` or, inside a ``with`` block,
:func:`symident.switches` with the name as a keyword; the block wins over the
environment. Switches change how a detection runs, never what it reports as proven. The
active ones are recorded in ``info["settings"]["switches"]``, and an unknown name gives a
warning. From R, every ``SYMIDENT_*`` variable and every option ``dMod.sym.<name>``
reaches symident.

An exception that a route turns into a failure is recorded in ``info["errors"]``;
``DEBUG`` raises it instead, and the test suite runs with it. ``DIAG`` turns every
diagnostic output on. The route switches turn a route or a shortcut off (``NO...``) or
force one, to isolate a route when a result looks wrong.

Debugging:

``DEBUG``
   exceptions a route would record in info['errors'] are raised, except failures of SymPy.
``DIAG``
   every diagnostic switch on.

Diagnostic output:

``TIMING``
   stage times.
``LIEDIAG``
   saturation, Lie order caps and sufficiency proofs.
``ROBUSTDIAG``
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

Numbers:

``LIEPLATEAU``
   orders the rank stays flat before the saturation ends (3).
``LIEPLATEAU_BLOCK``
   the plateau per condition block; 0 for one plateau.
``VERIFY_MARGIN``
   orders the saturation guard checks past the reported one (6).
``JETWORK``
   characters of jets the jet route may form (1000000).
``DIRECTSECS``
   seconds msolve may take for the direct kernel (120).

Installation of msolve:

``MSOLVE``
   path of the msolve executable.
``CACHE``
   directory of the symident cache.
``LIBS_CACHE``
   build directory of install_msolve().
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
``OMP_STUB_H``
   guard of the OpenMP stub header of the msolve build.
