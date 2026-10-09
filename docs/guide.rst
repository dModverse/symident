User guide
==========

The guide follows one model through detection and reduction: a precursor ``x1`` is made
at a basal rate and on a stimulus ``u``, and is either converted into the observed
product ``x2`` or degraded. The system rests before the stimulus, which is switched on at
:math:`t = 0` with two doses.

The model
---------

:class:`symident.Model` takes a reaction network with the species, the stoichiometry
(one row per reaction, one column per species) and the rates:

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

The right-hand sides can also be given directly, as a dict from state to expression
(``^`` or ``**`` for powers), followed by the observables:

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
Exponential and logarithm       ``exp``, ``exp10``, ``b^x`` with a positive rational ``b``,
                                ``log``, ``ln``, ``log2``, ``log10``, ``log(x, b)``
Roots and powers                ``sqrt``, rational powers such as ``x^(1/2)``, free
                                exponents ``x^n`` with a parameter ``n``
Trigonometric                   ``sin``, ``cos``, ``tan``, ``cot``, ``sec``, ``csc``,
                                ``asin``, ``acos``, ``atan``, ``atan2``
Hyperbolic                      ``sinh``, ``cosh``, ``tanh``, ``coth``, ``sech``, ``csch``,
                                ``asinh``, ``acosh``
Switches                        ``piecewise``, ``ifelse``, ``Heaviside``, ``sign``,
                                ``abs``, ``max``, ``min``, comparisons and logical
                                operators (see Switches)
==============================  ===========================================================

A logarithm, a root or a fractional power needs an argument that is positive on the
coordinates declared ``positive``. A power is rational only when written as a fraction:
``x^(1/2)``, not ``x^0.5``. ``time`` is the clock and may appear anywhere. Any other
function is rejected by the default engine with a message that names the term;
``engine="symbolic"`` accepts every SymPy function but is meant for small models.

A function not covered here can be requested from the maintainer, Simon Beyer
(simon.beyer@fdm.uni-freiburg.de), or as an issue at
https://github.com/dModverse/symident/issues.

Switches
~~~~~~~~

Right-hand sides and observables may switch between regimes:
``piecewise(v1, c1, v2, c2, ..., otherwise)`` in SBML order (``0`` where no condition
holds and no ``otherwise`` is given), ``ifelse(c, a, b)``, the comparisons ``<``, ``>``,
``<=``, ``>=``, the operators ``&&``, ``||`` and ``!``, ``Heaviside()``, ``sign()``,
``abs()``, ``max()`` and ``min()``. A condition that the coordinates declared
``positive`` decide is resolved, so ``abs(x)`` is ``x`` and ``x > 0`` is true for a
positive ``x``. Any other condition makes the model hybrid:

.. code-block:: python

   m = si.Model({"x": "piecewise(-k1*x, x > thr, -k2*x)"}, {"y": "c*x"})
   print(si.detect(m, reconstruct=True))

.. code-block:: text

   Result:  rank 4 / 5  |  1 non-identifiable direction

   Generators  X = Σᵢ η(i) ∂ᵢ

   Scalings:
     X₁
         η(c)   :  - c
         η(thr) :  + thr
         η(x)   :  + x

Both regimes keep the scaling of ``x`` and ``c``, and the threshold moves along with
``x``, so of ``c``, ``thr`` and the initial value only ``c*thr`` and ``c*x`` are
determined. With a known threshold, ``x > 1`` or
``fixed=["thr"]``, the switch breaks the scaling and the model is identifiable.

A direction is reported only when the observables of every regime stay unchanged along
it, every switching surface whose switch shows in the observables is mapped to itself,
and every event and the given initial values are kept. Every sign pattern of the
switching functions counts as a regime, whether or not a trajectory reaches it; only a
pattern that the declared signs contradict is dropped, such as ``x > c`` together with
``x < c - d`` for positive ``d``. A parameter that acts only in a regime the trajectory
never enters is therefore reported identifiable, while a reported direction always
exists. :doc:`method` states the rule and how it is decided.

A switch is *visible* when the observables differ between the regimes on its two sides
on the surface. A switch that acts only on states that never reach the observables is
invisible, and its surface may move:

.. code-block:: python

   m = si.Model({"x": "-k*x", "z": "piecewise(-a*z, z > c, -b*z)"}, {"y": "s*x"})
   si.detect(m).info["switching"]["conditions"][0]["visibility"]
   # {'c - z = 0': 'invisible'}: a, b, c and z are free, and so are s and x up to s*x

``regimes`` names the regimes an experiment visits, by condition: each entry is a
comparison or a list of them, the first the regime at the start. Then only these regimes
and the surfaces between them count, and a parameter of a regime outside them is
reported non-identifiable:

.. code-block:: python

   m = si.Model({"x": "piecewise(-k1*x, x > c, -k2*x)"}, {"y": "x"}, trafo={"x": "2"},
                regimes=["x > c"])

Here ``k2`` and ``c`` are free. A start that the initial values contradict is an error;
a start they do not decide is an assumption, listed in ``info["switching"]``. Without
``regimes`` the result holds for every initial state; with it, for the named
experiment.

A switch on time alone, such as ``time > tau``, is a surface like the others: a
symmetry keeps its switching time when the switch is visible. Events may also fire at a
surface: an event with ``root`` in place of ``time`` resets its state each time the root
changes sign,

.. code-block:: python

   si.Model({"x": "a - k*x"}, {"y": "x"}, trafo={"x": "0"},
            events=[{"var": "x", "root": "x - c", "value": "0", "method": "replace"}])

``info["switching"]`` lists the surfaces, the regimes counted, the sign patterns
excluded and, per condition, the regimes, the visibility of each surface and the
assumptions.

The answer is certified from both sides. Necessary conditions, the observability rows
of every regime from the start of every condition, from generic points of the visible
surfaces and after a switch between two regimes, bound the rank from below; directions
proven symbolically bound it from above. When the two bounds do not meet, ``detect()``
raises an error naming the candidates it could not prove, instead of returning a result.
The same happens for ``x == c`` (a condition that holds only on a surface), for a switch
on a switched expression, and for ``gauge``, ``reduce_cq`` and ``free_initial`` together
with switches. With ``equilibrate``, every resting state of every regime must have a
closed form.

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

   Generators  X = Σᵢ η(i) ∂ᵢ

   Scalings:
     X₁
         η(kb)  :  + kb
         η(kin) :  + kin
         η(s)   :  - s

   General:
     X₂
         η(k1)  :  + k1
         η(kb)  :  - kb
         η(kin) :  - kin
         η(kl)  :  - k1

Of the six parameters only four combinations are identifiable. A generator
:math:`X = \sum_i \eta_i\,\partial_{z_i}` is a direction along which the observables do
not change:

- :math:`X_1` raises both production rates and lowers the scale ``s`` by the same
  factor. ``x2`` grows in proportion, and ``s*x2`` stays the same: the absolute
  production rates are not identifiable, only ``kb*s`` and ``kin*s``.
- :math:`X_2` shifts flux between the two fates of ``x1``: it raises ``k1``, lowers
  ``kl`` by the same amount and lowers the production rates. The total outflow
  ``k1 + kl`` is unchanged, so ``x1`` relaxes at the same rate, and so is the flux into
  ``x2``, ``k1*kb/(k1 + kl)`` at rest and ``k1*kin/(k1 + kl)`` per dose. ``x2`` sees only
  these combinations, never the split itself.

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

``summary()`` adds how the result was obtained: the Lie order, whether the saturation is
certified, the gauge and the route of each closed form. The raw result is available with
``to_dict()``, each direction as a :class:`~symident.Symmetry`.

Reduction
---------

:func:`symident.reduce` (or ``res.reduce()``) returns a reparametrisation that removes the
directions. A scaling is removed by fixing one of its coordinates; a general direction by
its invariants, the combinations that stay constant along it, and each invariant becomes a
new parameter ``q_1, q_2, ...``:

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

   Sections: the invariants name each orbit, the chart takes one point on it
     {X₁}  s = 1
         pin: a scaling orbit is a ray, so any positive value meets it exactly once
     {X₂}  kl = 0
         face: every orbit in the positive orthant reaches it exactly once, with
         every other coordinate positive. A face is tried first: it switches rates
         off, and each remaining coordinate of the block becomes the q_<k> that
         holds the value it takes there

The reduced model fixes the scale at ``s = 1`` and moves all outflow of ``x1`` into the
conversion, ``kl = 0``. Its parameters are the identifiable combinations found above:
``q_1`` is the total outflow rate of ``x1``, ``q_2`` and ``q_3`` are the basal and the
stimulated flux into ``x2``, each in units of the measured signal. ``k2`` is identifiable
and stays as it is.

``red.trafo`` holds the transformation as a dict of strings, ready to be composed with
the model's parameters:

.. code-block:: python

   red.trafo
   # {'k1': 'q_1', 'k2': 'k2', 'kb': 'q_2', 'kin': 'q_3', 'kl': '0', 's': '1'}

Settings
--------

:func:`symident.reconst_control` collects the settings of the saturation and the
reconstruction, among them the degree bounds of the fits, the relevance caps and a time
limit; pass the result as ``control`` to :func:`~symident.detect`. Under ``equilibrate``,
directions that stay open are last fitted along parameter lines and may then name the
resting states; ``homotopy=False`` skips that route.
