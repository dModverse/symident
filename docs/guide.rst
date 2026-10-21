User guide
==========

One model runs through detection, reduction and a variant with a switch: a
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

As ODEs, the right-hand sides are a dict from state to expression, followed by the
observables:

.. code-block:: python

   m = si.Model({"x1": "kb + kin*u - (k1 + kl)*x1", "x2": "k1*x1 - k2*x2", "u": "0"},
                {"y": "s*x2"},
                inputs=["u"],
                events=[{"var": "u", "time": 0, "value": "dose", "method": "replace"}],
                conditions={"rows": ["low", "high"], "cols": {"dose": [1, 10]}})

Arguments of :class:`~symident.Model`:

``inputs``
   input states, zero at the start and at rest; an event sets them.
``unknown_inputs``
   unknown functions of time (see `Unknown inputs`_).
``known_inputs``
   known functions of time of no assumed form, such as a recorded forcing: their values and
   time derivatives are known symbols.
``events``
   a list of dicts with ``var``, ``time``, ``value`` and ``method`` (``replace``, ``add``
   or ``multiply``); ``root`` in place of ``time`` fires where an expression changes sign.
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
``totals``
   conserved totals, name to expression, each conserved by the right-hand sides; computed
   from ``reactions`` when omitted. ``reduce_cq`` eliminates one species per total.

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
                                operators (see `Switches`_)
==============================  ===========================================================

These are the expressions of cppDE (``?cppDE::expressions``): ``acot(x)`` is
``atan(1/x)``, and likewise for the other inverse reciprocal functions. ``pi`` is the
number π and cannot name a symbol; ``E`` is a name and ``exp(1)`` is Euler's number;
``time`` is the clock. A logarithm, a root or a fractional power needs an argument that is
positive on the coordinates declared ``positive``, and a power is rational only as a
fraction: ``x^(1/2)``, not ``x^0.5``. ``floor``, ``ceiling`` and ``round`` are refused;
write their steps with ``piecewise()``. Any other function, ``erf`` and ``erfc`` among
them, needs ``engine="symbolic"``, which accepts every SymPy function but is meant for
small models. Further functions can be requested at
https://github.com/dModverse/symident/issues.

Detection
---------

:func:`symident.detect` returns the non-identifiable directions as a
:class:`~symident.Symmetries` object. With ``equilibrate=True`` every condition starts at
its steady state with the inputs at zero:

.. code-block:: python

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

Of the six parameters only four combinations are identifiable. Each direction is printed
as its generator: ``ξ(θ)`` is the rate at which the coordinate ``θ`` (a parameter or a free
initial value) changes along the direction, and coordinates that do not change are left
out. Along each direction the observables stay the same:

- :math:`X_1` raises both production rates and lowers the scale ``s`` by the same
  factor. ``x2`` grows in proportion, and ``s*x2`` stays the same: the absolute
  production rates are not identifiable, only ``kb*s`` and ``kin*s``.
- :math:`X_2` shifts flux between the two fates of ``x1``: it raises ``k1``, lowers
  ``kl`` by the same amount and lowers the production rates. The total outflow
  ``k1 + kl`` is unchanged, so ``x1`` relaxes at the same rate, and so is the flux into
  ``x2``, ``k1*kb/(k1 + kl)`` at rest and ``k1*kin/(k1 + kl)`` per dose. ``x2`` sees only
  these combinations, never the split itself.

Every direction has a ``level``: ``exact`` (proven over Q), ``verified`` (checked at
random points with the error probability in ``info["bound"]``), ``checked`` (the same
test without an error bound, or on part of the entries) or ``open`` (no closed form). ``res.unproven()`` returns the
verified and checked directions.

A locally identifiable model may still have a discrete symmetry, such as two rate constants
that enter only through their sum and product. ``res.solution_count()`` counts the values of
the coordinates that give the output of a random point modulo a prime, over the algebraic
closure: a value outside the declared domain, such as ``-k`` where only ``k^2`` enters,
counts too. A count above one is a candidate symmetry, found at that point, not proven.
It needs msolve (see :doc:`install`) and one condition without events, switches or unknown
inputs, detected without ``equilibrate`` and ``reduce_cq``; for a model that is not rational
it returns None.

The rank, and with it the number of directions, is ``proven``, ``verified`` (with a
probability) or a ``lower bound``, in which case the model may have fewer
non-identifiable directions. ``info["rank_status"]`` gives the status and the reason, and
the printout says when the rank is not proven. :doc:`method` states when each level and
status applies.

Options of :func:`~symident.detect`:

``reconstruct``
   return general directions as exact rational functions (otherwise by their support).
``equilibrate``
   start every condition at its steady state, with the inputs at zero.
``positive``
   the coordinates known to be positive, all by default; logarithms, roots and switch
   conditions are decided on this domain. An entry may be an expression, such as
   ``"1-x"`` for a fraction ``x`` below one; :func:`~symident.reduce` then prefers
   sections inside that bound.
``gauge``
   ``False`` reconstructs every direction; ``None`` fixes one coordinate per scaling,
   chosen to keep the general directions narrow; a list ranks the coordinates to fix.
``cores``
   threads and worker processes.

:func:`~symident.detect` documents the others.

``res.summary()`` adds how the result was obtained: the Lie order, the rank proof, the
suggested gauge and whatever stopped part of the detection. ``res.info`` holds the
details, listed under :class:`~symident.Symmetries`. While ``info["limits"]`` is not
empty, a direction reported without closed form may have one. ``info["assumptions"]``
holds the declared domain and switch starts the result assumes, and the error probability
of a verified result. ``res.symmetries`` lists the directions as
:class:`~symident.Symmetry`; ``to_dict()`` returns the raw result.

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

Each line under ``Sections`` picks one point on every orbit of the listed directions;
``red.summary(detailed=True)`` explains why. The section ``kl = 0`` lies on the boundary
of the parameter space, which the orbits reach only in a limit; the model is continuous up
to it, so the limit gives the output of the orbit.

``red.level`` is ``exact`` when every claim is proven, ``verified`` or ``checked`` when a
direction rests on a closed form of that level, and ``open`` when a claim of a section is
not proven; the printout names such claims. ``red.checks`` holds the check of the trafo
over Q and the detection on the reduced model (``check=True``); the summary line says when
that detection does not confirm the reduction. A direction of level ``open`` is an error
unless ``allow_open=True``, which leaves it in place. A ``checked`` direction, or a result
whose saturation guard saw the rank still grow, is an error unless ``allow_checked=True``;
a ``verified`` one gives a warning.

``red.trafo`` is the transformation as a dict of strings, ready to be composed with
the model's parameters:

.. code-block:: python

   red.trafo
   # {'k1': 'q_1', 'k2': 'k2', 'kb': 'q_2', 'kin': 'q_3', 'kl': '0', 's': '1'}

An entry for a symbol that the condition grid sets goes into the cells of the grid. Under
``equilibrate`` an entry named like a state pins the resting value it holds, not an initial
value; the model takes a numeric one in ``fixed``. The closed loop of ``check=True`` applies
the trafo this way.

PEtab problems
--------------

:func:`symident.read_petab` (or :meth:`symident.Model.from_petab`) reads a
`PEtab <https://petab.readthedocs.io>`_ problem [Schmiester2021]_, format version 1 or 2, into
a :class:`~symident.Model`. It needs the extra ``symident[petab]`` (python-libsbml and
PyYAML). Reading the model of [Raia2011]_ from the
`benchmark collection <https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab>`_
[Hass2019]_:

.. code-block:: python

   m = si.read_petab("Raia_CancerResearch2011/Raia_CancerResearch2011.yaml")
   res = si.detect(m)
   print(res)

.. code-block:: text

   Result:  rank 21 / 23  |  2 non-identifiable directions

   Generators  X = Σᵢ ξ(i) ∂ᵢ

   Scalings:
     X₁  (exact)
         ξ(SOCS3_accumulation)   :  + SOCS3_accumulation
         ξ(SOCS3mRNA_production) :  + SOCS3mRNA_production
         ξ(scaling_SOCS3mRNA)    :  - scaling_SOCS3mRNA

     X₂  (exact)
         ξ(CD274mRNA_production) :  + CD274mRNA_production
         ξ(scaling_CD274mRNA)    :  - scaling_CD274mRNA

   exact = proven over Q.

The scale of each transcript readout trades against its production rate; ``res.reduce()``
fixes ``SOCS3_accumulation = 1`` and ``CD274mRNA_production = 1``.

Estimated parameters stay symbols; every other parameter takes its nominal value. Each
experiment with measurements becomes one condition, a preequilibration sets
``equilibrate``, and the estimated parameters on a log scale or with a positive lower
bound become ``Model.positive``, the default of ``detect(positive=)``. The noise model does
not enter. An SBML or PEtab feature without a counterpart in the model, such as an
algebraic rule, a delay or a fast reaction, raises :class:`~symident.PEtabError`.
``m.petab`` lists the experiments, the estimated parameters and the fixed values;
:func:`~symident.read_petab` documents the mapping.

Unknown inputs
--------------

An input whose course nobody recorded, such as an emission history, a forcing to be
reconstructed or an intake, is an *unknown input*: a function of time of no assumed form,
written as a symbol in the right-hand sides and observables without an equation of its own
and listed in ``unknown_inputs``. The question is whether the parameters are identifiable
whatever the input does. A direction may then change the input along with the parameters;
the input itself is not asked about and does not count in ``rank`` and ``dim``.

An input of known form, such as dosing steps, a physical law or the spline a fit will use,
belongs in the model with its coefficients as ordinary parameters, and the result holds for
that form only. The two answers differ: a parameter that is non-identifiable with an unknown
input is non-identifiable under every parametrization flexible enough, while a parameter
that is identifiable under a parametrization may be so only because the form is taken as
exact. A recorded input whose form is not modeled goes into ``known_inputs``: its jets are
known symbols, so the result holds for a generic recorded course.

A glacier length ``L`` that relaxes to the
temperature anomaly ``T`` on the time scale ``tau``, with only ``L`` measured:

.. code-block:: python

   m = si.Model({"L": "-(c*T + L)/tau"}, {"y": "L"}, unknown_inputs=["T"])
   res = si.detect(m, reconstruct=True)
   print(res)

.. code-block:: text

   Result:  rank 1 / 3  |  2 non-identifiable directions

   Generators  X = Σᵢ ξ(i) ∂ᵢ

   Scalings:
     X₁  (exact)
         ξ(c) :  + c
         input change:
         ξ(T) :  - T

     X₂  (exact)
         ξ(tau) :  + tau
         input change:
         ξ(T) :  + (L + T*c)/c

   exact = proven over Q.

Only the initial length is identifiable. The sensitivity ``c`` trades against the scale of
``T``, and every ``tau`` fits the record with the temperature
``T = -(tau*L' + L)/c`` that its input change describes. ``s.input_change`` holds
:math:`\xi(T)` for each exact direction; the type and the reduction refer to the parameters
and initial values, so ``res.reduce()`` fixes ``c = 1`` and ``tau = 1``.

Every condition sees the same unknown input; a string in ``conditions`` renames it in a
condition, which gives that condition an input of its own when the new name is listed in
``unknown_inputs`` too, and a number fixes it there. An event leaves the input as it is:
the input is one analytic function across it, so a jump of a state can separate parameters
that the input absorbs otherwise. Under ``equilibrate`` the input is zero at rest, like
every input, and arbitrary from the start on. Switches, an event or a ``trafo`` entry for
the input and ``engine="symbolic"`` are refused.

.. warning::

   A polynomial stand-in for an unknown input, a chain ``w' = w1, ..., wd' = 0`` written as
   states with free initial values, is a family of :math:`d + 1` numbers, not an unknown
   input. Once the Lie order passes :math:`d + 1` the output determines them, and
   parameters that trade against the input look identifiable: with a cubic stand-in for
   ``T`` the glacier model above reports ``tau`` as identifiable. A spline is a polynomial
   on each segment, so the local analysis sees the same.

``info["unknown_inputs"]`` lists the jets of each input (see :doc:`method`), the
Lie order up to which they are exact, and whether the analysis reached past it. The jets
grow with the model; ``reconst_control(input_jet_order=)`` sets their number, and a chain
shorter than the Lie order needs makes the rank a lower bound with an entry in
``info["limits"]``. Unless the rank is full or exact directions fill the kernel, it is a
lower bound, since the jets grow with the Lie order and no order is proven sufficient. A
direction without a lift that is a function of the states, the parameters and the input
stays ``checked``, with no input change.

Stationary profiles
~~~~~~~~~~~~~~~~~~~

A profile observed at steady state, :math:`0 = F(x, u, u_x, u_{xx}, \theta)` in one space
dimension, is an ODE in :math:`x`. Written as a first-order system with :math:`x` in the
role of time, it is an ordinary model. The morphogen gradient :math:`D u'' = k u`, with the
profile ``u`` measured:

.. code-block:: python

   m = si.Model({"u": "v", "v": "k*u/D"}, {"y": "u"})
   print(si.detect(m, reconstruct=True))

.. code-block:: text

   Result:  rank 3 / 4  |  1 non-identifiable direction

   Generators  X = Σᵢ ξ(i) ∂ᵢ

   Scalings:
     X₁  (exact)
         ξ(D) :  + D
         ξ(k) :  + k

   exact = proven over Q.

Only the decay length :math:`\sqrt{D/k}` is identifiable. The initial values are the
profile and its slope at the first point. A source term whose shape is not known,
``v = (k*u - s)/D`` with ``unknown_inputs=["s"]``, frees ``D`` and ``k`` separately.

Switches
--------

Right-hand sides and observables may switch between regimes:
``piecewise(v1, c1, v2, c2, ..., otherwise)`` in SBML order (``0`` where no condition
holds and no ``otherwise`` is given), ``ifelse(c, a, b)``, the comparisons ``<``, ``>``,
``<=``, ``>=``, the operators ``&&``, ``||`` and ``!``, ``Heaviside()``, ``sign()``,
``abs()``, ``max()`` and ``min()``. A condition that the coordinates declared
``positive`` decide is resolved, so ``abs(x)`` is ``x`` and ``x > 0`` is true for a
positive ``x``. ``x == c`` counts as false and ``x != c`` as true unless both sides agree
identically.

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

A direction is reported when the observables of every regime stay unchanged along it and
it keeps the visible switching surfaces, the events and the given initial values. Every
sign pattern of the switching functions counts as a regime, whether or not a trajectory
reaches it, so a parameter that acts only in a regime the trajectory never enters is
reported identifiable; ``regimes`` below restricts the regimes. :doc:`method` states the
rule.

A switch is *visible* when the observables tell its two sides apart on the surface, as
here; ``info["switching"]["conditions"]`` gives the visibility per condition. A switch
that acts only on states that never reach the observables is invisible, and its surface
may move:

.. code-block:: python

   m2 = si.Model({"x": "-k*x", "z": "piecewise(-a*z, z > c, -b*z)"}, {"y": "s*x"})
   r2 = si.detect(m2, reconstruct=True)
   r2.info["switching"]["conditions"][0]["visibility"]
   # {'c - z = 0': 'invisible'}

``a``, ``b``, ``c`` and the initial value of ``z`` are free, and ``s`` and the initial
value of ``x`` are determined only as the product ``s*x``. A switch that is neither
visible nor invisible is ``undecided``.

``regimes`` lists the regimes an experiment visits, one list for every condition or a
dict from condition name to list. Each entry is a comparison or a list of comparisons; the
first is the regime at the start. Then only these regimes and the surfaces between them
count, and a parameter of a regime outside them is reported non-identifiable. A model
without a switch that the declared signs leave open rejects ``regimes``. An experiment
that stays below the threshold never sees ``kl``:

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

``kl`` and ``thr`` are free, each on its own. The start below the threshold is declared,
not derived from the initial values, so the result holds for this experiment, and
``info["assumptions"]["starts"]`` lists the declaration. A start that the initial values
contradict is an error.

A switch on time alone, such as ``time > tau``, is a surface like the others. An event
with ``root`` in place of ``time`` resets its state each time the root expression changes
sign.

When the rule of the switches is not proven for every direction, the rank is a lower
bound and ``info["switching"]["uncertified"]`` says why. The directions the rule leaves
open are no symmetries of the result: the printout lists them as candidates, and
``info["switching"]["candidates"]`` holds them. A model with
switches rejects ``gauge``, ``reduce_cq``, ``free_initial`` and ``engine="symbolic"``, and
a switch on parameters alone needs its side in ``regimes``. With ``equilibrate``, every
resting state of every regime needs a closed form.

Settings
--------

:func:`symident.reconst_control` collects the settings of the saturation and the
reconstruction; pass the result as ``control`` to :func:`~symident.detect`. Among them,
with their defaults:

``degree_cap``, ``degree_cap_max``
   the degree bound of the dense fit (4) and the degree up to which a wider entry is
   probed and fitted (8).
``gap_order_cap``
   the highest order between events after the start (8).
``timeout``
   the time limit of the reconstruction in seconds, once per detection (no limit).
``seed``
   the seed of every random choice (0).

Raise a cap when ``info["limits"]`` names it; :func:`~symident.reconst_control` documents
the others. The same model, settings and seed give the same result, whatever ``cores``.

Diagnostics
-----------

``detect(trace=True)`` prints the stage times; ``trace="full"`` adds the Lie order
diagnostics. Further diagnostics and route choices are set by an environment variable
``SYMIDENT_<NAME>``, or as a keyword of :func:`symident.switches` inside a ``with`` block,
which wins over the environment:

``DEBUG``
   raise the exceptions a route would record in ``info["errors"]``.
``DIAG``
   every diagnostic output on.

They change how a result is reached, not what it reports as proven. The active ones are
recorded in ``info["settings"]["switches"]``, an unknown name gives a warning, and
:doc:`api` lists every switch. :doc:`install` describes
``SYMIDENT_MSOLVE`` and ``SYMIDENT_CACHE``, and how the switches are set from R.

References
----------

.. [Hass2019] H. Hass, C. Loos, E. Raimúndez-Álvarez, J. Timmer, J. Hasenauer and C. Kreutz.
   Benchmark problems for dynamic modeling of intracellular processes. *Bioinformatics*
   35(17), 3073-3082, 2019.

.. [Raia2011] V. Raia, M. Schilling, M. Böhm, B. Hahn, A. Kowarsch, A. Raue, C. Sticht,
   S. Bohl, M. Saile, P. Möller, N. Gretz, J. Timmer, F. Theis, W.-D. Lehmann, P. Lichter and
   U. Klingmüller. Dynamic mathematical modeling of IL13-induced signaling in Hodgkin and
   primary mediastinal B-cell lymphoma allows prediction of therapeutic targets. *Cancer
   Research* 71(3), 693-704, 2011.

.. [Schmiester2021] L. Schmiester, Y. Schälte, F. T. Bergmann et al. PEtab: interoperable
   specification of parameter estimation problems in systems biology. *PLOS Computational
   Biology* 17(1), e1008646, 2021.
