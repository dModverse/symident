User guide
==========

Running example: a precursor ``x1``, made basally and on a stimulus ``u``, becomes the
observed product ``x2`` or is degraded; from rest, ``u`` switches on at :math:`t = 0` with
one of two doses.

Model formulation
-----------------

.. math::

   \dot x_1 = k_b + k_{in}\, u - (k_1 + k_l)\, x_1, \qquad
   \dot x_2 = k_1 x_1 - k_2 x_2, \qquad \dot u = 0, \qquad y = s\, x_2 .

:class:`symident.Model` takes a reaction network (species, stoichiometry with one row per
reaction, rates) or ODEs; both give the same model:

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

The input ``u`` is zero at rest until the event. As ODEs:

.. code-block:: python

   m = si.Model({"x1": "kb + kin*u - (k1 + kl)*x1", "x2": "k1*x1 - k2*x2", "u": "0"},
                {"y": "s*x2"},
                inputs=["u"],
                events=[{"var": "u", "time": 0, "value": "dose", "method": "replace"}],
                conditions={"rows": ["low", "high"], "cols": {"dose": [1, 10]}})

Arguments of :class:`~symident.Model`:

``inputs``
   input states, zero at the start and at rest; events set them.
``unknown_inputs``
   unknown functions of time (see `Unknown inputs`_).
``known_inputs``
   known functions of time of no assumed form, such as a recording; values and time
   derivatives are known symbols, as many as the Lie order needs (``info["known_inputs"]``).
``events``
   dicts with ``var``, ``time``, ``value`` and ``method`` (``replace``, ``add``,
   ``multiply``); ``root`` in place of ``time`` fires where an expression changes sign.
``conditions``
   one row per condition, as ``{"rows": [...], "cols": {name: [values]}}`` or a data frame;
   a number fixes a symbol, a string replaces it by an expression or name. Free initial
   values are shared unless the grid or ``trafo`` sets them.
``trafo``
   a parameter transformation; an entry named like a state sets the initial value.
``fixed``
   known symbols; they take generic values.
``parameters``
   further symbols treated as parameters; the coordinates stay the same.
``totals``
   conserved totals, name to expression, conserved by the right-hand sides; computed from
   ``reactions`` when omitted. ``reduce_cq`` eliminates one species per total.

.. _expressions:

Expression language
~~~~~~~~~~~~~~~~~~~

Expressions use ``+``, ``-``, ``*``, ``/``, powers (``^`` or ``**``) and these functions:

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
                                operators (see `Switches and regimes`_)
==============================  ===========================================================

The language is that of cppDE (``?cppDE::expressions``):

- ``acot(x)`` is ``atan(1/x)``, likewise for the other inverse reciprocal functions.
- ``pi`` is π and cannot name a symbol; ``E`` is a name; ``exp(1)`` is Euler's number;
  ``time`` is the clock.
- Logarithms, roots, fractional powers and bases of free exponents need an argument
  positive on the ``positive`` coordinates, also in initial and event values.
- A rational power is a fraction: ``x^(1/2)``, not ``x^0.5``.
- ``floor``, ``ceiling`` and ``round`` are refused; write their steps with ``piecewise()``.
- Other functions, such as ``erf`` and ``erfc``, need ``engine="symbolic"`` (every SymPy
  function, small models); requests to https://github.com/dModverse/symident/issues.

Symmetry detection
------------------

:func:`symident.detect` returns the non-identifiable directions as
:class:`~symident.Symmetries`. ``equilibrate=True`` starts every condition at a steady state
with the inputs at zero:

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

``ξ(θ)`` is the rate of change of the coordinate ``θ``, a parameter or free initial value,
along the direction; unchanged coordinates are omitted. The observables stay the same:

- :math:`X_1` scales ``kb`` and ``kin`` up and ``s`` down: only ``kb*s`` and ``kin*s``
  are identifiable;
- along :math:`X_2` the outflow ``k1 + kl`` and the flux into ``x2``, ``k1*kb/(k1 + kl)``
  at rest and ``k1*kin/(k1 + kl)`` per dose, stay fixed, which is all ``x2`` sees.

The ``level`` of a direction is

``exact``
   proven over Q;
``verified``
   checked at random points, error probability in ``info["bound"]``;
``checked``
   the same test without an error bound, or on part of the entries;
``open``
   no closed form.

``res.unproven()`` returns the verified and checked directions. The rank is ``proven``,
``verified`` (with a probability) or a ``lower bound`` (then the model may have fewer
non-identifiable directions); ``info["rank_status"]`` gives status and reason, the printout
flags an unproven rank, and :doc:`method` states when each applies.

A discrete symmetry, such as two rate constants entering only through sum and product,
survives local identifiability. ``res.solution_count()`` counts, over the algebraic
closure, the coordinate values giving the output of a random point modulo a prime, also
outside the declared domain (``-k`` where only ``k^2`` enters); a count above one is a
candidate at that point, not proven. It needs msolve (:doc:`install`) and one condition
without events, switches or unknown inputs, detected without ``equilibrate`` and
``reduce_cq``, and returns None for a non-rational model.

Options of :func:`~symident.detect` (the others are documented there):

``reconstruct``
   general directions as exact rational functions, otherwise by their support.
``equilibrate``
   start every condition at a steady state, inputs at zero. Each conserved total adds a
   free resting value; a direction moving it is reconstructed, on the total, with
   ``reduce_cq=True``, else it may stay without closed form (``info["limits"]``).
``positive``
   coordinates known to be positive (default all); logarithms, roots and switch conditions
   are decided on this domain. An entry may be an expression, such as ``"1-x"`` for a
   fraction ``x`` below one; :func:`~symident.reduce` then prefers sections inside it.
``gauge``
   ``False`` reconstructs every direction; ``None`` fixes one coordinate per scaling,
   chosen to keep the general directions narrow; a list ranks the coordinates to fix.
``verify``
   check the rank where the Lie order is not proven sufficient; ``False`` skips it.
``scalings_only``
   only the scaling symmetries, from an exact integer kernel.
``cores``
   threads and worker processes.

``res.summary()`` adds Lie order, rank proof, suggested gauge and what stopped part of the
detection. ``res.info`` (keys under :class:`~symident.Symmetries`) holds the details; a
nonempty ``info["limits"]`` means a direction without closed form may have one, and
``info["assumptions"]`` holds declared domain, switch starts and the error probability of a
verified result. ``res.symmetries`` lists :class:`~symident.Symmetry` objects, ``to_dict()``
the raw result.

Reduction to identifiable parameters
------------------------------------

:func:`symident.reduce` (or ``res.reduce()``) removes the directions: a scaling by fixing
one coordinate of it, a general direction by the invariants, the combinations constant
along it, which become new parameters ``q_1, q_2, ...``:

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

``q_1`` is the outflow rate of ``x1``, ``q_2`` and ``q_3`` the basal and stimulated flux
into ``x2`` in units of the signal; ``k2`` stays. Each line under ``Sections`` is one block
of jointly reduced directions and picks one point per orbit; ``red.summary(detailed=True)``
says why. ``kl = 0`` lies on the boundary, which the orbits reach only in a limit; the model
is continuous there, so the limit gives the output of the orbit.

``red.level`` is ``exact`` when every claim is proven, ``verified`` or ``checked`` when a
direction rests on a closed form of that level, and ``open`` when a section claim is not
proven (named in the printout).
``red.checks`` holds the check of the trafo over Q and, with ``check=True``, the detection on
the reduced model, flagged in the summary line when it does not confirm the reduction.
Unproven directions:

- ``open``: an error unless ``allow_open=True``, which leaves the direction in place;
- ``checked``, or a saturation guard that saw the rank still grow: an error unless
  ``allow_checked=True``;
- ``verified``: a warning.

``red.trafo`` is the transformation as a dict of strings:

.. code-block:: python

   red.trafo
   # {'k1': 'q_1', 'k2': 'k2', 'kb': 'q_2', 'kin': 'q_3', 'kl': '0', 's': '1'}

An entry for a symbol of the condition grid goes into the grid cells. Under
``equilibrate`` an entry named like a state pins the resting value; a numeric one goes into ``fixed``.
``check=True`` applies the trafo this way.

PEtab problems
--------------

:func:`symident.read_petab` (or :meth:`symident.Model.from_petab`) reads a
`PEtab <https://petab.readthedocs.io>`_ problem [Schmiester2021]_, format 1 or 2, into a
:class:`~symident.Model`; it needs ``symident[petab]``. The model of [Raia2011]_ from the
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

Each transcript readout scale trades against the production rate of the transcript;
``res.reduce()`` fixes ``SOCS3_accumulation = 1`` and ``CD274mRNA_production = 1``.

The mapping, documented in :func:`~symident.read_petab` and recorded in ``m.petab``
(experiments, estimated parameters, fixed values, renamed ids):

- estimated parameters stay symbols, the others take their nominal values;
- each experiment with measurements is one condition; a preequilibration sets
  ``equilibrate``;
- estimated parameters on a log scale or with a positive lower bound form
  ``Model.positive``, the default of ``detect(positive=)``;
- the noise model does not enter;
- an id the expression language cannot read, such as one with a leading underscore, is
  renamed;
- an SBML or PEtab feature without counterpart, such as an algebraic rule, a delay or a
  fast reaction, raises :class:`~symident.PEtabError`.

Unknown inputs
--------------

An *unknown input*, such as an emission history or an intake, is a function of time of no
assumed form: a symbol without equation, listed in ``unknown_inputs``. The parameters are
analyzed whatever the input does; a direction may change the input, which does not count in
``rank`` and ``dim``.

An input of known form (dosing steps, a physical law, the spline a fit will use) belongs in
the model, coefficients as parameters, and the result holds for that form only.
Non-identifiable with an unknown input implies non-identifiable under every flexible enough
parametrization; the converse fails, since identifiability under a parametrization may
rest on taking the form as exact. A recorded input of unmodeled form goes into
``known_inputs``; the result then holds for a generic recorded course.

A glacier length ``L`` relaxing to the temperature anomaly ``T`` on the time scale ``tau``:

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

Only the initial length is identifiable: ``c`` trades against the scale of ``T``, and every
``tau`` fits with ``T = -(tau*L' + L)/c``. ``Symmetry.input_change`` holds :math:`\xi(T)`
per exact direction; type and reduction concern parameters and initial values only
(``res.reduce()`` fixes ``c = 1``, ``tau = 1``).

- All conditions share the input; a string in ``conditions`` renames it (a separate input
  if the new name is in ``unknown_inputs``), a number fixes it.
- The input is analytic across an event, so a state jump can separate parameters the
  input absorbs otherwise.
- Under ``equilibrate`` it is zero at rest and arbitrary from the start on.
- With switches each condition needs a separate input (:doc:`method`).
- Refused: an event or ``trafo`` entry for the input, ``engine="symbolic"``.

.. warning::

   A polynomial stand-in for an unknown input, a chain ``w' = w1, ..., wd' = 0`` of states
   with free initial values, is :math:`d + 1` numbers, not an unknown input. Past Lie order
   :math:`d + 1` the output determines them, and parameters trading against the input look
   identifiable: with a cubic stand-in for ``T`` the glacier model reports ``tau``
   identifiable. A spline, a polynomial per segment, looks the same to the local analysis.

``info["unknown_inputs"]`` lists per input the jets (:doc:`method`), the Lie order up to
which they are exact, and whether the analysis went past it. ``reconst_control(input_jet_order=)``
sets the number of jets, which grows with the model; too short a chain makes the rank a
lower bound (``info["limits"]``). A direction without a lift as a function of states,
parameters and input stays ``checked``, without input change.

Stationary spatial profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A steady-state profile :math:`0 = F(x, u, u_x, u_{xx}, \theta)` in one space dimension is
an ordinary model, a first-order system with :math:`x` as time. The morphogen gradient
:math:`D u'' = k u`:

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

Only the decay length :math:`\sqrt{D/k}` is identifiable; the initial values are profile
and slope at the first point. A source of unknown shape, ``v = (k*u - s)/D`` with
``unknown_inputs=["s"]``, frees ``D`` and ``k`` separately.

Switches and regimes
--------------------

The switch functions are those of the expression table, with the comparisons ``<``,
``>``, ``<=``, ``>=`` and the operators ``&&``, ``||``, ``!``.
``piecewise(v1, c1, v2, c2, ..., otherwise)`` takes SBML order and is ``0`` where no
condition holds and no ``otherwise`` is given; ``ifelse(c, a, b)``. Conditions decided by the
``positive`` coordinates are resolved: for a positive ``x``, ``abs(x)`` is ``x`` and
``x > 0`` true. ``x == c`` is false and ``x != c`` true unless both sides agree
identically.

A coordinate is non-identifiable only if it is so in every regime the declared signs allow;
a parameter of an excluded regime, such as ``k2`` in ``piecewise(-k1*x, x > 0, -k2*x)``
with a positive ``x``, is reported non-identifiable. With degradation of ``x1`` only above a
threshold ``thr``:

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

:math:`X_2` is gone: the observable registers the change of outflow from ``k1`` to
``k1 + kl`` at the threshold, so ``k1`` and ``kl`` are determined separately. The scaling remains, with ``thr`` moving along;
``fixed=["thr"]`` breaks it too.

A reported direction leaves the observables of every regime unchanged and keeps the visible
switching surfaces, the events and the given initial values. Every sign pattern of the
switching functions is a regime, reached or not, so a parameter of a regime never entered is
reported identifiable unless ``regimes`` excludes that regime. The state at a switch is any
point of the surface; :doc:`method` states the rule and the exception of states that stay
at zero (``info["zero_states"]``).

A switch is *visible* when the observables tell the two sides apart on the surface, as
here, *invisible* when it acts only on states that never reach the observables, so that the
surface may move, and *undecided* otherwise; ``info["switching"]["conditions"]`` gives the
visibility per condition:

.. code-block:: python

   m2 = si.Model({"x": "-k*x", "z": "piecewise(-a*z, z > c, -b*z)"}, {"y": "s*x"})
   r2 = si.detect(m2, reconstruct=True)
   r2.info["switching"]["conditions"][0]["visibility"]
   # {'c - z = 0': 'invisible'}

``a``, ``b``, ``c`` and the initial value of ``z`` are free; ``s`` and the initial value of
``x`` enter only as ``s*x``.

``regimes`` lists the visited regimes, one list for every condition or a dict by condition
name; an entry is a comparison or a list of them, the first the regime at the start. Only
these regimes and the surfaces between them count; a parameter of another regime is
reported non-identifiable. A model without a switch left open by the
declared signs rejects ``regimes``. An experiment below the threshold never sees ``kl``:

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

``kl`` and ``thr`` are free, independently. The start is declared, not derived, so the
result holds for this experiment; ``info["assumptions"]["starts"]`` lists it, and a start
the initial values contradict is an error.

A switch on time alone, such as ``time > tau``, is a surface like the others. An event with
``root`` resets ``var`` each time the root expression changes sign. Where the rule is not
proven for every direction, the rank is a lower bound, ``info["switching"]["uncertified"]``
says why, and the printout lists the open directions as candidates
(``info["switching"]["candidates"]``), not as symmetries. :doc:`method` lists the options
and models that switches refuse.

Reconstruction settings
-----------------------

:func:`symident.reconst_control` builds the ``control`` of :func:`~symident.detect`
(saturation and reconstruction); defaults in parentheses:

``degree_cap``, ``degree_cap_max``
   the degree bound of the dense fit (4) and the degree up to which a wider entry is
   probed and fitted (8).
``gap_order_cap``
   the highest order between events after the start (8).
``timeout``
   the time limit of the reconstruction in seconds, once per detection (no limit).
``seed``
   the seed of every random choice (0).

Raise a cap that ``info["limits"]`` names; :func:`~symident.reconst_control` lists the
others. Model, settings and seed fix the result for any ``cores``.

Diagnostic output
-----------------

``detect(trace=True)`` prints stage times, ``trace="full"`` also Lie order diagnostics.
Switches set further diagnostics and route choices, as an environment variable
``SYMIDENT_<NAME>`` or, taking precedence, a keyword of :func:`symident.switches` in a
``with`` block:

``DEBUG``
   raise the exceptions a route would record in ``info["errors"]``.
``DIAG``
   every diagnostic output on.

Switches change how a result is reached, not what it proves. ``info["settings"]["switches"]``
records the active ones; an unknown name warns. :doc:`api` lists all, :doc:`install`
covers ``SYMIDENT_MSOLVE``, ``SYMIDENT_CACHE`` and R.

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
