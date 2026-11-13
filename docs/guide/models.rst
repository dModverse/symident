Models
======

A model in symident is a set of ordinary differential equations with observables, together
with the experimental conditions, events and inputs under which it is observed. As running
example we consider a precursor ``x1`` that is made basally and on a stimulus ``u``, and that
becomes the observed product ``x2`` or is degraded. From rest, ``u`` switches on at
:math:`t = 0` with one of two doses:

.. math::

   \dot x_1 = k_b + k_{in}\, u - (k_1 + k_l)\, x_1, \qquad
   \dot x_2 = k_1 x_1 - k_2 x_2, \qquad \dot u = 0, \qquad y = s\, x_2 .

The *coordinates* of a model are the quantities whose identifiability is decided: the
estimated parameters, here ``kb``, ``kin``, ``k1``, ``kl``, ``k2`` and ``s``, and the initial
values that neither the model nor ``trafo`` fixes, the *free initial values*. Symbols listed in
``fixed`` are known; they take *generic values*, i.e., values outside a set of measure zero,
random values in the computation.

Reaction networks and ODEs
--------------------------

:class:`symident.Model` takes either a reaction network, i.e., species, a stoichiometry with
one row per reaction, and rate expressions, or the ODEs directly. Both give the same model:

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

The input ``u`` is zero at rest until the event sets it. As ODEs the same model reads

.. code-block:: python

   m = si.Model({"x1": "kb + kin*u - (k1 + kl)*x1", "x2": "k1*x1 - k2*x2", "u": "0"},
                {"y": "s*x2"},
                inputs=["u"],
                events=[{"var": "u", "time": 0, "value": "dose", "method": "replace"}],
                conditions={"rows": ["low", "high"], "cols": {"dose": [1, 10]}})

Arguments
---------

The further arguments of :class:`~symident.Model` describe how the model is observed:

``inputs``
   input states, i.e., inputs that events set, as ``u`` above; they are zero at the start and
   at rest.
``unknown_inputs``, ``known_inputs``
   unknown functions of time, and recorded functions of time of no assumed form
   (:doc:`inputs`).
``events``
   dicts with ``var``, ``time``, ``value`` and ``method`` (``replace``, ``add`` or
   ``multiply``). ``root`` in place of ``time`` fires where an expression changes sign, and
   ``cond`` lists the conditions, by their index from zero, in which the event occurs; by
   default it occurs in all.
``conditions``
   the *condition grid*, one row per condition, i.e., per experiment, as
   ``{"rows": [...], "cols": {name: [values]}}`` or a data frame.
   A number fixes a symbol, and a string replaces it by an expression or another name. A
   column named like a state gives the initial value of that state. Free initial values are
   shared between conditions unless the grid or ``trafo`` sets them.
``trafo``
   a parameter transformation, i.e., a dict of expressions by symbol; an entry named like a
   state sets its initial value. A list gives one dict per condition and must have as many
   entries as the grid has rows.
``fixed``
   known symbols; they take generic values and are no coordinates.
``parameters``
   further symbols treated as parameters; the coordinates stay the same.
``totals``
   conserved totals, name to expression, which the right-hand sides must conserve. When they
   are omitted, they are computed from the linear conservation laws, from the stoichiometry of
   ``reactions`` or from the right-hand sides of the ODEs, and named ``total1``, ``total2``,
   .... ``reduce_cq`` eliminates one species per total (:doc:`detection`).
``equilibrate``, ``positive``, ``reduce_cq``
   defaults for :func:`~symident.detect`, which a call can override; not given, they are
   ``False``, ``True`` and ``False``.

.. _expressions:

Expression language
-------------------

Expressions use ``+``, ``-``, ``*``, ``/``, powers (``^`` or ``**``) and the following
functions:

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
                                operators (:doc:`switches`)
==============================  ===========================================================

The language is that of cppDE (``?cppDE::expressions``), with the following conventions:

- ``acot(x)`` is ``atan(1/x)``, and likewise for the other inverse reciprocal functions.
- ``pi`` is π and cannot name a symbol; ``E`` is a name, ``exp(1)`` is Euler's number, and
  ``time`` is the clock.
- Logarithms, roots, fractional powers and bases of free exponents need an argument that is
  positive on the ``positive`` coordinates, also in initial and event values.
- A rational power is written as a fraction, ``x^(1/2)``, not as ``x^0.5``.
- ``floor``, ``ceiling`` and ``round`` are refused; their steps are written with
  ``piecewise()``.
- Other functions, such as ``erf`` and ``erfc``, need ``engine="symbolic"``, which accepts
  every SymPy function but suits small models only. Requests for further functions are
  welcome at https://github.com/dModverse/symident/issues.
