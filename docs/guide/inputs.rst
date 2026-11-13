Unknown and known inputs
========================

Unknown inputs
--------------

Often a model is driven by a quantity that is not measured and whose course in time is not
known, e.g., an unmeasured stimulus or an intake. symident treats such a quantity as an
*unknown input*: a function of time of no assumed form, given as a symbol without equation and
listed in ``unknown_inputs``. The parameters are then analyzed for whatever the input does. A
non-identifiable direction may change the input along with the parameters; the input itself
does not count in ``rank`` and ``dim``, the rank and the number of coordinates.

An input of known form, such as dosing steps, a physical law or the spline a fit will use,
belongs in the model, with its coefficients as parameters, and the result then holds for that
form only. The two questions are related: a direction that is non-identifiable with an unknown
input is non-identifiable under every flexible enough parametrization of the input. The
converse fails, since identifiability under a parametrization may rest on taking the form as
exact. A recorded input of unmodeled form goes into ``known_inputs``, and the result then holds
for a generic recorded course.

As an example we consider a readout ``x`` that follows an unmeasured stimulus ``u`` with gain
``k`` on the time scale ``tau``:

.. code-block:: python

   import symident as si

   m = si.Model({"x": "(k*u - x)/tau"}, {"y": "x"}, unknown_inputs=["u"])
   res = si.detect(m, reconstruct=True)
   print(res)

.. code-block:: text

   Result:  rank 1 / 3  |  2 non-identifiable directions

   Generators  X = Σᵢ ξ(i) ∂ᵢ

   Scalings:
     X₁  (exact)
         ξ(k) :  + k
         input change:
         ξ(u) :  - u

     X₂  (exact)
         ξ(tau) :  + tau
         input change:
         ξ(u) :  + (k*u - x)/k

   exact = proven over Q.

Only the initial value of ``x`` is identifiable: ``k`` trades against the scale of ``u``, and
every ``tau`` fits with ``u = (tau*x' + x)/k``. ``Symmetry.input_change`` holds the component
:math:`\xi(u)` of each exact direction; the type of a direction and the reduction concern
parameters and initial values only, so ``res.reduce()`` fixes ``k = 1`` and ``tau = 1``.

A cubic in place of ``u``, i.e., the states ``u' = u1, u1' = u2, u2' = u3, u3' = 0`` with free
initial values, makes ``tau`` identifiable: four numbers, which the output determines, replace
a function of time. A spline does the same.

The following rules apply to unknown inputs:

- All conditions share the input. A string in ``conditions`` renames it, which gives a
  separate input if the new name is in ``unknown_inputs``, and a number fixes it.
- The input is analytic across an event, so a state jump can separate parameters that the
  input absorbs otherwise.
- Under ``equilibrate`` the input is zero at rest and arbitrary from the start on.
- With switches each condition needs a separate input (:doc:`../method`).
- An event or ``trafo`` entry for the input and ``engine="symbolic"`` are refused.

``info["unknown_inputs"]`` lists per input the jets (:doc:`../method`), the Lie order up to
which they are exact, and whether the analysis went past it.
``reconst_control(input_jet_order=)`` sets the number of jets, which grows with the model; too
short a chain makes the rank a lower bound (``info["limits"]``). A direction without a *lift*,
i.e., without components on states and input as functions of states, parameters and input that
prove it (:doc:`../method`), stays ``checked``, without input change.

Known inputs
------------

A *known input* is a function of time that is recorded but has no assumed form. It is listed in
``known_inputs``; its values and time derivatives at the start are known symbols, as many as
the Lie order needs, and ``info["known_inputs"]`` reports them. The result holds for a generic
recorded course of the input.
