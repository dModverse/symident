Switches and regimes
====================

Many models switch between different dynamics, e.g., a degradation that sets in above a
threshold. symident accepts such switches in the right-hand sides and observables. Each
undecided comparison defines a *switching function*, whose zero set is a *switching surface*,
and each sign pattern of the switching functions a *regime*, the smooth model on one side of
every surface. The *declared signs* are the domain given by ``positive`` and the start sides
given by ``regimes``. The switch
functions are those of the expression table (:ref:`expressions`), with the comparisons ``<``,
``>``, ``<=``, ``>=`` and the operators ``&&``, ``||``, ``!``.
``piecewise(v1, c1, v2, c2, ..., otherwise)`` follows the SBML order and is ``0`` where no
condition holds and no ``otherwise`` is given; ``ifelse(c, a, b)`` is the two-way case.
Conditions that the ``positive`` coordinates decide are resolved: for a positive ``x``,
``abs(x)`` is ``x`` and ``x > 0`` is true. ``x == c`` is false and ``x != c`` is true unless
both sides agree identically.

A coordinate is non-identifiable only if it is so in every regime the declared signs allow.
A parameter of a regime that the declared signs exclude, such as ``k2`` in
``piecewise(-k1*x, x > 0, -k2*x)`` with a positive ``x``, is reported non-identifiable. As an example we take the running example of
:doc:`models` and let the loss of ``x1`` with rate ``kl`` act only above a threshold ``thr``:

.. code-block:: python

   import symident as si

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

The general direction of the model without threshold (:doc:`detection`) is gone: the
observable registers the change of the outflow from ``k1`` to ``k1 + kl`` at the threshold, so
``k1`` and ``kl`` are determined separately. The scaling remains, with ``thr`` moving along;
``fixed=["thr"]`` breaks it too.

A reported direction leaves the observables of every regime unchanged and keeps the visible
switching surfaces, the events and the given initial values. Every sign pattern of the
switching functions is a regime, reached or not. A parameter of a regime that is never entered
is therefore reported identifiable unless ``regimes`` excludes that regime. The state at a
switch is any point of the surface; :doc:`../method` states the rule and the exception of
states that stay at zero (``info["zero_states"]``).

Visibility
----------

A switch is *visible* when the observables tell the two sides apart on the surface, as in the
example above. It is *invisible* when the observables agree on both sides and every state they
read has the same right-hand side on both sides, as when the switch acts only on states that
never reach the observables; the surface may then move. Otherwise it is *undecided*.
``info["switching"]["conditions"]`` gives the visibility per condition:

.. code-block:: python

   m2 = si.Model({"x": "-k*x", "z": "piecewise(-a*z, z > c, -b*z)"}, {"y": "s*x"})
   r2 = si.detect(m2, reconstruct=True)
   r2.info["switching"]["conditions"][0]["visibility"]
   # {'c - z = 0': 'invisible'}

Here ``a``, ``b``, ``c`` and the initial value of ``z`` are free, and ``s`` and the initial
value of ``x`` enter only as ``s*x``.

Visited regimes
---------------

``regimes`` lists the visited regimes, one list for every condition or a dict by condition
name. An entry is a comparison or a list of them, and the first entry is the regime at the
start. Only these regimes and the surfaces between them count, and a parameter of another
regime is reported non-identifiable. A model without a switch that the declared signs leave
open rejects ``regimes``. An experiment below the threshold never sees ``kl``:

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

``kl`` and ``thr`` are now free, independently of each other. The start is declared, not
derived, so the result holds for this experiment; ``info["assumptions"]["starts"]`` lists it,
and a start that the initial values contradict is an error.

A switch on time alone, such as ``time > tau``, is a surface like the others. An event with
``root`` resets ``var`` each time the root expression changes sign. Where the rule is not
proven for every direction, the rank is a lower bound, ``info["switching"]["uncertified"]``
says why, and the printout lists the open directions as candidates
(``info["switching"]["candidates"]``), not as symmetries. :doc:`../method` lists the options
and models that switches refuse.
