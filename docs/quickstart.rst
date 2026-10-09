Quickstart
==========

A model with two states, one observed up to an unknown scale:

.. code-block:: python

   import symident as si

   m = si.Model({"A": "-k1*A", "B": "k1*A - k2*B"}, {"y": "s*B"})
   res = si.detect(m, reconstruct=True)
   print(res)

.. code-block:: text

   Result:  rank 4 / 5  |  1 non-identifiable direction

   Generators  X = Σᵢ η(i) ∂ᵢ

   Scalings:
     X₁
         η(A) :  + A
         η(B) :  + B
         η(s) :  - s

The coordinates are the parameters ``k1``, ``k2``, ``s`` and the initial values of ``A``
and ``B``. Four combinations of them are identifiable; the fifth direction scales both
initial values up and ``s`` down without changing ``y``. :meth:`~symident.Symmetries.reduce`
removes it:

.. code-block:: python

   print(res.reduce())

.. code-block:: text

   Reduced 1 of 1 direction.

   Trafo (non-identity entries)
     A = 1

   Sections: the invariants name each orbit, the chart takes one point on it
     {X₁}  A = 1
         pin: a scaling orbit is a ray, so any positive value meets it exactly once

Fixing the initial value of ``A`` at one leaves an identifiable model.
