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

   Generators  X = Σᵢ ξ(i) ∂ᵢ

   Scalings:
     X₁  (exact)
         ξ(A) :  + A
         ξ(B) :  + B
         ξ(s) :  - s

   exact = proven over Q.

The coordinates are the parameters ``k1``, ``k2``, ``s`` and the initial values of ``A``
and ``B``. Four combinations of them are identifiable; the fifth direction scales both
initial values up and ``s`` down without changing ``y``. The printout gives the direction
as a vector field: ``ξ(A) = A`` says that ``A`` changes at the rate ``A``, so ``A`` and
``B`` grow by the same factor while ``s`` shrinks by it. The :doc:`guide` explains the
notation. :meth:`~symident.Symmetries.reduce` removes the direction:

.. code-block:: python

   print(res.reduce())

.. code-block:: text

   Reduced 1 of 1 direction.

   Trafo (non-identity entries)
     A = 1

   Sections:
     {X₁}  A = 1
     summary(detailed=True) explains each section.

Fixing the initial value of ``A`` at one leaves an identifiable model: every orbit of the
scaling is a ray and meets ``A = 1`` exactly once.
