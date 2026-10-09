symident
========

Structural identifiability of ODE models by Lie symmetries over finite fields.

symident finds the non-identifiable directions of an ODE model with observables,
conditions, events and inputs, returns them as exact generators where possible, and
reduces the model to identifiable parameters. The observability rank is computed modulo
primes in a C++ kernel; closed forms are reconstructed by sparse interpolation and
verified at fresh primes.

.. code-block:: python

   import symident as si

   m = si.Model({"A": "-k1*A", "B": "k1*A - k2*B"}, {"y": "s*B"})
   res = si.detect(m, reconstruct=True)
   red = res.reduce()

.. toctree::
   :maxdepth: 2

   install
   quickstart
   guide
   method
   api
   changelog
