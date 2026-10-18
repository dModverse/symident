symident
========

Structural identifiability of ODE models by Lie symmetries over finite fields.

symident finds the non-identifiable directions of an ODE model with observables,
conditions, events and inputs, returns them as exact generators where possible, and
reduces the model to identifiable parameters. The rank of the OIM is computed over
:math:`\mathbb{F}_p` in a C++ kernel; closed forms are reconstructed by sparse
interpolation and proven over Q, or checked at random points modulo a large prime, with an
error bound where one applies.

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
