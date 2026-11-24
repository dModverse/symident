PEtab problems
==============

Many published models are available as `PEtab <https://petab.readthedocs.io>`_ problems
:cite:p:`Schmiester2021`, which bundle an SBML model with conditions, observables and the estimated
parameters. :func:`symident.read_petab`, or :meth:`symident.Model.from_petab`, reads such a
problem, format 1 or 2, into a :class:`~symident.Model`; it needs ``symident[petab]``. As an
example we take the model of :cite:p:`Raia2011` from the
`benchmark collection <https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab>`_
:cite:p:`Hass2019`:

.. code-block:: python

   import symident as si

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

Each readout scale of a transcript trades against the production rate of the transcript, and
``res.reduce()`` fixes ``SOCS3_accumulation = 1`` and ``CD274mRNA_production = 1``.

The mapping is documented in :func:`~symident.read_petab` and recorded in ``m.petab``
(experiments, estimated parameters, fixed values, renamed ids):

- estimated parameters stay symbols, the others take their nominal values;
- each experiment with measurements is one condition. A preequilibration sets
  ``equilibrate``; it has to be part of every experiment or of none, since symident starts all
  conditions alike. With conserved totals in a preequilibrated network, ``reduce_cq`` is set;
- an experiment without preequilibration whose measurements are all at time ``inf`` is a
  readout at rest: the system starts from the values of the condition, rests under it, and
  only the resting values of the observables enter. This assumes that a resting state exists
  and is the one the experiment reaches. ``m.petab["at_rest"]`` lists these experiments; an
  experiment with a finite time keeps the trajectory;
- estimated parameters on a log scale or with a positive lower bound form
  ``Model.positive``, the default of ``detect(positive=)``;
- the noise model does not enter;
- an id that the expression language cannot read, such as one with a leading underscore, is
  renamed;
- an SBML or PEtab feature without counterpart, such as an algebraic rule, a delay or a fast
  reaction, raises :class:`~symident.PEtabError`.
