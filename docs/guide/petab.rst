PEtab problems
==============

Many published models are available as `PEtab <https://petab.readthedocs.io>`_ problems
[Schmiester2021]_, which bundle an SBML model with conditions, observables and the estimated
parameters. :func:`symident.read_petab`, or :meth:`symident.Model.from_petab`, reads such a
problem, format 1 or 2, into a :class:`~symident.Model`; it needs ``symident[petab]``. As an
example we take the model of [Raia2011]_ from the
`benchmark collection <https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab>`_
[Hass2019]_:

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
- estimated parameters on a log scale or with a positive lower bound form
  ``Model.positive``, the default of ``detect(positive=)``;
- the noise model does not enter;
- an id that the expression language cannot read, such as one with a leading underscore, is
  renamed;
- an SBML or PEtab feature without counterpart, such as an algebraic rule, a delay or a fast
  reaction, raises :class:`~symident.PEtabError`.

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
