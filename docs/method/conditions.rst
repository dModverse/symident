Conditions, events, resting states and inputs
=============================================

A model is observed under several conditions, with events, from resting states, with inputs or
with switches. This chapter states how each enters the local analysis of :doc:`local` and the
proofs of :doc:`proofs`, and how a PEtab problem maps onto them; the global analysis takes them up
in :ref:`fiber-features`.

.. _conditions-events:

Conditions and events
---------------------

The rows of the conditions :math:`c = 1, \dots, C` are stacked over the shared coordinates, so
that :math:`\ker \mathrm{OIM} = \bigcap_c \ker \mathrm{OIM}^{(c)}`.

A segment starts from the end of the previous one, propagated over the *gap* :math:`\Delta t_i`
between the events. To separate directions that exist only for particular event timings from
generic ones, the gaps are placed on the line :math:`\Delta t_i = \varepsilon\, \Delta t_i^{0}`
through the actual gaps :math:`\Delta t_i^{0}`, and the rank is taken over the Laurent series
:math:`\mathbb{F}_p((\varepsilon))`. The kernel at :math:`\varepsilon = 1` is read from Padé forms
of the kernel entries that further coefficients confirm; this acceptance is heuristic and gives a
basis of the kernel only.

The coefficients of the rows after a gap belong to a family of functions fixed by the model, so
their rank at points uniform modulo :math:`p` bounds the rank over the series from above. The *gap
order* :math:`M`, the number of coefficients kept, rises until the truncated rank reaches this
bound, which proves :math:`M` sufficient. Without a bound (``equilibrate``, recast terms, a
symbolic event time) or at ``reconst_control(gap_order_cap=)``, the rank is a lower bound, and
``info["limits"]`` says where :math:`M` stopped.

**Many conditions.** With many conditions of one segment each, the analysis runs on a subset
:math:`S` of the conditions, chosen at a random point so that each added condition raises the rank
or reaches a new coordinate. Since the rows of :math:`S` are rows of the whole problem,

.. math::

   \operatorname{rank} \mathrm{OIM} \;\ge\; \operatorname{rank} \mathrm{OIM}^{(S)} = d - k

for the :math:`k` directions :math:`\xi_j` found on :math:`S`. The upper bound needs each
:math:`\xi_j` to be a symmetry of every condition :math:`c \notin S`, proven when no coordinate of
the support reaches the output of :math:`c`, when :math:`\xi_j` is proportional to an exact scaling,
or when the lift of the closed form, or at rest the output identities, are proven on :math:`c`
(:ref:`proofs-over-q`). A condition where none applies joins :math:`S`. ``info["selection"]`` lists
:math:`S`; ``SYMIDENT_NOSELECT=1`` analyzes every condition.

.. _steady-states:

Resting states
--------------

With ``equilibrate=True`` every condition starts at rest: the initial value is a *resting state*
:math:`x^\ast(\theta)` with :math:`f(x^\ast, 0, \eta) = 0`, inputs at zero. The *balances* of a
condition are the components of :math:`f` cleared to polynomials, and the *resting variety*
:math:`W` is the closure of :math:`\{(\theta, x) : f(x, 0, \eta(\theta)) = 0\}`. No closed form of
the resting state is needed: at each point the balances are solved over :math:`\mathbb{F}_p`, by
elimination or by msolve :cite:p:`Berthomieu2021`. The kernel is taken in coordinates and resting
states jointly, with the tangency of a direction :math:`(\xi_\theta, \xi_x)` to :math:`W` as extra
rows. With :math:`N` the states that are neither the pivot of a conserved total, nor an input, nor
at zero at rest, and :math:`f_N` their balances, the tangency rows are

.. math::

   J_x\, \xi_{x_N} + J_\theta\, \xi_\theta = 0, \qquad J_x = \partial_{x_N} f_N, \quad
   J_\theta = \partial_\theta f_N .

Dropping the balance of each pivot is exact: each total :math:`\sum_i a_i x_i` gives :math:`\sum_i
a_i f_i = 0`, and the coefficient matrix :math:`C_P` of the totals at the pivots is invertible, so
:math:`f_P = -C_P^{-1} C_N f_N` vanishes wherever :math:`f_N` does. The square :math:`J_x` is the
*resting Jacobian*. A proven rank also needs the resting Jacobian of every condition invertible at
the base point; otherwise the rank is a lower bound.

**Branches.** A *branch* is an irreducible component of :math:`W` over :math:`\mathbb{Q}`. By
default the first condition draws the resting states and solves the balances for the *turnover
rates*, one rate per state that enters the balance of the state linearly (the *forward seed*), which
names the component that the drawn values parametrize. Where a balance has several roots, the one
taken is named in ``info["branches"]``, by the factor over :math:`\mathbb{Q}(\theta)` it solves, or
as *conjugate* where the balances have an irreducible eliminant; conjugate roots have equal rank,
and an eliminant irreducible at a point is irreducible over :math:`\mathbb{Q}(\theta)`. A
state rests at zero when the right-hand side, with the inputs at zero, is linear in it with a
coefficient nonzero at rest; a second branch with the state away from zero is named there and taken
by ``free_initial``. The same holds for a set of states whose right-hand sides all vanish where the
set is zero, an invariant face, when the Jacobian on the set is invertible at generic values; states
of a conserved total make it singular and stay off the face.

**Conserved totals.** A linear combination :math:`T = \sum_i a_i x_i` of states that the right-hand
sides conserve is a *total*; the coefficients may hold parameters, such as volumes, and the basis
read off is positive on the parameter space where possible. Under ``equilibrate`` each total leaves
one resting value free, held by one species of the total, the *pivot*, as initial value.
``free_initial`` names the pivots, otherwise they are chosen by the structure of the model, so that
the result does not depend on the names or the order of the states; other pivots give generators
with the same span. With ``reduce_cq`` the pivots are eliminated, :math:`x_P = C_P^{-1} (T - C_N
x_N)`, and the totals become coordinates, with the same rank and span of directions.

**Rank at the resting start.** The ceiling of the order certificate (:ref:`sufficient-lie-order`)
is the generic rank on the whole resting variety, so the computed rank cannot exceed the rank along
the trajectories from the declared start; it can only fall short of it, and is then reported as a
lower bound or covered by the error bound. The exact scalings bound the rank from above, so a rank
at that bound proves the order sufficient without random points, with ``info["order_bound"]`` zero.

**Diagonal start.** Conditions sharing :math:`\theta` have coupled resting states. Rank and kernel
are generic on the joint variety :math:`\{(\theta, x_1, \dots, x_C) : (\theta, x_c) \in W_c\}`, so
a generic point of it over :math:`\mathbb{F}_p((\varepsilon))` gives them. symident seeks
parameters whose value zero makes all conditions equal, so that a common resting point :math:`\hat
z` exists, and lifts each condition along :math:`\theta(\varepsilon) = \hat\theta + \varepsilon\,
\zeta` to a power series by Newton steps,

.. math::

   x_{c,k} = -J_c(\hat z)^{-1}\, [\varepsilon^k]\, f_c\big(x_{c,<k}, \theta(\varepsilon)\big),
   \qquad J_c = \partial_x f_c ,

the expansion of the unique component through :math:`\hat z` where :math:`J_c(\hat z)` is
invertible. The rank over the truncated series bounds the generic rank from below, and
``info["series_certified"]`` says whether the *series order* is proven sufficient.
``SYMIDENT_NODIAG`` turns the diagonal start off.

**Curve samples.** Where a direct solve of the balances at a random point needs msolve past short
work limits, reconstruction and checks take *curve samples* instead of resting points. A point
:math:`\theta^\ast` is reached on the line :math:`\theta(\varepsilon) = \hat\theta + \varepsilon
(\theta^\ast - \hat\theta)` from a forward seed :math:`\hat z = (\hat\theta, \hat x)` at
:math:`Z = 0`: the seed of :math:`\theta^\ast` itself, whose turnover rates then move along the
line, or an earlier seed that differs from :math:`\theta^\ast` in fewer coordinates. Each
condition is lifted as above, the state columns are eliminated from the series rows, and the
kernel in the coordinates at :math:`\varepsilon = 1`, the kernel at :math:`\theta^\ast`, is the
Padé form of each entry, accepted where the known coefficients determine it with one to spare.
Only :math:`\xi_\theta` is read off: the state components follow from :math:`J_x \xi_{x_N} =
-J_\theta \xi_\theta`. Where an entry has no such form, the series order of the curve samples is
doubled for the rest of the analysis, at most twice from ``gap_order_cap``; past that the point is
skipped like a pole and ``info["limits"]`` says so. The Lie order of a continued condition is
certified at a resting point drawn forward (:ref:`sufficient-lie-order`). ``SYMIDENT_RESTCURVE``
and ``SYMIDENT_NORESTCURVE`` force curve samples or direct solves.

**Existence of a resting state.** Before the analysis the balances are solved over :math:`\mathbb{Q}`
at random rational parameter values. Balances without a complex solution raise an ``InputError``,
unless the modular decision below proves a resting state; a warning is given where no real root lies
in the declared domain. This is a check at random points, not a proof; stability is not considered.
At an integer point :math:`\theta^\ast`, each coordinate uniform in :math:`\{1, \dots, M\}` with
:math:`M = 2^{64}`, the balances and :math:`w \prod_i \mathrm{den}_i - 1` get a Gröbner basis modulo
a prime, and ``info["resting"]["exists"]`` is

- ``"proven"`` where the quotient is finite and every point modulo :math:`p` is simple: Hensel's
  lemma lifts the point to a solution over :math:`W(\mathbb{F}_{p^r})[[t]]`, :math:`\theta =
  \theta^\ast + t`, a domain whose field of fractions holds :math:`\mathbb{Q}(\theta)`, so the
  balances have a solution over :math:`\overline{\mathbb{Q}(\theta)}`;
- ``"no"`` where the basis is :math:`\{1\}` and msolve finds no solution over :math:`\mathbb{Q}`
  either, wrong with probability at most

  .. math::
     :label: eq-rest-none

     \frac{D_\theta}{M}

  per point (``bound``), :math:`D_\theta` the Bezout number of the balances and the saturation:
  the values with an empty fiber of a dominating component :math:`Y` lie in a hypersurface of
  degree at most :math:`\deg Y \le D_\theta` :cite:p:`Heintz1983`, met with probability at most
  :math:`D_\theta / M` by :cite:p:`Schwartz1980`;
- ``"open"`` otherwise.

.. _inputs:

Unknown and known inputs
------------------------

An *unknown input* :math:`w(t)` is an analytic function of time that is left open. A direction of
:math:`\theta` is non-identifiable when nearby :math:`\tilde\theta` with some :math:`\tilde w` give
the output of :math:`(\theta, w)` for a generic :math:`w`.

The Lie derivative :math:`L_F^k h` holds the input only through the *jets* :math:`w_j =
w^{(j)}(t_0)`, :math:`j \le k`. The chain :math:`\dot w = w_1, \dots, \dot w_{K-1} = 0` of length
:math:`K` therefore joins the states :cite:p:`Martinelli2015,Villaverde2019`; it is exact for every
row up to order :math:`K - 1` and is lengthened while the order reaches past it. The jet values
:math:`\hat w` at :math:`t_0` are *nuisance coordinates*: generic, unknown, and not part of
:math:`\theta`. With :math:`\mathrm{OIM}_\theta` and :math:`\mathrm{OIM}_{\hat w}` the columns for
:math:`\theta` and for :math:`\hat w`, the identifiable dimension is

.. math::
   :label: eq-nuisance

   \operatorname{rank} \big[\mathrm{OIM}_\theta \mid \mathrm{OIM}_{\hat w}\big]
   - \operatorname{rank} \mathrm{OIM}_{\hat w} ,

the dimension of the part of the row space without jet components, the counterpart of the
separation of the observable codistribution in :cite:p:`Martinelli2019`; :cite:p:`Villaverde2019`
ask instead for observability of the extended state, jets included.

The chain grows with the order, so the bound of :ref:`sufficient-lie-order` proves no order
sufficient here. The rank is ``proven`` when it is full or when independent exact directions fill
the kernel, else it is a lower bound, in both cases as far as the value of Equation
:eq:`eq-nuisance` holds. That value is proven where the jet columns have full column rank,
verified with the chance by :cite:p:`Schwartz1980` that a maximal minor of
:math:`\mathrm{OIM}_{\hat w}` vanishes at the point, and a lower bound where two points differ. The
chain passes every gap, so the input is analytic across events; at a switch it is only continuous
(:ref:`switching`).

A direction is proven as a field :math:`Y = \xi_\theta\, \partial_\theta + \xi_x\, \partial_x +
\xi_w\, \partial_w`, rational in states, parameters and finitely many jets, with the identities of
:ref:`proofs-over-q` extended by :math:`(\partial_w f)\, \xi_w` and :math:`(\partial_w h)\, \xi_w`;
the jets need no identity, since :math:`\xi_{w_{j+1}} = F(\xi_{w_j})` prolongs :math:`\xi_w`. The
published *input change* is :math:`\xi_w` of a proven field.

A *known input* enters as the same chain, with known symbols as jet values, like the extended Lie
derivatives of :cite:p:`Karlsson2012`, lengthened while the order reaches past it
(``info["known_inputs"]``). The order certificate takes the ranks on the columns of states and
coordinates only: the field of the jets depends on neither, so these rows still form the filtration
of :ref:`sufficient-lie-order`, and a start with special jets, such as a step, can only miss the
ceiling.

.. _switching:

Switching models
----------------

Each switch condition, a comparison or the argument of ``Heaviside()``, ``sign()``, ``abs()``,
``max()`` or ``min()``, is a sign condition :math:`\gamma_j(x, \eta, t) > 0`. The *declared signs*
are the domain declared by ``positive`` and the start sides declared by ``regimes``. Conditions that
the declared signs decide are resolved; a factor of odd multiplicity of an undecided one is a
*switching function* :math:`\gamma_j`, with *switching surface* :math:`\Gamma_j = \{\gamma_j = 0\}`.
A sign vector :math:`\sigma` gives a smooth *regime* :math:`(F_\sigma, h_\sigma)`. A *root event*
(``root`` in place of ``time``) adds the root expression as a switching function and resets the
target state on :math:`\Gamma_j`.

**Regimes and entry points.** A direction counts only if it is a symmetry of every regime the
declared signs allow, from the start of each condition and from every point of a surface, since the
state at a switch is unknown. Without ``regimes``, no sign vector is dropped for parameter values
or for not being reached, so the result holds for every initial state. A state that every allowed
regime keeps at zero from a zero start is dropped (``info["zero_states"]``). Other invariant sets of
the start are not used for a copy switched into a regime, so the rank with switches can exceed the
rank along the trajectories from the declared start; without switches it cannot
(:ref:`steady-states`).

**Visibility.** Two regimes are *neighbors* at :math:`\Gamma_j` when they differ only in the sign of
:math:`\gamma_j`. The switch is *visible* when for some neighbors, observable :math:`h_i` and order
:math:`k`

.. math::

   \big(L_{F_\sigma}^k h_{\sigma,i} - L_{F_{\sigma'}}^k h_{\sigma',i}\big)\big|_{\Gamma_j}
   \ne 0 ,

proven by a nonzero Taylor coefficient at a point of :math:`\Gamma_j` modulo a prime; *hidden*
when the observables and the right-hand sides of every state reaching them agree; and *undecided*
otherwise. A root event surface counts as visible.

**The rule.** Let :math:`\tau_e` be the time and :math:`E_e` the map of the states of event
:math:`e`. A direction :math:`v` is reported non-identifiable when, in every condition, it lies at
the start in a distribution :math:`\mathcal{D} = \operatorname{span}\{Y_1, \dots, Y_m\}` of fields
on parameters and states with

.. math::

   [Y_i, F_\sigma] \in \mathcal{D}, \qquad Y_i(h_\sigma) = 0 \qquad
   \text{for every regime } \sigma,

.. math::

   Y_i(\gamma_j) = 0 \ \text{on } \Gamma_j \ \text{for every visible surface}, \qquad
   Y_i(\tau_e) = 0, \qquad \mathrm{D}E_e\, Y_i(z) \in \mathcal{D}\big(E_e(z)\big) \
   \text{for every event and reset},

and, at a surface that is not visible and that some :math:`Y_i` moves, for every pair of neighbors

.. math::

   \big(F_\sigma - F_{\sigma'}\big)\big|_{\Gamma_j} \in \mathcal{D}, \qquad
   \big(h_\sigma - h_{\sigma'}\big)\big|_{\Gamma_j} = 0 .

No :math:`Y_i` moves a fixed symbol, an input or time. Along every regime flow, :math:`v` stays in
:math:`\mathcal{D}` and keeps the observables to first order; at a visible surface :math:`v` is
tangent, so the crossing time stays; at a hidden one the crossing may move, but the jump lies in
:math:`\mathcal{D}` and the observables agree. The rule therefore never reports a direction that
changes the observables, for any sequence of the allowed regimes.

**Lower bound.** The directions of the rule satisfy the rows of copies of the model: each regime
from the start of each condition, from generic points of the surfaces, and switched at fixed times
into the other regimes, with outputs along *words* :math:`L_{F_\sigma}^a L_{F_{\sigma'}}^b
h_{\sigma'}`. A kernel dimension :math:`k_\Delta` of these rows, projected onto :math:`\theta` off
the *fresh coordinates*, the entry states at surface points, gives the lower bound :math:`d -
k_\Delta`. The projection is proven where the fresh columns have full column rank at the sample
point, ``verified`` with the chance by :cite:p:`Schwartz1980` that a maximal minor vanishes at a
point uniform modulo the verification prime, else a lower bound (``info["projection"]``,
``info["rank_status"]``).

**Upper bound.** The candidates are the scalings of :ref:`scalings`, the closed forms of the kernel
and the unit direction of every coordinate; those failing a condition of the rule are dropped until
the family passes, each identity proven by symbolic simplification. :math:`\ell` candidates left,
independent at a point, bound the rank by :math:`d - \ell`. If :math:`\ell = k_\Delta`, the rank is
:math:`d - k_\Delta` and the candidates span the non-identifiable directions; otherwise :math:`d -
k_\Delta` is a lower bound, ``info["switching"]["uncertified"]`` names the rejected candidates,
and the unproven kernel directions appear under
``info["switching"]["candidates"]``, not as symmetries. For an undecided switch the rule accepts
tangency or a hidden jump and the rows omit the surface, so both bounds stay valid.

**Unknown inputs.** The input is continuous at a crossing and analytic between crossings. The input
direction :math:`G_\sigma = \partial_w F_\sigma` joins the bracket condition,

.. math::

   [Y_i, F_\sigma] = \sum_k c_{ik}\, Y_k + a_{i\sigma}\, G_\sigma ,

and :math:`v = \sum_i \beta_i Y_i` changes the input in :math:`\sigma` by :math:`-\sum_i \beta_i
a_{i\sigma}`. Where :math:`G_\sigma \notin \mathcal{D}`, the change must not jump where a trajectory
crosses :math:`\Gamma_j` into a neighbor :math:`\sigma'`: on :math:`\Gamma_j`,

.. math::

   a_{i\sigma'} - a_{i\sigma}
   - \frac{Y_i(\gamma_j)}{F_\sigma(\gamma_j)} \sum_k e_k\, a_{k\sigma'} = 0 ,
   \qquad F_\sigma - F_{\sigma'} = \sum_k e_k Y_k ,

the last term counting only at a hidden surface that :math:`Y_i` moves. The jets of each copy are
nuisance coordinates (:ref:`inputs`). The rule holds per condition, so each condition needs a
separate input, renamed in ``conditions``.

**Listed regimes.** With ``regimes``, the rule, the rows and the candidates use the listed regimes
of each condition, and a surface counts only with listed regimes on both sides. Parameters of other
regimes stay coordinates and are reported non-identifiable. A start side not decided by the declared
signs is listed under ``info["assumptions"]["starts"]``.

**Limits.** Each of the following is an error, never a result: a surface in the parameters alone
whose side ``regimes`` does not give; a condition on a switched expression; more than 4096 sign
vectors; ``gauge``, ``reduce_cq``, ``free_initial``, ``engine="symbolic"``; a state named
``time``; under ``equilibrate``, a resting state without closed form; an infeasible listed regime or
a start that contradicts the initial values; with unknown inputs, an input read by several
conditions, a switch on an input, an event after the start, or an initial or event value that reads
an input.

.. _petab-mapping:

PEtab problems
--------------

:func:`~symident.read_petab` maps a PEtab problem :cite:p:`Schmiester2021` onto these elements.
Each experiment with measurements is one condition; the estimated parameters are the coordinates,
the other parameters take their nominal values. A preequilibration is the resting start of
``equilibrate`` (:ref:`steady-states`), with the preequilibration condition in place of the inputs
at zero and the condition of the experiment as events at the start; it has to be part of every
experiment or of none.

An experiment without preequilibration whose measurements are all at :math:`t = \infty` is a
*readout at rest*: the output is :math:`h(x^\ast(\theta), \eta(\theta))` at the resting state of the
condition, under the premise that the trajectory converges to that resting state. Only the rows of
order zero at the resting state enter, with the tangency rows of :ref:`steady-states`. A readout at
rest next to an experiment that starts without preequilibration is read as a trajectory, with a
warning.
