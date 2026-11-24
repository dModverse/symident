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

A segment starts from the end of the previous one, propagated over the *gap*
:math:`\Delta t_i` between the events. To separate directions that exist only for particular
event timings from generic ones, the gaps are placed on the line
:math:`\Delta t_i = \varepsilon\, \Delta t_i^{0}` through the actual gaps :math:`\Delta
t_i^{0}`, and the rank is taken over the Laurent series :math:`\mathbb{F}_p((\varepsilon))` in
the formal variable :math:`\varepsilon`. The kernel at :math:`\varepsilon = 1` is read from
Padé forms, i.e., rational functions of :math:`\varepsilon` matching the series of each kernel
entry, that further coefficients confirm; this acceptance is heuristic and gives a basis of the
kernel only.

The rows after a gap are power series in :math:`\varepsilon` whose coefficients belong to a
family of functions fixed by the model. Their rank, at points uniform modulo :math:`p`,
therefore gives an upper bound of the rank over the series, with error terms that enter the
order bound. The *gap order* :math:`M`, the number of coefficients kept, rises until the
truncated rank reaches this bound, which proves :math:`M` sufficient; a truncated rank above
the bound refutes it, and nothing is certified. A symbolic event time gives no bound, since it
moves the boundary and shifts the series and the event state outside the family. Without bound
(``equilibrate``, recast terms, such an event time) or at ``reconst_control(gap_order_cap=)``,
the rank is a lower bound, and ``info["limits"]`` says where :math:`M` stopped.

**Many conditions.** With 24 or more conditions of one segment each, without unknown inputs,
recast terms or, at rest, conserved totals, the analysis runs on a subset :math:`S` of the
conditions. The conditions are taken one of each *structure* in turn, a structure being the set
of coordinates that reach the output of a condition at some Lie order. At a random point, a
condition joins :math:`S` where the rows of the condition raise the rank of the rows of the
conditions before it, or where it is the first to reach a coordinate; the rows go to two orders
past the largest first order of a coordinate, and again to the order the analysis of :math:`S`
reaches, with the conditions of :math:`S` first. At rest the rows of a condition are those on
:math:`\theta` after the resting states of the condition are eliminated. Since the rows of
:math:`S` are rows of the whole problem,

.. math::

   \operatorname{rank} \mathrm{OIM} \;\ge\; \operatorname{rank} \mathrm{OIM}^{(S)} = d - k

for the :math:`k` directions :math:`\xi_1, \dots, \xi_k` found on :math:`S`. The upper bound
needs each :math:`\xi_j` to be a symmetry of every condition :math:`c \notin S`, which is
proven in one of four ways: no coordinate of the support of :math:`\xi_j` reaches the output of
:math:`c`, so that every row of :math:`c` vanishes on the support; :math:`\xi_j` is
proportional to an exact scaling of the whole model, checked on one condition per group of
equal monomials; the lift of the closed form of :math:`\xi_j` is proven on :math:`c` as in
:ref:`proofs-over-q`; or, at rest, the output identities hold at the resting states of :math:`c`.
Then :math:`\ker \mathrm{OIM} \supseteq \operatorname{span}(\xi_1, \dots, \xi_k)` and
the two ranks agree, proven when every :math:`\xi_j` is exact on :math:`S`. A condition where a
direction leaves the kernel at a random point, or where none of the four applies, joins
:math:`S` and the analysis repeats; where :math:`S` would hold more than half of the conditions,
or a direction has no exact closed form, every condition is analyzed. ``info["selection"]``
lists :math:`S` and how the other conditions are covered; ``SYMIDENT_NOSELECT=1`` analyzes every
condition.

.. _steady-states:

Resting states
--------------

With ``equilibrate=True`` every condition starts at rest: the initial value is a *resting
state* :math:`x^\ast(\theta)` with :math:`f(x^\ast, 0, \eta) = 0`, inputs at zero. The
*balances* of a condition are the components of :math:`f` cleared to polynomials, and the
*resting variety* :math:`W` is the closure of :math:`\{(\theta, x) : f(x, 0, \eta(\theta)) =
0\}`. No closed form of the resting state is needed: at each point the balances are solved over
:math:`\mathbb{F}_p`, by linear elimination, by resultants for coupled subsystems of one or two
states, or by msolve :cite:p:`Berthomieu2021`. Only msolve excludes states at zero, by a Rabinowitsch
variable :math:`\tau` with :math:`\tau \prod_i x_i = 1`, and a resting state solved modulo one
prime is not resampled at another. The kernel is taken in coordinates and resting states
jointly, with the tangency of a direction :math:`(\xi_\theta, \xi_x)` to :math:`W` as extra
rows rather than through the implicit function theorem. Let :math:`N` be the states that are
neither the pivot of a conserved total (below), nor an input, nor at zero at rest, and
:math:`f_N` their balances. The tangency rows are

.. math::

   J_x\, \xi_{x_N} + J_\theta\, \xi_\theta = 0, \qquad J_x = \partial_{x_N} f_N, \quad
   J_\theta = \partial_\theta f_N,

where the columns of :math:`\partial_x f` at the pivots are columns of :math:`J_\theta`, since
the resting value of a pivot is a coordinate; inputs and states at zero enter as rows that hold
them at zero. One balance per total is dropped, that of the pivot. This is exact: each total
:math:`\sum_i a_i x_i` gives :math:`\sum_i a_i f_i = 0`, and the coefficient matrix
:math:`C_P` of the totals at the pivots is invertible, so :math:`f_P = -C_P^{-1} C_N f_N`
vanishes wherever :math:`f_N` does. The full :math:`\partial_x f` is singular with a total, the
reduced :math:`J_x` is square; it is the *resting Jacobian*. A proven rank also needs the
resting Jacobian of every condition solved at the point itself invertible at the base point;
otherwise the rank is a lower bound.

**Branches.** A *branch* is an irreducible component of :math:`W` over :math:`\mathbb{Q}`. By
default the first condition draws the resting states and solves the balances for the
*turnover rates*, one rate per state that enters the balance of the state linearly (the *forward seed*);
this names the component that the drawn values parametrize. Where a balance has several roots
over :math:`\mathbb{F}_p`, the first nonzero one found is taken and named in
``info["branches"]``: by the factor over :math:`\mathbb{Q}(\theta)` it solves where a balance
splits into factors, else as *conjugate*. Conjugate means that the lexicographic Gröbner basis
of the balances, states at zero removed, has at a point the *shape*
:math:`x_1 - g_1(x_n), \dots, x_{n-1} - g_{n-1}(x_n), P(x_n)` with an irreducible *eliminant*
:math:`P`. Other points take their root on that branch or, without one, are left out.
Conjugate roots have equal rank, and an eliminant irreducible at a point is irreducible over
:math:`\mathbb{Q}(\theta)`.

**Zero at rest.** A state rests at zero when the right-hand side of the state, with the inputs at zero, is
linear in it with a coefficient that is nonzero while the other such states are zero. In a
reaction network, a species that is consumed but not produced rests at zero as the single
reactant of a consuming reaction, as does the single reactant of positive weight in a reaction
that lowers a weight no reaction raises (a *sink*; a weight assigns each species a number in
:math:`[0, 1]`). Of several such reactants, as in :math:`A + B \to 0`, the first is set to zero
(``info["branches"]``). A state with right-hand side zero at rest, and the pivot of a conserved
total (below), has a free resting value, the initial value. Rates are read at generic values,
so a catalyst at rest is nonzero. Where the coefficient of a state of a total at zero holds
other states, as for :math:`\dot B = B\,(k_1 A - k_2)`, the resting set has a second branch
with the state away from zero, where the coefficient vanishes; ``info["branches"]`` names it,
and ``free_initial`` naming the state takes it. ``free_initial`` naming a state at zero on every
resting state is an error.

**Conserved totals.** A linear combination :math:`T = \sum_i a_i x_i` of states that the
right-hand sides conserve is a *total*; the coefficients may hold parameters, such as volumes.
Totals are read from the stoichiometry of a reaction network or from the linear laws of the
ODEs, unless given. The basis read off consists of nonnegative laws where these span the laws,
the extreme rays of the cone of nonnegative laws with the fewest species first, so that it does
not depend on the order of the species and each total is positive on the parameter space; a
total with coefficients of both signs is named in ``info["limits"]`` under ``reduce_cq``, which
takes it positive.

Under ``equilibrate`` each total leaves one resting value free, which one species of the total,
the *pivot*, holds as initial value. The pivots form a set :math:`P` of species, one per
independent total, with an invertible coefficient matrix :math:`C_P`. ``free_initial`` names
them; otherwise they are chosen by the structure of the model, in this order: not at zero at
rest, not an input and not the base of a free power, a coefficient without parameters, in the
fewest totals, observed. Where several sets tie by these criteria, up to six are analyzed and the
result with the best rank status, rank, levels and kinds is kept, so that the result does not
depend on the names or the order of the states and totals. With another pivot the resting
value of another state is the coordinate, so the generators differ, but their spans agree after
the map to the parameters and all resting states at a common resting point. Where directions
stay open, other valid sets are tried as well. ``info["settings"]["free_initial"]`` holds the
pivots of the result.

With ``reduce_cq`` the pivots are eliminated instead, :math:`x_P = C_P^{-1} (T - C_N x_N)`, and
the values :math:`T` of the totals become coordinates (``info["settings"]["eliminated"]``); the
pivots are chosen as above, except that unobserved species are eliminated first. An event that changes
a species :math:`x_i` of a total changes the total by :math:`a_i\,(E(x_i) - x_i)`, with
:math:`E(x_i)` the value after the event; like every event value it reads the states before the
events. The total is then a state without dynamics, so the segments after the event see the new
total, and an event on an eliminated species changes only the totals. Both parametrizations
describe the same resting variety, so ``reduce_cq`` gives the same rank and the same span of the
directions, mapped to the same space, as the pivots.

**Rank at the resting start.** Under ``equilibrate`` the rank is that at the declared resting
start, whose coordinates fix the level set of each total. The ceiling of the order certificate
(:ref:`sufficient-lie-order`) is the generic rank on the whole resting variety, so the computed rank
cannot exceed the rank along the trajectories from the declared start; it can only fall short of
it, and is then reported as a lower bound (a singular resting Jacobian, an order not proven
sufficient) or covered by the error bound (a random point). The exact scalings bound the rank
from above, so a rank at that bound proves the order sufficient without random points, with
``info["order_bound"]`` zero.

**Diagonal start.** Conditions sharing :math:`\theta` have coupled resting states. Rank and
kernel are generic on the joint variety of the conditions,
:math:`\{(\theta, x_1, \dots, x_C) : (\theta, x_c) \in W_c\}`, so any point of it over
:math:`\mathbb{F}_p((\varepsilon))` off an exceptional subvariety gives them. symident seeks a
set :math:`P_Z` of free parameters whose value zero makes all conditions equal to the first,
and the largest set :math:`X_Z` of states whose value zero is then invariant; at
:math:`\theta_{P_Z} = 0`, :math:`x_{X_Z} = 0` the forward seed is a resting point
:math:`\hat z` of all conditions. Along :math:`\theta(\varepsilon) = \hat\theta + \varepsilon\,
\zeta`, with :math:`\zeta` the values of :math:`P_Z` at the sample point, Newton steps lift each
condition to a power series,

.. math::

   x_{c,k} = -J_c(\hat z)^{-1}\, [\varepsilon^k]\, f_c\big(x_{c,<k}, \theta(\varepsilon)\big),
   \qquad J_c = \partial_x f_c ,

with :math:`[\varepsilon^k]` the coefficient of :math:`\varepsilon^k` and :math:`x_{c,<k}` the
series up to order :math:`k - 1`. With :math:`J_c(\hat z)` invertible, the series is the
expansion of the unique component through :math:`\hat z`; a trial at a random point checks
this and that the states of :math:`X_Z` leave zero. The rank over the truncated series bounds
the generic rank from below, and ``info["series_certified"]`` says whether the number of
coefficients, the *series order*, is proven sufficient. The kernel at :math:`\varepsilon = 1`
comes from Padé forms of the kernel entries, each accepted only when the coefficients beyond
those determining it agree. ``SYMIDENT_NODIAG`` turns the diagonal start off.

**Existence of a resting state.** Before the analysis, the balances of each resting model are
solved over :math:`\mathbb{Q}` by msolve at three random rational parameter values, positive
where declared positive, with the poles of the right-hand sides excluded. Balances without a
solution over the complex numbers at all three values raise an ``InputError``, unless the
modular decision below proves a resting state: a system with solutions at generic
parameter values has them outside a proper subvariety. The modular decision, a Gröbner basis
modulo a prime at three random integer points, gives ``info["resting"]["exists"]``:
``"proven"``, ``"no"`` with the probability ``bound`` of a resting state for generic values, or
``"open"``. Where the real roots,
isolated by msolve, include none in the declared domain (real, positive where declared positive),
a warning is given. ``info["resting"]`` holds the counts: ``points``, ``consistent``,
``in_domain``, and ``undecided`` for an isolating interval that contains zero or a resting set of
positive dimension. This is a check at random points, not a proof. Stability of the resting
state is not considered. The check is skipped for resting models with leaves of transcendental
terms or more than twelve resting states.

**The modular decision.** At an integer point :math:`\theta^\ast`, each coordinate uniform in
:math:`\{1, \dots, M\}` with :math:`M = 2^{64}` and a random sign where the domain does not fix
it, the balances :math:`f_i(\theta^\ast, x)` and :math:`w \prod_i \mathrm{den}_i - 1` (numerators
and denominators over :math:`\mathbb{Q}(\theta)`, cleared to integers) get a Gröbner basis modulo a
prime with exact linear algebra.

- With as many balances as resting states, none of them zero at :math:`\theta^\ast` or modulo
  :math:`p`, a finite quotient and as many distinct values of a random form as the dimension of the
  quotient, every point modulo :math:`p` is simple, the Jacobian is invertible there, and Hensel's
  lemma lifts the point to a solution over :math:`W(\mathbb{F}_{p^r})[[t]]`, :math:`\theta =
  \theta^\ast + t`, a domain whose field of fractions holds :math:`\mathbb{Q}(\theta)`. The balances
  then have a solution over :math:`\overline{\mathbb{Q}(\theta)}`: a resting state exists for
  generic parameter values, proven (``exists: "proven"``).
- Where the basis is :math:`\{1\}` and msolve finds no solution over :math:`\mathbb{Q}` at
  :math:`\theta^\ast` either, there is no resting state for generic values except with probability

  .. math::
     :label: eq-rest-none

     \frac{D_\theta}{M}

  per point (``exists: "no"``, ``bound``; independent points multiply), :math:`D_\theta` the
  Bezout number of the balances and the saturation in :math:`(\theta, x, w)`. If a component
  :math:`Y` of the resting variety dominates the coordinates, the values with an empty fiber lie
  in a hypersurface of degree at most :math:`\deg Y \le D_\theta` :cite:p:`Heintz1983`, met with
  probability at most :math:`D_\theta / M` by :cite:p:`Schwartz1980` for coordinates drawn independently
  with no value more likely than :math:`1/M`.
- Otherwise the decision is open (``exists: "open"``).

.. _inputs:

Unknown and known inputs
------------------------

An *unknown input* :math:`w(t)` is an analytic function of time that is left open. A direction
of :math:`\theta` is non-identifiable when nearby :math:`\tilde\theta` with some :math:`\tilde w`
give the output of :math:`(\theta, w)` for a generic :math:`w`.

The Lie derivative :math:`L_F^k h` holds the input only through the *jets*
:math:`w_j = w^{(j)}(t_0)`, :math:`j \le k`. The chain :math:`\dot w = w_1, \dots,
\dot w_{K-1} = 0` of length :math:`K` therefore joins the states :cite:p:`Martinelli2015,Villaverde2019`; it is exact for every row up to order :math:`K - 1` and is lengthened while
the order reaches past it. Rows up to order :math:`k` see :math:`k + 1` jets of an infinite
family, so they never pin the input to a finite form. The jet values :math:`\hat w` at
:math:`t_0` are *nuisance coordinates*: generic, unknown, and not part of :math:`\theta`. With
:math:`\mathrm{OIM}_\theta` and :math:`\mathrm{OIM}_{\hat w}` the columns of the OIM for
:math:`\theta` and for :math:`\hat w`, the identifiable dimension is

.. math::
   :label: eq-nuisance

   \operatorname{rank} \big[\mathrm{OIM}_\theta \mid \mathrm{OIM}_{\hat w}\big]
   - \operatorname{rank} \mathrm{OIM}_{\hat w} ,

the kernel of the full matrix modulo the kernel within :math:`\hat w`, projected onto
:math:`\theta`; ``rank`` and ``dim`` count :math:`\theta` only. This is the dimension of the
part of the row space without jet components, the counterpart of the separation of the
observable codistribution, the span of the gradients of the Lie derivatives, in
:cite:p:`Martinelli2019`; :cite:p:`Villaverde2019` ask instead for observability of the extended state, jets
included.

The bound of :ref:`sufficient-lie-order` needs a fixed state space, and the chain grows with the
order, so no order is proven sufficient here. The rank is ``proven`` when it is full or when
independent exact directions fill the kernel, else it is a lower bound, in both cases as far
as the value of Equation :eq:`eq-nuisance` holds. Where :math:`\mathrm{OIM}_{\hat w}` drops
rank, that value can exceed the generic one. The value is

- proven where the jet columns that the rows up to the orders taken can hold have full column
  rank, at the base point or at a point uniform modulo :math:`p` with the same value;
- verified with the same value but no full column rank, with the chance by :cite:p:`Schwartz1980` that
  a maximal minor of :math:`\mathrm{OIM}_{\hat w}` vanishes there;
- a lower bound where the values differ.

The chain is part of the state and passes every gap, so the input is analytic across events;
at a switch it is only continuous (see :ref:`switching`). A row after gap order :math:`M`
holds jets up to the Lie order of the row plus :math:`M`, so the Lie order is held to :math:`K - 1 - M`.
Under ``equilibrate`` the chain rests at zero, and a jump at the start sets the nuisance
values.

A direction is proven as a field
:math:`Y = \xi_\theta\, \partial_\theta + \xi_x\, \partial_x + \xi_w\, \partial_w`, rational in
states, parameters and finitely many jets, with the identities of :ref:`proofs-over-q` extended by
:math:`(\partial_w f)\, \xi_w` per state and :math:`(\partial_w h)\, \xi_w` per observable, and
with the chain in :math:`F`. The jets need no identity, since :math:`\xi_{w_{j+1}} =
F(\xi_{w_j})` prolongs :math:`\xi_w`; a component with the last jet of the chain is not proven.
The published *input change* is :math:`\xi_w` of a proven field; a kernel vector gives it only
at :math:`t_0`, and a field proven in the log coordinate of an input under a fractional power has no
closed form in the input (``None``). That log coordinate is the nuisance coordinate of the input.

A *known input* enters as the same chain, with known symbols as jet values, like the input
derivatives of the extended Lie derivatives of :cite:p:`Karlsson2012`. Past order :math:`K - 1` the
chain is a polynomial, i.e., a specialization of the input, so the Lie order and the order
certificate are held to :math:`K - 1 - M`, and the chain is doubled, up to three times, while
the order reaches past it (``info["known_inputs"]``).

The order certificate (:ref:`sufficient-lie-order`) takes the ranks of a known input's model on the
columns of the states and coordinates only. The jets would add a column per order, so the ranks
with them never stop rising. Without them the lemma holds as stated: the field of the jets,
:math:`w_j \mapsto w_{j+1}`, depends on neither the states nor the coordinates, so
:math:`\partial_z F(\varphi) = (\partial_z \varphi)\, \partial_z f + F(\partial_z \varphi)` for
every :math:`\varphi` with :math:`z = (x, \theta)`, and the restricted rows form a filtration over
the rational functions in :math:`z` and all jets. The ranks of the filtration rise strictly up to the ceiling, which
is therefore reached within :math:`\dim z` orders. The OIM needs no jet column, since the start
jets do not depend on :math:`\theta`. The jets are free in :math:`S`, and the proposition holds
for every input: a start whose jets are special (a step) can only miss the ceiling, which leaves
the order not proven sufficient. Models with events at several times and steady states keep the
ranks with the jet columns.

.. _switching:

Switching models
----------------

Each switch condition, a comparison or the argument of ``Heaviside()``, ``sign()``,
``abs()``, ``max()`` or ``min()``, is a sign condition :math:`\gamma_j(x, \eta, t) > 0`,
:math:`j = 1, \dots, n_\gamma`. The *declared signs* are the domain declared by ``positive``
and the start sides declared by ``regimes``. Conditions that the declared signs decide are
resolved; a factor of odd multiplicity of an undecided one is a *switching function*
:math:`\gamma_j`, with *switching surface* :math:`\Gamma_j = \{\gamma_j = 0\}`. A sign vector
:math:`\sigma \in \{+1, -1\}^{n_\gamma}` gives a smooth *regime* :math:`(F_\sigma, h_\sigma)`,
the field and the observables where every :math:`\gamma_j` has the sign :math:`\sigma_j`. A
*root event* (``root`` in place of ``time``) adds the root expression as a switching function
and resets the target state, the *event state*, on :math:`\Gamma_j`.

**Regimes.** A direction counts only if it is a symmetry of every regime the declared signs
allow. Without ``regimes``, every sign vector counts unless the declared positivity decides
the sign of :math:`\gamma_a - \gamma_b` or :math:`\gamma_a + \gamma_b` against it; none is
dropped for parameter values or for not being reached, so the result holds for every initial
state. A parameter that acts only in a regime excluded by a decided condition stays a
coordinate, with the unit direction :math:`\partial_{\theta_i}` as exact direction. Each
allowed regime enters as a *copy* of the model, i.e., the rows of that regime with initial
states of their own and the parameters shared, from the start of the condition and, with
other regimes allowed, as a copy switched into it (``info["switching"]["conditions"]``).

**Entry points.** The state at a switch is unknown, so a direction counts only if it is a
symmetry from every point of the surface. A state that starts at zero in every condition, that
every allowed regime keeps at zero with finite derivatives in such states, and that no later
event moves, is zero along every trajectory. It is dropped (``info["zero_states"]``), and a
coordinate that only such states read is a separate direction :math:`\partial_{\theta_i}`;
``info["limits"]`` names a condition with further such states. For a copy switched into a
regime, other invariant sets of the start, such as the level set of a conserved total, are not
used, so the rank of a model with switches can exceed the rank along the trajectories from the
declared start. This concerns switches only; under ``equilibrate`` without switches the rank
cannot exceed it (:ref:`steady-states`).

**Visibility.** Two regimes :math:`\sigma, \sigma'` are *neighbors* at :math:`\Gamma_j` when
they differ only in the sign of :math:`\gamma_j`. The switch is *visible* when for some
neighbors, observable :math:`h_i` and order :math:`k`

.. math::

   \big(L_{F_\sigma}^k h_{\sigma,i} - L_{F_{\sigma'}}^k h_{\sigma',i}\big)\big|_{\Gamma_j}
   \ne 0 .

A nonzero Taylor coefficient at a point of :math:`\Gamma_j`, modulo a prime, proves this. The
switch is *invisible*, or *hidden*, when for every pair the observables agree and every state
reaching them has the same right-hand side, so that every Lie derivative agrees, and
*undecided* otherwise. A root event surface counts as visible.

**The rule.** Let :math:`\tau_e` be the time and :math:`E_e` the map of the states of event
:math:`e`. A direction :math:`v` is reported non-identifiable when, in every condition, it lies
at the start in a distribution :math:`\mathcal{D} = \operatorname{span}\{Y_1, \dots, Y_m\}` of
fields on parameters and states with

.. math::

   [Y_i, F_\sigma] \in \mathcal{D}, \qquad Y_i(h_\sigma) = 0 \qquad
   \text{for every regime } \sigma,

.. math::

   Y_i(\gamma_j) = 0 \ \text{on } \Gamma_j \ \text{for every visible surface}, \qquad
   Y_i(\tau_e) = 0, \qquad \mathrm{D}E_e\, Y_i(z) \in \mathcal{D}\big(E_e(z)\big) \
   \text{for every event and reset},

and, at a surface that is not visible and that some :math:`Y_i` moves, for every pair of
neighbors

.. math::

   \big(F_\sigma - F_{\sigma'}\big)\big|_{\Gamma_j} \in \mathcal{D}, \qquad
   \big(h_\sigma - h_{\sigma'}\big)\big|_{\Gamma_j} = 0 .

The events and resets are those of the condition: an event that names conditions (``cond``) acts in
those only. No :math:`Y_i` moves a fixed symbol, an input or time. The direction is
:math:`v = \big(\xi, X(x_0)\big)`, with the derivative of the initial values along :math:`X` as
state part (under ``equilibrate``, of the resting state of each regime). Along every regime
flow, :math:`v` stays in :math:`\mathcal{D}` and keeps the observables to first order; at a
visible surface :math:`v` is tangent, so the crossing time stays; at a hidden one the crossing
may move, but the jump lies in :math:`\mathcal{D}` and the observables agree. The rule therefore
never reports a direction that changes the observables, for any sequence of the listed regimes.

**Lower bound.** The directions of the rule satisfy the rows of copies of the model, whose rank
bounds the rank from below:

- each regime from the start of each condition, with the events of the condition;
- each regime from a generic point of each visible surface linear in a state, where a direction
  of the rule is tangent to the surface;
- each regime from a generic point, when a switch of the condition is not visible;
- each copy switched at a fixed time into every other regime of the condition, with outputs
  along *words* of two regimes, :math:`L_{F_\sigma}^a L_{F_{\sigma'}}^b h_{\sigma'}`;
- the switching time :math:`t_j(\theta)` of every visible switch in time alone.

The *fresh coordinates* are the entry states of the copies that start at surface points. The
kernel of all these rows, projected onto :math:`\theta`, contains every direction of the rule;
a kernel dimension :math:`k_\Delta` gives the lower bound :math:`d - k_\Delta`. A minor that is
nonzero at the sample point modulo a prime is nonzero as a rational function over
:math:`\mathbb{Q}`, so the rank modulo a prime at a point bounds the generic rank from below.
The projection off the fresh coordinates is a difference of two ranks,
:math:`\operatorname{rank} J - \operatorname{rank} J_F` with :math:`J_F` the fresh columns, and can
rise at a special point. It is proven where the fresh columns that enter a row have full column
rank at the sample point: generically :math:`\operatorname{rank} J_F` is then their number, and
:math:`\operatorname{rank} J` at the point bounds the generic one from below. A fresh column
enters no row when the state it starts reaches no observable of the copy. The reach is taken
with the *held* states of the copy at their start values: a set of states whose start values
are free of fresh coordinates and whose right-hand sides vanish identically there, with finite
derivatives, in every regime of the copy. Such states keep their start values, so every Lie
derivative of the copy is one of the fields restricted to that level set.

Rows of fewer copies keep every direction of the rule in their kernel, so their projection is
an upper bound on the dimension of the rule. The copies without fresh coordinates are taken
first: rows of full rank there prove the full rank, and no other copy is stacked. Otherwise,
where the fresh columns of some start points drop rank at the sample point, the rows of the
other copies give a proven dimension; equal to the dimension of all rows, it proves the
projection. Else the projection is checked with the kernel at a point uniform modulo the
verification prime, and proven there by the same column test. With equal dimension there and
no such proof it is ``verified``, with the chance by :cite:p:`Schwartz1980` that a maximal minor of the
stacked rows or of their fresh columns vanishes there, else a lower bound
(``info["projection"]``, ``info["rank_status"]``).

**Upper bound.** The candidates are the scalings of :ref:`scalings`, the closed forms of the kernel
of the rows above and the unit direction :math:`\partial_{\theta_i}` of every coordinate; those
failing a condition of the rule are dropped until the family passes. Each identity is tested at
a point with rational coordinates and then proven by symbolic simplification to zero;
:math:`\ell` candidates left, independent at such a point, bound the rank by :math:`d - \ell`.
If :math:`\ell = k_\Delta`, the rank is :math:`d - k_\Delta` and the candidates span the
non-identifiable directions; otherwise :math:`d - k_\Delta` is a lower bound,
``info["switching"]["uncertified"]`` names the rejected candidates, and the unproven kernel
directions, which satisfy only necessary conditions, appear under
``info["switching"]["candidates"]``, not as symmetries. For an undecided switch the rule accepts
tangency or a hidden jump and the rows omit the surface, so both bounds stay valid.

**Unknown inputs.** The input is continuous where a trajectory crosses a surface and analytic
between crossings; the derivatives of the input may jump at a crossing. The input direction
:math:`G_\sigma = \partial_w F_\sigma`, over the inputs that no observable of :math:`\sigma`
reads, joins the bracket condition,

.. math::

   [Y_i, F_\sigma] = \sum_k c_{ik}\, Y_k + a_{i\sigma}\, G_\sigma ,

with rational functions :math:`c_{ik}` and :math:`a_{i\sigma}`, and
:math:`v = \sum_i \beta_i Y_i` changes the input in :math:`\sigma` by :math:`-\sum_i \beta_i
a_{i\sigma}`; an input that an observable reads stays unchanged. Where :math:`G_\sigma \in
\mathcal{D}` the change is free. Otherwise it must not jump where a trajectory crosses
:math:`\Gamma_j` from :math:`\sigma` into a neighbor :math:`\sigma'`: on :math:`\Gamma_j`,

.. math::

   a_{i\sigma'} - a_{i\sigma}
   - \frac{Y_i(\gamma_j)}{F_\sigma(\gamma_j)} \sum_k e_k\, a_{k\sigma'} = 0 ,
   \qquad F_\sigma - F_{\sigma'} = \sum_k e_k Y_k ,

the last term counting only at a hidden surface that :math:`Y_i` moves, whose crossing then
shifts. A switch is visible only through orders :math:`k \le r_i`, with :math:`r_i` the lowest
order at which an input can enter :math:`L^k h_i`, read from the symbols of each order; higher
orders hold input derivatives. The jets of each copy are nuisance coordinates (see
:ref:`inputs`), with a start of their own per copy and, at a surface point, the
input value shared by the two neighbors. A copy from a generic point of each regime replaces the
copy switched at a fixed time, which would restart the input off the surfaces, where the changes
of two regimes differ. The rank status is the weaker of the projections off the jets and off the
fresh coordinates. The rule holds per condition, so each condition needs a separate input,
renamed in ``conditions``.

**Listed regimes.** With ``regimes``, the rule, the rows and the candidates use the listed
regimes of each condition, and a surface counts only with listed regimes on both sides.
Parameters of other regimes stay coordinates and are reported non-identifiable. A sign of
:math:`\gamma_j(x_0)` decided by the declared signs must agree with the first entry, else an
error is raised; an undecided one is declared by ``regimes`` and listed under ``declared`` in
``info["switching"]["conditions"]`` and in ``info["assumptions"]["starts"]``.

**Limits.** Each of the following is an error, never a result:

- a surface in the parameters alone whose side ``regimes`` does not give;
- a condition on a switched expression;
- more than 4096 sign vectors;
- ``gauge``, ``reduce_cq``, ``free_initial``, ``engine="symbolic"``;
- a state named ``time``;
- under ``equilibrate``, a resting state without closed form;
- an infeasible listed regime, or a start that contradicts the initial values;
- with unknown inputs: an input read by several conditions, a switch on an input, an event after
  the start, an initial or event value that reads an input.


.. _petab-mapping:

PEtab problems
--------------

:func:`~symident.read_petab` maps a PEtab problem :cite:p:`Schmiester2021` onto these elements.
Each experiment with measurements is one condition; the estimated parameters are the
coordinates, the other parameters take their nominal values. A preequilibration is the resting
start of ``equilibrate`` (:ref:`steady-states`), with the preequilibration condition in place of
the inputs at zero, and the condition of the experiment as events at the start; it has to be part
of every experiment or of none, since all conditions start alike.

An experiment without preequilibration whose measurements are all at :math:`t = \infty` is a
*readout at rest*. The output of the experiment is the observables at the resting state :math:`x^\ast(\theta)` of
the condition, :math:`h(x^\ast(\theta), \eta(\theta))`, which is the output of the experiment
under the premise that the trajectory from the start of the condition converges to that resting
state. The experiment enters as a resting start under the condition of the experiment, with no
event after it: the trajectory stays at rest, the output is constant, and only the rows of order
zero at the resting state enter, with the tangency rows of :ref:`steady-states`. Readouts at rest and preequilibrated experiments mix; a
readout at rest next to an experiment that starts without preequilibration is read as a
trajectory, with a warning.
