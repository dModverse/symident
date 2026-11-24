Reconstruction and proofs
=========================

The kernel at a point gives the non-identifiable directions as numbers. To read them as
generators, and to reduce the model, symident reconstructs them in closed form and then tries to
prove them over :math:`\mathbb{Q}`.

.. _scalings:

Scalings
--------

A *scaling* is a symmetry :math:`\Phi_\lambda(\theta)_i = e^{\alpha_i \lambda}\, \theta_i` with
integer *weights* :math:`\alpha_i`, so :math:`\xi_i = \alpha_i \theta_i`. A function is
*homogeneous* of weight :math:`a` when the scaling multiplies it by :math:`e^{a \lambda}`. With
weights :math:`\alpha_x` of the states as well, the scaling is a symmetry when every
right-hand side has the weight of the state, every observable, event time and multiplicative
event value the weight zero, and every initial value and replaced or added event value the
weight of the state. These conditions are linear in the weights and read off the exponents of
the monomials of all expressions together; their integer solutions form a lattice, the
*integer kernel*, and a basis of it gives the scalings exactly. They are taken out first and checked
again symbolically. A power :math:`x^e` whose exponent :math:`e` holds no state and only
coordinates of weight zero has the weight :math:`e\,\alpha_x`, since
:math:`(e^{\alpha_x \lambda} x)^e = e^{e \alpha_x \lambda} x^e` on the principal branch, also for
a state that starts at 0; the recast :math:`E = \theta^e` of a free power of a coordinate has
the weight :math:`e\,\alpha_\theta`, and :math:`L = \log \theta` is translated by
:math:`\alpha_\theta`. A scaling that holds only on an invariant set, such as
:math:`(k_1, k_2) \mapsto (e^\lambda k_1, e^\lambda k_2)` for :math:`\dot x = k_1 - k_2 x`
started at rest, is not an exact symmetry and fails the check. The scaling of the example in
Equation :eq:`eq-example` is exact, with weights :math:`(0, 0, -1, 1, 1)`.

.. _general-directions:

General directions
------------------

The other non-identifiable directions are *general directions*. The coordinates are ordered with
the initial values of the free states first, in the order of the model, followed by the
parameters in a fixed sorted order. Gauss-Jordan elimination of the rows takes the pivot
columns from left to right, and a *free coordinate* is one without pivot, whose column depends
on the columns to the left. Each kernel vector has the component one on one free coordinate and
zero on the others; every other component is then a rational function of the coordinates.
Alternatively, a direction is fitted in log coordinates, :math:`\xi_i / \theta_i`, and multiplied
back, or chosen with the smallest support outside the scalings, with the component one on the
leftmost coordinate of the support.

The components are reconstructed from samples. Perturbation finds the variables of each entry;
the entry is sampled at many points and primes and fitted by dense interpolation, by sparse
interpolation after :cite:p:`BenOr1988` or by univariate rational interpolation along lines, and it is
lifted to rational coefficients by the Chinese remainder theorem and rational reconstruction
:cite:p:`Wang1982`. The *route*, as ``Symmetry.route`` names it, is one of

- ``fit``: rational reconstruction from samples over several primes; ``read-off``: a fast pass
  that reads Laurent monomials on the support, with the other coordinates held; ``fit per
  prime``: points that each prime solves, joined by the Chinese remainder theorem; each possibly
  ``in log coordinates`` and with the kind of fit in parentheses, ``constant``, ``dense`` (a
  monomial table up to ``degree_cap``) or ``sparse``;
- ``forward fit per prime``: under ``equilibrate``, with the resting states drawn and the
  turnover rates solved, as in the forward seed;
- ``narrow fit``: the kernel restricted to the columns of the support of one direction, in which
  it is one-dimensional, with ``exchange groups`` where coordinates that enter only through their
  sum or their product were held at the base point and substituted back; ``narrow read-off``:
  entries constant, or constant multiples of the coordinates (a scaling on the support), read at
  the base point and held at two fresh points; ``narrow (a column without rows)``: the direction
  along a coordinate whose column of the kernel matrix vanishes; ``jets (order k)``: the
  generator read from the Taylor coefficients of the kernel entries along a support of at most
  six coordinates. With many conditions the narrow route reads the kernels of a *cover*, the
  conditions whose rows reach the rank of all of them at the base point, and finds the
  coordinates that move a direction by whether the direction still annihilates the rows with a
  group of coordinates perturbed, from jets seeded along the direction; each closed form is then
  checked on every condition;
- ``homotopy fit``: the fit along lines below.

A *sample bank* of points over the union of the relevant coordinates, with kernels per point and
prime, is shared by all dense fits. Before it is reported, every closed form, and a direction
along one coordinate that fills the kernel, is checked at seeded points uniform modulo the
verification prime, the *kernel check* of :ref:`kernel-check`; under ``equilibrate`` on the
coordinates the closed form states, from which the resting-state copies per condition follow.

A direction that no sampling route closes is fitted along lines :math:`\theta_b + \mu\,
\delta` through the base point: the kernel entries become power series in :math:`\mu` (under
``equilibrate`` with the resting state continued along the line), and Padé forms on a *star*,
several lines in independent directions :math:`\delta`, are fitted as rational functions. An
entry that is not rational in the parameters alone is fitted in parameters and resting states
modulo their balances, the state :math:`x` of condition :math:`c` being named ``x_c<c>``. Each
entry is checked at a prime outside those of the fit, and the direction against the kernel at
two points uniform modulo the verification prime. ``reconst_control(homotopy=False)`` turns
this route off.

.. _orbit-geometry:

Orbit geometry
--------------

Each direction with a closed form is classified by the geometry of the orbits,
``Symmetry.kind``, up to a common factor :math:`c(\theta)` of the components:

- a *scaling*, :math:`\xi_i = c\, w_i \theta_i` with numbers :math:`w_i`, has lines in
  logarithmic coordinates as orbits and monomial invariants; a pin
  :math:`\theta_i = 1` meets every orbit once;
- a *translation*, :math:`\xi_i = c\, v_i` with numbers :math:`v_i`, has lines in linear
  coordinates as orbits and linear invariants. It is proven like a scaling, from the linear
  structure of the model: the right-hand sides, observables, initial values and events do not
  change along :math:`v`, checked over :math:`\mathbb{Q}`, with the field on the states where
  :math:`v` moves initial values. In the positive orthant an orbit ends on the boundary, so the
  section lies on a face, :math:`\theta_i = 0`, reached in the limit;
- every other direction is *curved*.

Translations and curved directions are general directions (``Symmetry.type``). The labels
:math:`X_1, X_2, \dots` of the report and of the reduction count the scalings first, then the
translations and the curved directions.

.. _invariant-fields:

Invariant fields and gauge
--------------------------

Some directions follow from the model equations without sampling. For a model with resting
states, a *global symmetry* is a field of one regime, found by a detection of that regime with
free initial values and extended to the initial and event values, that lies in the kernel. A
*static subsystem* is a set of states at rest that no event moves; parameters that only the
balances of the subsystem hold give fields that leave every such rest in place. Both are reported with these
names as route. Where every direction of the kernel lies in the span of fields that satisfy the
conditions of :ref:`proofs-over-q` at random points, the *invariant fields*, the rank holds at every
order (``info["rank_fields"]``), with an error bound in ``info["rank_fields_bound"]``. (A) is
checked as a zero and (B), (C) as membership in the span, at two random points; a wrong identity
passes only where a polynomial vanishes there, which the bound covers. Membership at random
points shows (B) and (C) as identities over the rational functions, the hypothesis of the
proposition in :ref:`proofs-over-q`, with coefficients that may have poles on the trajectory.

With ``gauge``, the detection first finds the scalings and fixes one coordinate per scaling, as
a basis of the weights in the order of preference given and, with general directions, the one
that leaves the fewest nonzero entries in the remaining kernel. A second pass with these
coordinates fixed reconstructs the general directions. Without ``gauge``,
``info["gauge_suggestion"]`` names this choice.

.. _proofs-over-q:

Proofs over Q
-------------

A closed form is *proven* when it is shown to be a symmetry without random points. For a scaling
this is the check of :ref:`scalings`. For a general direction, symident *lifts* the direction to a
field :math:`Y = \xi_\theta\, \partial_\theta + \xi_x\, \partial_x` on parameters and states,
i.e., it finds state components :math:`\xi_x`, and proves the conditions below over
:math:`\mathbb{Q}` for every regime of a condition, with event maps :math:`E_e` between the
regimes. They are stated for fields :math:`X_1, \dots, X_m` per regime and the unit directions
:math:`\partial_u` of parameters :math:`u` that enter no regime, whose span is
:math:`\mathcal{D}`; :math:`X'_l` are the fields of the regime after an event, and :math:`v` is
the direction lifted onto the start :math:`z_0`.

.. math::
   :label: eq-fields

   \begin{aligned}
   &\text{(A)} \quad X_l(h) = 0 \ \text{for every output } h, \\
   &\text{(B)} \quad [F, X_l] = \textstyle\sum_k c_{lk}\, X_k + \sum_u c_{lu}\, \partial_u, \\
   &\text{(C)} \quad \mathrm{D}E_e\, X_l = \textstyle\sum_k c'_{lk}\, X'_k \circ E_e + \sum_u c'_{lu}\, \partial_u, \\
   &\text{(D)} \quad v = \textstyle\sum_l a_l\, X_l(z_0) + \sum_u a_u\, \partial_u,
   \ \text{the } X_l \text{ regular at } z_0, \text{ and } v(\tau_e) = 0 \text{ for every event time } \tau_e,
   \end{aligned}

with :math:`a` numbers and :math:`c, c'` rational functions of states and parameters, which may
have poles. A proof over :math:`\mathbb{Q}` takes the lift as the one field of each regime, with
:math:`[F, Y] = 0` and :math:`\mathrm{D}E_e\, Y = Y' \circ E_e`; the invariant fields of
:ref:`invariant-fields` take the fields of the regimes. Where :math:`[F, Y] \neq 0`, the
proof takes as :math:`X_l` the lift together with the lifts proven before, and (B) and (C) with
coefficients solved at a random point and checked over :math:`\mathbb{Q}` on the other
coordinates: :math:`[F, Y] = \lambda Y + \sum_j \mu_j Z_j` holds, e.g., for a translation of
two initial values whose sum the dynamics keeps, where the flow scales :math:`Y`. With unknown
inputs the jets of an input, the input included, take no identity: their components are the
prolongation :math:`\xi_{w_{j+1}} = F(\xi_{w_j}) + \lambda \xi_{w_j} + \sum_j \mu_j \xi^{Z_j}_{w_j}`,
which makes (B) hold there, so a field whose parameter components hold the input is allowed.
Components are tried as found, canonical, over a common denominator and divided by each of their
first four components.

**Proposition.** Under (A) to (D) the direction lies in the kernel at every Lie order and every
gap length.

*Proof.* Let :math:`\mathcal{A}` be the rational functions :math:`\varphi` of a regime with :math:`X_l(\varphi) = 0`
and :math:`\partial_u \varphi = 0` for all :math:`l, u`. By (A), :math:`h \in \mathcal{A}`. By (B),
:math:`X_l(F \varphi) = F(X_l \varphi) - [F, X_l](\varphi) = 0`, and :math:`[F, \partial_u] = 0`, so :math:`F`
maps :math:`\mathcal{A}` into itself; by (C), :math:`\varphi \circ E_e \in \mathcal{A}` for :math:`\varphi` in :math:`\mathcal{A}` of the
next regime. Let :math:`\mathcal{A}_k` be the functions of segment :math:`k` built from the outputs of the segment
and from :math:`\varphi \circ E_e` with :math:`\varphi \in \mathcal{A}_{k+1}` by Lie derivatives along
:math:`F`. They lie in :math:`\mathcal{A}` and are analytic along the trajectory, as is every expression
of the model where it is defined. With :math:`z^{(k)}` the start of segment :math:`k` and
:math:`\varphi_t` the flow of :math:`F`, the claim is that the derivative of
:math:`\chi(z^{(k)})` along :math:`v` vanishes for every :math:`\chi \in \mathcal{A}_k`. For
:math:`k = 1` it is :math:`\sum_l a_l\, X_l(\chi)(z_0) + \sum_u a_u\, \partial_u \chi(z_0) = 0`
by (D). If it holds for
:math:`k`, the derivative of :math:`\chi(\varphi_t(z^{(k)}))` along :math:`v` is analytic
in :math:`t` on the segment, and the derivatives at :math:`t = 0` are those of
:math:`L_F^j \chi \in \mathcal{A}_k` at :math:`z^{(k)}`, which vanish, so it vanishes on the
whole segment. With :math:`\chi` an output this is the kernel of every row; with
:math:`\chi = \varphi \circ E_e` at the event time, which :math:`v` does not move, it is the claim
for :math:`k + 1`. :math:`\square`

The proof needs (B) and (C) as identities of rational functions only. The flow need not keep
the span :math:`\mathcal{D}`: for :math:`X = x^2\, \partial_y` and :math:`F = \partial_x`,
:math:`[F, X] = (2/x)\, X`, yet the flow moves :math:`\partial_y` from :math:`x = -1` to
:math:`x = 0`, where :math:`\mathcal{D} = 0`. The proof follows the functions that the fields
annihilate instead, which a drop of rank does not affect. The identities are proven over
:math:`\mathbb{Q}` per state equation, observable, initial value and event, by symbolic
cancellation. The state components are the closed form read on the states, the initial-value
identities solved for the parameters that set the initial values, a combination of the
symmetries of the model equations, or a solution on the *support* of the direction. The
identities of the lift are linear in :math:`\xi_x`. The support starts at the states whose
equation, initial value or output reads a parameter that the direction moves and grows by the
states whose equations read it; on it, :math:`\xi_x` is a polynomial of degree one in the states,
and of degree two on at most 12 states, solved from the identities at random points modulo a
prime. The lift of :math:`c(\theta)\, Y` is :math:`c(\theta)` times the lift of :math:`Y`, so the
coefficients are read as rational numbers, from two primes, times a factor :math:`c(\theta)`:
one, a component, a component divided by the coordinate it moves, or the inverse of the common
denominator of the components, where the quotient agrees at two parameter points. Otherwise up to
8 coefficients are solved over the rational functions of the parameters from as many independent
identities at points with small coordinates. The solution is a candidate; the identities over
:math:`\mathbb{Q}` decide. The regimes of conditions with the same initial values try the lift of
the first such regime first.

An identity is first evaluated modulo the prime :math:`2^{61} - 1` at a random point; a nonzero
value refutes it exactly. Otherwise the numerator is expanded in a polynomial ring over
:math:`\mathbb{Q}` with a cap on the terms of every polynomial and on the term products of one
multiplication; past a cap the identity is undecided, and the direction keeps the level of the
kernel check.

.. _unproven:

Unproven directions
-------------------

``info["proof"]`` lists every direction that no proof closes, with the ``support`` and the
``reason`` of each failed route:

- ``lift of the closed form: [Y, F] on x does not vanish``: the field with state components read
  from the closed form is no symmetry, and no other lift was found.
- ``output is not a polynomial in t``: the output identities decide only outputs polynomial in
  time.
- ``no combination of the regime fields moves the parameters alike``, ``no combined lift`` (with
  the cause, such as more than 8 regimes or work above ``reconst_control(exact_work_cap=)``): the
  regimes need state components that agree on the parameters.
- ``events after the start``, ``an event at the start``, ``an input switches on at the start``:
  the proof at the resting states covers a start at rest with the inputs at rest that no event
  follows or changes. With events, states that no event reaches and that rest alone, such as a
  subsystem driving the states the events set, become constant coordinates on their resting
  variety, parametrized by a linearly entering parameter; the field holds the component of each
  resting state that keeps the variety tangent, and the following states :math:`s` get components
  :math:`\xi_s = A s + a`, with a matrix :math:`A` and a vector :math:`a` solved from the
  identities of regimes, outputs and events. This covers one resting condition, with reason
  ``resting subsystem as coordinates`` and the failing identity.
- ``an inverted power recast``, ``the resting states have no rational parametrization``, ``more
  than 8 resting states per condition``: a balance solved for a recast power, without rational
  parametrization by the parameters, or with more than the 8 states for which the resting
  Jacobian is inverted symbolically stays unproven, the last also in ``info["limits"]``.
- ``no exact proof with these nuisance coordinates``: the unknown-input coordinates have no lift.

.. _kernel-check:

Error probability of a closed form
----------------------------------

**The kernel check.** A closed form :math:`v` passes the check at a point if it lies in the
kernel there. A point where the kernel is not defined, at a pole or without a resting state, is
skipped. A wrong closed form gives a nonzero *kernel identity*
:math:`\sum_j \mathrm{OIM}_{lj}\, v_j` for some row :math:`l`, which cleared is a polynomial of
degree :math:`D`, so it passes at one point with probability at most :math:`D / p` and at two
points with probability at most :math:`(D / p)^2`. With :math:`n_k` the most points a check drew
for the two that passed (``info["bound"]["draws"]``), the pair is one of
:math:`\binom{n_k}{2}`. For :math:`n_v` verified directions and the largest :math:`D`,

.. math::
   :label: eq-kernel-bound

   \Pr\big[\text{a wrong direction is verified}\big] \le
   n_v \binom{n_k}{2} \Big(\frac{D}{p}\Big)^2 + \text{order bound},

which ``info["bound"]`` holds. The bound needs a rational model without root events, event times
rational in the parameters, and Lie and gap orders proven sufficient, since the check runs at the
orders used. It covers the closed forms of the routes checked at points uniform modulo
:math:`p` (``fit``, ``read-off``, ``narrow``, ``jets``, ``homotopy fit``); a form of another
route, such as a static subsystem or the forward fit, is checked at particular points only and
stays ``checked``.

**The bound at rest.** Under ``equilibrate`` a check point draws the coordinates uniformly modulo
:math:`p` and solves the balances :math:`f_i` of each condition there. A point counts where,
unless linear elimination solves the balances, every state is nonzero. The resting states are
algebraic, not rational, in the coordinates, so the Schwartz-Zippel lemma (:ref:`order-bound`) is applied on the resting
variety. Let :math:`Z` be an irreducible component of the resting variety with
:math:`\dim Z = d`, and :math:`P` a polynomial of degree :math:`D` that does not vanish on
:math:`Z`; :math:`V(P)` denotes the zero set of :math:`P` and a *hypersurface* the zero set of one nonzero
polynomial. Then :math:`Z \cap V(P)` has degree at most :math:`D \deg Z \le D \prod_i \deg f_i`,
the *Bezout number* of the balances :cite:p:`Heintz1983`, the projection of this intersection to the coordinates lies in a
hypersurface of at most that degree :cite:p:`Heintz1983`, and therefore

.. math::
   :label: eq-rest-bound

   \Pr_\theta\big[\exists x:\ (\theta, x) \in Z,\ P(\theta, x) = 0\big] \le
   \frac{D\, \deg Z}{p}.

Let :math:`D_\theta` and :math:`D_x` be the columns of :math:`D_N` for the coordinates and for the
states, and :math:`\operatorname{adj}` the adjugate matrix. With :math:`J_x` invertible, a closed
form :math:`v` extends to the kernel of the joint rows exactly where

.. math::

   \det(J_x)\, D_\theta v - D_x \operatorname{adj}(J_x)\, J_\theta v = 0 .

Multiplied by :math:`\det(J_x)`, this identity also vanishes at a check point with a singular
Jacobian, and it does not vanish on the component where :math:`J_x` is invertible at one
resting point of the condition, which is checked. The degree of the identity is the degree :math:`D` of the
cleared kernel identity with the start as coordinates plus :math:`2 \sum_i (\deg f_i - 1)`. A
closed form that reads resting states of further conditions, ``x_c<c>``, lives on the product of
their varieties over :math:`\theta`. The solves leave the component only above a proper
subvariety of the coordinates of degree at most :math:`\prod_i \deg f_i`, the zeros of the
pivots of a linear elimination or the projection of the other components, provided every root
taken lies on one component over :math:`\mathbb{Q}(\theta)`: a rest solved by linear elimination,
or interior roots conjugate over :math:`\mathbb{Q}(\theta)` (``info["branches"]``), on a split
factor where a balance splits. In Equation :eq:`eq-kernel-bound` the degree :math:`D` is then
replaced by :math:`(D + 2 \sum_i (\deg f_i - 1)) \prod_i \deg f_i + \sum \prod_i \deg f_i`, the
products and the sum taken over the conditions read. Without such a component (branches not
decided), with resting states continued along a line at the check points, or with a power or
exponential in the balances, a direction that the proof leaves open is ``checked``, and
``info["bound"]`` gives the reason.

**What the bounds exclude.** The bounds above are probabilities over the points modulo :math:`p`. They
exclude the chance that :math:`p` divides the *content* of an integer identity, the greatest
common divisor of the coefficients, which then vanishes modulo :math:`p`, or that :math:`p` is
one of the finitely many primes at which an identity nonzero on a resting component vanishes
there modulo :math:`p`. The counts that follow have no bad primes: the exact one has no random
choice, the verified one runs over :math:`\mathbb{Q}`.

.. _levels:

Levels and rank status
----------------------

The *level* of a closed form says how far it is established:

- ``exact``: proven over :math:`\mathbb{Q}`, as a scaling of the integer kernel or as a field of
  :ref:`proofs-over-q`.
- ``verified``: in the kernel at two seeded points modulo the verification prime, with the error
  probability of :ref:`kernel-check` in ``info["bound"]``.
- ``checked``: the same test where no bound applies (``info["bound"]`` says why), or a test on the
  entries that can be evaluated at the points.
- ``open``: no closed form.

The rank (``info["rank_status"]``) is

- *proven* when it is full, when it plus the number of independent exact directions is :math:`d`,
  or when every direction is proven over :math:`\mathbb{Q}`;
- *verified*, with a probability, when the Lie, gap and series orders are proven sufficient and
  every direction is verified with a bound; when every direction lies in the span of invariant
  fields, checked at random points with a bound (``info["rank_fields_bound"]``); or when a rank
  modulo unknown inputs holds at a random point;
- a *lower bound* otherwise, i.e., the generic rank may be higher and some reported directions may
  not exist.

Besides proofs, a result rests on two assumptions (``info["assumptions"]``,
``Reduction.assumptions``), both part of the declared model: the domain :math:`\Theta`
(``positive``) and, with switches, the start sides that ``regimes`` declares where the initial
values do not decide them. A verified closed form or rank is wrong with probability at most
``error_bound``, including a wrong certified order. Floating point enters the choice of a gauge
and of the ``reduce_cq`` pivots, never a rank, a sign or a closed form. Every other step is
proven or named in the result: a ``checked`` direction, a ``lower bound`` rank, a section claim
under ``unproven``, a limit under ``info["limits"]``, a resting branch under
``info["branches"]``.
