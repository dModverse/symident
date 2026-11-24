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
again symbolically. A power :math:`x^e` whose exponent holds no state and only coordinates of
weight zero has the weight :math:`e\,\alpha_x`, also for a state that starts at 0. A scaling that holds only on an invariant set, such as
:math:`(k_1, k_2) \mapsto (e^\lambda k_1, e^\lambda k_2)` for :math:`\dot x = k_1 - k_2 x`
started at rest, is not an exact symmetry and fails the check. The scaling of the example in
Equation :eq:`eq-example` is exact, with weights :math:`(0, 0, -1, 1, 1)`.

.. _general-directions:

General directions
------------------

The other non-identifiable directions are *general directions*. The coordinates are ordered with
the initial values of the free states first, followed by the parameters in a fixed sorted order.
Gauss-Jordan elimination of the rows takes the pivot columns from left to right, and a *free
coordinate* is one without pivot. Each kernel vector has the component one on one free coordinate
and zero on the others; every other component is then a rational function of the coordinates.
Alternatively, a direction is fitted in log coordinates, :math:`\xi_i / \theta_i`, or chosen with
the smallest support outside the scalings.

The components are reconstructed from samples at many points and primes, by dense or sparse
interpolation :cite:p:`BenOr1988` or by rational interpolation along lines, and lifted to rational
coefficients by the Chinese remainder theorem and rational reconstruction :cite:p:`Wang1982`. The
*route*, ``Symmetry.route``, says how: ``fit``, ``read-off`` and ``fit per prime`` sample the kernel
(``forward fit per prime`` with drawn resting states), the ``narrow`` routes and ``jets (order k)``
restrict it to the support of one direction, and ``homotopy fit`` fits Padé forms along lines through
the base point, also in parameters and resting states (the state :math:`x` of condition :math:`c`
named ``x_c<c>``).

Before it is reported, every closed form is checked at seeded points uniform modulo the
verification prime, the *kernel check* of :ref:`kernel-check`.

.. _orbit-geometry:

Orbit geometry
--------------

Each direction with a closed form is classified by the geometry of the orbits, ``Symmetry.kind``,
up to a common factor :math:`c(\theta)` of the components:

- a *scaling*, :math:`\xi_i = c\, w_i \theta_i` with numbers :math:`w_i`, has lines in logarithmic
  coordinates as orbits and monomial invariants; a pin :math:`\theta_i = 1` meets every orbit once;
- a *translation*, :math:`\xi_i = c\, v_i` with numbers :math:`v_i`, has lines as orbits and linear
  invariants. It is proven like a scaling, from the linear structure of the model. In the positive
  orthant an orbit ends on the boundary, so the section lies on a face, :math:`\theta_i = 0`,
  reached in the limit;
- every other direction is *curved*.

Translations and curved directions are general directions (``Symmetry.type``). The labels
:math:`X_1, X_2, \dots` of the report and of the reduction count the scalings first, then the
translations and the curved directions.

.. _invariant-fields:

Invariant fields and gauge
--------------------------

Some directions follow from the model equations without sampling: a *global symmetry*, a field of
one regime with free initial values that lies in the kernel, and a *static subsystem*, a set of
states at rest that no event moves, whose parameters give fields that leave the rest in place.
Where every direction of the kernel lies in the span of fields that satisfy the conditions of
:ref:`proofs-over-q` at random points, the *invariant fields*, the rank holds at every order
(``info["rank_fields"]``), with an error bound in ``info["rank_fields_bound"]``: (A) is checked as
a zero and (B), (C) as membership in the span at two random points, which shows them as identities
over the rational functions, the hypothesis of the proposition in :ref:`proofs-over-q`.

With ``gauge``, the detection first finds the scalings and fixes one coordinate per scaling, in the
order of preference given, then reconstructs the general directions with these coordinates fixed.
Without ``gauge``, ``info["gauge_suggestion"]`` names this choice.

.. _proofs-over-q:

Proofs over Q
-------------

A closed form is *proven* when it is shown to be a symmetry without random points. For a scaling
this is the check of :ref:`scalings`. For a general direction, symident *lifts* the direction to a
field :math:`Y = \xi_\theta\, \partial_\theta + \xi_x\, \partial_x` on parameters and states and
proves the conditions below over :math:`\mathbb{Q}` for every regime of a condition, with event maps
:math:`E_e` between the regimes. They are stated for fields :math:`X_1, \dots, X_m` per regime and
the unit directions :math:`\partial_u` of parameters :math:`u` that enter no regime, whose span is
:math:`\mathcal{D}`; :math:`X'_l` are the fields of the regime after an event, and :math:`v` is the
direction lifted onto the start :math:`z_0`.

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
have poles. The fields are the lift alone, with :math:`[F, Y] = 0`, or the lift together with the
lifts proven before, with coefficients solved at a random point and checked over :math:`\mathbb{Q}`.
With unknown inputs the jets take no identity, since the prolongation of the input component makes
(B) hold there.

**Proposition.** Under (A) to (D) the direction lies in the kernel at every Lie order and every
gap length.

*Proof.* Let :math:`\mathcal{A}` be the rational functions :math:`\varphi` of a regime with
:math:`X_l(\varphi) = 0` and :math:`\partial_u \varphi = 0` for all :math:`l, u`. By (A),
:math:`h \in \mathcal{A}`. By (B), :math:`X_l(F \varphi) = F(X_l \varphi) - [F, X_l](\varphi) = 0`,
so :math:`F` maps :math:`\mathcal{A}` into itself; by (C), :math:`\varphi \circ E_e \in \mathcal{A}`
for :math:`\varphi` in :math:`\mathcal{A}` of the next regime. Let :math:`\mathcal{A}_k` be the
functions of segment :math:`k` built from the outputs and from :math:`\varphi \circ E_e`,
:math:`\varphi \in \mathcal{A}_{k+1}`, by Lie derivatives along :math:`F`; they are analytic along
the trajectory. The claim is that the derivative of :math:`\chi(z^{(k)})` along :math:`v` vanishes
for every :math:`\chi \in \mathcal{A}_k`, :math:`z^{(k)}` the start of segment :math:`k`. For
:math:`k = 1` this is (D). If it holds for :math:`k`, the derivative of
:math:`\chi(\varphi_t(z^{(k)}))` along :math:`v` is analytic in :math:`t` with vanishing
derivatives :math:`L_F^j \chi` at :math:`t = 0`, so it vanishes on the segment; with :math:`\chi` an
output this is the kernel of every row, with :math:`\chi = \varphi \circ E_e` at the event time,
which :math:`v` does not move, the claim for :math:`k + 1`. :math:`\square`

The proof needs (B) and (C) as identities of rational functions only. The flow need not keep the
span :math:`\mathcal{D}`: for :math:`X = x^2\, \partial_y` and :math:`F = \partial_x`, :math:`[F, X]
= (2/x)\, X`, yet :math:`\mathcal{D} = 0` at :math:`x = 0`. The proof follows the functions that the
fields annihilate instead, which a drop of rank does not affect.

The identities are proven over :math:`\mathbb{Q}` per state equation, observable, initial value and
event, by symbolic cancellation. They are linear in :math:`\xi_x`, whose candidates come from the
closed form, the initial values, symmetries of the model equations, or a solve at random points; the
identities decide. A nonzero value modulo a prime refutes an identity exactly; an identity past the
work caps is undecided, and the direction keeps the level of the kernel check.

.. _unproven:

Unproven directions
-------------------

``info["proof"]`` lists every direction that no proof closes, with the ``support`` and the
``reason`` of each failed route: a lift whose bracket with :math:`F` does not vanish on the states,
an output that is not polynomial in time, regimes whose fields do not move the parameters alike,
events after a resting start, a resting state without rational parametrization, or unknown-input
coordinates without a lift. Where resting subsystems with events are taken as constant coordinates,
the reason is ``resting subsystem as coordinates`` with the failing identity; a limit is also named
in ``info["limits"]``.

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
:math:`p` and solves the balances :math:`f_i` of each condition there. The resting states are
algebraic in the coordinates, so the Schwartz-Zippel lemma (:ref:`order-bound`) is applied on the
resting variety. Let :math:`Z` be an irreducible component of the resting variety and :math:`P` a
polynomial of degree :math:`D` that does not vanish on :math:`Z`. Then :math:`Z \cap V(P)` has
degree at most :math:`D \deg Z \le D \prod_i \deg f_i`, the *Bezout number* of the balances
:cite:p:`Heintz1983`, the projection to the coordinates lies in a hypersurface of at most that
degree, and therefore

.. math::
   :label: eq-rest-bound

   \Pr_\theta\big[\exists x:\ (\theta, x) \in Z,\ P(\theta, x) = 0\big] \le
   \frac{D\, \deg Z}{p}.

With :math:`D_\theta` and :math:`D_x` the columns of :math:`D_N` for coordinates and states, a
closed form :math:`v` extends to the kernel of the joint rows exactly where

.. math::

   \det(J_x)\, D_\theta v - D_x \operatorname{adj}(J_x)\, J_\theta v = 0 ,

an identity of degree :math:`D + 2 \sum_i (\deg f_i - 1)` that does not vanish on the component
where :math:`J_x` is invertible at one resting point, which is checked. The solves leave the
component only above a subvariety of degree at most :math:`\prod_i \deg f_i`, provided every root
taken lies on one component over :math:`\mathbb{Q}(\theta)` (``info["branches"]``). In Equation
:eq:`eq-kernel-bound` the degree :math:`D` is then replaced by :math:`(D + 2 \sum_i (\deg f_i - 1))
\prod_i \deg f_i + \sum \prod_i \deg f_i`, over the conditions read. Without such a component, or
with a power or exponential in the balances, a direction that the proof leaves open is ``checked``,
and ``info["bound"]`` gives the reason.

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
