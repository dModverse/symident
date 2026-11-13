Method
======

This page states the mathematics behind symident: the notion of identifiability it decides,
how the rank of the observability-identifiability matrix is computed and when its order is
sufficient, how non-identifiable directions are obtained in closed form and proven, which
statements rest on random points and with which error probability, and how a model is reduced
to identifiable parameters. A small model serves as example throughout the first part.

Notation and terms
------------------

**Model.** symident analyzes models given by ordinary differential equations (ODEs),

.. math::
   :label: eq-model

   \dot x = f(x, u, \eta), \qquad x(t_0) = x_0(\eta), \qquad y = h(x, \eta),

with states :math:`x \in \mathbb{R}^{n_x}`, parameters :math:`\eta`, start time :math:`t_0`,
inputs :math:`u = u(t, \eta)`, i.e., given functions of time such as a dose set by an event,
and an observation function :math:`h = (h_1, \dots, h_{n_y})` with one component per
observable. The *coordinates* :math:`\theta \in \Theta \subseteq \mathbb{R}^d` are the
estimated parameters and the free initial values; they give the parameters by a
*parametrization* :math:`\eta = \eta(\theta)` and the initial value :math:`x_0(\theta)`, and
:math:`\Theta` is an open set, the *parameter space*. The observed quantities along the
solution,

.. math::

   y(t; \theta) = h\big(x(t), \eta(\theta)\big),

form the *prediction*. Right-hand sides, initial values and observables are rational
functions, or they are made rational by auxiliary states (see `Non-rational terms`_). A
*direction* at :math:`\theta` is a vector :math:`\xi \in \mathbb{R}^d` of rates of change of
the coordinates.

**Extended state.** Adding parameters and time to the states, :math:`z = (x, \eta, t)` with
:math:`\dot\eta = 0` and :math:`\dot t = 1`, makes the model the autonomous vector field
:math:`F = f\, \partial_x + \partial_t` [Tunali1987]_. :math:`F(\varphi)` is the rate of
change of a function :math:`\varphi(z)` along a solution, and the *start* is
:math:`z_0(\theta) = (x_0(\theta), \eta(\theta), t_0)`. A set :math:`S` of points :math:`z` is
*invariant* under :math:`F` when :math:`F` is tangent to it at each of its points, so that a
solution that starts in :math:`S` stays in :math:`S`.

**Conditions and events.** A *condition* is one experiment; conditions share :math:`\theta`
and differ in fixed values, inputs or renamings. An *event* changes a state or an input
instantaneously at a given time and splits a condition into *segments*.

**Algebra.** A *variety* is the common zero set of finitely many polynomials; it is
*irreducible* when it is not the union of two smaller varieties. A statement holds for
*generic* points of an irreducible variety :math:`V` when it holds outside a proper subvariety
of :math:`V`; on :math:`\Theta` the exceptions have measure zero. The *generic rank* of a
matrix of rational functions on :math:`V` is its rank at generic points of :math:`V`, which is
its largest rank on :math:`V`. The *degree* of a polynomial is its total degree, that of a
vector of rational functions the largest degree of the numerators and of the common
denominator; to *clear* an expression is to multiply it by its denominator. The degree
:math:`\deg V` of a variety of dimension :math:`m` is the number of its points on a generic
affine subspace of complementary dimension.

**Primes.** Ranks are computed over finite fields :math:`\mathbb{F}_p`. The four *engine
primes* :math:`p_1, \dots, p_4`, close to :math:`1.5 \cdot 10^9`, are fixed. The *verification
prime* :math:`p`, a prime below :math:`1.4 \cdot 10^9` distinct from them, is drawn from the
seed of each detection. A point *uniform modulo* :math:`p` has independent coordinates, each
uniform in :math:`\{0, \dots, p - 1\}`, drawn from a seed of its own. The *base point*
:math:`\theta_b` has the coordinates 2, 3, 5, ..., the first primes.

**Closed forms.** A *closed form* of a direction gives each component :math:`\xi_i` as a
rational expression of the coordinates. Its *level* (``exact``, ``verified``, ``checked``,
``open``) says how far it is established (`Levels and rank status`_), and its *route* how it
was obtained (`General directions`_).

As example we consider two compartments, a substance :math:`A` that is converted into
:math:`B` with rate :math:`k_1`, a loss of :math:`B` with rate :math:`k_2`, and an observation
of :math:`B` up to an unknown scale :math:`s`,

.. math::
   :label: eq-example

   \dot A = -k_1 A, \qquad \dot B = k_1 A - k_2 B, \qquad y = s\, B,

with free initial values :math:`A_0` and :math:`B_0`, so that
:math:`\theta = (k_1, k_2, s, A_0, B_0)` and :math:`d = 5`.

Theoretical background
----------------------

Structural identifiability and Lie symmetries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The model is *locally structurally identifiable* at :math:`\theta` if no nearby
:math:`\tilde\theta \ne \theta` has the same prediction. *Structural* means that the question
is decided by the equations alone, as if the observables were known exactly and continuously,
and for generic :math:`\theta`.

A continuous family of coordinate values with the same prediction is the orbit of a *Lie
symmetry*, a local one-parameter group :math:`\Phi_\lambda` of transformations of
:math:`\Theta` with

.. math::
   :label: eq-symmetry

   y\big(t; \Phi_\lambda(\theta)\big) = y(t; \theta)

[Yates2009]_ [Merkt2015]_ [Massonis2020]_. The group is described by its *generator*, the
vector field

.. math::
   :label: eq-generator

   X = \sum_{i=1}^d \xi_i(\theta)\, \partial_{\theta_i}, \qquad
   \xi(\theta) = \partial_\lambda \Phi_\lambda(\theta)\big|_{\lambda = 0},

whose value :math:`\xi(\theta)` at :math:`\theta` is a direction. The integral curve of
:math:`X` through :math:`\theta` is the *orbit* :math:`\mathcal{O}(\theta)`. The coordinates
with :math:`\xi_i \not\equiv 0` form the *support* of the generator. A function :math:`I(\theta)`
with :math:`X(I) = 0` is an *invariant* of :math:`X`; it is constant on orbits, and the
prediction depends on :math:`\theta` only through invariants. With :math:`r` generators whose
values are linearly independent at generic :math:`\theta`, :math:`d - r` functionally
independent combinations of the coordinates are *identifiable*. In Equation :eq:`eq-example`,
multiplying both initial values by a factor and dividing :math:`s` by it leaves :math:`y`
unchanged; the generator is

.. math::

   X = A_0\, \partial_{A_0} + B_0\, \partial_{B_0} - s\, \partial_s,

and :math:`k_1`, :math:`k_2`, :math:`s A_0` and :math:`s B_0` are its invariants.

Lie derivatives and the observability-identifiability matrix
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The *Lie derivatives* of the output are

.. math::
   :label: eq-lie

   L_F^0 h = h, \qquad L_F^{k+1} h = F\big(L_F^k h\big).

For analytic models they determine the prediction: near :math:`t_0` it is the Lie series

.. math::
   :label: eq-lie-series

   y(t; \theta) = \sum_{k \ge 0} \frac{(t - t_0)^k}{k!}\, L_F^k h\big(z_0(\theta)\big),

and elsewhere its analytic continuation, so equal Lie derivatives at :math:`t_0` give equal
predictions [Pohjanpalo1978]_. We collect the Lie derivatives up to the *Lie order* :math:`N`
and differentiate them with respect to :math:`\theta`,

.. math::
   :label: eq-oim

   \Psi_N(\theta) = \big(h, L_F^1 h, \dots, L_F^N h\big)\big(z_0(\theta)\big), \qquad
   \mathrm{OIM}_N(\theta) = \frac{\partial \Psi_N}{\partial \theta},

which gives the *observability-identifiability matrix* (OIM) [Villaverde2016]_. Its *rows of
order* :math:`k` are the gradients :math:`\partial_\theta\, L_F^k h_i(z_0(\theta))`,
:math:`i = 1, \dots, n_y`. A generator leaves :math:`\Psi_N` unchanged, so its value lies in
:math:`\ker \mathrm{OIM}_N(\theta)` for every :math:`N`. Conversely, where the rank of
:math:`\mathrm{OIM}_N` is its largest over all orders and all :math:`\theta`, which holds for
generic :math:`\theta`, every vector of the kernel is the value of a generator: the kernels
have constant dimension there, so a smooth vector field with values in them extends any
kernel vector, and along its flow every Lie derivative, and by Equation :eq:`eq-lie-series`
the prediction, stays constant. The model is therefore locally structurally identifiable when
:math:`\mathrm{OIM}_N` has rank :math:`d` for some :math:`N`, and the *non-identifiable
directions* are the vectors of the kernel.

For the example of Equation :eq:`eq-example` the first Lie derivatives are

.. math::

   L_F^0 h = s B, \qquad L_F^1 h = s\,(k_1 A - k_2 B), \qquad
   L_F^2 h = s\,\big(k_2^2 B - k_1 (k_1 + k_2) A\big),

evaluated at :math:`(A, B) = (A_0, B_0)`. The rows of :math:`\mathrm{OIM}_N` reach rank four,
and the kernel is spanned by the value of the generator above.

Sufficient Lie order
~~~~~~~~~~~~~~~~~~~~

The Lie order :math:`N` has to stop, and a rank that stays constant for a few orders is not
final at a special start, such as an initial value fixed to a number, a resting state or the
state after a dose. For :math:`\dot x = \theta`, :math:`y = x^m` and :math:`x_0 = 0`, every
row below order :math:`m` vanishes, and only the row of order :math:`m` shows that
:math:`\theta` is identifiable. symident therefore stops at a proven upper bound of the rank,
the *ceiling* below [HermannKrener1977]_.

Let :math:`D_k` be the matrix of the gradients :math:`\partial_z L_F^j h_i`, :math:`j \le k`,
:math:`i \le n_y`, and :math:`S` an irreducible variety of points :math:`z` that is invariant
under :math:`F`. By the chain rule,
:math:`\mathrm{OIM}_k(\theta) = D_k\big(z_0(\theta)\big)\, \partial_\theta z_0(\theta)`.

**Lemma.** At a generic point of :math:`S`, once :math:`\operatorname{rank} D_{k+1} =
\operatorname{rank} D_k` for one :math:`k`, the rank stays the same for every later order.

*Proof.* Let :math:`\Omega_k` be the span of the rows of :math:`D_k` over the rational
functions on :math:`S`. A row is a one-form :math:`\omega = \sum_i \omega_i\, \mathrm{d}z_i`,
and its Lie derivative along :math:`F` is
:math:`(\mathcal{L}_F \omega)_i = \sum_j F_j\, \partial_{z_j} \omega_i + \sum_j \omega_j\,
\partial_{z_i} F_j`. It is additive, satisfies :math:`\mathcal{L}_F(a\, \omega) = F(a)\, \omega
+ a\, \mathcal{L}_F \omega` for functions :math:`a`, and by the chain rule
:math:`\mathcal{L}_F\, \partial_z \varphi = \partial_z F(\varphi)`, so it maps the row of
:math:`L_F^j h_i` to that of :math:`L_F^{j+1} h_i`. Hence

.. math::
   :label: eq-filtration

   \mathcal{L}_F \sum_{j \le k} a_j\, \partial_z L_F^j h_i
   = \sum_{j \le k} F(a_j)\, \partial_z L_F^j h_i + \sum_{j \le k} a_j\, \partial_z L_F^{j+1} h_i .

If :math:`\Omega_{k+1} = \Omega_k`, the right-hand side lies in :math:`\Omega_k` for every
element of :math:`\Omega_k`, so :math:`\Omega_k` is invariant under :math:`\mathcal{L}_F` and
contains every later row. :math:`\square`

Let :math:`\rho_S`, the *ceiling*, be this final rank at a generic point of :math:`S`.

**Proposition.** Let :math:`S` contain the start :math:`z_0(\theta)` of every
:math:`\theta \in \Theta`. If at one start :math:`z_0(\theta^\ast)` the rank of :math:`D_N`
equals :math:`\rho_S`, then :math:`\ker \mathrm{OIM}_k(\theta) = \ker \mathrm{OIM}_N(\theta)`
for all :math:`k \ge N`, both at :math:`\theta^\ast` and at generic :math:`\theta`.

*Proof.* By the lemma no :math:`D_k` exceeds rank :math:`\rho_S` on :math:`S`, so every minor
of size :math:`\rho_S + 1` vanishes on :math:`S`, and :math:`\operatorname{rank} D_k \le
\rho_S` at every point of :math:`S`. The rank of :math:`D_N(z_0(\theta))` is the rank of a
matrix rational in :math:`\theta`, so at generic :math:`\theta` it is at least its value at
:math:`\theta^\ast`, hence equal to :math:`\rho_S`. Wherever it equals :math:`\rho_S`, the row
spaces of :math:`D_k(z_0(\theta))`, :math:`k \ge N`, contain that of :math:`D_N` and have
dimension at most :math:`\rho_S`, so they coincide, and so do the kernels after multiplication
by :math:`\partial_\theta z_0`. :math:`\square`

The proposition is what the checks of closed forms need: they run at random coordinate values,
at the order certified at one start. A *resting point* :math:`z_*` is one with
:math:`f(z_*) = 0`; for an autonomous model, where :math:`t` is dropped from :math:`z`,
:math:`F(z_*) = 0` there, and the rows form a *Krylov sequence*, each the previous one times
the same matrix,

.. math::
   :label: eq-krylov

   \partial_z L_F^{k+1} h(z_*) = \partial_z L_F^k h(z_*)\; \partial_z F(z_*),

so there one order without change of rank is final at the point itself.

**The set** :math:`S`. Without ``equilibrate``, :math:`S` consists of the starts shifted by a
subspace :math:`U` of the states,
:math:`S = \{z_0(\theta) + (b, 0, 0) : \theta \in \Theta,\ b \in U\}`, with :math:`U` the
smallest subspace for which :math:`F` stays in :math:`U \times \{0\} \times \mathbb{R}` on
:math:`S`. It is built from :math:`U = \{0\}` by adding the state part of :math:`F` at a random
point of the current :math:`S` until that part lies in :math:`U`. This covers a start at rest,
a subsystem at rest, states that stay at zero, fixed initial values, and linear relations the
dynamics keeps. Under ``equilibrate`` every start is a resting point, and :math:`S` is the
resting variety of the condition (`Steady states`_), on which :math:`f = 0`; its generic point
is a random coordinate value with the resting state solved there (the first of 24 draws with a
root). In both cases the rank at the start is taken at the base point. Where the start does
not reach :math:`\rho_S`, e.g., on the level set of a nonlinear conserved quantity, no order
is proven sufficient. The order then stops after a *plateau*, three orders without change of
rank (``SYMIDENT_LIEPLATEAU``), and the result says that the Lie order is not proven
sufficient.

**Conditions and events.** The rows of several conditions are stacked, and their row space
stops changing once that of every condition has stopped, so every condition is expanded to
its own order. An event moves the start of the next segment: a state replaced by a value of
the coordinates is removed from :math:`U`, a value added keeps :math:`U`, and an event that
multiplies a state or reads the states adds that state's direction to :math:`U`. The
proposition extends to the power series of `Conditions and events`_ and is checked at a
vanishing gap.

**Random points.** :math:`U` and :math:`\rho_S` are taken at points uniform modulo the
verification prime :math:`p`. They are wrong only where a polynomial vanishes: the state part
of :math:`F` at the step that ends :math:`U`, or a minor of size :math:`\rho_S` of the rows of
an order, both composed with the start, whose degree :math:`\deg z_0` multiplies theirs, and
on a resting variety times the Bezout number of its balances (`Error bounds`_). Since the
generic ranks rise strictly up to their final value, orders up to the number of columns plus
one reveal a wrong plateau. The degrees are those of the rational system the modular engine
evaluates, with exponentials, free powers and trigonometric functions as separate states. A
point is the first of its draws without a pole of the starts, up to 8, or on a resting variety
the first with a root, up to 24. It is wrong only if one of the points drawn is, so each term is
multiplied by the number of points drawn for its point, and the terms add up to the order bound
of `Error bounds`_.

**Without a stopping rule.** A direction that is an exact symmetry, i.e., a scaling under
which every expression is homogeneous (`Scalings`_) or a field with the identities of
`Proofs over Q`_, lies in the kernel at every order, so a rank that together with such
independent directions gives :math:`d` is proven for all orders. ``engine="symbolic"`` computes
the Lie derivatives over the rational functions and confirms the order by an exact rank:
where the rows at the initial values reach the generic rank, a nonvanishing minor makes every
later row a combination of earlier ones. Where no order is proven sufficient, a *saturation
guard* (``verify=True``) takes the rank at the base point again six orders further
(``SYMIDENT_VERIFY_MARGIN``) and the gap order of `Conditions and events`_ two orders further.
If the rank grows, the detection is repeated from that order, up to three times; if it still
grows, the rank is a lower bound, ``info["verification"]`` says where, and a reduction is
refused.

Computation
-----------

Ranks over finite fields
~~~~~~~~~~~~~~~~~~~~~~~~

symident evaluates :math:`\mathrm{OIM}_N` exactly over :math:`\mathbb{F}_p` in a compiled
*modular engine*: the Lie derivatives as power series in time truncated at order :math:`N`,
the derivatives with respect to :math:`\theta` as dual numbers, all rows at a point in one pass
[Sedoglavic2002]_. A point can only underestimate the generic rank, and a random point does so
with small probability [Schwartz1980]_ [Zippel1979]_. The rank of the result is taken at the
base point, as the largest over the four engine primes. Randomness enters through the points
uniform modulo the verification prime: one point that checks the rank of the *saturation*,
i.e., of the order chosen, and the points that check closed forms.

Conditions and events
~~~~~~~~~~~~~~~~~~~~~

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

Steady states
~~~~~~~~~~~~~

With ``equilibrate=True`` every condition starts at rest: its initial value is a *resting
state* :math:`x^\ast(\theta)` with :math:`f(x^\ast, 0, \eta) = 0`, inputs at zero. The
*balances* of a condition are the components of :math:`f` cleared to polynomials, and its
*resting variety* :math:`W` is the closure of :math:`\{(\theta, x) : f(x, 0, \eta(\theta)) =
0\}`. No closed form of the resting state is needed: at each point the balances are solved over
:math:`\mathbb{F}_p`, by linear elimination, by resultants for coupled subsystems of one or two
states, or by msolve [Berthomieu2021]_. Only msolve excludes states at zero, by a Rabinowitsch
variable :math:`\tau` with :math:`\tau \prod_i x_i = 1`, and a resting state solved modulo one
prime is not resampled at another. The kernel is taken in coordinates and resting states
jointly, with the tangency of a direction :math:`(\xi_\theta, \xi_x)` to :math:`W`,

.. math::

   J_x\, \xi_x + J_\theta\, \xi_\theta = 0, \qquad J_x = \partial_x f, \quad J_\theta =
   \partial_\theta f,

as extra rows rather than through the implicit function theorem; :math:`J_x` is the *resting
Jacobian*. The rank holds where the resting Jacobian of every condition solved at the point
itself is invertible at the base point; otherwise it is a lower bound.

**Branches.** A *branch* is an irreducible component of :math:`W` over :math:`\mathbb{Q}`. By
default the first condition draws the resting states and solves the balances for the
*turnover rates*, one rate per state that enters its balance linearly (the *forward seed*);
this names the component that the drawn values parametrize. Where a balance has several roots
over :math:`\mathbb{F}_p`, the first nonzero one found is taken and named in
``info["branches"]``: by the factor over :math:`\mathbb{Q}(\theta)` it solves where a balance
splits into factors, else as *conjugate*. Conjugate means that the lexicographic Gröbner basis
of the balances, states at zero removed, has at a point the *shape*
:math:`x_1 - g_1(x_n), \dots, x_{n-1} - g_{n-1}(x_n), P(x_n)` with an irreducible *eliminant*
:math:`P`. Other points take their root on that branch or, without one, are left out.
Conjugate roots have equal rank, and an eliminant irreducible at a point is irreducible over
:math:`\mathbb{Q}(\theta)`.

**Zero at rest.** A state rests at zero when its right-hand side, with the inputs at zero, is
linear in it with a coefficient that is nonzero while the other such states are zero. In a
reaction network, a species that is consumed but not produced rests at zero as the single
reactant of a consuming reaction, as does the single reactant of positive weight in a reaction
that lowers a weight no reaction raises (a *sink*; a weight assigns each species a number in
:math:`[0, 1]`). Of several such reactants, as in :math:`A + B \to 0`, the first is set to zero
(``info["branches"]``). A state with right-hand side zero at rest, and the pivot of a conserved
total (below), has a free resting value, its initial value. Rates are read at generic values,
so a catalyst at rest is nonzero.

**Conserved totals.** A linear combination :math:`\sum_i a_i x_i` of states that the right-hand
sides conserve is a *total*; totals are read from the stoichiometry of a reaction network or
from the integer linear laws of the ODEs, unless given. Under ``equilibrate`` each total leaves
one resting value free, which one species of the total, its *pivot*, holds as its initial value;
the pivots are chosen per total, preferring ``free_initial`` and avoiding states at zero, inputs
and bases of free powers. Where directions stay open, other choices of pivots are tried, up to
six. With ``reduce_cq`` the pivot :math:`x_{i_0}` is eliminated instead,
:math:`x_{i_0} = (T - \sum_{i \ne i_0} a_i x_i)/a_{i_0}`, and the value :math:`T` of the total
becomes a parameter.

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

Unknown and known inputs
~~~~~~~~~~~~~~~~~~~~~~~~

An *unknown input* :math:`w(t)` is an analytic function of time that is left open. A direction
of :math:`\theta` is non-identifiable when nearby :math:`\tilde\theta` with some :math:`\tilde w`
give the prediction of :math:`(\theta, w)` for a generic :math:`w`.

The Lie derivative :math:`L_F^k h` holds the input only through its *jets*
:math:`w_j = w^{(j)}(t_0)`, :math:`j \le k`. The chain :math:`\dot w = w_1, \dots,
\dot w_{K-1} = 0` of length :math:`K` therefore joins the states [Martinelli2015]_
[Villaverde2019]_; it is exact for every row up to order :math:`K - 1` and is lengthened while
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
[Martinelli2019]_; [Villaverde2019]_ ask instead for observability of the extended state, jets
included.

The bound of `Sufficient Lie order`_ needs a fixed state space, and the chain grows with the
order, so no order is proven sufficient here. The rank is ``proven`` when it is full or when
independent exact directions fill the kernel, else it is a lower bound, in both cases as far
as the value of Equation :eq:`eq-nuisance` holds. Where :math:`\mathrm{OIM}_{\hat w}` drops
rank, that value can exceed the generic one. The value is

- proven where the jet columns that the rows up to the orders taken can hold have full column
  rank, at the base point or at a point uniform modulo :math:`p` with the same value;
- verified with the same value but no full column rank, with the chance by [Schwartz1980]_ that
  a maximal minor of :math:`\mathrm{OIM}_{\hat w}` vanishes there;
- a lower bound where the values differ.

The chain is part of the state and passes every gap, so the input is analytic across events;
at a switch it is only continuous (see `Switching models`_). A row after gap order :math:`M`
holds jets up to its Lie order plus :math:`M`, so the Lie order is held to :math:`K - 1 - M`.
Under ``equilibrate`` the chain rests at zero, and a jump at the start sets the nuisance
values.

A direction is proven as a field
:math:`Y = \xi_\theta\, \partial_\theta + \xi_x\, \partial_x + \xi_w\, \partial_w`, rational in
states, parameters and finitely many jets, with the identities of `Proofs over Q`_ extended by
:math:`(\partial_w f)\, \xi_w` per state and :math:`(\partial_w h)\, \xi_w` per observable, and
with the chain in :math:`F`. The jets need no identity, since :math:`\xi_{w_{j+1}} =
F(\xi_{w_j})` prolongs :math:`\xi_w`; a component with the last jet of the chain is not proven.
The published *input change* is :math:`\xi_w` of a proven field; a kernel vector gives it only
at :math:`t_0`.

A *known input* enters as the same chain, with known symbols as jet values, like the input
derivatives of the extended Lie derivatives of [Karlsson2012]_. Past order :math:`K - 1` the
chain is a polynomial, i.e., a specialization of the input, so the Lie order and the order
certificate are held to :math:`K - 1 - M`, and the chain is doubled, up to three times, while
the order reaches past it (``info["known_inputs"]``).

Switching models
~~~~~~~~~~~~~~~~

Each switch condition, a comparison or the argument of ``Heaviside()``, ``sign()``,
``abs()``, ``max()`` or ``min()``, is a sign condition :math:`\gamma_j(x, \eta, t) > 0`,
:math:`j = 1, \dots, n_\gamma`. The *declared signs* are the domain declared by ``positive``
and the start sides declared by ``regimes``. Conditions that the declared signs decide are
resolved; a factor of odd multiplicity of an undecided one is a *switching function*
:math:`\gamma_j`, with *switching surface* :math:`\Gamma_j = \{\gamma_j = 0\}`. A sign vector
:math:`\sigma \in \{+1, -1\}^{n_\gamma}` gives a smooth *regime* :math:`(F_\sigma, h_\sigma)`,
the field and the observables where every :math:`\gamma_j` has the sign :math:`\sigma_j`. A
*root event* (``root`` in place of ``time``) adds its root expression as a switching function
and resets its target state, the *event state*, on :math:`\Gamma_j`.

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
``info["limits"]`` names a condition with further such states. Other invariant sets of the
start, such as the level set of a conserved total, are not used, so the rank can exceed the
rank along the trajectories from the declared start.

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

No :math:`Y_i` moves a fixed symbol, an input or time. The direction is
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
nonzero at the sample point is nonzero as a rational function. The projection off the fresh
coordinates, a difference of two ranks, can rise at a special point. It is proven where every
kernel direction within the fresh coordinates alone is the unit direction of a fresh coordinate
entering no row (the entry value of a state reaching no observable of the copy), which is
checked at the sample point and then with the kernel at a point uniform modulo the verification
prime, whose projection then counts. With equal dimension there and no such proof it is
``verified``, with the chance by [Schwartz1980]_ that a maximal minor of the stacked rows or of
their fresh columns vanishes there, else a lower bound (``info["projection"]``,
``info["rank_status"]``).

**Upper bound.** The candidates are the scalings of `Scalings`_, the closed forms of the kernel
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
between crossings; its derivatives may jump at a crossing. The input direction
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
`Unknown and known inputs`_), with a start of their own per copy and, at a surface point, the
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

Non-rational terms
~~~~~~~~~~~~~~~~~~

The modular engine needs rational functions. Elementary functions are therefore recast into
rational form [Savageau1987]_, type by type in a fixed order and innermost first within a type;
a nesting across that order, such as :math:`\sin(\exp x)` or :math:`\exp(\arctan x)`, is
refused. A term :math:`\nu = \varphi(\psi)` with :math:`\psi` rational in the states becomes a
new state with right-hand side :math:`\varphi'(\psi)\, \sum_j \partial_{x_j} \psi \, f_j`,
which is rational in :math:`\nu` and :math:`\psi` for

.. math::

   \varphi(\psi) = \psi^\alpha:\ \ \frac{\alpha\, \nu}{\psi}, \qquad
   \exp(\psi):\ \ \nu, \qquad \log(\psi):\ \ \frac{1}{\psi}, \qquad
   \arctan(\psi):\ \ \frac{1}{1 + \psi^2}, \qquad \operatorname{artanh}(\psi):\ \ \frac{1}{1 -
   \psi^2},

and for :math:`\arcsin`, :math:`\arccos`, :math:`\operatorname{arsinh}` and
:math:`\operatorname{arcosh}` contains a square root, which is recast in turn. Trigonometric
functions go through :math:`\tan(\psi/2)`. A *free power* :math:`b^\mu`, with a parameter
:math:`\mu` as exponent, becomes a state :math:`B = b^\mu` with companion :math:`L_b = \log b`,
so that the field is rational and :math:`\mu` enters through :math:`\partial B / \partial \mu
= B L_b`.

Under ``equilibrate`` the resting model has no exponential, so a power :math:`b^\mu` of a
positive rational number :math:`b` is rewritten first: next to a power :math:`c^\mu` of a symbol
:math:`c` with the same exponent, :math:`c` stands for :math:`b c` throughout,
:math:`(b c)^\mu` being a power of one symbol, and the result is mapped back. Otherwise
:math:`b` is a constant with :math:`\log b` taken independent of the other constants, as for
logarithms of numbers in log coordinates (named in ``info["limits"]``).

A new state starts at :math:`\varphi(\psi)` of the initial state. Where this depends on the
coordinates, a transcendental :math:`\varphi` becomes an input of the modular engine of its own,
a *leaf*, drawn independently of its argument and tied to it by relation rows
:math:`\mathrm{d}\nu = \varphi'(\psi)\, \mathrm{d}\psi`: a transcendental graph lies in no proper
variety, so a random point on it is generic. With several transcendental constants, such as
:math:`e` and :math:`\pi`, this needs their algebraic independence, which follows from
Schanuel's conjecture and is not proven. An algebraic :math:`\varphi`, a root or rational power,
is exact: through :math:`L = \log \psi` when :math:`\psi` is affine in one coordinate, else by
the root modulo the prime at each sample point, over primes in which every root of the model
exists. Logarithms, fractional powers and bases of free powers need an argument positive on
:math:`\Theta`, also in initial and event values; otherwise an error names the symbols not
declared positive. An observable :math:`\kappa \sum_i a_i \log \rho_i + \beta`, with rational
:math:`\rho_i` and constants :math:`\kappa, a_i, \beta`, is strictly monotone in
:math:`\prod_i \rho_i^{a_i} e^{\beta/\kappa}` and is analyzed as that product.

Closed forms and proofs
-----------------------

The kernel at a point gives the non-identifiable directions as numbers. To read them as
generators, and to reduce the model, symident reconstructs them in closed form and then tries to
prove them over :math:`\mathbb{Q}`.

Scalings
~~~~~~~~

A *scaling* is a symmetry :math:`\Phi_\lambda(\theta)_i = e^{\alpha_i \lambda}\, \theta_i` with
integer *weights* :math:`\alpha_i`, so :math:`\xi_i = \alpha_i \theta_i`. A function is
*homogeneous* of weight :math:`a` when the scaling multiplies it by :math:`e^{a \lambda}`. With
weights :math:`\alpha_x` of the states as well, the scaling is a symmetry when every
right-hand side has the weight of its state, every observable, event time and multiplicative
event value the weight zero, and every initial value and replaced or added event value the
weight of its state. These conditions are linear in the weights and read off the exponents of
the monomials of all expressions together; their integer solutions form a lattice, the
*integer kernel*, and its basis gives the scalings exactly. They are taken out first and checked
again symbolically. A scaling that holds only on an invariant set, such as
:math:`(k_1, k_2) \mapsto (e^\lambda k_1, e^\lambda k_2)` for :math:`\dot x = k_1 - k_2 x`
started at rest, is not an exact symmetry and fails the check. The scaling of the example in
Equation :eq:`eq-example` is exact, with weights :math:`(0, 0, -1, 1, 1)`.

General directions
~~~~~~~~~~~~~~~~~~

The other non-identifiable directions are *general directions*. The coordinates are ordered with
the initial values of the free states first, in the order of the model, followed by the
parameters in a fixed sorted order. Gauss-Jordan elimination of the rows takes its pivot
columns from left to right, and a *free coordinate* is one without pivot, whose column depends
on the columns to its left. Each kernel vector has the component one on its free coordinate and
zero on the others; every other component is then a rational function of the coordinates.
Alternatively, a direction is fitted in log coordinates, :math:`\xi_i / \theta_i`, and multiplied
back, or chosen with the smallest support outside the scalings, with the component one on the
leftmost coordinate of the support.

The components are reconstructed from samples. Perturbation finds the variables of each entry;
the entry is sampled at many points and primes and fitted by dense interpolation, by sparse
interpolation after [BenOr1988]_ or by univariate rational interpolation along lines, and it is
lifted to rational coefficients by the Chinese remainder theorem and rational reconstruction
[Wang1982]_. The *route*, as ``Symmetry.route`` names it, is one of

- ``fit``: rational reconstruction from samples over several primes; ``read-off``: a fast pass
  that reads Laurent monomials on the support, with the other coordinates held; ``fit per
  prime``: points that each prime solves, joined by the Chinese remainder theorem; each possibly
  ``in log coordinates`` and with the kind of fit in parentheses, ``constant``, ``dense`` (a
  monomial table up to ``degree_cap``) or ``sparse``;
- ``forward fit per prime``: under ``equilibrate``, with the resting states drawn and the
  turnover rates solved, as in the forward seed;
- ``narrow fit``: the kernel restricted to the columns of the support of one direction, in which
  it is one-dimensional, with ``exchange groups`` where coordinates that enter only through their
  sum or their product were held at the base point and substituted back; ``jets (order k)``: the
  generator read from the Taylor coefficients of the kernel entries along a support of at most
  six coordinates;
- ``homotopy fit``: the fit along lines below.

A *sample bank* of points over the union of the relevant coordinates, with kernels per point and
prime, is shared by all dense fits. Before it is reported, every closed form, and a direction
along one coordinate that fills the kernel, is checked at seeded points uniform modulo the
verification prime, the *kernel check* of `Error bounds`_; under ``equilibrate`` on the
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

Invariant fields and gauge
~~~~~~~~~~~~~~~~~~~~~~~~~~

Some directions follow from the model equations without sampling. For a model with resting
states, a *global symmetry* is a field of one regime, found by a detection of that regime with
free initial values and extended to the initial and event values, that lies in the kernel. A
*static subsystem* is a set of states at rest that no event moves; parameters that only its
balances hold give fields that leave every such rest in place. Both are reported with these
names as route. Where every direction of the kernel lies in the span of fields that satisfy the
conditions of `Proofs over Q`_ at random points, the *invariant fields*, the rank holds at every
order (``info["rank_fields"]``), with an error bound in ``info["rank_fields_bound"]``. (A) is
checked as a zero and (B), (C) as membership in the span, at two random points; a wrong identity
passes only where a polynomial vanishes there, which the bound covers. Membership at random
points shows (B) and (C) as identities over the rational functions, the hypothesis of the
proposition in `Proofs over Q`_, with coefficients that may have poles on the trajectory.

With ``gauge``, the detection first finds the scalings and fixes one coordinate per scaling, as
a basis of the weights in the order of preference given and, with general directions, the one
that leaves the fewest nonzero entries in the remaining kernel. A second pass with these
coordinates fixed reconstructs the general directions. Without ``gauge``,
``info["gauge_suggestion"]`` names this choice.

Proofs over Q
~~~~~~~~~~~~~

A closed form is *proven* when it is shown to be a symmetry without random points. For a scaling
this is the check of `Scalings`_. For a general direction, symident *lifts* the direction to a
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
`Invariant fields and gauge`_ take the fields of the regimes.

**Proposition.** Under (A) to (D) the direction lies in the kernel at every Lie order and every
gap length.

*Proof.* Let :math:`\mathcal{A}` be the rational functions :math:`g` of a regime with :math:`X_l(g) = 0`
and :math:`\partial_u g = 0` for all :math:`l, u`. By (A), :math:`h \in \mathcal{A}`. By (B),
:math:`X_l(F g) = F(X_l g) - [F, X_l](g) = 0`, and :math:`[F, \partial_u] = 0`, so :math:`F`
maps :math:`\mathcal{A}` into itself; by (C), :math:`g \circ E_e \in \mathcal{A}` for :math:`g` in :math:`\mathcal{A}` of the
next regime. Let :math:`\mathcal{A}_k` be the functions of segment :math:`k` built from its outputs
and from :math:`g \circ E_e` with :math:`g \in \mathcal{A}_{k+1}` by Lie derivatives along
:math:`F`. They lie in :math:`\mathcal{A}` and are analytic along the trajectory, as is every expression
of the model where it is defined. With :math:`z^{(k)}` the start of segment :math:`k` and
:math:`\varphi_t` the flow of :math:`F`, the claim is that the derivative of
:math:`\chi(z^{(k)})` along :math:`v` vanishes for every :math:`\chi \in \mathcal{A}_k`. For
:math:`k = 1` it is :math:`\sum_l a_l\, X_l(\chi)(z_0) + \sum_u a_u\, \partial_u \chi(z_0) = 0`
by (D). If it holds for
:math:`k`, the derivative of :math:`\chi(\varphi_t(z^{(k)}))` along :math:`v` is analytic
in :math:`t` on the segment, and its derivatives at :math:`t = 0` are those of
:math:`L_F^j \chi \in \mathcal{A}_k` at :math:`z^{(k)}`, which vanish, so it vanishes on the
whole segment. With :math:`\chi` an output this is the kernel of every row; with
:math:`\chi = g \circ E_e` at the event time, which :math:`v` does not move, it is the claim
for :math:`k + 1`. :math:`\square`

The proof needs (B) and (C) as identities of rational functions only. The flow need not keep
the span :math:`\mathcal{D}`: for :math:`X = x^2\, \partial_y` and :math:`F = \partial_x`,
:math:`[F, X] = (2/x)\, X`, yet the flow moves :math:`\partial_y` from :math:`x = -1` to
:math:`x = 0`, where :math:`\mathcal{D} = 0`. The proof follows the functions that the fields
annihilate instead, which a drop of rank does not affect. The identities are proven over
:math:`\mathbb{Q}` per state equation, observable, initial value and event, by symbolic
cancellation. The state components are the closed form read on the states, the initial-value
identities solved for the parameters that set the initial values, or a combination of the
symmetries of the model equations.

An identity is first evaluated modulo the prime :math:`2^{61} - 1` at a random point; a nonzero
value refutes it exactly. Otherwise its numerator is expanded in a polynomial ring over
:math:`\mathbb{Q}` with a cap on the terms of every polynomial and on the term products of one
multiplication; past a cap the identity is undecided, and the direction keeps the level of its
kernel check.

Unproven directions
~~~~~~~~~~~~~~~~~~~

``info["proof"]`` lists every direction that no proof closes, with its ``support`` and the
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

Error bounds
------------

Several steps above take place at random points. A statement found there can be wrong, and
symident bounds the probability of this where a degree bound of the polynomials involved is
known. The random choices are the verification prime :math:`p`, drawn per detection from its
seed, and points uniform modulo :math:`p`, each from a seed of its own. All bounds rest on the
Schwartz-Zippel lemma [Schwartz1980]_ [Zippel1979]_: a nonzero polynomial of degree :math:`D` in
:math:`m` variables over a field vanishes at a point drawn uniformly from :math:`A^m`, for a
finite subset :math:`A` of the field, with probability at most :math:`D / |A|`.

**The order bound.** A wrong certificate of the Lie or gap order needs a polynomial to vanish at
one of the random points of `Sufficient Lie order`_ or `Conditions and events`_: the step that
ends the subspace :math:`U`, or a minor of the rows of an order, on a resting variety with the
degree of Equation :eq:`eq-rest-bound`. With the degrees of the rational system, the
probabilities add up to ``info["order_bound"]``, each multiplied by the number of points drawn
until its point was taken, since the point taken is wrong only if one of them is;
``info["order_draws"]`` is the largest such number, and the points of the gap order are drawn
once. ``info["order_bound_why"]`` names the reason for a missing bound.

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
:math:`p` (``fit``, ``read-off``, ``narrow fit``, ``jets``, ``homotopy fit``); a form of another
route, such as a static subsystem or the forward fit, is checked at particular points only and
stays ``checked``.

**The bound at rest.** Under ``equilibrate`` a check point draws the coordinates uniformly modulo
:math:`p` and solves the balances :math:`f_i` of each condition there. A point counts where,
unless linear elimination solves the balances, every state is nonzero. The resting states are
algebraic, not rational, in the coordinates, so the lemma above is applied on the resting
variety. Let :math:`Z` be an irreducible component of the resting variety with
:math:`\dim Z = d`, and :math:`P` a polynomial of degree :math:`D` that does not vanish on
:math:`Z`; :math:`V(P)` denotes its zero set and a *hypersurface* the zero set of one nonzero
polynomial. Then :math:`Z \cap V(P)` has degree at most :math:`D \deg Z \le D \prod_i \deg f_i`,
the *Bezout number* of the balances [Heintz1983]_, its projection to the coordinates lies in a
hypersurface of at most that degree [Heintz1983]_, and therefore

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
resting point of the condition, which is checked. Its degree is the degree :math:`D` of the
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

**What the bounds exclude.** The bounds are probabilities over the points modulo :math:`p`. They
exclude the chance that :math:`p` divides the *content* of an integer identity, the greatest
common divisor of its coefficients, which then vanishes modulo :math:`p`, or that :math:`p` is
one of the finitely many primes at which an identity nonzero on a resting component vanishes
there modulo :math:`p`.

Levels and rank status
----------------------

The *level* of a closed form says how far it is established:

- ``exact``: proven over :math:`\mathbb{Q}`, as a scaling of the integer kernel or as a field of
  `Proofs over Q`_.
- ``verified``: in the kernel at two seeded points modulo the verification prime, with the error
  probability of `Error bounds`_ in ``info["bound"]``.
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

Discrete symmetries
-------------------

Local identifiability leaves finitely many coordinate values with the prediction of a generic
one; more than one such value is a *discrete symmetry*, which no generator shows.
:meth:`~symident.Symmetries.solution_count` equates the Taylor coefficients of the observables up
to an order with their values at a random point modulo a prime and counts the solutions with
msolve [Berthomieu2021]_, over the algebraic closure, away from zeros of coordinates and
denominators, as in the global identifiability test of [Hong2020]_. The count holds at that point
and is no proof.

Reduction and sections
----------------------

A reduction replaces the coordinates by combinations that the output determines [Chappell1998]_
[Evans2000]_ [Massonis2023]_, the invariants :math:`I_l(\theta)` with :math:`X(I_l) = 0`
[Olver1993]_. The directions are reduced in *blocks*, sets of directions with overlapping
supports, each block with the group :math:`G` generated by the flows of its generators.
symident searches, within degree bounds, monomial, polynomial and rational invariants (with a
denominator in one coordinate), Darboux polynomials and exponential factors [Prelle1983]_
[Goriely2001]_, i.e., polynomials :math:`P` and exponentials :math:`e^{P/Q}` that :math:`X`
multiplies by a polynomial, and invariants by separable quadrature, i.e., by integrating
:math:`\mathrm{d}\theta_i / \xi_i = \mathrm{d}\theta_j / \xi_j` where it separates, and by an
integrating factor. The invariants become new parameters, :math:`I_l(\theta) = q_l`, and the
remaining coordinates are expressed through them.

A scaling is removed by fixing one of its coordinates at one, such as an initial value or a
scale, a *pin*. Every orbit of a scaling meets the hypersurface :math:`\theta_i = 1` of a moved
coordinate once, so the result is a *global section*: a one-to-one map of the new parameters onto
a *global cross-section*, a set meeting every orbit exactly once, with the invariants as global
coordinates on the orbit space :math:`\Theta / G`. In the example of Equation :eq:`eq-example`,
:math:`A_0 = 1` meets every orbit of the scaling once. A direction along one coordinate is pinned
like a scaling; every orbit meets the pin where a *sign certificate*, a proof that an expression
has one sign on :math:`\Theta`, shows that the generator component on the pinned coordinate has
one sign, else the orbits can be the intervals between the zeros of that component, and the pin
is listed under ``unproven``.

The pinned coordinates of a block of scalings, its *transversal*, are chosen among the
coordinates of the support that are not fixed: subsets :math:`T` of the size of the block, taken
in the order of the coordinates, for which the columns :math:`W_T` of the weight matrix
:math:`W` are invertible and :math:`W_T^{-1} W` has integer entries, so that every other
coordinate becomes a monomial of the new parameters; for many candidates a greedy choice by the number of scalings moving each
coordinate. Without a certified transversal the block keeps its invariants only
(``invariant_only``). Before the search for invariants, a block of several general generators is
split by *module reduction*: an elimination order with few new nonzero entries, found at random
points modulo a prime and repeated exactly, gives combinations of the generators with smaller
supports, and independent ones form separate blocks.

For a block of :math:`r` general generators the invariants :math:`I(\theta) = q` are solved for
coordinates :math:`\theta_s = R(q, g)`, with :math:`r` *gauge coordinates* :math:`g` left. Every
level set :math:`\Lambda_q = \{\theta \in \Theta : I(\theta) = q\}` is one orbit when

1. :math:`R(q, g)` is the unique solution for every gauge value, so :math:`\Lambda_q` is the graph
   of :math:`R(q, \cdot)` over :math:`\mathcal{G}_q = \{g : (R(q, g), g) \in \Theta\}`;
2. the :math:`r \times r` minor of the generator components on :math:`g` has one sign on
   :math:`\Theta`, so every orbit is :math:`r`-dimensional and open in :math:`\Lambda_q` (for
   generators with poles, the minor of the fields times their denominators);
3. :math:`\mathcal{G}_q` is connected.

The set :math:`\mathcal{G}_q` is given by one inequality :math:`\pi_k(q, g) > 0` per positive
solved coordinate and square-root factor. It is connected when every inequality is convex in
:math:`g` (affine, a concave quadratic, or, in one positive gauge coordinate, a polynomial with at
most one sign change and so, by Descartes' rule of signs, at most one positive root), or when a
gauge coordinate enters every inequality affinely with a coefficient of known sign, so that the
*fibers*, the sets of that coordinate at fixed other gauge coordinates, are intervals and the set
is connected with its projection, decided the same way. Square roots in :math:`g`, or of a single
positive invariant, are first removed by a square substitution, which is one to one on the
positive half-line. The *section point* :math:`\hat\theta(q)` then lies in :math:`\Lambda_q` and,
by a *positivity certificate*, a proof that each entry is positive on the domain, in
:math:`\Theta`.

**Coverage.** ``reach`` says that every level set of the invariants is one orbit; with *total
coverage* every orbit then meets the section, under *partial coverage* exactly the orbits whose
invariants lie where the solved entries are positive. Under declared bounds, i.e., expressions in
``positive``, an entry without positivity certificate is kept, with partial coverage, and listed
under ``unproven``. Without bounds it is kept only when the search found no section certified on
the whole domain, and only where a sampled orbit has every entry positive: the transformation then
covers the orbits on which the ``sign_conditions``, the entries as inequalities in :math:`q`,
hold, the sampled one among them, and no others. A positivity certificate treats an *atom*, a
non-polynomial subterm such as :math:`\log x`, as positive only where its sign is proven; an
AM-GM certificate, a lower bound by the inequality of arithmetic and geometric means over pairs of
terms, is strict only by the positive remainder the pairs leave, on a term free of coordinates
that are not declared positive where there are any.

**Face sections.** A *face* is a boundary set :math:`\theta_f = 0` for coordinates
:math:`\theta_f` of the block; a section on a face lies outside :math:`\Theta` and is reached only
in a limit. It is tried first, and once one is found no other is searched; the new parameters are
the coordinates of the face point, not invariants. With :math:`\theta_f` as gauge coordinates,
the conditions above make every level set one orbit, and :math:`R(q, \theta_f) \to R(q, 0) > 0`
puts the face point :math:`(0, R(q, 0))` in the closure of every orbit of :math:`\Lambda_q`. The
output, constant along the orbit, equals that at the face point when the model is continuous
there: no model expression has a pole at the face, every denominator in the block parameters has
one sign there, and under ``equilibrate`` :math:`x^\ast(\theta)` has a limit. For that limit each
generator is lifted to a field :math:`\tilde X = X + \xi_x \partial_x` tangent to the resting
variety, by the state components of an exact direction or by the implicit derivative
:math:`\xi_x = -J_x^{-1} J_\theta \xi_\theta`; exactly verified invariants of the lifted fields
that solve for the moved states and stay positive at :math:`\theta_f = 0` give the limit along
every orbit. The implicit derivative needs :math:`J_x` regular: proven when :math:`\det J_x` has
one sign on :math:`\Theta`, else checked at random points. Without a lift, the resting state is
continuous when :math:`f` is affine in the states with a resting Jacobian of one sign at the face,
else it is checked at random face points; a checked resting state sets ``face_level`` and
``level`` of the block to ``checked``.

**Injectivity.** That every orbit meets the cross-section :math:`\mathcal{C}` at most once
(``injective``) is proven, for any number of generators, when the solved coordinates recover every
point of :math:`\mathcal{C}` from the invariants, :math:`\hat\theta(I(\theta)) = \theta` on
:math:`\mathcal{C}`, since two points of one orbit have equal invariants. The identity is shown
over :math:`\mathbb{Q}` on a parametrization of the positive points of :math:`\mathcal{C}`,
monomial where the section equates monomials, else by solving each equation for a coordinate of
degree one; the claim is made for sections with total coverage off a face. For one generator it
also holds when :math:`X` changes the pinned coordinate, or the logarithm of the quotient of the
two sides of the section equation, with one sign on :math:`\Theta`.

**Zero limits.** With ``zero_limits=True`` each block reports which coordinates can be set to zero:
everywhere, under a condition, or in a limit along the orbits when the face is mapped into itself
by the generators, and whether this is certain (``to_dict()["zero_compatibility"]``).

An unproven claim is listed under ``unproven`` of the block and makes the block ``open``; a block
is ``verified`` or ``checked`` when a direction or a face resting state has that level, else
``exact``. The transformation is checked over :math:`\mathbb{Q}`, and ``check=True`` runs the
detection again on the reduced model (``Reduction.checks["loop"]``).

References
----------

.. [BenOr1988] M. Ben-Or and P. Tiwari. A deterministic algorithm for sparse
   multivariate polynomial interpolation. *Proceedings of the 20th ACM Symposium on
   Theory of Computing*, 301-309, 1988.

.. [Berthomieu2021] J. Berthomieu, C. Eder and M. Safey El Din. msolve: a library for
   solving polynomial systems. *Proceedings of ISSAC 2021*, 51-58, 2021.

.. [Chappell1998] M. J. Chappell and R. N. Gunn. A procedure for generating locally
   identifiable reparameterisations of unidentifiable non-linear systems by the similarity
   transformation approach. *Mathematical Biosciences* 148(1), 21-41, 1998.

.. [Evans2000] N. D. Evans and M. J. Chappell. Extensions to a procedure for generating
   locally identifiable reparameterisations of unidentifiable systems. *Mathematical
   Biosciences* 168(2), 137-159, 2000.

.. [Goriely2001] A. Goriely. *Integrability and Nonintegrability of Dynamical Systems*.
   Advanced Series in Nonlinear Dynamics 19, World Scientific, 2001.

.. [Heintz1983] J. Heintz. Definability and fast quantifier elimination in algebraically
   closed fields. *Theoretical Computer Science* 24(3), 239-277, 1983.

.. [HermannKrener1977] R. Hermann and A. J. Krener. Nonlinear controllability and
   observability. *IEEE Transactions on Automatic Control* 22, 728-740, 1977.

.. [Hong2020] H. Hong, A. Ovchinnikov, G. Pogudin and C. Yap. Global identifiability of
   differential models. *Communications on Pure and Applied Mathematics* 73(9), 1831-1879,
   2020.

.. [Karlsson2012] J. Karlsson, M. Anguelova and M. Jirstrand. An efficient method for
   structural identifiability analysis of large dynamic systems. *IFAC Proceedings Volumes*
   45(16), 941-946, 2012.

.. [Martinelli2015] A. Martinelli. Extension of the observability rank condition to
   nonlinear systems driven by unknown inputs. *23rd Mediterranean Conference on Control and
   Automation*, 589-595, 2015.

.. [Martinelli2019] A. Martinelli. Nonlinear unknown input observability: extension of the
   observability rank condition. *IEEE Transactions on Automatic Control* 64(1), 222-237,
   2019.

.. [Massonis2020] G. Massonis and A. F. Villaverde. Finding and breaking Lie symmetries:
   implications for structural identifiability and observability in biological modelling.
   *Symmetry* 12(3), 469, 2020.

.. [Massonis2023] G. Massonis, J. R. Banga and A. F. Villaverde. AutoRepar: a method to
   obtain identifiable and observable reparameterizations of dynamic models with mechanistic
   insights. *International Journal of Robust and Nonlinear Control* 33(9), 5039-5057, 2023.

.. [Merkt2015] B. Merkt, J. Timmer and D. Kaschek. Higher-order Lie symmetries in
   identifiability and predictability analysis of dynamic models. *Physical Review E* 92(1),
   012920, 2015.

.. [Olver1993] P. J. Olver. *Applications of Lie Groups to Differential Equations*.
   Graduate Texts in Mathematics 107, Springer, 2nd edition, 1993.

.. [Pohjanpalo1978] H. Pohjanpalo. System identifiability based on the power series
   expansion of the solution. *Mathematical Biosciences* 41, 21-33, 1978.

.. [Prelle1983] M. J. Prelle and M. F. Singer. Elementary first integrals of differential
   equations. *Transactions of the American Mathematical Society* 279(1), 215-229, 1983.

.. [Savageau1987] M. A. Savageau and E. O. Voit. Recasting nonlinear differential equations
   as S-systems: a canonical nonlinear form. *Mathematical Biosciences* 87(1), 83-115, 1987.

.. [Schwartz1980] J. T. Schwartz. Fast probabilistic algorithms for verification of
   polynomial identities. *Journal of the ACM* 27(4), 701-717, 1980.

.. [Sedoglavic2002] A. Sedoglavic. A probabilistic algorithm to test local algebraic
   observability in polynomial time. *Journal of Symbolic Computation* 33(5), 735-755,
   2002.

.. [Tunali1987] E. T. Tunali and T. J. Tarn. New results for identifiability of nonlinear
   systems. *IEEE Transactions on Automatic Control* 32(2), 146-154, 1987.

.. [Villaverde2016] A. F. Villaverde, A. Barreiro and A. Papachristodoulou. Structural
   identifiability of dynamic systems biology models. *PLOS Computational Biology* 12(10),
   e1005153, 2016.

.. [Villaverde2019] A. F. Villaverde, N. Tsiantis and J. R. Banga. Full observability and
   estimation of unknown inputs, states and parameters of nonlinear biological models.
   *Journal of the Royal Society Interface* 16(156), 20190043, 2019.

.. [Wang1982] P. S. Wang, M. J. T. Guy and J. H. Davenport. P-adic reconstruction of
   rational numbers. *ACM SIGSAM Bulletin* 16(2), 2-3, 1982.

.. [Yates2009] J. W. T. Yates, N. D. Evans and M. J. Chappell. Structural identifiability
   analysis via symmetries of differential equations. *Automatica* 45(11), 2585-2591,
   2009.

.. [Zippel1979] R. Zippel. Probabilistic algorithms for sparse polynomials.
   *Symbolic and Algebraic Computation, EUROSAM '79*, Lecture Notes in Computer Science
   72, 216-226, 1979.
