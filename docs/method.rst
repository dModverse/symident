Method
======

Model and parameter space
-------------------------

symident analyzes ODE models

.. math::

   \dot x = f(x, u, \eta), \qquad x(t_0) = x_0(\eta), \qquad y = h(x, \eta),

with states :math:`x \in \mathbb{R}^n`, parameters :math:`\eta`, an input
:math:`u = u(t, \eta)` and an observation function :math:`h`. A *parametrization*
:math:`\theta \mapsto \eta(\theta)` may give the parameters; :math:`\theta`, the estimated
parameters and free initial values, lies in an open *parameter space*
:math:`\Theta \subseteq \mathbb{R}^d`. The *prediction* is
:math:`g(t; \theta) = h\big(x(t), \eta(\theta)\big)`. Right-hand sides, initial values and
observables are rational, or become rational by the recasts below.

Structural identifiability
--------------------------

The model is *locally structurally identifiable* at :math:`\theta` if no nearby
:math:`\tilde\theta \ne \theta` has the same prediction. *Structural* means decided by the
equations alone, as if the observables were known exactly and continuously, for generic
:math:`\theta`, outside a set of measure zero.

A continuous family of parameter values with the same prediction is the orbit of a *Lie
symmetry*, a local one-parameter group :math:`\Phi_s` of transformations of :math:`\Theta`
with :math:`g\big(t; \Phi_s(\theta)\big) = g(t; \theta)` [Yates2009]_ [Merkt2015]_
[Massonis2020]_, with *generator*

.. math::

   X = \sum_{i=1}^d \xi_i(\theta)\, \partial_{\theta_i}, \qquad
   \xi(\theta) = \partial_s \Phi_s(\theta)\big|_{s = 0},

and the integral curve of :math:`X` through :math:`\theta` is the *orbit*
:math:`\mathcal{O}(\theta)`. The coordinates with :math:`\xi_i \ne 0` are the *support*. An
*invariant*, :math:`X(I) = 0`, is constant on orbits, and the prediction depends on
:math:`\theta` only through invariants. The number of identifiable combinations is
:math:`d` minus the number of independent generators.

Lie derivatives and the OIM
---------------------------

With :math:`z = (x, \eta, t)`, :math:`\dot \eta = 0`, :math:`\dot t = 1`, the model is the
vector field :math:`F = f\, \partial_x + \partial_t` with initial value :math:`z_0(\theta)`
[Tunali1987]_, and :math:`F(\varphi)` is the rate of change of :math:`\varphi` along a
solution. The *Lie derivatives* of the output are

.. math::

   L_F^0 h = h, \qquad L_F^{k+1} h = F\big(L_F^k h\big).

For analytic models they fix the prediction,

.. math::

   g(t; \theta) = \sum_{k \ge 0} \frac{(t - t_0)^k}{k!}\, L_F^k h\big(z_0(\theta)\big),

so equal Lie derivatives at :math:`t_0` give equal predictions [Pohjanpalo1978]_. Up to
order :math:`N`,

.. math::

   \Psi_N(\theta) = \big(h, L_F^1 h, \dots, L_F^N h\big)\big(z_0(\theta)\big), \qquad
   \mathrm{OIM}_N(\theta) = \frac{\partial \Psi_N}{\partial \theta},

the *observability-identifiability matrix* [Villaverde2016]_. Generators leave
:math:`\Psi_N` unchanged, so their tangent vectors lie in
:math:`\ker \mathrm{OIM}_N(\theta)`; at generic :math:`\theta` they span it. The model is
locally structurally identifiable when :math:`\mathrm{OIM}_N` has rank :math:`d` for some
:math:`N`.

Rank over finite fields
-----------------------

:math:`\mathrm{OIM}_N` is evaluated exactly over :math:`\mathbb{F}_p`, :math:`p < 2^{31}`:
Lie derivatives as truncated power series in time, :math:`\theta`-derivatives as dual
numbers, all rows at a point in one pass [Sedoglavic2002]_. A point can only underestimate
the generic rank, a random one with small probability [Schwartz1980]_ [Zippel1979]_. The
base point is not random: the coordinates are the primes 2, 3, 5, ..., the rank the largest
over four primes. Randomness enters through one saturation check point and the verification
points, uniform modulo a prime :math:`q` drawn from the seed per detection.

**Sufficient order.** With the parameters fixed, the states at the start of a segment lie
on a smallest invariant affine space, on which the rows have a generic rank no order exceeds
[HermannKrener1977]_; an event setting a state to a value of the states frees that state.
Where the rows at the start reach this rank, the order is *sufficient* and the rank generic,
not a false plateau; otherwise the order stops after a plateau and the result says the Lie
order is not proven sufficient. Affine space and generic rank are taken at points uniform
modulo :math:`q` and fail only where a polynomial of the degree of the field (a step of the
affine space) or a maximal minor of the rows of an order vanishes; the ranks rise strictly
up to the plateau, so orders up to the number of columns plus one cover a wrong plateau. By
[Schwartz1980]_ the order is sufficient except with probability at most
``info["order_bound"]``, the sum of these terms, which the bound of a verified closed form
includes. The degrees are those of the rational system the kernel runs, with exponentials,
free powers and trigonometric functions as separate states. Under ``equilibrate`` the
resting states are solved modulo the engine primes only and the certified order has no bound;
``info["order_bound_why"]`` names the reason for any missing bound.

``engine="symbolic"`` computes the Lie derivatives over the rational functions. The order
is found at a point and confirmed by an exact rank; where the rows at the initial values
reach the generic rank [HermannKrener1977]_, a nonvanishing minor makes every later row a
combination of earlier ones, so the kernel is final and a direction annihilating the rows
identically is proven.

Conditions and events
---------------------

Conditions share parameters and differ in fixed values or renamings. Their rows are stacked
over the shared coordinates, :math:`\ker \mathrm{OIM} = \bigcap_c \ker \mathrm{OIM}^{(c)}`.

An *event* changes a state or an input instantaneously at a known or estimated time and
splits a condition into segments, each starting from the previous one propagated over the
gap :math:`\Delta t`. With the gaps on the line :math:`\Delta t_i = \varepsilon\, \delta_i`
through the actual gaps, the rank is taken over the Laurent series
:math:`\mathbb{F}_p((\varepsilon))`, which separates directions of particular event timings
from generic ones. The kernel at :math:`\varepsilon = 1` is read from Padé forms of the
kernel entries that further coefficients confirm, a heuristic acceptance that gives a basis
of the kernel only.

The rows after a gap are power series in :math:`\varepsilon` of a fixed family of
functions, so the rank of all coefficients, at points uniform modulo :math:`q`, bounds their
rank (error terms in ``info["order_bound"]``). The gap order :math:`M` rises until the
truncated rank reaches the bound, which proves it sufficient; a truncated rank above the
bound refutes it and nothing is certified. A symbolic event time gives no bound, since it
moves the boundary and shifts the series and the event state outside the family. Without
bound (``equilibrate``, recast terms, such an event time) or at
``reconst_control(gap_order_cap=)`` the rank is a lower bound, and ``info["limits"]`` says
where :math:`M` stopped.

Unknown inputs
--------------

An *unknown input* :math:`w(t)` is an analytic function of time left open. A direction of
:math:`\theta` is non-identifiable when nearby :math:`\tilde\theta` with some :math:`\tilde w`
give the prediction of :math:`(\theta, w)` for a generic :math:`w`.

:math:`L_F^k h` holds the input only through the jet :math:`w(t_0), \dots, w^{(k)}(t_0)`.
The chain :math:`\dot w = w_1, \dots, \dot w_{K-1} = 0` joins the states [Martinelli2015]_
[Villaverde2019]_, exact for every row up to order :math:`K - 1` and lengthened while the
order reaches past it. Rows up to order :math:`k` see :math:`k + 1` jets of an infinite
family, so they never pin the input to a finite form. The jet values :math:`j` at
:math:`t_0` are *nuisance coordinates*, generic and unknown, outside :math:`\theta`. The
identifiable dimension is

.. math::

   \operatorname{rank} \big[\mathrm{OIM}_\theta \mid \mathrm{OIM}_j\big]
   - \operatorname{rank} \mathrm{OIM}_j ,

the kernel of the full matrix modulo the kernel within :math:`j`, projected onto
:math:`\theta`; ``rank`` and ``dim`` count :math:`\theta` only. It is the dimension of the
row space part without jet components, the counterpart of the separation of the observable
codistribution in [Martinelli2019]_; [Villaverde2019]_ ask instead for observability of the
extended state, jets included.

[HermannKrener1977]_ needs a fixed state space and the chain grows with the order, so no
order is proven sufficient. The rank is proven when full or when independent exact
directions fill the kernel, else a lower bound, in both cases as far as the value of the
difference holds. Where :math:`\mathrm{OIM}_j` drops rank that value can exceed the generic
one. The value is

- proven where the jet columns the rows up to the orders taken can hold have full column
  rank, at the base point or at a point uniform modulo :math:`q` with the same value;
- verified with the same value but no full column rank, with the chance by
  [Schwartz1980]_ that a maximal minor of :math:`\mathrm{OIM}_j` vanishes there;
- a lower bound where the values differ.

The chain, part of the state, passes every gap, so the input is analytic across events;
at a switch it is only continuous (see `Switching models`_). A row after gap order
:math:`M` holds jets up to the row's Lie order plus :math:`M`, so the Lie order is held to
:math:`K - 1 - M`. Under ``equilibrate`` the chain rests at zero, and a jump at the start
sets the nuisance values.

A direction is proven as a field
:math:`Y = \xi_\theta\, \partial_\theta + \xi_x\, \partial_x + \xi_w\, \partial_w` rational in
states, parameters and finitely many jets, with the identities of `Closed forms and levels`_
extended by :math:`(\partial_w f)\, \xi_w` per state and :math:`(\partial_w h)\, \xi_w` per
observable, and the jet flow in :math:`F`. The jets need no identity, as
:math:`\xi_{w^{(k+1)}} = F\big(\xi_{w^{(k)}}\big)` prolongs :math:`\xi_w`; a component with
the last jet of the chain is not proven. The published *input change* is :math:`\xi_w` of a
proven field; a kernel vector gives it only at :math:`t_0`.

A *known input* enters as the same chain with known symbols as jet values, like the input
derivatives of the extended Lie derivatives of [Karlsson2012]_. Past order :math:`K - 1` the
chain is a polynomial, a specialization of the input, so Lie order and order certificate
are held to :math:`K - 1 - M`, and the chain is doubled, up to three times, while the order
reaches past it (``info["known_inputs"]``).

Switching models
----------------

Each switch condition, a comparison or the argument of ``Heaviside()``, ``sign()``,
``abs()``, ``max()`` or ``min()``, is a sign condition :math:`\gamma_j(x, \eta, t) > 0`.
Conditions the declared signs decide are resolved; an odd factor of an undecided one is a
*switching function* :math:`\gamma_j`, with surface :math:`\Gamma_j = \{\gamma_j = 0\}`. A sign
vector :math:`\sigma \in \{+1, -1\}^J` gives a smooth regime :math:`(F_\sigma, h_\sigma)`. A
``root`` event adds the root as a switching function and resets the event state on
:math:`\Gamma_j`.

**Regimes.** A direction counts only as a symmetry of every regime the declared signs
allow. Without ``regimes`` every sign vector counts unless the declared positivity decides
the sign of :math:`\gamma_a - \gamma_b` or :math:`\gamma_a + \gamma_b` against it; none is
dropped for parameter values or for being unreached, so the result holds for every initial
state. A parameter acting only in a regime that a decided condition excludes stays a
coordinate, with exact direction :math:`\partial_\theta`. Each allowed regime enters as a
copy from the start of the condition and, with other regimes allowed, as a copy switched
into it (``info["switching"]["conditions"]``).

**Entry points.** The state at a switch is unknown, so a direction counts only as a
symmetry from every point of the surface. A state that starts at zero in every condition,
that every allowed regime keeps at zero with finite derivatives in such states, and that no
later event moves, is zero along every trajectory: it is dropped (``info["zero_states"]``),
and a coordinate only such states read is a separate direction :math:`\partial_\theta`;
``info["limits"]`` names a condition with further such states. Other invariant sets of the
start, such as the level set of a conserved total, are not used, so the rank can exceed the
rank along the trajectories from the declared start.

**Visibility.** Regimes :math:`\sigma, \sigma'` are neighbors at :math:`\Gamma_j` when they
differ only in the sign of :math:`\gamma_j`. The switch is *visible* when for some
neighbors, observable :math:`y` and order :math:`k`

.. math::

   \big(L_{F_\sigma}^k h_{\sigma,y} - L_{F_{\sigma'}}^k h_{\sigma',y}\big)\big|_{\Gamma_j}
   \ne 0 .

A nonzero Taylor coefficient at a point of :math:`\Gamma_j`, modulo a prime, proves it. It is
*invisible* when for every pair the observables agree and every state reaching them has the
same right-hand side, so every Lie derivative agrees, and *undecided* otherwise. A root
event surface counts as visible.

**The rule.** A direction :math:`v` is reported non-identifiable when, in every condition,
it lies at the start in a distribution
:math:`\Delta = \operatorname{span}\{Y_1, \dots, Y_r\}` of fields on parameters and states
with

.. math::

   [Y_i, F_\sigma] \in \Delta, \qquad Y_i(h_\sigma) = 0 \qquad
   \text{for every regime } \sigma,

.. math::

   Y_i(\gamma_j) = 0 \ \text{on } \Gamma_j \ \text{for every visible surface}, \qquad
   Y_i(\tau_e) = 0, \qquad \mathrm{D}E_e\, Y_i \in \Delta \circ E_e \
   \text{for every event and reset},

and, at a surface that is not visible and that some :math:`Y_i` moves, for every pair of
neighbors

.. math::

   \big(F_\sigma - F_{\sigma'}\big)\big|_{\Gamma_j} \in \Delta, \qquad
   \big(h_\sigma - h_{\sigma'}\big)\big|_{\Gamma_j} = 0 .

No :math:`Y_i` moves a fixed symbol, an input or time. The direction is
:math:`v = \big(\xi, X(x_0)\big)`, with the derivative of the initial values along :math:`X`
as state part (under ``equilibrate``, of the resting state of each regime). Along every
regime flow :math:`v` stays in :math:`\Delta`, keeping the observables to first order; at a
visible surface :math:`v` is tangent and the crossing time stays; at a hidden one the
crossing may move, but the jump lies in :math:`\Delta` and the observables agree. So the rule
never reports a direction that changes the observables, for any sequence of the listed
regimes.

**Lower bound.** The directions of the rule satisfy the observability rows of copies of the
model sharing the parameters, whose rank bounds the rank from below:

- each regime from the start of each condition, with the events of the condition;
- each regime from a generic point of each visible surface linear in a state, where a
  direction of the rule is tangent to the surface;
- each regime from a generic point, when a switch of the condition is not visible;
- each copy switched at a fixed time into every other regime of the condition, with outputs
  along words of two regimes, :math:`L_{F_\sigma}^a L_{F_{\sigma'}}^b h_{\sigma'}`;
- the switching time :math:`\tau_j(\theta)` of every visible switch in time alone.

The stacked kernel, projected onto the coordinates, contains every direction of the rule;
a kernel dimension :math:`k` gives the lower bound :math:`d - k`. A minor nonzero at the
sample point is nonzero as a rational function. The projection off the fresh coordinates of
the surface points, a difference of two ranks, can rise at a special point. It is proven
where every kernel direction within the fresh coordinates alone is the unit direction of a
fresh coordinate entering no row (the entry value of a state reaching no observable of the
copy), checked at the sample point, then with the kernel at a point uniform modulo the
verification prime, whose projection then counts. With equal dimension there and no such
proof it is ``verified``, with the chance by [Schwartz1980]_ that a maximal minor of the
stacked rows or of their fresh columns vanishes there, else a lower bound
(``info["projection"]``, ``info["rank_status"]``).

**Upper bound.** Candidates are the scalings of the integer kernel, the closed forms of the
stacked kernel and the unit field of every coordinate; those failing a condition of the rule
are dropped until the family passes. Each identity is tested at a rational point, then
proven by symbolic simplification to zero; :math:`\ell` candidates left, independent at a
rational point, bound the rank by :math:`d - \ell`. If :math:`\ell = k`, the rank is
:math:`d - k` and the candidates span the non-identifiable directions; otherwise
:math:`d - k` is a lower bound, ``info["switching"]["uncertified"]`` names the rejected
candidates, and the unproven stacked-kernel directions, which satisfy only necessary
conditions, appear under ``info["switching"]["candidates"]``, not as symmetries. For an
undecided switch the rule accepts tangency or hidden jump and the rows omit the surface, so
both bounds stay valid.

**Unknown inputs.** The input is continuous where a trajectory crosses a surface and analytic
between crossings; the input derivatives may jump at a crossing. The input direction
:math:`G_\sigma = \partial_w F_\sigma`, over the inputs no observable of :math:`\sigma` reads,
joins the bracket condition,

.. math::

   [Y_i, F_\sigma] = \sum_k c_{ik}\, Y_k + a_{i\sigma}\, G_\sigma ,

and :math:`v = \sum_i \lambda_i Y_i` changes the input in :math:`\sigma` by
:math:`-\sum_i \lambda_i a_{i\sigma}`; an input an observable reads stays unchanged. Where
:math:`G_\sigma \in \Delta` the change is free. Otherwise it must not jump where a trajectory
crosses :math:`\Gamma_j` from :math:`\sigma` into a neighbor :math:`\sigma'`: on
:math:`\Gamma_j`,

.. math::

   a_{i\sigma'} - a_{i\sigma}
   - \frac{Y_i(\gamma_j)}{F_\sigma(\gamma_j)} \sum_k d_k\, a_{k\sigma'} = 0 ,
   \qquad F_\sigma - F_{\sigma'} = \sum_k d_k Y_k ,

the last term counting only at a hidden surface that :math:`Y_i` moves, whose crossing then
shifts. A switch is visible only through orders :math:`k \le r_y`, the lowest order at which
an input can enter :math:`L^k h_y` by the symbols of each order; higher orders hold input
derivatives. The jets of each copy are nuisance coordinates (see `Unknown inputs`_), with a
start of their own per copy and, at a surface point, the input value shared by the two
neighbors. A copy from a generic point of each regime replaces the copy switched at a fixed
time, which would restart the input off the surfaces, where the changes of two regimes
differ. The rank status is the weaker of the projections off the jets and off the fresh
coordinates. The rule holds per condition, so each condition needs a separate input,
renamed in ``conditions``.

**Listed regimes.** With ``regimes`` the rule, rows and candidates use the listed regimes of
each condition, and a surface counts only with listed regimes on both sides. Parameters of
other regimes stay coordinates and are reported non-identifiable. A sign of
:math:`\gamma_j(x_0)` decided by the declared signs must agree with the first entry, else an
error; an undecided one is declared by ``regimes`` and listed under ``declared`` in
``info["switching"]["conditions"]`` and in ``info["assumptions"]["starts"]``.

**Limits.** Each is an error, never a result:

- a surface in the parameters alone whose side ``regimes`` does not give;
- a condition on a switched expression;
- more than 4096 sign vectors;
- ``gauge``, ``reduce_cq``, ``free_initial``, ``engine="symbolic"``;
- a state named ``time``;
- under ``equilibrate``, a resting state without closed form;
- an infeasible listed regime, or a start that contradicts the initial values;
- with unknown inputs: an input read by several conditions, a switch on an input, an event
  after the start, an initial or event value that reads an input.

Steady states
-------------

With ``equilibrate=True`` every condition starts at rest, :math:`f(x^\ast, \eta) = 0`,
binding :math:`x_0 = x^\ast` to :math:`\theta`. No closed form is needed: per sample point the
resting state is solved over :math:`\mathbb{F}_p` by linear elimination, by resultants for
small coupled cores, or by msolve [Berthomieu2021]_. Only msolve excludes zero states, by a
Rabinowitsch variable, and a resting state solved modulo one prime is not resampled at
another. The kernel is taken in parameters and resting states jointly, with the tangency
:math:`\partial_x f\, \xi_x + \partial_\theta f\, \xi_\theta = 0` as extra rows, not via the
implicit function theorem. The parameter rank holds where the resting Jacobian of every
directly solved condition is invertible at the base point, else it is a lower bound.

**Branches.** By default the first condition draws the resting states and solves for the
turnover rates (the forward seed), which names the component the drawn values parametrize.
Of several branches the first nonzero one over :math:`\mathbb{F}_p` is taken and named in
``info["branches"]``: by the factor over :math:`\mathbb{Q}(\theta)` it solves where a balance
splits, else as conjugate where the lex basis of the balances off the zero states is in
shape position at a point with an irreducible eliminant. Other points take their root on
that branch or, without one, are left out. Conjugate roots have equal rank, and an eliminant irreducible at
a point is irreducible over :math:`\mathbb{Q}(\theta)`.

**Zero at rest.** A state rests at zero when the right-hand side of the state, inputs at
zero, is linear in it with a coefficient nonzero while the other such states are zero. In a
network, a species consumed but not produced rests at zero as the single reactant of a
consuming reaction, as does the single positive-weight reactant of a reaction lowering a
weight no reaction raises (a sink). Of several such reactants, as in :math:`A + B \to 0`, the first
is set to zero (``info["branches"]``). A state with right-hand side zero at rest, and the
pivot of a held moiety, has a free resting value, the initial value. Rates are read at
generic values, so a catalyst at rest is nonzero.

**Diagonal start.** Conditions sharing :math:`\theta` have coupled resting states. Rank and
kernel are generic on the joint variety
:math:`W = \{(\theta, x_1, \dots, x_C) : f_c(x_c, \theta) = 0\}`, so any point of :math:`W`
over :math:`\mathbb{F}_p((\varepsilon))` off an exceptional set gives them. symident seeks free
parameters :math:`Z` whose zero makes all conditions equal the first, and the largest set
:math:`S` of states whose zero is then invariant; at :math:`\theta_Z = 0, x_S = 0` the
forward seed is a resting point :math:`P_0` of all conditions. Along
:math:`\theta(\varepsilon) = \theta_0 + \varepsilon\, \delta`, :math:`\delta` the values of
:math:`Z` at the sample point, Newton steps lift each condition:

.. math::

   x_{c,k} = -J_c(P_0)^{-1}\, [\varepsilon^k]\, f_c\big(x_{c,<k}, \theta(\varepsilon)\big),
   \qquad J_c = \partial_x f_c .

With :math:`J_c(P_0)` invertible the series is a germ of the unique component of :math:`W`
through :math:`P_0`; a trial at a random point checks this and that the states of :math:`S`
leave zero. The truncated series rank bounds the generic rank from below
(``info["series_certified"]``: series order proven sufficient). The kernel at
:math:`\varepsilon = 1` comes from Padé forms of the kernel entries, each accepted only when
the coefficients beyond those determining it agree. ``SYMIDENT_NODIAG`` turns this off.

Non-rational terms
------------------

Elementary functions are recast into rational form [Savageau1987]_, type by type in a fixed
order and innermost first within a type; a nesting across that order, such as
:math:`\sin(\exp x)` or :math:`\exp(\arctan x)`, is refused. A term :math:`w = \varphi(v)`
with states in :math:`v` becomes a state with right-hand side
:math:`\varphi'(v)\, \sum_j \partial_{x_j} v \, f_j`, rational in :math:`w` and :math:`v` for

.. math::

   \varphi(v) = v^c:\ \ \frac{c\, w}{v}, \qquad
   \exp(v):\ \ w, \qquad \log(v):\ \ \frac{1}{v}, \qquad
   \arctan(v):\ \ \frac{1}{1 + v^2}, \qquad \operatorname{artanh}(v):\ \ \frac{1}{1 - v^2},

and for :math:`\arcsin`, :math:`\arccos`, :math:`\operatorname{arsinh}` and
:math:`\operatorname{arcosh}` contains a square root, recast in turn. Trigonometric functions
go through :math:`\tan(v/2)`. A free exponent :math:`b^n` becomes :math:`E = b^n` with
companion :math:`L = \log b`, so the field is rational and :math:`n` enters through
:math:`\partial E / \partial n = E L`.

Under ``equilibrate`` the resting model has no exponential, so a power :math:`b^n` of a
positive rational :math:`b` is rewritten first: next to a power :math:`K^n` of the same
exponent, :math:`K` stands for :math:`bK` throughout, :math:`(bK)^n` being a power of one
symbol, and the result is mapped back. Otherwise :math:`b` is a constant with :math:`\log b`
taken independent of the other constants, as for logarithms of numbers in log coordinates
(named in ``info["limits"]``).

A new state starts at :math:`\varphi(v)` of the initial state. Where this depends on the
coordinates, a transcendental :math:`\varphi` becomes a generic leaf, drawn off the graph and
tied to the argument by relation rows :math:`\mathrm{d}w = \varphi'(v)\, \mathrm{d}v`: a
transcendental graph lies in no proper algebraic set, so a random point on it is generic.
With several transcendental constants, such as :math:`e` and :math:`\pi`, this needs their
algebraic independence, which follows from Schanuel's conjecture and is not proven. An
algebraic :math:`\varphi`, a root or rational power, is exact: through :math:`L = \log v`
when :math:`v` is affine in one coordinate, else by the root modulo the prime at each sample
point, over primes in which every root of the model exists. Logarithms, fractional powers
and bases of symbolic exponents need an argument positive on :math:`\Theta`, also in initial
and event values; otherwise an error names the symbols not declared positive. An observable
:math:`\kappa \sum_i a_i \log h_i + c`, strictly monotone in
:math:`\prod_i h_i^{a_i} e^{c/\kappa}`, is analyzed as that product.

Closed forms and levels
-----------------------

The directions are reconstructed in closed form. *Scalings*,
:math:`\Phi_s(\theta)_i = e^{w_i s}\, \theta_i` with integer weights :math:`w` and
:math:`\xi_i = w_i \theta_i`, follow exactly from an integer lattice of monomial weights and
are taken out first. The other kernel vectors, *general directions*, are normalized; each
entry is a rational function whose variables perturbation finds, sampled at many points and
primes, fitted by dense interpolation, sparse interpolation after [BenOr1988]_ or univariate
rational interpolation along lines, and lifted to rational coefficients by the Chinese
remainder theorem and rational reconstruction [Wang1982]_. Every closed form, and a
direction along one coordinate that fills the kernel, is checked before reporting at seeded
points uniform modulo the verification prime; under ``equilibrate`` on the coordinates the
closed form states, from which the resting-state copies per condition follow.

A direction no sampling route closes is fitted along lines :math:`\theta^0 + \lambda\,\delta`
through the base point: the kernel entries become power series in :math:`\lambda` (under
``equilibrate`` with the resting state continued along the line), and Padé approximants on a
star of lines are fitted as rational functions. An entry not rational in the parameters
alone is fitted in parameters and resting states modulo their equations, a state of condition
:math:`k` named ``x_c<k>``. Each entry is checked at a prime outside those of the lift, the
direction against the kernel at two points uniform modulo the verification prime.
``reconst_control(homotopy=False)`` turns the route off.

Levels of a closed form:

- ``exact``: proven over Q, as a scaling of the integer kernel or as a field on parameters
  and states whose identities per state equation, observable, initial value and event cancel
  symbolically; the state components are the closed form read on the states, the
  initial-value identities solved for the parameters setting the initial values, or a
  combination of the symmetries of the model equations.
- ``verified``: in the kernel at two seeded points modulo a verification prime :math:`q`.
  With a degree bound :math:`D` of the cleared kernel identity, a wrong closed form passes
  with probability at most :math:`(D/q)^2` [Schwartz1980]_. Points without kernel are
  skipped; with :math:`k` the most points a check drew for the passing two
  (``info["bound"]["draws"]``), the pair is one of :math:`\binom{k}{2}`, and
  ``info["bound"]`` holds :math:`n \binom{k}{2} (D/q)^2` for :math:`n` verified directions and
  the largest :math:`D`, plus ``info["order_bound"]``. The bound needs a rational model without
  resting states or root events, event times rational in the parameters, and Lie and gap
  orders proven sufficient, as the check runs at the orders used. It excludes the chance that
  :math:`q`, drawn per detection, divides the content of an integer identity, which then
  vanishes modulo :math:`q`.
- ``checked``: the same test where no bound applies (``info["bound"]`` says why), or a test
  on the entries that can be evaluated at the points.
- ``open``: no closed form.

The rank (``info["rank_status"]``) is

- *proven* when full, when it plus the number of independent exact directions is :math:`d`,
  or when every direction is proven over Q;
- *verified*, with a probability, when Lie, gap and series orders are proven sufficient and
  every direction is verified with a bound; when every direction lies in the span of
  invariant fields of the model, which hold the kernel at every order, checked at random
  points with a bound (``info["rank_fields_bound"]``); or when a rank modulo unknown inputs
  holds at a random point;
- a *lower bound* otherwise.

Unproven directions
-------------------

``info["proof"]`` lists a direction no proof closes with the ``support`` and the ``reason``
of each failed route:

- ``lift of the closed form: [Y, F] on x does not vanish``: the field with state components
  read from the closed form is no symmetry, and no other lift was found.
- ``output is not a polynomial in t``: the output identities decide only outputs polynomial
  in time.
- ``no combination of the regime fields moves the parameters alike``, ``no combined lift``
  (with the cause, such as more than 8 regimes or work above
  ``reconst_control(exact_work_cap=)``): the regimes need state components agreeing on the
  parameters.
- ``events after the start``, ``an event at the start``, ``an input switches on at the
  start``: the proof at the resting states covers a start at rest, inputs at rest, that no
  event follows or changes. With events, states that no event reaches and that rest alone,
  such as a subsystem driving the states the events set, become constant coordinates on
  their resting variety, parametrized by a linearly entering parameter; the field holds the
  component of each resting state keeping the variety tangent, and the following states get
  components :math:`\xi = G z + a` solved from the identities of regimes, outputs and events.
  This covers one resting condition, with reason ``resting subsystem as coordinates`` and the
  failing identity.
- ``an inverted power recast``, ``the resting states have no rational parametrization``,
  ``more than 8 resting states per condition``: a balance solved for a recast power, without
  rational parametrization by the parameters, or with more than the 8 states for which the
  resting Jacobian is inverted symbolically stays unproven, the last also in
  ``info["limits"]``.
- ``no exact proof with these nuisance coordinates``: the unknown-input coordinates have no
  lift.

Under ``equilibrate`` no error bound covers the resting states, which are solved at random
points modulo a prime, so a direction the proof leaves open is ``checked`` (reason in
``info["bound"]``).

Discrete symmetries
-------------------

Local identifiability leaves finitely many parameter values with the prediction of a generic
one; more than one is a discrete symmetry, which no generator shows.
:meth:`~symident.Symmetries.solution_count` equates the Taylor coefficients of the
observables up to an order with their values at a random point modulo a prime and counts the
solutions with msolve [Berthomieu2021]_, over the algebraic closure, away from zeros of
coordinates and denominators, as in the global identifiability test of [Hong2020]_. The count
holds at that point and is no proof.

Reduction and sections
----------------------

A reduction replaces the parameters by combinations the output determines [Chappell1998]_
[Evans2000]_ [Massonis2023]_, the invariants :math:`I_l(\theta)`, :math:`X(I_l) = 0`
[Olver1993]_. symident searches, within degree bounds, monomial, polynomial and rational
invariants (denominator in one coordinate), Darboux polynomials and exponential factors
[Prelle1983]_ [Goriely2001]_, and invariants by separable quadrature and by an integrating
factor. The invariants become new parameters, :math:`I_l(\theta) = q_l`, and the remaining
coordinates are expressed through them.

A scaling is removed by fixing one coordinate of the scaling at one, such as an initial
value or a scale. Every orbit is a ray meeting that value once, so the result is a *global
section*: a one-to-one map of the new parameters onto a *global cross-section*, a set meeting
every orbit exactly once, with the invariants as global coordinates on :math:`\Theta / G`.
A direction along one coordinate is pinned like a scaling; every orbit meets the pin where
the sign certificate proves the generator component on the pinned coordinate of one sign on
the domain, else the orbits can be the intervals between the zeros of that component and the
pin is listed under ``unproven``.

For a block of :math:`r` general generators the invariants are solved for coordinates
:math:`c = C(q, g)`, with :math:`r` gauge coordinates :math:`g` left. Every level set
:math:`L_q = \{\theta \in \Theta : I(\theta) = q\}` is one orbit when

1. :math:`C(q, g)` is the unique solution for every gauge value, so :math:`L_q` is the
   graph of :math:`C(q, \cdot)` over :math:`G_q = \{g : C(q, g) \in \Theta\}`;
2. the :math:`r \times r` minor of the generator components on :math:`g` has one sign on
   :math:`\Theta`, so every orbit is :math:`r`-dimensional and open in :math:`L_q` (for
   generators with poles, the minor of the fields times their denominators);
3. :math:`G_q` is connected.

:math:`G_q` is given by one condition :math:`p > 0` per positive solved coordinate and
square-root factor. It is connected when every condition is convex in :math:`g` (affine, a
concave quadratic, or, in one positive gauge coordinate, a polynomial with at most one sign
change and so, by Descartes' rule of signs, at most one positive root), or when a gauge
coordinate enters every condition affinely with a coefficient of known sign, so that the
fibers are intervals and the set is connected with the projection, decided the same way.
Square roots in :math:`g`, or of a single positive invariant, are first removed by a square
substitution, one to one on the positive half-line. The section point :math:`T(q)` then
lies in :math:`L_q` and, by the positivity certificate, in :math:`\Theta`.

**Coverage.** ``reach`` says every level set of the invariants is one orbit; with total
coverage every orbit then meets the section, under partial coverage exactly the orbits whose
invariants lie where the solved entries are positive. Under declared bounds an entry without
positivity certificate is kept, with partial coverage, and listed under ``unproven``.
Without bounds it is kept only when the search found no section certified on the whole
domain, and only where a sampled orbit has every entry positive: the trafo then covers the
orbits on which the ``sign_conditions`` hold, the sampled one among them, and no others. A
positivity certificate treats an atom such as :math:`\log x` as positive only where the sign
of the atom is proven; an AM-GM certificate is strict only by the positive budget the pairs
of the certificate leave, on a term free of real coordinates where there are any.

**Face sections.** A section on a face, :math:`Z = 0` for coordinates :math:`Z` of the block,
lies outside :math:`\Theta` and is reached only in a limit. It is tried first, and once found
no other is searched; the new parameters are the face-point coordinates, not invariants. With
:math:`Z` as gauge the conditions above make every level set one orbit, and
:math:`C(q, Z) \to C(q, 0) > 0` puts :math:`F(q) = (0, C(q, 0))` in the closure of every orbit
of :math:`L_q`. The output, constant along the orbit, equals that at :math:`F(q)` when the
model is continuous there: no model expression has a pole at the face, every denominator in
the block parameters has one sign there, and under ``equilibrate`` :math:`x^\ast(\theta)` has
a limit. For that limit each generator is lifted to a field
:math:`\tilde X = X + \xi_x \partial_x` tangent to :math:`f = 0`, by the state components of an
exact direction or by the implicit derivative :math:`\xi_x = -J^{-1} f_\theta \xi_\theta`;
exactly verified invariants of the lifted fields that solve for the moved states and stay
positive at :math:`Z = 0` give the limit along every orbit. The implicit derivative needs
:math:`J` regular: proven when :math:`\det J` has one sign on :math:`\Theta`, else checked at
random points. Without a lift, the resting state is continuous when :math:`f` is affine in
the states with a resting Jacobian of one sign at the face, else checked at random face
points; a checked resting state sets ``face_level`` and ``level`` of the block to
``checked``.

**Injectivity.** That every orbit meets the section :math:`S` at most once (``injective``) is
proven, for any number of generators, when the solved coordinates recover every point of :math:`S` from the
invariants, :math:`T(I(\theta)) = \theta` on :math:`S`, since two points of one orbit have
equal invariants. The identity is shown over Q on a parametrization of the positive points of
:math:`S`, monomial for a section of balances, else by solving each equation for a
coordinate of degree one; the claim is made for sections with total coverage off a face.
For one generator it also holds when :math:`X` changes the pinned coordinate, or the
log-ratio of the two sides of the section, with one sign on :math:`\Theta`.

An unproven claim is listed under ``unproven`` of the block and makes the block ``open``; a
block is ``verified`` or ``checked`` when a direction or a face resting state has that level,
else ``exact``. The transformation is checked over Q, and ``check=True`` runs the detection
again on the reduced model (``Reduction.checks["loop"]``).

Assumptions and error bounds
----------------------------

Besides proofs, a result rests on two assumptions (``info["assumptions"]``,
``Reduction.assumptions``), both part of the declared model: the domain :math:`\Theta`
(``positive``) and, with switches, the start sides ``regimes`` declares where the initial
values do not decide them. A verified closed form or rank is wrong with probability at most
``error_bound`` [Schwartz1980]_, including a wrong certified order. Floating point enters the
choice of a gauge and of the ``reduce_cq`` pivots, never a rank, a sign or a closed form.
Every other step is proven or named in the result: a ``checked`` direction, a
``lower bound`` rank, a section claim under ``unproven``, a limit under ``info["limits"]``, a
resting branch under ``info["branches"]``.
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
