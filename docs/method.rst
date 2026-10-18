Method
======

The model
---------

symident analyzes ODE models

.. math::

   \dot x = f(x, u, \eta), \qquad x(t_0) = x_0(\eta), \qquad y = h(x, \eta),

with states :math:`x \in \mathbb{R}^n`, parameters :math:`\eta`, an input
:math:`u = u(t, \eta)` and an observation function :math:`h`. The parameters may be given by
a *parametrization* :math:`\theta \mapsto \eta(\theta)`, with :math:`\theta` in an open
*parameter space* :math:`\Theta \subseteq \mathbb{R}^d`; :math:`\theta` holds the estimated
parameters and the initial values that are not fixed. The *prediction* is
:math:`g(t; \theta) = h\big(x(t), \eta(\theta)\big)`. The right-hand sides, initial values and
observables are rational functions, or become rational after the recasts described below.

Structural identifiability
--------------------------

The model is *locally structurally identifiable* at :math:`\theta` if no
:math:`\tilde\theta \ne \theta` near :math:`\theta` has the same prediction,
:math:`g(\cdot; \tilde\theta) = g(\cdot; \theta)`. The property is *structural* because it
is decided by the model equations alone, as if the observables were known exactly and
continuously in time, and it is stated for generic :math:`\theta`, that is for all values
outside a set of measure zero.

A continuous family of parameter values with the same prediction is the orbit of a *Lie
symmetry*, a local one-parameter group of transformations :math:`\Phi_s` of
:math:`\Theta` with :math:`g\big(t; \Phi_s(\theta)\big) = g(t; \theta)` [Yates2009]_. Its
*generator* is the vector field

.. math::

   X = \sum_{i=1}^d \xi_i(\theta)\, \partial_{\theta_i}, \qquad
   \xi(\theta) = \partial_s \Phi_s(\theta)\big|_{s = 0},

and the integral curve of :math:`X` through :math:`\theta` is the *orbit*
:math:`\mathcal{O}(\theta)`. The coordinates with :math:`\xi_i \ne 0` are the *support* of
the direction. A function :math:`I` with :math:`X(I) = 0` is an *invariant*: it is
constant on orbits, and the prediction depends on :math:`\theta` only through the
invariants. The number of identifiable combinations is :math:`d` minus the number of
independent generators.

Lie derivatives
---------------

The parameters and the time are added to the states, :math:`z = (x, \eta, t)` with
:math:`\dot \eta = 0` and :math:`\dot t = 1`, so that the model is the vector field
:math:`F = f\, \partial_x + \partial_t` with initial value :math:`z_0(\theta)`. Applied to
a function :math:`\varphi(z)`, it gives the rate of change along a solution,
:math:`F(\varphi) = \frac{\mathrm{d}}{\mathrm{d}t}\, \varphi\big(z(t)\big)`. The *Lie
derivatives* of the output are

.. math::

   L_F^0 h = h, \qquad L_F^{k+1} h = F\big(L_F^k h\big).

For analytic models they fix the prediction at the initial time,

.. math::

   g(t; \theta) = \sum_{k \ge 0} \frac{(t - t_0)^k}{k!}\, L_F^k h\big(z_0(\theta)\big),

so two parameter values with the same Lie derivatives at :math:`t_0` have the same
prediction [Pohjanpalo1978]_. Collected up to order :math:`N`,

.. math::

   \Psi_N(\theta) = \big(h, L_F^1 h, \dots, L_F^N h\big)\big(z_0(\theta)\big), \qquad
   \mathrm{OIM}_N(\theta) = \frac{\partial \Psi_N}{\partial \theta},

where :math:`\mathrm{OIM}_N` is the *observability-identifiability matrix* of order
:math:`N`. A generator leaves every component of :math:`\Psi_N` unchanged, so its tangent
vector lies in :math:`\ker \mathrm{OIM}_N(\theta)`. At a generic :math:`\theta` the kernel
consists exactly of these tangent vectors, and the model is locally structurally
identifiable when :math:`\mathrm{OIM}_N` has full rank :math:`d` for some order :math:`N`.

Rank over finite fields
-----------------------

symident evaluates :math:`\mathrm{OIM}_N` in exact arithmetic over the finite field
:math:`\mathbb{F}_p`, :math:`p < 2^{31}`. The Lie derivatives are computed as truncated
power series in time and the derivatives with respect to :math:`\theta` as dual numbers,
so one pass yields all rows of :math:`\mathrm{OIM}_N` at a random point, following the
probabilistic approach of [Sedoglavic2002]_. A random point can only underestimate the
generic rank, and it falls short with small probability [Schwartz1980]_ [Zippel1979]_;
the rank is confirmed at further primes and points.

The rank grows with the order :math:`N` until it stops. With the parameters fixed, the
states at the start of a segment lie on a smallest invariant affine space, on which the
rows have a generic rank that no order exceeds [HermannKrener1977]_. Where the rows at the
start reach this rank, the order is proven *sufficient*: the reported rank is the generic
one and not a false plateau. Otherwise the order stops after a plateau of constant rank,
and the result says that the Lie order is not proven sufficient.

``engine="symbolic"`` computes the Lie derivatives over the rational functions. Its order
is found at a point and confirmed by an exact rank; where the rows at the initial values
reach the generic rank [HermannKrener1977]_, a nonvanishing minor makes every later row a
combination of the earlier ones, so the kernel is final and a direction that annihilates
the rows identically is proven.

Conditions and events
---------------------

Several experimental conditions share parameters and differ in fixed values or
renamings. Their rows are stacked into one matrix over the shared coordinates, with
:math:`\ker \mathrm{OIM} = \bigcap_c \ker \mathrm{OIM}^{(c)}`, so a direction has to
leave the prediction of every condition unchanged.

An *event* changes a state or an input instantaneously at a known or estimated time and
splits a condition into segments. The state at the start of a segment is the state of the
previous segment propagated over the gap :math:`\Delta t` between the events. Every gap is
put on the line through the actual gaps, :math:`\Delta t_i = \varepsilon\, \delta_i`, and
the rank is taken over :math:`\mathbb{F}_p((\varepsilon))`, the field of formal Laurent
series in :math:`\varepsilon`, which separates directions that hold only for particular
event timings from generic ones.

The rows after a gap are power series in :math:`\varepsilon` of a fixed family of
functions, so the argument above bounds their rank by the rank of all coefficients. The gap
order :math:`M` is raised until the truncated rank reaches this bound, which proves it
sufficient. Without a bound (under ``equilibrate`` or with recast terms) or at
``reconst_control(gap_order_cap=)``, the rank is a lower bound and ``info["limits"]`` says
where :math:`M` stopped.

Switches
--------

Each condition of a switch, a comparison or the argument of ``Heaviside()``, ``sign()``,
``abs()``, ``max()`` or ``min()``, is a sign condition :math:`\gamma_j(x, \eta, t) > 0`.
Conditions the declared signs decide are resolved; an odd factor of an undecided
condition is a *switching function* :math:`\gamma_j`, and
:math:`\Gamma_j = \{\gamma_j = 0\}` its surface. A sign vector
:math:`\sigma \in \{+1, -1\}^J` fixes every condition and gives a smooth regime
:math:`(F_\sigma, h_\sigma)`. An event with a ``root`` adds its root as a switching
function and resets its state on :math:`\Gamma_j`.

**Regimes.** Without ``regimes``, every sign vector counts as a regime unless the
declared signs prove it empty: when :math:`\gamma_a - \gamma_b` or
:math:`\gamma_a + \gamma_b` has a sign that the declared positivity decides, the sign
vectors that contradict it are dropped. No regime is dropped because of parameter values or
because a trajectory does not reach it, so the result holds for every initial state.

**Visibility.** Two regimes :math:`\sigma, \sigma'` of a condition are neighbors at
:math:`\Gamma_j` when they differ only in the sign of :math:`\gamma_j`. The switch on
:math:`\Gamma_j` is *visible* when, for some pair of neighbors, some observable
:math:`y` and some order :math:`k`,

.. math::

   \big(L_{F_\sigma}^k h_{\sigma,y} - L_{F_{\sigma'}}^k h_{\sigma',y}\big)\big|_{\Gamma_j}
   \ne 0 .

A nonzero Taylor coefficient of the difference at a point of :math:`\Gamma_j`, modulo a
prime, proves visibility. The switch is *invisible* when, for every pair, the observables
agree and every state that reaches them has the same right-hand side in both regimes, so
that every Lie derivative agrees. Otherwise it is *undecided*. The surface of a root event
is always treated as visible.

**The rule.** A direction :math:`v` of the coordinates is reported non-identifiable
when, in every condition, it lies at the start in a distribution
:math:`\Delta = \operatorname{span}\{Y_1, \dots, Y_r\}` of vector fields on parameters
and states with

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
:math:`v = \big(\xi, X(x_0)\big)`: its state part is the derivative of the initial values
along :math:`X`, under ``equilibrate`` of the resting state of each regime. Along the flow
of every regime :math:`v` stays in :math:`\Delta`, so the observables keep their value to
first order. At a visible surface it is tangent, so the crossing time does not change; at a
hidden one the crossing may move, but the jump lies in :math:`\Delta` and the observables
agree there. The rule therefore never reports a direction along which the observables
change, for any sequence of the listed regimes.

**Lower bound.** Every direction of the rule satisfies linear conditions at a point,
and their rank bounds the rank from below. They are the observability rows of copies of
the model that share the parameters:

- each regime from the start of each condition, with its events;
- each regime from a generic point of each visible surface that is linear in a state,
  where a direction of the rule is tangent to the surface;
- each regime from a generic point, when a switch of the condition is not visible;
- each copy switched at a fixed time into every other regime of the condition, whose
  outputs contain the Lie derivatives along words of two regimes,
  :math:`L_{F_\sigma}^a L_{F_{\sigma'}}^b h_{\sigma'}`;
- the switching time :math:`\tau_j(\theta)` of every visible switch in time alone.

The kernel of the stacked rows, projected onto the coordinates, contains every direction
of the rule; its dimension :math:`k` gives the lower bound :math:`d - k`. A minor that is
nonzero at the sample point is nonzero as a rational function, so the bound needs no
probabilistic argument.

**Upper bound.** The candidates are the scalings of the integer kernel, the closed forms of
the stacked kernel and the unit field of every coordinate. A candidate that fails a
condition of the rule is dropped until the family passes. Every identity is first tested at
a rational point and then proven by symbolic simplification to zero; :math:`\ell`
remaining candidates independent at a rational point bound the rank from above by
:math:`d - \ell`.

When :math:`\ell = k`, the rank is :math:`d - k` and the candidates span the
non-identifiable directions. Otherwise :math:`d - k` is a lower bound and
``info["switching"]["uncertified"]`` names the rejected candidates. The directions of the
stacked kernel that the rule does not prove satisfy only the necessary conditions of the
stacked rows, so they are listed under ``info["switching"]["candidates"]``, not as
symmetries. For an undecided switch the rule accepts either the tangency or the hidden jump
and leaves out the rows of its surface, so both bounds stay valid.

**Listed regimes.** With ``regimes``, the rule, the rows and the candidates use the listed
regimes of each condition, and a surface counts only when listed regimes lie on both of its
sides. Parameters of the other regimes stay coordinates, so a parameter that acts only
outside the listed regimes is reported non-identifiable. A sign of :math:`\gamma_j(x_0)`
that the declared signs decide must agree with the first entry, else the analysis raises an
error; an undecided sign is declared by ``regimes`` and listed under ``declared`` in
``info["switching"]["conditions"]`` and under ``info["assumptions"]["starts"]``.

**Limits.** Each is an error, never a result: a surface in the parameters alone whose
side ``regimes`` does not give; a condition on a switched expression; more than 4096 sign
vectors; ``gauge``, ``reduce_cq``, ``free_initial`` and ``engine="symbolic"``; a state
named ``time``; under ``equilibrate``, a resting state without a closed form; a listed
regime that is not feasible; and a start that contradicts the initial values.

Steady states
-------------

With ``equilibrate=True`` every condition starts at its resting state
:math:`f(x^\ast, \eta) = 0`, which binds :math:`x_0 = x^\ast` to :math:`\theta`. No closed
form is needed: the resting state is solved over :math:`\mathbb{F}_p` at each sample
point, by linear elimination, by resultants for small coupled cores, or by msolve
[Berthomieu2021]_. The kernel is taken in the joint coordinates of parameters and resting
states, with the tangency condition
:math:`\partial_x f\, \xi_x + \partial_\theta f\, \xi_\theta = 0` as additional rows. By
default the first condition draws its resting states and solves for the turnover rates
instead (the forward seed). Where several resting branches exist, the first nonzero branch
over :math:`\mathbb{F}_p` is taken, and ``info["branches"]`` names it.

**Diagonal start.** Conditions that share :math:`\theta` have coupled resting states. Rank
and kernel are generic properties of the joint variety
:math:`W = \{(\theta, x_1, \dots, x_C) : f_c(x_c, \theta) = 0\}`, so any point of
:math:`W` over :math:`\mathbb{F}_p((\varepsilon))` outside an exceptional set gives them.
symident looks for free parameters :math:`Z` whose zero makes every condition coincide with
the first, and the largest set :math:`S` of states whose zero is then invariant. At
:math:`\theta_Z = 0, x_S = 0` the forward seed is a resting point :math:`P_0` of every
condition at once. Along :math:`\theta(\varepsilon) = \theta_0 + \varepsilon\, \delta`,
with :math:`\delta` the values of :math:`Z` at the sample point, each condition is lifted by
Newton steps

.. math::

   x_{c,k} = -J_c(P_0)^{-1}\, [\varepsilon^k]\, f_c\big(x_{c,<k}, \theta(\varepsilon)\big),
   \qquad J_c = \partial_x f_c .

If :math:`J_c(P_0)` is invertible, the series is a germ of the unique component of
:math:`W` through :math:`P_0`; a trial at a random point checks this and that the states of
:math:`S` leave zero. The truncated series rank bounds the generic rank from below; whether
the series order is proven sufficient is ``info["series_certified"]``. The kernel at the
sample point itself, :math:`\varepsilon = 1`, is read from Padé forms of its entries, each
accepted only when the coefficients beyond those that determine it agree with it.
``SYMIDENT_NODIAG`` switches the diagonal start off.

Non-rational terms
------------------

Elementary functions are recast into rational form, innermost first. A term
:math:`w = \varphi(v)` whose argument :math:`v` contains states becomes a new state; its
right-hand side follows from the chain rule, :math:`\varphi'(v)\, \sum_j \partial_{x_j} v
\, f_j`, which is rational in :math:`w` and :math:`v` for

.. math::

   \varphi(v) = v^c:\ \ \frac{c\, w}{v}, \qquad
   \exp(v):\ \ w, \qquad \log(v):\ \ \frac{1}{v}, \qquad
   \arctan(v):\ \ \frac{1}{1 + v^2}, \qquad \operatorname{artanh}(v):\ \ \frac{1}{1 - v^2},

and for :math:`\arcsin`, :math:`\arccos`, :math:`\operatorname{arsinh}` and
:math:`\operatorname{arcosh}` contains a square root, which is recast in turn. The
trigonometric functions are expressed through the half-angle tangent :math:`\tan(v/2)`. A
free exponent :math:`b^n` becomes :math:`E = b^n` with the companion :math:`L = \log b`,
after which the field is rational and :math:`n` enters through
:math:`\partial E / \partial n = E L`.

The initial value of a new state is :math:`\varphi(v)` at the initial state. Where it
depends on the coordinates, a transcendental :math:`\varphi` becomes a generic leaf tied to
its argument by :math:`\mathrm{d}w = \varphi'(v)\, \mathrm{d}v`: a transcendental graph
lies in no proper algebraic set, so a random point on it is generic. An algebraic
:math:`\varphi`, a root or rational power, has no such point and is handled exactly:
through :math:`L = \log v` when :math:`v` is affine in one coordinate, otherwise by taking
the root modulo the prime at each sample point, over primes in which every root of the
model exists. A logarithm or fractional power needs an argument positive on
:math:`\Theta`. An observable :math:`\kappa \sum_i a_i \log h_i + c` is a strictly
monotone function of :math:`\prod_i h_i^{a_i} e^{c/\kappa}` and is analyzed as that
product.

Closed forms
------------

The rank alone gives the number of non-identifiable directions; the directions
themselves are reconstructed in closed form.

*Scalings*, :math:`\Phi_s(\theta)_i = e^{w_i s}\, \theta_i` with integer weights
:math:`w`, have the generator components :math:`\xi_i = w_i \theta_i`. They follow
exactly from an integer lattice of monomial weights and are taken out first. Every other
direction is *general*.

*General directions* are the remaining kernel vectors. They are normalized, and each
entry is reconstructed as a rational function of the coordinates: the coordinates an
entry depends on are found by perturbation, the entry is sampled at many points and
primes, fitted by dense interpolation, by sparse interpolation in the style of
[BenOr1988]_, or by univariate rational interpolation along lines, and lifted to
rational coefficients by the Chinese remainder theorem and rational reconstruction
[Wang1982]_. Every closed form is checked at fresh seeded points before it is reported.

A direction that no sampling route closes is fitted along lines
:math:`\theta^0 + \lambda\,\delta` through the base point. Its kernel entries become power
series in :math:`\lambda`, under ``equilibrate`` with the resting state continued along the
line; their Padé approximants give the entries on a star of lines, from which each entry is
fitted as a rational function. An entry that is not rational in the parameters alone is
fitted in the parameters and the resting states modulo their equations, with a state of
condition :math:`k` named ``x_c<k>``. Each entry is checked at a prime outside its lift, and
the direction against the kernel at a fresh point. ``reconst_control(homotopy=False)``
turns the route off.

Every closed form has a *level*:

- ``exact``: proven over Q, as a scaling from the integer kernel or as a vector field on
  parameters and states whose identities per state equation, observable, initial value and
  event cancel symbolically.
- ``verified``: in the kernel at two seeded points modulo a verification prime :math:`q`.
  With a degree bound :math:`D` of the cleared kernel identity, a wrong closed form passes
  with probability at most :math:`(D/q)^2` [Schwartz1980]_; ``info["bound"]`` holds the sum
  over the verified directions. The bound needs a rational model without resting states or
  root events, with numeric event times, and Lie and gap orders proven sufficient, since
  the check runs at the orders used.
- ``checked``: the same test where no bound applies, or a test on the entries that can be
  evaluated at the points; ``info["bound"]`` says why a bound does not apply.
- ``open``: no closed form.

The rank is *proven* when it is full, when it plus the number of independent exact
directions equals :math:`d`, or when every direction is proven over Q. It is *verified*,
always with a probability, when the Lie, gap and series orders are proven sufficient and
every direction is verified with a bound, or when every direction lies in the span of
invariant vector fields of the model, checked at random points with a bound
(``info["rank_fields_bound"]``). Otherwise it is a *lower bound* (``info["rank_status"]``).

Reduction
---------

A reduction replaces the parameters by combinations that the output determines, the
invariants :math:`I_l(\theta)` with :math:`X(I_l) = 0` [Olver1993]_. symident searches
monomial, polynomial and rational invariants, Darboux polynomials and exponential factors
[Goriely2001]_ within degree bounds.

The invariants become new parameters, :math:`I_l(\theta) = q_l`, and the remaining
coordinates are expressed through them. A scaling is removed by fixing one of its
coordinates at one, for example an initial value or a scale. Every orbit of a scaling is a
ray and meets that value once, so the result is a *global section*: it maps the new
parameters one-to-one onto a *global cross-section*, a set that meets every orbit exactly
once, and the invariants are global coordinates on the orbit space :math:`\Theta / G`.

For a block of :math:`r` general generators, the invariants are solved for coordinates
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
change, hence at most one positive root by Descartes' rule of signs), or when a gauge
coordinate enters every condition affinely with a coefficient of known sign: the fibers are
then intervals, and the set is connected when its projection is, which is decided the same
way. Square roots in :math:`g`, or of a single positive invariant, are first removed by a
square substitution, which is one to one on the positive half-line. The section point
:math:`T(q)` then lies in :math:`L_q` and, by the positivity certificate, in
:math:`\Theta`, so every orbit meets the section (``reach``). Under partial coverage the
level sets are still single orbits, and an orbit meets the section exactly when its
invariants lie where the solved entries are positive.

A section on a face, :math:`Z = 0` for coordinates :math:`Z` of the block, lies outside
:math:`\Theta` and is reached only in a limit. With :math:`Z` as the gauge, the conditions
above make every level set one orbit, and :math:`C(q, Z) \to C(q, 0) > 0` puts the face
point :math:`F(q) = (0, C(q, 0))` in the closure of every orbit of :math:`L_q`. The output
is constant along the orbit, so :math:`F(q)` gives it when the model is continuous there:
no model expression has a pole at the face, every denominator in the parameters of the
block has one sign there, and under ``equilibrate`` the resting state :math:`x^\ast(\theta)`
has a limit. For that limit each generator is lifted to a field
:math:`\tilde X = X + \xi_x \partial_x` tangent to :math:`f = 0`, by the state components
of an exact direction or by the implicit derivative
:math:`\xi_x = -J^{-1} f_\theta \xi_\theta`; invariants of the lifted fields, verified
exactly, that solve for the moved states and stay positive at :math:`Z = 0` give the limit
along every orbit. The implicit derivative needs :math:`J` regular: proven when
:math:`\det J` has one sign on :math:`\Theta`, else checked at random points. Without a
lift, the resting state is continuous when :math:`f` is affine in the states with a
resting Jacobian of one sign at the face, else checked at random points of the face. A
checked resting state sets the block's ``face_level`` and ``level`` to ``checked``.

That every orbit meets the section :math:`S` at most once (``injective``) is proven, for any
number of generators, when the solved coordinates give back every point of :math:`S` from
its invariants, :math:`T(I(\theta)) = \theta` for :math:`\theta \in S`: two points of one
orbit have the same invariants and therefore coincide. The identity is shown over Q on a
parametrization of the positive points of :math:`S`, monomial for a section of balances,
else by solving each equation for a coordinate of degree one. The claim is made for
sections with total coverage that do not lie on a face. For a block with one generator it
also holds when :math:`X` changes the pinned coordinate, or the log-ratio of the two sides
of the section, with one sign on :math:`\Theta`.

A claim that is not proven is listed under the block's ``unproven`` and makes the block
``open``; a block is ``verified`` or ``checked`` when a direction or a face resting state
has that level, and ``exact`` otherwise. The transformation is checked over Q, and with
``check=True`` the detection runs again on the reduced model (``Reduction.checks["loop"]``).

What a result rests on
----------------------

Besides proofs, a result rests on two assumptions, which ``info["assumptions"]`` and
``Reduction.assumptions`` list: the declared domain :math:`\Theta` (``positive``) and, for a
model with switches, the sides that ``regimes`` declares at the start where the initial
values do not decide them. Both are part of the model as declared. A verified closed form
or rank is wrong with probability at most ``error_bound`` [Schwartz1980]_. The other steps
are proven, or the result names them: a ``checked`` direction, a rank that is a ``lower
bound``, a section claim under ``unproven``, a limit under ``info["limits"]`` and a resting
branch under ``info["branches"]``.

References
----------

.. [BenOr1988] M. Ben-Or and P. Tiwari. A deterministic algorithm for sparse
   multivariate polynomial interpolation. *Proceedings of the 20th ACM Symposium on
   Theory of Computing*, 301-309, 1988.

.. [Berthomieu2021] J. Berthomieu, C. Eder and M. Safey El Din. msolve: a library for
   solving polynomial systems. *Proceedings of ISSAC 2021*, 51-58, 2021.

.. [Goriely2001] A. Goriely. *Integrability and Nonintegrability of Dynamical Systems*.
   Advanced Series in Nonlinear Dynamics 19, World Scientific, 2001.

.. [HermannKrener1977] R. Hermann and A. J. Krener. Nonlinear controllability and
   observability. *IEEE Transactions on Automatic Control* 22, 728-740, 1977.

.. [Olver1993] P. J. Olver. *Applications of Lie Groups to Differential Equations*.
   Graduate Texts in Mathematics 107, Springer, 2nd edition, 1993.

.. [Pohjanpalo1978] H. Pohjanpalo. System identifiability based on the power series
   expansion of the solution. *Mathematical Biosciences* 41, 21-33, 1978.

.. [Schwartz1980] J. T. Schwartz. Fast probabilistic algorithms for verification of
   polynomial identities. *Journal of the ACM* 27(4), 701-717, 1980.

.. [Sedoglavic2002] A. Sedoglavic. A probabilistic algorithm to test local algebraic
   observability in polynomial time. *Journal of Symbolic Computation* 33(5), 735-755,
   2002.

.. [Wang1982] P. S. Wang, M. J. T. Guy and J. H. Davenport. P-adic reconstruction of
   rational numbers. *ACM SIGSAM Bulletin* 16(2), 2-3, 1982.

.. [Yates2009] J. W. T. Yates, N. D. Evans and M. J. Chappell. Structural identifiability
   analysis via symmetries of differential equations. *Automatica* 45(11), 2585-2591,
   2009.

.. [Zippel1979] R. Zippel. Probabilistic algorithms for sparse polynomials.
   *Symbolic and Algebraic Computation, EUROSAM '79*, Lecture Notes in Computer Science
   72, 216-226, 1979.
