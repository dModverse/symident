Method
======

The model
---------

symident analyzes ODE models

.. math::

   \dot x = f(x, u, p), \qquad x(t_0) = x_0(p), \qquad y = h(x, p),

with states :math:`x \in \mathbb{R}^n`, parameters :math:`p`, an input
:math:`u = u(t, p)` and an observation function :math:`h`. The parameters may be given by
a *parametrization* :math:`\theta \mapsto p(\theta)`, with :math:`\theta` in an open
*parameter space* :math:`\Theta \subseteq \mathbb{R}^d`; :math:`\theta` holds the estimated
parameters and the initial values that are not fixed. The *prediction* is
:math:`g(t; \theta) = h\big(x(t), p(\theta)\big)`. The right-hand sides, initial values and
observables are rational functions, or become rational after the recasts described below.

Structural identifiability
--------------------------

The model is *locally structurally identifiable* at :math:`\theta` if no
:math:`\tilde\theta \ne \theta` near :math:`\theta` has the same prediction,
:math:`g(\cdot; \tilde\theta) = g(\cdot; \theta)`. The property is *structural* because it
is decided by the model equations alone, as if the observables were known exactly and
continuously in time, and it is stated for generic :math:`\theta`, that is for all values
outside a set of measure zero. A structurally non-identifiable model cannot be fitted
uniquely from any amount of data; practical identifiability, which asks whether the data
at hand suffice, is a separate question.

A continuous family of parameter values with the same prediction is the orbit of a *Lie
symmetry*, a local one-parameter group of transformations :math:`\Phi_s` of
:math:`\Theta` with :math:`g\big(t; \Phi_s(\theta)\big) = g(t; \theta)` [Yates2009]_. Its
*generator* is the vector field

.. math::

   X = \sum_{i=1}^d \xi_i(\theta)\, \partial_{\theta_i}, \qquad
   \xi(\theta) = \partial_s \Phi_s(\theta)\big|_{s = 0},

and the integral curve of :math:`X` through :math:`\theta` is the *orbit*
:math:`\mathcal{O}(\theta)`. The components :math:`\xi_i` say how fast each coordinate
moves along the orbit, and the coordinates with :math:`\xi_i \ne 0` are the *support* of
the direction. A function :math:`I` with :math:`X(I) = 0` is an *invariant*: it is
constant on orbits, and the prediction depends on :math:`\theta` only through the
invariants. Each independent generator removes one identifiable combination, so the number
of identifiable combinations is :math:`d` minus the number of independent generators.

Lie derivatives
---------------

The parameters and the time are added to the states, :math:`z = (x, p, t)` with
:math:`\dot p = 0` and :math:`\dot t = 1`, so that the model is the vector field

.. math::

   F = \sum_{j=1}^n f_j\big(x, u(t, p), p\big)\, \partial_{x_j} + \partial_t ,

written :math:`F = f\, \partial_x + \partial_t` for short, with initial value
:math:`z_0(\theta)`. A vector field is written in the basis of partial derivatives: the
coefficient of :math:`\partial_{z_j}` is the velocity of the coordinate :math:`z_j`, here
:math:`f_j` for a state, :math:`0` for a parameter and :math:`1` for the time. The
notation reads the field as the derivative along the flow: applied to a function
:math:`\varphi(z)`, it gives the rate of change of :math:`\varphi` along a solution,

.. math::

   F(\varphi) = \sum_{j=1}^n f_j\, \frac{\partial \varphi}{\partial x_j}
   + \frac{\partial \varphi}{\partial t} = \frac{\mathrm{d}}{\mathrm{d}t}\,
   \varphi\big(z(t)\big)

(see `Vector field
<https://en.wikipedia.org/wiki/Vector_field#Vector_fields_on_subsets_of_Euclidean_space>`_
on Wikipedia). The generator :math:`X` above is a vector field of the same kind, on the
parameter space. The *Lie derivatives* of the output along :math:`F` are

.. math::

   L_F^0 h = h, \qquad L_F^{k+1} h = F\big(L_F^k h\big),

the time derivatives of the prediction along a solution. For analytic models the
prediction is fixed by them at the initial time,

.. math::

   g(t; \theta) = \sum_{k \ge 0} \frac{(t - t_0)^k}{k!}\, L_F^k h\big(z_0(\theta)\big),

and two parameter values with the same Lie derivatives at :math:`t_0` have the same
prediction [Pohjanpalo1978]_. Collected up to order :math:`N`,

.. math::

   \Psi_N(\theta) = \big(h, L_F^1 h, \dots, L_F^N h\big)\big(z_0(\theta)\big), \qquad
   \mathrm{OIM}_N(\theta) = \frac{\partial \Psi_N}{\partial \theta},

where :math:`\mathrm{OIM}_N` is the *observability-identifiability matrix* of order
:math:`N`. A generator leaves every component of :math:`\Psi_N` unchanged,
:math:`X\big(L_F^k h(z_0)\big) = 0`, so its tangent vector :math:`X|_\theta` lies in
:math:`\ker \mathrm{OIM}_N(\theta) \subseteq T\Theta|_\theta`. At a generic
:math:`\theta` the kernel consists exactly of these tangent vectors, and the model is
locally structurally identifiable when :math:`\mathrm{OIM}_N` has full rank :math:`d` for
some order :math:`N`.

Rank over finite fields
-----------------------

Symbolic Lie derivatives grow quickly with the order. symident evaluates
:math:`\mathrm{OIM}_N` in exact arithmetic modulo a prime :math:`p < 2^{31}`, over the
finite field :math:`\mathbb{F}_p`. The model is compiled to an instruction tape; the Lie
derivatives are computed as truncated power series in time and the derivatives with
respect to :math:`\theta` as dual numbers, so one pass of the tape yields all rows of
:math:`\mathrm{OIM}_N` at a random point. This follows the probabilistic approach of
[Sedoglavic2002]_. Exact arithmetic avoids the conditioning problems of floating point. A
random point can only underestimate the generic rank, and it falls short with small
probability [Schwartz1980]_ [Zippel1979]_; the rank is confirmed at further primes and
points.

The rank grows with the order :math:`N` until it stops. With the parameters fixed, the
states through the start of a segment lie on a smallest invariant affine space, and on it
the rows have a generic rank that no order exceeds [HermannKrener1977]_. Where the rows at
the start reach this rank, the order is proven *sufficient* per condition and segment:
beyond it the rank does not grow, so the reported rank is the generic one and not a false
plateau. Where it does not apply, for example for large stacked systems or a non-affine
invariant set, the order stops after a plateau of constant rank, and the result says that
the Lie order is not proven sufficient. For the gap order :math:`M` the rows after a gap
are power series in :math:`\varepsilon`, and their rank over
:math:`\mathbb{F}_p((\varepsilon))` is at most the rank of all their coefficients. These
are the rows of :math:`L_{F_1}^k \big( L_{F_2}^j h_2 \circ E \big)` at the start of the
segment before the gap, nested through the earlier gaps: Lie derivatives along one field of
a fixed family of functions, so the same argument makes their rank final from the first
order that adds nothing, once the rows at the start reach it. :math:`M` is raised until the
truncated rank reaches this bound, which proves it sufficient; where no bound applies or the
cap ``reconst_control(gap_order_cap=)`` comes first, the rank is a lower bound and
``info["limits"]`` says where :math:`M` stopped. A verified direction is checked at the
orders used, so its bound holds only where both orders are proven sufficient.

``engine="symbolic"`` proves its order exactly. With the parameters as constant states,
the rows :math:`\partial_z L_F^k h` stop growing at the first order that adds no rank over
the rational functions [HermannKrener1977]_; that order is found at a point and confirmed
by an exact rank. Where the rows at the initial values reach this rank, a nonvanishing
minor makes every later row there a combination of the earlier ones: the kernel is final,
and a direction that annihilates the rows identically is proven. States that stay at
their initial value are replaced by it; exponentials of single symbols and one logarithm
of a number enter as further symbols.

Conditions and events
---------------------

Several experimental conditions share parameters and differ in fixed values or
renamings. Their rows are stacked into one matrix over the shared coordinates, with
:math:`\ker \mathrm{OIM} = \bigcap_c \ker \mathrm{OIM}^{(c)}`, so a direction has to
leave the prediction of every condition unchanged.

An *event* changes a state or an input instantaneously at a known or estimated time and
splits a condition into segments. The state at the start of a segment is the state of the
previous segment propagated over the gap :math:`\Delta t` between the events. The gaps
enter as formal power series variables: every gap is put on the line through the actual
gaps, :math:`\Delta t_i = \varepsilon\, \delta_i`, and the rank is taken over
:math:`\mathbb{F}_p((\varepsilon))`, the field of formal Laurent series in
:math:`\varepsilon`. Directions that hold only for particular event timings are thereby
separated from generic ones.

Switches
--------

Each condition of a switch, a comparison or the argument of a step, ``sign()``,
``abs()``, ``max()`` or ``min()``, is a sign condition :math:`\gamma_j(x, p, t) > 0`.
Conditions the declared signs decide are resolved; an odd factor of an undecided
condition is a *switching function* :math:`\gamma_j`, and
:math:`\Gamma_j = \{\gamma_j = 0\}` its surface. A sign vector
:math:`\sigma \in \{+1, -1\}^J` fixes every condition and gives a smooth regime
:math:`(F_\sigma, h_\sigma)`. An event with a ``root`` adds its root as a switching
function and resets its state on :math:`\Gamma_j`.

**Regimes.** Without ``regimes``, every sign vector counts as a regime unless the
declared signs prove it empty: when :math:`\gamma_a - \gamma_b` or
:math:`\gamma_a + \gamma_b` has a sign that the declared positivity decides in every
condition, the sign vectors that contradict it are dropped (``x > c`` with ``x < c - d``
for positive ``d``). No regime is dropped because of parameter values or because a
trajectory does not reach it; the result holds for every initial state. With
``regimes``, each condition uses the regimes it lists, and the result holds for that
experiment (see below).

**Visibility.** Two regimes :math:`\sigma, \sigma'` of a condition are neighbors at
:math:`\Gamma_j` when they differ only in the sign of :math:`\gamma_j`. The switch on
:math:`\Gamma_j` is *visible* when, for some pair of neighbors, some observable
:math:`y` and some order :math:`k`,

.. math::

   \big(L_{F_\sigma}^k h_{\sigma,y} - L_{F_{\sigma'}}^k h_{\sigma',y}\big)\big|_{\Gamma_j}
   \ne 0 .

It is decided exactly: :math:`\gamma_j = 0` is solved for a symbol in which it is linear,
a state first, and a nonzero Taylor coefficient of the difference at a seeded point of
:math:`\Gamma_j`, modulo a prime, proves visibility. The switch is *invisible* when, for every pair, the observables agree
and every state that reaches them (the closure of the states in the observables under the
right-hand sides of both regimes) has the same right-hand side in both regimes; then every
Lie derivative agrees. Otherwise, or when :math:`\gamma_j` is linear in no symbol, the
switch is *undecided*. The surface of a root event is always treated as visible.

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
   \big(h_\sigma - h_{\sigma'}\big)\big|_{\Gamma_j} = 0 ;

no :math:`Y_i` moves a fixed symbol, an input or time, and :math:`v` is
:math:`\big(\xi, X(x_0)\big)` with the given initial values :math:`x_0`, its own value at a
free initial value, at each resting state of each regime under ``equilibrate``. Along the
flow of every regime the direction stays in :math:`\Delta`, so every observable keeps its
value to first order; at a visible surface the direction is tangent, so the crossing time
does not change; at a hidden one the crossing may move, but the jump of the dynamics lies
in :math:`\Delta` and the observables agree there. The rule therefore never reports a
direction along which the observables change, for any sequence of the listed regimes. It
is a rule over regimes, not over one trajectory: without ``regimes``, a parameter that
acts only in a regime the trajectory never reaches is still reported identifiable.

**Lower bound.** Every direction of the rule satisfies linear conditions at a point,
and their rank bounds the rank from below. They are the observability rows, over
:math:`\mathbb{F}_p` as above, of copies of the model that share the parameters:

- each regime from the start of each condition, with its events;
- each regime from a generic point of each visible surface that is linear in a state,
  the free coordinates of the point shared by the regimes and projected out: there a
  direction of the rule is tangent to the surface;
- each regime from a generic point, when a switch of the condition is not visible;
- each copy switched at a fixed time into every other regime of the condition: the
  outputs after the switch contain the Lie derivatives along words of two regimes,
  :math:`L_{F_\sigma}^a L_{F_{\sigma'}}^b h_{\sigma'}`, which a direction of an invariant
  distribution leaves unchanged;
- the switching time :math:`\tau_j(\theta)` of every visible switch in time alone.

The kernel of the stacked rows, projected onto the coordinates, contains every direction
of the rule; its dimension :math:`k` gives the lower bound :math:`d - k`. A minor that is
nonzero at the sample point is nonzero as a rational function, so the bound needs no
probabilistic argument.

**Upper bound.** Candidates are the scalings of the integer kernel (the terms of every
visible :math:`\gamma_j` scaled alike, initial values and events as constraints), the
closed forms of the stacked kernel and the unit field of every coordinate. With the unit
fields of the states they form the family :math:`Y_i`; a member that fails a condition of
the rule is dropped until the family passes, and a candidate counts when its start vector
lies in :math:`\Delta`. Every identity, membership in :math:`\Delta` included, is proven
by symbolic simplification to zero, after a nonzero value at a rational point has had the
chance to refute it; :math:`\ell` such candidates independent at a rational point bound the
rank from above by :math:`d - \ell`.

When :math:`\ell = k`, the rank is :math:`d - k` and the candidates span the
non-identifiable directions. Otherwise the directions of the stacked kernel that the rule
does not prove are reported as ``verified``, the rank :math:`d - k` is a lower bound, and
``info["switching"]["uncertified"]`` lists the rejected candidates. For an undecided switch the rule accepts either the tangency or the
hidden jump; neither option reports a direction along which the observables change, and
the rows of its surface are left out, so the two bounds stay valid. ``scalings_only``
returns the proven scalings alone.

**Listed regimes.** ``regimes`` lists, per condition, the regimes an experiment visits:
each entry is a comparison or a list of them and selects the regimes on those sides; the
first entry is the regime at the start. The rule, the rows and the candidates then use
these regimes, and a surface counts only when listed regimes lie on both of its sides.
Parameters of the other regimes stay coordinates, so a parameter that acts only outside
the listed regimes is reported non-identifiable. The start is checked: a sign of
:math:`\gamma_j(x_0)` that the declared signs decide must agree with the first entry, else
the analysis raises an error; an undecided sign is an assumption, listed under
``info["switching"]["conditions"]``.

**Limits.** Each is an error, never a result: a surface in the parameters alone whose
side ``regimes`` does not give; a condition on a switched expression; a visible surface
that is linear in no symbol when a candidate must be checked on it; more than
:math:`2^{12}` sign vectors; ``gauge``, ``reduce_cq`` and ``free_initial``; under
``equilibrate``, a resting state without a closed form; a listed regime that is not
feasible; and a start that contradicts the initial values.

Steady states
-------------

With ``equilibrate=True`` every condition starts at its resting state
:math:`f(x^\ast, p) = 0`, which binds :math:`x_0 = x^\ast` to :math:`\theta`. The resting
state is solved over :math:`\mathbb{F}_p` at each sample point: by linear elimination
where possible, by resultants for small coupled cores, and by msolve [Berthomieu2021]_
for coupled nonlinear systems; no closed form is needed. The kernel is taken in the joint
coordinates of parameters and resting states, with the tangency condition
:math:`\partial_x f\, \xi_x + \partial_\theta f\, \xi_\theta = 0` as additional rows. By
default the first condition draws its resting states and solves for the turnover rates
instead (the forward seed); the other conditions solve for their states. Where several
resting branches exist, the solve takes the first nonzero branch over
:math:`\mathbb{F}_p`.

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
:math:`\operatorname{arcosh}` contains a square root, which is recast in turn.
:math:`\sin` and :math:`\cos` are expressed through the half-angle tangent
:math:`\tan(v/2)`. A free exponent :math:`b^n` becomes :math:`E = b^n` with the companion
:math:`L = \log b`, after which the field is rational and :math:`n` enters through
:math:`\partial E / \partial n = E L`.

The initial value of a new state is :math:`\varphi(v)` at the initial state. Where it
depends on coordinates, a transcendental function (exponential, logarithm, arctangent)
becomes a generic leaf tied to its argument by the linearized relation
:math:`\mathrm{d}w = \varphi'(v)\, \mathrm{d}v`: a transcendental graph lies in no proper
algebraic set, so a random point on it is generic. An algebraic function has no such
generic point and is handled exactly instead. When its argument is affine in one
coordinate, :math:`v = a + b\,\theta_i`, the analysis runs in :math:`L = \log v` in place
of :math:`\theta_i`; otherwise the root is taken modulo the prime at every sample point. A
square root exists for half the points, and a point without one is redrawn; an odd root of
degree :math:`\ell` is unique for every value once :math:`\ell` is prime to
:math:`p - 1`, since :math:`x \mapsto x^\ell` is then a bijection of
:math:`\mathbb{F}_p`, and a rational power :math:`v^{k/\ell}` combines the two. Roots of
numeric constants and odd degrees choose the primes: the analysis runs over primes that
contain every root of the model. A logarithm or fractional power needs an argument that is
positive on the parameter space. An observable :math:`a \log h + c` with a numeric
:math:`a` contains the same information as :math:`h` and is analyzed as such.

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
:math:`\theta^0 + \lambda\,\delta` through the base point. Under ``equilibrate`` the
resting state is continued as a power series in :math:`\lambda`; without it the leaves
follow the line and the initial values with radicals become series. The kernel entries
become series, and their Padé approximants give the entries on a star of lines, from
which each entry is fitted as a rational function of the parameters. An entry that is not
rational in the parameters alone is fitted in the parameters and the resting states (or
the initial values with radicals), modulo their equations: states held at zero drop out,
and states that enter an equation linearly are eliminated or, when all do, substituted. A
remaining state of condition :math:`k` is named ``x_c<k>``, or ``x`` when the direction
needs a single condition. With events after the start the kernel takes the series along
the line on an axis of its own, separate from the formal gaps between the events, and the
resting state seeds the first segment. Each entry is checked at a prime outside its lift
and the direction against the kernel at a fresh point. In the recast coordinates of a free
exponent, :math:`L = \log b` moves with the parameters, and :math:`E = b^n` and :math:`L`
are substituted back into the closed form. The route does not apply to the transient
recast coordinates, to kernels that depend on the gaps between events, or to event times
that depend on the coordinates; ``reconst_control(homotopy=False)`` turns it off.

Every closed form has a *level*. It is ``exact`` when proven over Q: a scaling from the
integer kernel, or a direction lifted to a vector field on parameters and states (the
closed form itself, its canonical representative, its multiple by the common denominator,
or a combination of the fields of each regime) whose identities per state equation,
observable, initial value and event cancel symbolically. Otherwise it is ``verified``: it
lies in the kernel at two seeded points modulo a verification prime :math:`q`, and from a
degree bound :math:`D` of the observability rows a wrong closed form passes with
probability at most :math:`(D/q)^2` per direction [Schwartz1980]_ (``info["bound"]``).
The check runs at the Lie and gap orders used, so the bound holds only where both are
proven sufficient. A direction without closed form is ``open``.

A rank is *proven* when the rank plus the number of independent exact scalings (and
static symmetries) equals :math:`d`, or when every direction is proven over Q. When every
direction lies in the span of invariant vector fields of the model, the invariance covers
every Lie order, but it is checked modulo a prime at one point, so the rank is *verified*,
not proven. It is also *verified* when both orders are proven sufficient and every
direction has a closed form, and a *lower bound* otherwise (``info["rank_status"]``).

Reduction
---------

A reduction replaces the parameters by combinations that the output determines, the
invariants :math:`I_l(\theta)` with :math:`X(I_l) = 0` [Olver1993]_. symident searches
monomial, polynomial and rational invariants, Darboux polynomials and exponential factors
[Goriely2001]_ within degree bounds.

The invariants become new parameters, :math:`I_l(\theta) = q_l`, and the remaining
coordinates are expressed through them. The result is a *global section*: it maps the new
parameters one-to-one onto a *global cross-section*, a set that meets every orbit exactly
once, so the invariants are global coordinates on the orbit space :math:`\Theta / G`. A
scaling is removed by fixing one of its coordinates, for example an initial value or a
scale at one; every orbit of a scaling is a ray and meets such a value once. Positivity
of the resulting expressions on the parameter space is certified, and a block whose
cross-section is not met by every orbit in the parameter space is reported as partial.
The trafo is checked over Q, and the detection on the reduced model has to leave exactly
the remaining directions. Each block is ``exact`` when every claim is proven,
``verified`` when a direction rests on a verified closed form, and ``assumed`` when a
claim rests on an assumption that the result names.

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
