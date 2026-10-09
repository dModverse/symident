Method
======

The model
---------

symident analyses ODE models with observables,

.. math::

   \dot x = f(x, \theta), \qquad x(0) = x_0(\theta), \qquad y = g(x, \theta),

with states :math:`x \in \mathbb{R}^m`, parameters :math:`\theta` and observables
:math:`y`. The right-hand sides, initial values and observables are rational functions,
or become rational after the recasts described below. The unknowns of the analysis, the
parameters together with the initial values that are not fixed, are collected in the
coordinates :math:`z \in \mathbb{R}^n`.

Structural identifiability
--------------------------

A coordinate is *identifiable* if the observables determine its value: whenever two
coordinate vectors :math:`z` and :math:`z'` produce the same observables for all times,
:math:`y(t; z) = y(t; z')`, the coordinate agrees in both. It is *locally* identifiable
if this holds for :math:`z'` in a neighbourhood of :math:`z`.

The property is *structural* when it is decided by the model equations alone, as if
the observables were known exactly and continuously in time, and when it holds for
generic coordinates, that is for all values outside a set of measure zero. A model that
is structurally non-identifiable cannot be fitted uniquely from any amount of data;
practical identifiability, which asks whether the data at hand suffice, is a separate
question that requires structural identifiability first.

A model is structurally non-identifiable when the coordinates can move along a curve
without changing the observables. Such a curve is the flow of a vector field

.. math::

   X = \sum_{i=1}^n \eta_i(z)\, \partial_{z_i},

the *generator* of a one-parameter group of transformations of the coordinates that
leaves every observable invariant: a *Lie symmetry* of the model's input-output map
[Yates2009]_.
The components :math:`\eta_i` say how fast each coordinate moves along the direction,
and the coordinates with :math:`\eta_i \ne 0` are its *support*. Each independent
generator removes one identifiable combination of the coordinates, so the number of
identifiable combinations is :math:`n` minus the number of independent directions.

Lie derivatives
---------------

The *Lie derivative* of a function :math:`h(x, \theta)` along the right-hand side
:math:`f` is its rate of change along the trajectories,

.. math::

   L_f h = \sum_{j=1}^m f_j \, \frac{\partial h}{\partial x_j},
   \qquad L_f^{k+1} h = L_f \big(L_f^k h\big), \qquad L_f^0 h = h .

Along a solution, :math:`\frac{d^k}{dt^k}\, g(x(t), \theta) = (L_f^k g)(x(t), \theta)`.
For analytic models the observables are therefore fixed by their Taylor coefficients at
the initial time,

.. math::

   y(t) = \sum_{k \ge 0} \frac{t^k}{k!}\, \big(L_f^k g\big)(x_0(\theta), \theta),

and two coordinate vectors give the same observables exactly when they give the same
Lie derivatives at :math:`t = 0` [Pohjanpalo1978]_. Identifiability becomes a question about the map

.. math::

   \Phi : z \mapsto \Big( \big(L_f^k g\big)(x_0, \theta) \Big)_{k \ge 0}.

A vector field :math:`X` is a non-identifiable direction when it annihilates every
component of :math:`\Phi`,

.. math::

   X\big(L_f^k g\big) = 0 \quad \text{for all } k \ge 0 .

At a point :math:`z` these conditions are linear in :math:`\eta(z)`: the directions are
the kernel of the *observability-identifiability matrix*

.. math::

   \mathcal{O}_N = \Big[\, \partial_z \big(L_f^k g\big)(x_0, \theta) \,\Big]_{k = 0, \dots, N},

and the model is structurally locally identifiable when :math:`\mathcal{O}_N` has full
rank :math:`n` at a generic point for some order :math:`N`. The kernel dimension at a
generic point is the number of independent non-identifiable directions.

Rank over finite fields
-----------------------

Symbolic Lie derivatives grow quickly with the order. symident evaluates
:math:`\mathcal{O}_N` numerically, but in exact arithmetic modulo a prime
:math:`p < 2^{31}`. The model is compiled to an instruction tape; the Lie derivatives are
computed as truncated power series in time and the derivatives with respect to
:math:`z` as dual numbers, so one pass of the tape yields all rows of
:math:`\mathcal{O}_N` at a random point. This follows the probabilistic approach of
[Sedoglavic2002]_. Exact arithmetic avoids the conditioning problems of floating point,
and a random point gives the generic rank with high probability [Schwartz1980]_
[Zippel1979]_; the rank is confirmed at further primes and points.

The rank grows with the order :math:`N` until it saturates. For each condition or block
the saturated order is certified: beyond the certified order the rank provably does not
grow, so the reported rank is the generic one rather than a plateau.

Conditions and events
---------------------

Several experimental conditions share parameters and differ in fixed values or
renamings. Their rows are stacked into one matrix over the shared coordinates, so a
direction has to leave the observables of every condition unchanged.

Events reset or shift states at given times and split a condition into segments. The
state at the start of a segment depends on the previous segment and on the time between
events; these gap lengths enter as formal power series variables, so directions that
hold only for particular event timings are separated from generic ones.

Switches
--------

Each condition of a switch, a comparison or the argument of a step, ``sign()``,
``abs()``, ``max()`` or ``min()``, is a sign condition :math:`g_j(x, \theta, t) > 0`.
Conditions the declared signs decide are resolved; an odd factor of an undecided
condition is a *switching function* :math:`g_j`, and :math:`S_j = \{g_j = 0\}` its
surface. A sign vector :math:`\sigma \in \{+1, -1\}^m` fixes every condition and gives
a smooth regime :math:`(f_\sigma, h_\sigma)`. An event with a ``root`` adds its root as
a switching function and resets its state on :math:`S_j`.

**Regimes.** Without ``regimes``, every sign vector counts as a regime unless the
declared signs prove it empty: when :math:`g_a - g_b` or :math:`g_a + g_b` has a sign
that the declared positivity decides in every condition, the sign vectors that
contradict it are dropped (``x > c`` with ``x < c - d`` for positive ``d``). No regime is
dropped because of parameter values or because a trajectory does not reach it; the
result holds for every initial state. With ``regimes``, each condition uses the regimes
it names, and the result holds for that experiment (see below).

**Visibility.** Two regimes :math:`\sigma, \sigma'` of a condition are neighbours at
:math:`S_j` when they differ only in the sign of :math:`g_j`. The switch on
:math:`S_j` is *visible* when, for some pair of neighbours, some observable :math:`y`
and some order :math:`k`,

.. math::

   \big(L_{F_\sigma}^k h_{\sigma,y} - L_{F_{\sigma'}}^k h_{\sigma',y}\big)\big|_{S_j} \ne 0 ,

with :math:`F_\sigma = \sum_k f_{\sigma,k}\,\partial_{x_k}` (and :math:`\partial_t` for a
model in time). It is decided exactly: :math:`g_j = 0` is solved for a symbol in which
it is linear, a state first, and a nonzero rational value at a rational point of
:math:`S_j` proves visibility. The switch is *invisible* when, for every pair, the
observables agree and every state that reaches them (the closure of the states in the
observables under the right-hand sides of both regimes) has the same right-hand side in
both regimes; then every jet agrees. Otherwise, or when :math:`g_j` is linear in no
symbol, the switch is *undecided*. The surface of a root event is always kept as
visible.

**The rule.** A direction :math:`v` of the coordinates is reported non-identifiable
when, in every condition, it lies at the start in a distribution
:math:`D = \operatorname{span}\{Y_1, \dots, Y_r\}` of vector fields on parameters and
states with

.. math::

   [Y_i, F_\sigma] \in D, \qquad Y_i(h_\sigma) = 0 \qquad \text{for every regime } \sigma,

.. math::

   Y_i(g_j) = 0 \ \text{on } S_j \ \text{for every visible surface}, \qquad
   Y_i(\tau_e) = 0, \qquad DE_e\, Y_i \in D \circ E_e \ \text{for every event and reset},

and, at a surface that is not visible and that some :math:`Y_i` moves, for every pair of
neighbours

.. math::

   \big(F_\sigma - F_{\sigma'}\big)\big|_{S_j} \in D, \qquad
   \big(h_\sigma - h_{\sigma'}\big)\big|_{S_j} = 0 ;

no :math:`Y_i` moves a fixed symbol, an input or time, and :math:`v` is
:math:`\big(\eta, X(x_0(\theta))\big)` with the given initial values :math:`x_0`, its
own value at a free initial value, at each resting state of each regime under
``equilibrate``. A direction of the flow of every regime stays in :math:`D`, so every
observable keeps its value to first order; at a visible surface the direction is
tangent, so the crossing time stays; at a hidden one the crossing may move, but the jump
of the dynamics lies in :math:`D` and the observables agree there. The rule therefore
never reports a direction along which the observables change, for any sequence of the
named regimes. It is a rule over regimes, not over one trajectory: without ``regimes``, a
parameter that acts only in a regime the trajectory never reaches is still reported
identifiable.

**Lower bound.** Every direction of the rule satisfies linear conditions at a point,
and their rank bounds the rank from below. They are the observability rows, over
:math:`GF(p)` as above, of copies of the model that share the parameters:

- each regime from the start of each condition, with its events;
- each regime from a generic point of each visible surface that is linear in a state,
  the free coordinates of the point shared by the regimes and projected out: there a
  direction of the rule is tangent to the surface;
- each regime from a generic point, when a switch of the condition is not visible;
- each copy switched at a fixed time into every other regime of the condition: the
  outputs after the switch hold the Lie derivatives along words of two regimes,
  :math:`L_{F_\sigma}^a L_{F_{\sigma'}}^b h_{\sigma'}`, which a direction of an invariant
  distribution keeps;
- the switching time :math:`\tau_j(\theta)` of every visible switch in time alone.

The kernel of the stacked rows, projected onto the coordinates, contains every direction
of the rule; its dimension :math:`k` gives the lower bound :math:`n - k`. A minor that is
nonzero at the sample point is nonzero as a rational function, so the bound needs no
probabilistic argument.

**Upper bound.** Candidates are the scalings of the integer kernel (the terms of every
visible :math:`g_j` scaled alike, initial values and events as constraints), the closed
forms of the stacked kernel and the unit field of every coordinate. With the unit fields
of the states they form the family :math:`Y_i`; a member that fails a condition of the
rule is dropped until the family passes, and a candidate counts when its start vector
lies in :math:`D`. Every identity, membership in :math:`D` included, is proven by
symbolic simplification to zero; :math:`d` such candidates independent at a rational
point bound the rank from above by :math:`n - d`.

When :math:`d = k`, the rank is :math:`n - k` and the candidates span the
non-identifiable directions; otherwise the analysis raises an error that names the
rejected candidates. For an undecided switch the rule accepts either the tangency or the
hidden jump; neither option reports a direction along which the observables change, and
the rows of its surface are left out, so the two bounds stay valid. ``scalings_only``
returns the proven scalings alone.

**Named regimes.** ``regimes`` lists, per condition, the regimes an experiment visits:
each entry is a comparison or a list of them and selects the regimes on those sides; the
first entry is the regime at the start. The rule, the rows and the candidates then use
these regimes, and a surface counts only when named regimes lie on both of its sides.
Parameters of the other regimes stay coordinates, so a parameter that acts only outside
the named regimes is reported non-identifiable. The start is checked: a sign of
:math:`g_j(x_0)` that the declared signs decide must agree with the first entry, else
the analysis raises an error; an undecided sign is an assumption, listed under
``info["switching"]["conditions"]``.

**Limits.** Each is an error, never a result: a condition ``==`` or ``!=``, which holds
only on a surface; a condition on a switched expression; a visible surface that is
linear in no symbol when a candidate must be checked on it; more than :math:`2^{12}`
sign vectors; ``gauge``, ``reduce_cq`` and ``free_initial``; under ``equilibrate``, a
resting state without a closed form; a named regime that is not feasible or a start
that contradicts the initial values; and every model where the two bounds do not meet,
for instance when the only directions are not rational fields.

Steady states
-------------

With ``equilibrate=True`` every condition starts at its resting state
:math:`f(x^\ast, z) = 0`. The resting state is solved over :math:`GF(p)` at each sample
point: by linear elimination where possible, by resultants for small coupled cores, and
by msolve [Berthomieu2021]_ for coupled nonlinear systems. The kernel is taken in the
joint coordinates of parameters and resting states, with the tangency condition
:math:`\partial_x f\, \xi_x + \partial_z f\, \xi_z = 0` as additional rows. Where a
resting state cannot be solved backward, the turnover rates are solved forward from
chosen states.

Non-rational terms
------------------

Elementary functions are recast into rational form, innermost first. A term
:math:`w = \varphi(u)` whose argument :math:`u` holds states becomes a new state; its
right-hand side follows from the chain rule, :math:`\varphi'(u)\, \sum_j \partial_{x_j} u
\, f_j`, which is rational in :math:`w` and :math:`u` for

.. math::

   \varphi(u) = u^c:\ \ \frac{c\, w}{u}, \qquad
   \exp(u):\ \ w, \qquad \log(u):\ \ \frac{1}{u}, \qquad
   \arctan(u):\ \ \frac{1}{1 + u^2}, \qquad \operatorname{artanh}(u):\ \ \frac{1}{1 - u^2},

and for :math:`\arcsin`, :math:`\arccos`, :math:`\operatorname{arsinh}` and
:math:`\operatorname{arcosh}` holds a square root, which is recast in turn.
:math:`\sin` and :math:`\cos` are expressed through :math:`\tan(u/2)`.

The initial value of a new state is :math:`\varphi(u)` at the initial state. Where it
depends on coordinates, a transcendental function (exponential, logarithm, arctangent)
becomes a generic leaf tied to its argument by the linearised relation
:math:`dw = \varphi'(u)\, du`: a transcendental graph lies in no proper algebraic set, so
a random point on it is generic. An algebraic function has no such generic
point and is handled exactly instead: by a chart when its argument is affine in one
coordinate, :math:`u = a + b\,v`, which replaces :math:`v` by :math:`L = \log u`, and
otherwise by taking the root modulo the prime at every sample point. A square root
exists for half the points, and a point without one is redrawn; an odd root of degree
:math:`q` is unique for every value once :math:`q` is prime to :math:`p - 1`, since
:math:`x \mapsto x^q` is then a bijection of :math:`GF(p)`, and a rational power
:math:`u^{m/n}` combines the two. Roots of numeric constants and odd degrees choose the
primes: the analysis runs over primes that hold every root of the model. A logarithm or fractional power needs an
argument that is positive on the declared domain. An observable :math:`a \log h + c` with
a numeric :math:`a` holds the same information as :math:`h` and is analysed as such.

Closed forms
------------

The rank alone gives the number of non-identifiable directions; the directions
themselves are reconstructed in closed form.

*Scalings*, :math:`z_i \mapsto \lambda^{w_i} z_i`, have the generator
:math:`\eta_i = w_i z_i`. They follow exactly from an integer lattice of monomial
weights and are taken out first.

*General directions* are the remaining kernel vectors. They are normalised, and each
entry is reconstructed as a rational function of the coordinates: the coordinates an
entry depends on are found by perturbation, the entry is sampled at many points and
primes, fitted by dense interpolation, by sparse interpolation in the style of
[BenOr1988]_, or by univariate rational interpolation along lines, and lifted to
rational coefficients by the Chinese remainder theorem and rational reconstruction
[Wang1982]_. Every closed form is checked at a fresh prime before it is reported.

Under ``equilibrate``, a direction that no sampling route closes is fitted along lines
:math:`\theta_0 + t\,d` through the base point. The resting state is continued as a
power series in :math:`t`, the kernel entries become series, and their Pade
approximants give the entries on a star of lines, from which each entry is fitted as a
rational function of the parameters. An entry that is not rational in the parameters
alone is fitted in the parameters and the resting states, modulo the resting equations;
resting states that these equations give linearly are substituted. A remaining resting
state of condition :math:`k` is named ``x_c<k>``, or ``x`` when the direction needs a
single condition. Each entry is checked at a prime outside its lift and the direction
against the kernel at a fresh point. The route does not apply to recast coordinates or
to events after the start, and ``reconst_control(homotopy=False)`` turns it off.

A rank is *proven* when the rank plus the number of independent exact scalings (and
static symmetries) equals the dimension, or when every direction lies in the span of
invariant vector fields of the model.

Reduction
---------

A reduction removes the non-identifiable directions by choosing one point on every
orbit. A scaling is removed by fixing one of its coordinates, for example an initial
value or a scale at one. A general direction is removed through its *invariants*, the
functions :math:`I(z)` that are constant along the direction, :math:`X(I) = 0`
[Olver1993]_. symident searches monomial, polynomial and rational invariants, Darboux
polynomials and exponential factors [Goriely2001]_ within degree bounds.

The invariants become new parameters :math:`q_k`, and the remaining coordinates are
expressed through them on a *section*, a set of points that every orbit meets exactly
once. Positivity of the resulting expressions on the declared domain is certified.

References
----------

.. [BenOr1988] M. Ben-Or and P. Tiwari. A deterministic algorithm for sparse
   multivariate polynomial interpolation. *Proceedings of the 20th ACM Symposium on
   Theory of Computing*, 301-309, 1988.

.. [Berthomieu2021] J. Berthomieu, C. Eder and M. Safey El Din. msolve: a library for
   solving polynomial systems. *Proceedings of ISSAC 2021*, 51-58, 2021.

.. [Goriely2001] A. Goriely. *Integrability and Nonintegrability of Dynamical Systems*.
   Advanced Series in Nonlinear Dynamics 19, World Scientific, 2001.

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
