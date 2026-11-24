Global analysis
===============

Local identifiability leaves finitely many coordinate values with the output of a generic one.
A *discrete symmetry* is a map :math:`\sigma` with :math:`\Psi_N(\sigma(\theta)) =
\Psi_N(\theta)` and :math:`\sigma(\theta) \ne \theta`, which no generator shows. It matters
for estimation only where :math:`\sigma(\theta) \in \Theta`: :math:`\theta \mapsto -\theta`
of a positive coordinate that enters as :math:`\theta^2` is no symmetry of the model on
:math:`\Theta`. Counts over the algebraic closure, as in :cite:p:`Hong2020,ReyBarreiro2023`,
include such values; the global analysis (``detect(mode="global")``,
:meth:`~symident.Symmetries.discrete`) counts in :math:`\Theta` as well.

The global analysis sets up the fiber of a generic point as a polynomial system
(:ref:`fiber`, :ref:`fiber-features`, :ref:`rational-form`), counts the points modulo a prime and
reconstructs the maps between them (:ref:`maps`), and certifies the count, exactly by the field of
the output coefficients (:ref:`exact-count`), with an error bound over :math:`\mathbb{Q}`
(:ref:`verified-count`, :ref:`several-experiments`), or not (:ref:`certificates`). Points
proposed numerically and certified by interval arithmetic (:ref:`monodromy`) propose maps only.

.. _fiber:

The fiber
---------

A model with directions is reduced first (:func:`~symident.reduce`); the reduced model is
locally identifiable, so the *fiber* :math:`\Psi_N^{-1}(\Psi_N(\theta^\ast))` is finite. The
fiber at a point :math:`\theta^\ast` is the solution set of the polynomial system that equates the
Taylor coefficients of the observables up to order :math:`N` with their values at
:math:`\theta^\ast`, away from zeros of coordinates and denominators. msolve
:cite:p:`Berthomieu2021` counts the points of the system over the algebraic closure modulo a prime, an upper bound
for the size of the fiber of the model, which more equations can only lower. As in SIAN
:cite:p:`Hong2020`, the derivatives of the states at the start are unknowns of the system, tied by the
Taylor coefficients of :math:`D(x)\,\dot x = N(x)` and :math:`E(x)\, g = M(x)`. Every
observable starts one order above the highest of the orders whose next coefficient adds no rank to
the Jacobian in the coordinates: at a given start, a coefficient that adds no rank can still remove
points. The orders rise together until the count is the same at two successive steps, or equals
the size of the orbit of the proven candidates, one if there are none, which no higher order
changes, since more equations give a fiber within the one before and the orbit stays in it. The
count of a block is the number of distinct values of a random linear form in the coordinates on
the points of the block: msolve gives the reduced Gröbner basis, the standard monomials span the
quotient, and the number of distinct eigenvalues of the multiplication by the form is the degree of
the square-free part of the minimal polynomial of that multiplication. Points that differ only in the derivatives of the states, or that are
multiple, count once. The series of a known input is a number, a variable
that a linear equation fixes to a number is substituted, a derivative that enters one equation
only, there linearly with a numeric coefficient, leaves with that equation, and the rest splits
into blocks without a shared variable, whose fibers multiply. msolve runs over :math:`\mathbb{F}_p`
with the probabilistic sparse linear algebra of msolve.

*Stages.* A block is counted stage by stage. Let :math:`A` be a set of variables of the block, :math:`P_A`
the equations and factors of the saturation that read only variables in :math:`A`, and :math:`P_R`
the others. A point of the block is a pair :math:`(a, r)` with :math:`P_A(a) = 0` and
:math:`P_R(a, r) = 0`, the factors nonzero, so the points of the block are the disjoint union over
the points :math:`a` of :math:`P_A` of the points of :math:`P_R(a, \cdot)`, and the count of the
block is the sum of the counts of the rest at each :math:`a`. This holds for every :math:`A`; the
block triangular form of the incidence of equations and variables :cite:p:`Dulmage1958` only proposes one:
a maximum matching of equations to variables, the graph in which a variable points to the variables
the matched equation of the variable reads, and as :math:`A` the smallest proper set closed under this graph that
holds a strongly connected part of more than one variable, so that the equations in :math:`P_A`
match every variable of :math:`A`. The block is first counted at once under an eighth of ``msolve_entries``,
since the rest after a stage is often small. Then msolve gives the rational parametrization of
:math:`P_A`; where every distinct root of the eliminating polynomial lies in
:math:`\mathbb{F}_p` (in :math:`\mathbb{Q}` for the count over :math:`\mathbb{Q}`, at most 16
roots), each point is substituted, the fixings above run again, and the rest is counted by stages;
otherwise the block is counted at once. Over :math:`\mathbb{Q}` the stage is first
solved modulo a prime and skipped where a point lies outside :math:`\mathbb{F}_p`, since rational
points reduce to points in :math:`\mathbb{F}_p` for a prime that divides no denominator; the choice
of the route never changes the count. Points :math:`a` that differ only in the derivatives of the states count apart, so
the sum is never below the number of points with distinct coordinates.

.. _fiber-features:

Conditions, events, resting states and switches
-----------------------------------------------

The conditions are stacked as in the local analysis: each has separate derivatives, the
coordinates are shared. After an event the gaps are :math:`\varepsilon\, \Delta t_i^0` as in
:ref:`conditions-events`: the state at the end of a gap is the series of the segment before
evaluated at :math:`\varepsilon\, \Delta t_i^0`, the event maps it to the start of the next
segment, and every coefficient after the event is a power series in :math:`\varepsilon`, kept up
to the *gap order*, which rises with the order. A segment whose start holds coordinates, an event
at an estimated time, is expanded at the start of the condition on the flow continued backward,
so that the coefficients of the segment depend on the time of the event. Under ``equilibrate`` the start of a
condition is the solution of the balances over :math:`\mathbb{Q}(\theta)`, exponentials of the coordinates
included, with the inputs at zero,
the states that are coordinates at their free resting value and the states without dynamics at
their initial value; where the balances have more than one rational solution, the one with the
most states away from zero, and none without one rational solution. States that no observable
depends on are left out. With switches, as in :ref:`switching`, each condition enters from the
start in the regime of the start, decided at a random point, and each allowed regime as a copy
whose start is free on a switching surface :math:`\gamma_j = 0` to an allowed neighbor: at the point
the surface is solved for a state in which it is linear, and in the fiber
:math:`\gamma_j(z', \theta') = 0` is an equation; the clock and the jets of known inputs of a copy
start at known values. A map is ``exact`` with a map of the states in every regime that takes each
surface to itself, :math:`\gamma_j(\varphi(x), \sigma(\theta)) = c\, \gamma_j(x, \theta)` with
:math:`c` free of the states and positive on :math:`\Theta`. A switch on the parameters alone puts
the model in one regime and is not handled. A map reconstructed from the fiber is
checked on the coefficients four orders further up, in :math:`t` and in :math:`\varepsilon`; a
map that fails shows a fiber too large at that order, and the orders rise.

.. _rational-form:

Rational form of the fiber
--------------------------

The fiber needs a model rational in the states and coordinates. Three rewrites before the count
keep the fiber. A coordinate :math:`\theta` that enters only through :math:`b^{c\theta}`, with
rational :math:`c`, is read as :math:`X = b^{\theta/L}`, :math:`L` the common denominator of the
exponents, a bijection of :math:`\mathbb{R}` onto the positive reals; a map is reported back as
:math:`\theta' = L \log_b \sigma(X)`, so a swap stays a swap and a discrete scaling becomes a
translation. An observable :math:`h` for which :math:`e^{c h}` is rational, :math:`c \in \{1,
\log 10, \log 2\}`, is replaced by :math:`e^{c h}`, one of the form :math:`e^{r}` by :math:`r`,
and an odd power :math:`r^{p/q}` by :math:`r`: these maps are injective where the observable is
real, so two values of the coordinates have the same output before the rewrite if and only if
they have it after. A factor :math:`e^{a t}`, with :math:`a` free of the states and the clock,
becomes a state :math:`E` with :math:`\dot E = a E` and :math:`E(0) = 1`, whose unique solution it
is; this needs conditions that start at :math:`t = 0`. A coordinate :math:`\theta` that
enters only through :math:`e^{c/\theta}`, with rational :math:`c`, is read as :math:`X = e^{a/\theta}`, :math:`a`
the positive rational greatest common divisor of these :math:`c`, so that :math:`X` enters with integer powers;
:math:`\theta \mapsto X` is a bijection of the positive reals onto :math:`(1, \infty)` and of the negative reals
onto :math:`(0, 1)`, and a map is reported back as :math:`\theta' = a / \log \sigma(X)`. The counts in the domain
take :math:`X > 0`, which holds :math:`\theta` real but not positive. A state without dynamics that starts at a
number and that every event sets to a number, such as a clamped input, is that number within each segment, and is
substituted there where a non-rational term reads it. ``info["discrete"]["rewrites"]`` lists the rewrites, and a
term that stays non-rational is named in the error.

**Non-rational terms.** A term :math:`\varphi(\psi)` of the states that these rewrites
leave, :math:`\varphi` an exponential, a logarithm or a power :math:`b^\mu` with a non-integer
exponent and :math:`\psi` rational, becomes :math:`W \nu` (:math:`W + \nu` for the logarithm),
innermost first, with a new state :math:`\nu` and a new coordinate :math:`W = \varphi(\psi_0)`,
:math:`\psi_0` the value at the start:

.. math::

   \nu = e^{\psi - \psi_0},\ \Big(\frac{b}{b_0}\Big)^{\mu},\ \log \frac{\psi}{\psi_0}, \qquad
   \dot\nu = \nu\, \dot\psi,\ \ \mu\, \nu\, \frac{\dot b}{b},\ \ \frac{\dot\psi}{\psi}, \qquad
   \nu(0) = 1,\ 1,\ 0 .

A term of the coordinates alone becomes a new coordinate :math:`P`, the :math:`W` of a term with the
same start or an integer power of it, and a coordinate that no
equation reads afterwards is dropped where one :math:`P` determines it on the positive reals
(:math:`b^\mu` or :math:`\log b` with :math:`b = c\, \theta^{\pm 1}`, or :math:`e^{a \theta^{\pm 1} + d}`,
with :math:`c, \mu, a, d` free of :math:`\theta`). A positive state :math:`x` that the model reads only through
:math:`\log x`, through the factor :math:`x` of :math:`\dot x = x R` and through observables :math:`x^k` with an
integer :math:`k \neq 0` is replaced first by :math:`L = \log x`, with :math:`\dot L = R`, the observable
:math:`k L` and the start :math:`P = \log x_0`. This keeps the fiber: :math:`x = e^L` is the unique positive
solution, :math:`x^k \mapsto k \log x` is injective on the positive reals, and :math:`x_0 \mapsto P` is a
bijection onto the reals, so :math:`x_0` is dropped and no relation between a :math:`W` and :math:`x_0` is lost. A term with a known input or the clock, and a root or rational
power, whose start is algebraic, are not rewritten. The *relaxed model* is rational in
:math:`\tilde\theta = (\theta, W, P)`, and the output of the model is the relaxed output on the graph
:math:`\Gamma` of :math:`\theta \mapsto \omega(\theta)`, the starts of the new coordinates.
``discrete()`` counts the relaxed model modulo the continuous directions, as above, and
``info["discrete"]["relaxed"]`` lists the terms, the starts and the dropped coordinates.

*Proposition.* Suppose (a) every direction of the relaxed model that moves a :math:`W` is exact, a
scaling or a shift of the :math:`W` it moves by constants, and moves no coordinate read by a start
:math:`\omega`, with an invertible matrix of the weights and shifts on the :math:`W` these
directions move, (b) no direction moves a coordinate read by the start
of a :math:`W` that no scaling moves, and (c) these starts are algebraically independent over
:math:`\mathbb{Q}(\theta)`. Then at a generic point the fiber of the model has at most :math:`k`
connected components, :math:`k` the count of the relaxed model modulo the continuous directions, and exactly
:math:`k` where :math:`k = 1` or where the exact maps of the relaxed model that fix every
:math:`W` and every coordinate read by a start have an orbit of :math:`k` points.

*Proof.* By (a) these directions reach every value of the :math:`W` they move from any point (every
positive value for a scaling),
and by (c) the image of :math:`\theta \mapsto (\theta, \omega(\theta))` on the other :math:`W` is
Zariski dense, so a generic :math:`\theta` lies on a generic orbit of the relaxed model. A point
:math:`\theta'` of the fiber of the model gives the point :math:`(\theta', \omega(\theta'))` of
the relaxed fiber. On an orbit :math:`O` of the relaxed fiber, the :math:`W` that no scaling
moves and, by (b), the coordinates their starts read are constant, so either no point of
:math:`O` or every point after one group element meets these starts; the element :math:`s(q)` of
the directions of (a) that brings :math:`q \in O` to the starts of the moved :math:`W` is unique
and continuous in :math:`q` by (a), since these directions fix what the starts read. So :math:`O \cap \Gamma` is
empty or the image of the connected :math:`O` under :math:`q \mapsto s(q)\, q`, and the fiber of
the model is the union of at most :math:`k` connected sets. A map that fixes the :math:`W` and
what the starts read takes :math:`\Gamma` to itself, so an orbit of :math:`k` points meets
:math:`k` relaxed orbits on :math:`\Gamma`. :math:`\square`

For exponentials alone, (c) holds where the Jacobian of the exponents :math:`\psi_0` in the
coordinates has full rank at a random point: an integer combination of the :math:`\psi_0` that
is constant has a zero differential, and the lemma of Kolchin and Ostrowski in :ref:`non-rational`
gives the independence. For logarithms alone the same holds with :math:`\mathrm{d} \log
\psi_0`: a combination :math:`\sum_i c_i \log \psi_{0,i}` that is rational has a differential with
simple poles only, so it is constant. Otherwise (c) is a premise, named in
``certificate["premise"]``, and the certificate is ``axiom``. Where (a) or (b) fails,
``certificate["relaxed_why"]`` names the direction, the count is not reported (``count`` None,
the relaxed count under ``relaxed["count"]``), and the certificate is ``open``.

**Free powers of a state at 0.** A power :math:`x^{c_0 + c\alpha}` of a state
:math:`x` that starts at the number 0, with integers :math:`c_0 \ge 0`, :math:`c > 0` and a
coordinate :math:`\alpha` declared positive, has no start :math:`\psi_0 \neq 0` to relax. With
:math:`k` and :math:`c_x` as in :ref:`singular-starts`, :math:`x^\alpha = Z N` with :math:`Z =
c_x^\alpha` and the *normalized power* :math:`N = (x/c_x)^\alpha = \tilde S Q`, :math:`\tilde S =
t^{k\alpha}`, a series in :math:`t` and :math:`\tilde S` whose coefficients are polynomials in
:math:`\alpha`, the coefficients of :math:`x` and :math:`1/c_x`. The model with :math:`x^\alpha = 0`
gives the coefficients of :math:`x` of layer 0; :math:`k` is the first order at which the coefficient is
nonzero at a random point, which shows it nonzero as a function, and the coefficients below
:math:`k` are computed over :math:`\mathbb{Q}(\theta)` and vanish identically.

:math:`Z` leaves the model in one of three ways. (i) It cancels against a factor
:math:`c_x^{-\alpha}`: the model is written in :math:`N`. (ii) The model is invariant under
:math:`Z \mapsto \lambda Z`, :math:`P_i \mapsto \lambda^{q_i} P_i` for the new coordinates
:math:`P_i = b_i^{q_i \alpha}` of the relaxation above: then the output depends on
:math:`Z` and :math:`P_i` only through :math:`\tilde P_i = P_i Z^{-q_i} = (b_i / c_x)^{q_i \alpha}`,
the model is taken on the slice :math:`Z = 1` with :math:`\tilde P_i` in place of :math:`P_i`, and a
:math:`\tilde P_i` determines a dropped coordinate as before where :math:`c_x` is positive on the
domain and free of it. The map from :math:`\theta` to the slice is then a bijection, so this step
needs no premise. (iii) Otherwise :math:`Z` is a new coordinate :math:`W` with the start
:math:`c_x^\alpha`, under the proposition above. :math:`c_x` must be positive on the domain by the
declared signs and free of the known inputs.

In the prolonged model each state has the coefficients :math:`x_{n,m}` of index :math:`n` and
layer :math:`m`, standing for :math:`t^{n - km + km\alpha}`, and the derivative of index :math:`n`
is :math:`\lambda_{n+1,m}\, x_{n+1,m}`, :math:`\lambda_{n,m} = n - km + km\alpha`. The
coefficients :math:`Q_n` are unknowns with :math:`n\, c_x Q_n = \sum_{j=1}^n (\alpha j + j - n)
x_{k+j} Q_{n-j}`, :math:`Q_0 = 1`, and :math:`N` has :math:`Q_{n-k}` in index :math:`n` and layer
:math:`m+1` for layer :math:`m` of :math:`Q`; :math:`c_x = x_{k,0}` is saturated. Let :math:`v_N = 0`
and :math:`v_i = 1 + \min_j v_j` over the states and :math:`N` that the right-hand side of state
:math:`i` reads, and :math:`v_h = \min_j v_j` over those an observable reads. By induction on the
index, every term of layer :math:`m \ge 1` of state :math:`i` has :math:`n - km \ge v_i` where
the base has :math:`v_x \ge k`, so that every term of :math:`Q` has :math:`n - km \ge 0`; a power
with :math:`v_x < k` enters the dynamics of the base below :math:`t^k` and is refused. Unknowns
exist only for these terms, layer :math:`m` up to :math:`\lfloor N/k \rfloor` at order :math:`N`,
and only up to the index that the outputs read: an observable reads the states at the order of the observable, the
equation of index :math:`n` of a state reads the right-hand side at index :math:`n - 1`, and
:math:`N` reads the base at the same index. The unknowns above are determined by the ones below and
change no count.

**Lemma.** Let :math:`c_x` have one sign on the domain, and let the coefficient :math:`d =
h_{v_h + k, 1}` of an observable :math:`h` have one sign on the domain. Then every point
:math:`\theta'` of the domain with the output of a generic :math:`\theta` has :math:`\alpha' = \alpha`.

*Proof.* The expansions converge (:ref:`singular-starts`) and have exponents :math:`e = (n - km) + km
\alpha'`. At :math:`\theta'`, :math:`c_x \neq 0` gives the same :math:`k`; the terms of layer
:math:`m \ge 1` have :math:`n - km \ge v_h`, so for :math:`\alpha' > 0` the least exponent among
them is :math:`e' = v_h + k\alpha'`, attained by :math:`d(\theta') \neq 0` alone. Likewise
:math:`e = v_h + k\alpha` with :math:`d(\theta) \neq 0`, and :math:`e \notin \mathbb{Q}` for
generic :math:`\alpha`. The layers 0 have integer exponents. If :math:`e < e'`, the term
:math:`t^e` has the coefficient :math:`d(\theta) \neq 0` at :math:`\theta` and none at
:math:`\theta'`, as :math:`e` is no integer and lies below :math:`e'`; this contradicts the lemma of
:ref:`singular-starts`. If :math:`e' < e` and :math:`k\alpha'` is no integer, the same holds for
:math:`t^{e'}`. If :math:`k\alpha'` is an integer, every exponent at
:math:`\theta'` is an integer and :math:`t^e` again has no coefficient at :math:`\theta'`. So :math:`e
= e'` and :math:`\alpha' = \alpha`. :math:`\square`

The coefficients :math:`c_x` and :math:`d` are computed over :math:`\mathbb{Q}(\theta)`, and one
sign means numerator and denominator positive or negative by the declared signs; where this is not
shown the condition is a premise, ``certificate["premise"]``, and a count otherwise ``verified`` is
``axiom``. The fiber then has the equation :math:`\alpha' = \alpha` and the equations of all
layers, and the bound of the verified count takes the degrees of all of them. The maps are
reconstructed and checked as before but have no state maps, since :math:`N` is no rational
function of the states, and the exact count of the field of the outputs is not taken.
``info["discrete"]["start_powers"]`` names per power :math:`k`, :math:`c_x`, the observable,
:math:`d` and the premises. One such power per condition is taken, without events.

**Resting states inside a term.** Under ``equilibrate`` the start of the relaxed model is a resting
state, and a state :math:`x_H` inside a transcendental term has a resting value that solves a
transcendental balance. Such resting values :math:`R_H` become coordinates of a *resting chart*: the
transcendental terms at rest are leaves :math:`W = \varphi(\psi(R_H, \theta))`, the balances of the other states
are solved one at a time where each is linear in the state it belongs to, and one coordinate :math:`p_h` per state of
:math:`H` that enters the remaining balances linearly is solved from them, with an invertible coefficient matrix
(a nonzero determinant at a random rational point, which proves it nonzero as a function). Coordinates without a
declared sign come first, then those in the fewest balances. The chart

.. math::

   (\theta \setminus p,\ R_H) \mapsto \big(\theta,\ x^\ast\big), \qquad
   p = p(\theta \setminus p, R_H, W), \quad x^\ast_H = R_H, \quad x^\ast_N = x^\ast_N(\theta, R_H, W),

is a bijection onto the points :math:`(\theta, x^\ast)` of the resting set where the coefficient matrix and the
coefficients of the solved states do not vanish: :math:`R_H` reads :math:`x^\ast_H` off a point, and the
balances, linear in :math:`x^\ast_N` and :math:`p`, have the given point as their unique solution. The output of
a point of the resting set is the output from the resting state of the point, so the fiber of the model, as a set of points
:math:`(\theta, x^\ast)`, is the fiber in the chart; the chart model, rational up to the leaves :math:`W` and
with the start :math:`(R_H, x^\ast_N)`, is relaxed as above. In the chart the leaves are exponentials of
coordinates, so (c) follows from the rank of their exponents as before. Several resting states with the same
:math:`\theta` are distinct points of the fiber. The declared sign of :math:`p` is no sign of a chart coordinate,
so the counts in the domain do not hold it (``info["limits"]``). An event on a state inside a term, a known input,
a condition grid and a trafo of the states are not handled. A model with directions and a term of the states is
relaxed first; where the relaxed count bounds the fiber (no ``relaxed_why``), the reduction of the model itself is
skipped.

**Unknown inputs.** An unknown input enters the fiber through the jets at the start,
unknowns that are no coordinates: the chain of :ref:`inputs` with free starts. A
point of the fiber is then a value of the coordinates with some input of the same output, and
points that differ only in the jets count once. An observable of *relative degree* :math:`r` in the
input :math:`w`, the least order whose Lie derivative reads a jet of :math:`w`, has

.. math::

   L^{r + m} h = \partial_w (L^r h)\, w^{(m)} + (\text{terms in } w, \dots, w^{(m - 1)}),
   \qquad \partial_w L^r h \ne 0,

by induction on :math:`m`, since :math:`L^r h` reads :math:`w` but no derivative of it. So the
Taylor coefficient of order :math:`n` reads exactly the jets up to :math:`n - r`; :math:`r` comes
from symbolic Lie derivatives over :math:`\mathbb{Q}`. At the orders of a count, the jets above the
highest order that an observable reads are zero: no output coefficient up to these orders depends
on them, so the coordinates of the fiber are the same, and the equations of the derivatives of the
states, which read them, keep no free direction. An unknown of the system that is no coordinate
and enters no equation is left out, which keeps the coordinates of every point. The model has one
condition, no events, switches or ``equilibrate``, and the exact count by the field of the
outputs is not taken. A map :math:`\sigma` is ``checked`` where the equations of the fiber four
orders higher, with the coordinates at :math:`\sigma(\theta)`, have a solution modulo a prime at
random points, and ``exact`` where the map of the states fails without the inputs and a map
:math:`\psi(x, \theta, w)` of the input values, solved where linear, satisfies

.. math::

   F(\varphi(x), \sigma(\theta), \psi) = D\varphi(x)\, F(x, \theta, w), \qquad
   h(\varphi(x), \sigma(\theta), \psi) = h(x, \theta, w), \qquad
   \varphi(x_0(\theta)) = x_0(\sigma(\theta))

over :math:`\mathbb{Q}(x, \theta, w)`, :math:`\varphi` reading no input. Then :math:`(\varphi(x(t)),
\psi(x(t), \theta, w(t)))` solves the model at :math:`\sigma(\theta)` with the output of the
solution at :math:`\theta`; ``state_maps`` lists :math:`\psi` under the name of the input.

.. _maps:

Maps between the points of the fiber
------------------------------------

Permutations of the coordinates are tested before the count. If :math:`\sigma` permutes the
coordinates and keeps every Taylor coefficient :math:`c`, then :math:`c(\sigma(\theta)) =
c(\theta)` gives :math:`\partial_i c(\theta) = (\partial_{\sigma(i)} c)(\sigma(\theta))`, so
:math:`c` depends on :math:`\theta_i` if and only if it depends on :math:`\theta_{\sigma(i)}`, and
:math:`\sigma` maps each coordinate onto one with the same set of coefficients that depend on it.
These sets come from one evaluation of the Jacobian at a random point modulo a prime. The
transpositions within each class of equal sets, and then the products of two transpositions that
fail alone, are tested on the coefficients up to four orders past the rank criterion at a random
point modulo the prime, and a candidate that passes is proven as below or kept as ``checked``.
The candidates join the maps of the fiber; they need no count, which only decides whether the
maps are complete.

Where the other points are rational functions :math:`\sigma_j(\theta)`, they are reconstructed
without matching points across samples. For a coordinate :math:`\theta_i`, the characteristic
polynomial :math:`\prod_j (Z - \sigma_{j,i}(\theta))` over the fiber is invariant under
permutations of the points, so the coefficients are rational functions of :math:`\theta`; msolve
gives them modulo a prime from the rational univariate representation, and they are
interpolated as the generators are. A linear factor over :math:`\mathbb{Q}(\theta)` is a rational
:math:`\sigma_{j,i}`. Coordinates with the fewest arguments come first, until their rational
roots separate the points of the fiber; each other coordinate is then interpolated on each
point, labeled by those roots. A map is ``checked`` when the coefficients agree at random
rational points, and ``exact`` with a map :math:`\varphi` of the states such that

.. math::

   F(\varphi(x), \sigma(\theta)) = D\varphi(x)\, F(x, \theta), \qquad
   h(\varphi(x), \sigma(\theta)) = h(x, \theta), \qquad
   \varphi(x_0(\theta)) = x_0(\sigma(\theta))

over :math:`\mathbb{Q}(x, \theta)`: :math:`\varphi` is :math:`\sigma` on a state whose initial value
is a coordinate, a different one per state, and the identity on the others, with :math:`x \mapsto x^c` for a state
that starts at 1 with :math:`\dot x = x\, R` where :math:`R(\sigma) = c\, R` for a rational
:math:`c`, or, where these fail, the
solution of :math:`h(\varphi(x), \sigma(\theta)) = h(x, \theta)` and the Lie derivatives of these
equations where they are linear in :math:`\varphi`, or :math:`\varphi` affine in the states of the
block that :math:`\sigma` reaches, the states whose right-hand side or start holds a coordinate
that :math:`\sigma` moves, closed under the states that read them, and the identity elsewhere, with
coefficients solved from the identities where they are linear in them. With events, each segment has such a map, and
at each event :math:`\varphi_{\text{after}}(E(x, \theta)) = E(\varphi_{\text{before}}(x),
\sigma(\theta))`, in every condition. The ``exact`` maps generate a group :math:`G`; at a point
outside the fixed sets of the elements of :math:`G` the orbit is :math:`|G|` points of the fiber. When
:math:`|G|` equals the count, the count is final (``count_final``): the exact maps give a lower
bound equal to the upper one. ``info["discrete"]["certificate"]`` lists the state maps, :math:`|G|`,
the count and the prime. ``complete`` is ``"exact"`` where the field fixed by the exact maps lies
in the field of the output coefficients, which proves the fiber to be the orbit (:ref:`exact-count`); else ``"verified"`` where a count over :math:`\mathbb{Q}` at a random integer point
equals the orbit, with the probability ``count_bound`` that the fiber at a generic point has more
points. Otherwise the certificate names the step that is not proven: that the count modulo the
prime at a random point is the count over :math:`\mathbb{C}` at a generic point, an axiom
(``complete: "axiom"``, with ``count_bound_why``). A map with a positivity certificate for each
image coordinate preserves :math:`\Theta`; a map with an image coordinate negative on
:math:`\Theta` never meets it; otherwise the conditions on :math:`\theta` are listed. Where the fiber is
proven to be the orbit and every element of the group either preserves :math:`\Theta` or never meets
it, ``count_domain`` is one plus the number of elements that preserve it, exactly. Ordering the
coordinates and keeping in each orbit the point that is least in the first coordinate a map
moves gives a *fundamental domain*, a set of inequalities such as :math:`k_{01} < k_{12}`.

The counts in :math:`\Theta` at the sample points end at the work limits of the global analysis, and
where the proven maps account for the count, the points are the orbits. Over :math:`\mathbb{Q}`,
msolve isolates the real points in boxes, refined until each box decides
the sign of every coordinate declared positive; the count in :math:`\Theta` follows at rational
sample points. Where the rational maps of the next paragraph account for every point counted at a
sample point, and the images give the coefficients of the point exactly, the point and the images are the
fiber, and the count needs no isolation. The number of points in :math:`\Theta` is constant only between the hypersurfaces
on which two points meet or a point crosses the boundary of :math:`\Theta`, so the count is
reported per point and never taken as a global verdict.

.. _exact-count:

The exact count
---------------

Let :math:`L = \mathbb{Q}(\theta)`, :math:`G` the group of the ``exact`` maps
of :ref:`maps`, acting on :math:`L` by :math:`\sigma(f) = f \circ \sigma`, and
:math:`c` the Taylor coefficients of the observables at the start of the experiments, the first
segment of each condition, with the known inputs at the values drawn for them. Let :math:`T` be
the fiber at a generic point :math:`\theta` for these inputs: the values :math:`\sigma` over
:math:`\overline{L}` with the output of :math:`\theta`. A value with the output of :math:`\theta`
for every input lies in :math:`T`.

*Theorem.* Let :math:`F = \mathbb{Q}(c)` and :math:`\alpha \in L` with :math:`P_\alpha(Z) =
\prod_{\sigma \in G} (Z - \sigma(\alpha)) \in F[Z]` and :math:`L = F(\alpha)`. Then :math:`T` is the orbit of
:math:`\theta` under :math:`G`, of :math:`|G|` points.

*Proof.* The output is a function of :math:`\theta` whose coefficients generate a field
:math:`\mathbb{Q}(\Psi) \supseteq F`, and :math:`L` is finite over it, since it is finite over
:math:`F`. A point :math:`\sigma \in T` has coordinates algebraically independent over
:math:`\mathbb{Q}` (the output there has transcendence degree :math:`n`), so :math:`\theta \mapsto \sigma`
defines an embedding of :math:`L` into :math:`\overline{L}` over :math:`\mathbb{Q}(\Psi)`, and every
such embedding gives a point of :math:`T`; so :math:`|T| = [L : \mathbb{Q}(\Psi)]` in characteristic
0. The exact maps keep the output, so the orbit of :math:`\theta` lies in :math:`T`. :math:`\alpha`
is a root of :math:`P_\alpha \in F[Z]`, of degree :math:`|G|`, so :math:`|T| = [L :
\mathbb{Q}(\Psi)] \le [L : F] = [F(\alpha) : F] \le |G|`. :math:`\square`

The test of the theorem has no random step: random points only propose candidates, and each
candidate is checked exactly. Membership in :math:`F` is proven as a tower: a function is in
:math:`F` once it is a rational function of output coefficients and of functions already shown in
:math:`F`, :math:`f = A(w)/B(w)`, by the identity :math:`A(w(\theta))\, \mathrm{den}_f(\theta) =
\mathrm{num}_f(\theta)\, B(w(\theta))` with :math:`B(w(\theta)) \ne 0`, checked as polynomials in
:math:`\theta` over a common denominator. The output coefficients enter as quotients of polynomials in
:math:`\theta` over powers of a shared list of factors, the denominators of the starts of the states and the values of the
denominators of the right-hand sides and observables at the start, from the Taylor recursion of
:math:`D(x)\, x' = N(x)` order by order: a sum takes the largest power of each factor, so that no gcd is
needed and the denominators grow linearly with the order. The candidates :math:`A, B` come from linear algebra
modulo primes at random points, with rising degree, rebuilt over :math:`\mathbb{Q}`. The functions
shown in :math:`F` are:

- *Coefficients of input-output relations.* Let :math:`\phi_1, \dots, \phi_r` be the monomials of
  degree at most :math:`\delta` in the derivatives of each observable up to an order chosen per observable,
  which leaves it out where negative, and in the known inputs that the right-hand sides read, with
  their derivatives, the configurations taken with the fewest monomials first (for more
  than three observables each one alone and all at one order) and those holding one that gave
  relations left out, and :math:`M(c)` the
  matrix of the Taylor coefficients of :math:`\phi_j` along the trajectory of the first condition,
  whose entries are polynomials in output coefficients and the input values. A vector
  :math:`\lambda(\theta)` with :math:`\sum_j \lambda_j \phi_j(g, u) = 0` along every trajectory
  lies in the kernel of :math:`M(c(\theta))`; this is checked exactly, the derivatives of the
  observables as Lie derivatives, a polynomial identity in the states, the inputs and the
  coordinates. Where the rank of :math:`M` at one point modulo a prime is :math:`r - s` and
  :math:`s` such vectors are proven, each the identity on the same :math:`s` columns, they are the
  basis of the kernel over :math:`\mathbb{Q}(\theta)` with that identity, since a nonzero minor at
  one point is nonzero over :math:`\mathbb{Q}(\theta)`; the same basis solves the system over
  :math:`F`, so the entries of the basis lie in :math:`F` by Cramer's rule. They are interpolated as
  rational functions of the parameters from the relations among the :math:`\phi_j` at random states
  for random values of the parameters, the other coordinates held at one value, as polynomials of all
  entries at once, else over the denominator of one entry; the identity check decides. Rational right-hand sides and observables enter as rational functions of the states.
- *Coefficients of projections.* A *profile* :math:`h` gives per observable :math:`g_r` a number
  :math:`h_r` such that the derivatives :math:`g_r^{(k)}`, :math:`k < h_r`, of all observables are
  algebraically independent over :math:`L` and the next derivative of each one is algebraic over
  them; the rank of their Jacobian in the states at a random point modulo a prime proposes it, in
  rounds over the observables and one observable after the other. The *projection* of
  :math:`g_i` is a polynomial relation of :math:`g_i^{(h_i)}` with these derivatives and the input
  states read, the generator of the ideal of the relations in these variables :cite:p:`Dong2023`. The
  monomials :math:`\phi_j` of the projection are the support of the kernel of the values of the monomials at random
  states, for one random value of the parameters, modulo a prime: first with rising total degree
  and degree in :math:`g_i^{(h_i)}`, at most 1500 monomials, then inside bounds on the Newton
  polytope of the projection, at most 10000, then again by degrees, at most 6000: the largest
  sets of degrees below a cap that doubles, since one without a relation holds none of the
  smaller ones, and where the kernel has more than one vector, the least set with one. The bounds
  on the Newton polytope are counts of points. Let :math:`\Phi` be the map from the states to the
  base functions (the derivatives and the inputs of the projection), the parameters at their
  value, on a slice of the states: those whose columns keep the Jacobian of :math:`\Phi` at rank
  :math:`|\Phi| - 1` stay free, the others take random values, so that the image is dense in the
  hypersurface :math:`P = 0` of the projection and a generic point of it has the same number
  :math:`k` of preimages. For a weight vector :math:`w` of nonnegative integers and random vectors
  :math:`c, b`, the points of the slice with :math:`\Phi(x) = c + b\, t^w` lie over the roots of
  :math:`P(c + b\, t^w)`, a polynomial in :math:`t` of degree :math:`\max_{m \in \mathrm{supp}\,
  P} \langle w, m \rangle`, :math:`k` points over each; with :math:`c = 0` and :math:`t \ne 0`
  over the nonzero roots, :math:`\max - \min` of :math:`\langle w, m \rangle` in number. msolve
  :cite:p:`Berthomieu2021` counts the solutions of these systems over the algebraic closure of
  :math:`\mathrm{GF}(p)`, the denominators of the base functions inverted by a new unknown, and
  :math:`k` as the number of solutions of :math:`\Phi(x) = \Phi(x_0)` at a random :math:`x_0`.
  The weights are the total degree, each base function alone, the order of each derivative plus
  one and the reverse order, and random weights in :math:`\{0, \dots, 3\}` while more than 10000
  monomials meet the bounds. A point modulo a prime that is not generic can make a count too small
  and leave a monomial of :math:`P` out; then the kernel has no vector or the identity check below
  fails, so the bounds, like the rest of the support, only propose. :cite:p:`Dong2023` computes the
  projections by resultants over :math:`\mathbb{Q}(\theta)`, the extraneous factors removed by a
  membership test along power series solutions; here the support comes from one kernel at one
  value of the parameters, and the coefficients as follows. The coefficients are normalized at the one of
  least degree on a random line and interpolated as rational functions of the parameters over one
  denominator :math:`D`: a monomial where the restriction to every line parallel to an axis is a
  power of the variable, else interpolated from the values :math:`D(\theta) = \ell(1) / \ell(0)`
  with :math:`D(\beta) = 1`, :math:`\ell` the denominator of a random combination of the
  coefficients on the line :math:`\beta + t (\theta - \beta)`; the numerators :math:`a_j D` by
  sparse interpolation :cite:p:`BenOr1988` or in the box of their degrees per parameter, rebuilt over
  :math:`\mathbb{Q}` from more primes, at most 32, until two successive reconstructions agree. The projections of
  a profile are found first and taken with the fewest monomials first. A derivative is
  :math:`N / (Q^a E^b)`, :math:`Q` the product of the distinct denominators of the right-hand sides
  and :math:`E` the denominator of the observable, by :math:`L(N / (Q^a E^b)) = (\tilde N Q E - N
  (a \tilde Q E + b \tilde E Q)) / (Q^{a + 2} E^{b + 1})` with :math:`L(P) = \tilde P / Q`, so that
  the relation :math:`\sum_j a_j \phi_j = 0` is checked as one identity of polynomials in the states
  and coordinates, each term over the powers of :math:`Q` and :math:`E` that it lacks, without gcds.
  With rank :math:`r - 1` of :math:`M` at one point modulo a prime, :math:`r` the number of
  monomials, the coefficients lie in :math:`F` as in the item before. Every step that proposes
  (the profile, the monomials, the degrees, the interpolation) can only make the search fail,
  never prove a wrong function.
- *Coefficients of relations by elimination.* Where the projections give no proof, the states are
  eliminated as in :cite:p:`Hong2020` (StructuralIdentifiability.jl), with python-flint: the equation of an
  observable is differentiated along the flow, the derivatives of the states are removed with the
  equations of the model, and one state at a time is removed by a resultant over
  :math:`\mathbb{Z}[\theta]`, the determinant of the Bezout matrix (the Sylvester matrix where one
  degree is one) after the greatest common divisors of the rows and columns. Every polynomial of
  this process vanishes along every trajectory: the equations of the model do, and derivatives,
  combinations and resultants of such polynomials do. Of the irreducible factors of such a
  polynomial and of the divisors taken out, one vanishes along every trajectory: along one
  trajectory their product is zero in the domain of power series, so one factor is, and the
  coordinates and starts at which a given factor vanishes along the trajectory form a closed subset
  of the open set without a pole at the start, so one of these finitely many subsets is all of it.
  A factor nonzero at a point of a trajectory at random coordinates modulo a prime is not that one,
  so the product of the factors that vanish at three such points vanishes along every trajectory,
  without a random step. The relation of an observable at the end is the only one on the support
  where the rank of the Taylor coefficients of the monomials along the trajectory is one less than
  their number at a point modulo a prime, as for the projections; the coefficients over one of
  fewest terms then lie in :math:`F`, and those whose gradients in the parameters raise the rank
  at a random point modulo a prime, fewest terms first, with as many more, join the functions
  shown. Where a state :math:`x_i` is removed, an equation :math:`A x_i + B` of degree one in it
  (one of the two equations, else their first subresultant, a combination of both) whose
  coefficient :math:`A` is nonzero at a point of a trajectory gives :math:`x_i(0) = -B / A`, rational
  in the output coefficients, the coordinates and the starts of the states removed after it: the
  start lies in :math:`F` once these do. The elimination runs in a separate process while the share of
  ``search_entries`` for the relations is not spent and ends at ``elimination_terms``.
- the coordinates that :math:`G` fixes and the coefficients of :math:`P_\alpha`, each a rational
  function of the functions before it on the same coordinates, then of small systems of them (the
  functions shown in :math:`F` whose coordinates not yet shown lie in those of the function and at
  most two more, each divided by the largest monomial in the coordinates shown that divides the
  numerator and by the denominator where that reads coordinates shown only, again a function in
  :math:`F`, with the coordinates shown that they read, in degrees counted apart for both), then of
  all of them, then with the output coefficients added lowest order first; the traces :math:`\sum_{\sigma \in G} \sigma(\theta_i)` of the moved
  coordinates and sums over :math:`G` of monomials in them join where they are proven.

:math:`\alpha` is a moved coordinate with :math:`|G|` distinct images, else a combination of two
or three, with the simplest images. :math:`L = F(\alpha)` is shown coordinate by coordinate, each a
member of :math:`F(\alpha)` once shown, in two ways:

- as a rational function of functions in :math:`F`, of :math:`\alpha` and of the coordinates shown
  before, checked as above;
- for the start of a state :math:`x_i`, by a relation :math:`\sum_j \lambda_j \phi_j(g, u) + x_i
  \sum_k \mu_k \psi_k(g, u) = 0` along every trajectory, :math:`\psi_k` the monomials of one degree
  less, found and checked as the input-output relations, with :math:`\sum_k \mu_k \psi_k` nonzero
  also at the start, and with coefficients rational in parameters already shown in
  :math:`F(\alpha)`: at the start, :math:`x_i(0) = -\sum_j \lambda_j \phi_j / \sum_k \mu_k \psi_k`,
  with :math:`\phi_j, \psi_k` there polynomials in output coefficients and input values. Fixed
  starts join :math:`F` the same way. Such a state :math:`x_i(t)` is the same quotient along the
  whole trajectory, so the Taylor coefficients of the state lie in :math:`F` as well, and it joins the
  observables of the relations that follow. After the relations among at most 40 of these
  monomials, the projection of :math:`x_i` gives it: the relation of
  :math:`x_i` with the derivatives :math:`g_r^{(k)}`, :math:`k < h_r`, of degree one in
  :math:`x_i`, found, interpolated and checked as a projection, with :math:`\sum_k \mu_k \psi_k`
  nonzero at the start shown by the value at one point modulo a prime. Where :math:`x_i` is algebraic
  of higher degree over these derivatives, the base takes also :math:`g_r^{(h_r)}` of one observable, at a degree
  below the degree :math:`d_r` of the projection of :math:`g_r` in it: the monomials of lower degree in :math:`g_r^{(h_r)}` are a
  normal form modulo the projection, so no multiple of it enters the kernel.

The relations are tried in the order: monomials of the item on input-output relations up to 120,
projections, the elimination, then the others up to 300; the relations of a start among the
monomials up to 40 and then up to 300, after the projection of the start. A configuration or a
profile that needs the derivative of one with more than 5000 terms is left out. The search ends at
the work limits ``search_entries``, ``msolve_entries`` and ``elimination_terms``, or where the relations are tried and none is found. Past a
limit the count is reported without the exact certificate, and the certificate names what was not
reached. A call of msolve fails past the address space ``memory`` of ``discrete()``, 8 GiB by
default.
Constants of the model that are no coordinates are known symbols :math:`\kappa`: :math:`\mathbb{Q}`
is replaced by :math:`\mathbb{Q}(\kappa)` throughout, and the theorem holds for generic values of the
constants. ``info["discrete"]["certificate"]["exact"]`` lists the functions with their expressions,
the relations used and :math:`\alpha`, and ``complete: "exact"``. A short count modulo a prime comes first: where it equals the orbit of the proven candidates, or ends without a count, the
exact count is tried with the group of the candidates, and where it holds, neither the full count
nor the reconstruction of the maps runs. A second search, with the group of all exact maps, starts
from the functions the first one proved to lie in :math:`F`.

Where the search stops before :math:`L = F(\alpha)` is shown, the functions :math:`f_1, \dots,
f_m` it proved to lie in :math:`F` are counted instead of expressed. Every point of :math:`T` has
the values :math:`f_i(\theta)`, so :math:`T` lies in the fiber of :math:`\theta \mapsto (f(\theta),
c_0(\theta), \dots, c_N(\theta))` for any orders :math:`N`, the starts entering through the output
coefficients. The orders are those of the rank criterion with the gradients of the :math:`f_i` in the
basis first, so that the :math:`f_i` stand for the higher coefficients; the route runs only where
they lower the orders. The :math:`f_i` of least degree whose gradients raise the rank, with as many
more, are counted first, then all of them. As in the count, the orders start one above and rise,
at most to those of the rank criterion of the outputs alone, until the count modulo the prime at a
random point is below :math:`2|G|`; then the count over
:math:`\mathbb{Q}` of the next paragraph, with these :math:`f_i` as pins, bounds :math:`|T|`, and
a count :math:`k` with :math:`|G| \le k < 2|G|` gives :math:`|T| = |G|`, since :math:`|G|` divides
:math:`|T|` (*The orbit* in :ref:`several-experiments`). The smaller variety of the bound takes the
:math:`n` rows from the :math:`f_i` and the output coefficients, those of least degree first.
``certificate["pinned"]`` lists the functions counted, the orders, the counts modulo the prime and
over :math:`\mathbb{Q}`, the bound with the bits of the point and the Bezout number, or why the
route gave none.

*Coefficients of a large projection, verified.* Where the elimination gives no relation for the
observable of a projection with more than 1500 terms, a few coefficients of the projection join the
:math:`f_i` with an error bound. Let
:math:`S` be the support of :math:`r` monomials :math:`\phi_j` in the base functions :math:`b_v =
N_v / \prod_k F_k^{e_{vk}}`, found as above.

(a) The Taylor coefficients of the :math:`\phi_j` along the trajectory have rank :math:`r - 1` at a
random point modulo a prime, so at most one relation on :math:`S` holds along the trajectory, and if
one exists, the coefficients over one of them lie in :math:`F` by Cramer's rule, as above.

(b) The relation exists, verified. A trial takes :math:`r + 6` random states and a random
:math:`\theta` modulo a prime :math:`q` uniform among the primes from :math:`2^{29}` to the largest
engine prime. Times :math:`\prod_k F_k^{T_k}`, :math:`T_k = \max_j t_{jk}`, :math:`t_{jk} = \sum_v
m_{jv} e_{vk}`, a row of the values of the :math:`\phi_j` becomes the values of the polynomials
:math:`P_j = \prod_v N_v^{m_{jv}} \prod_k F_k^{T_k - t_{jk}}` of degree :math:`d_j`. Without a
relation, the determinant of the first :math:`r` rows is a nonzero polynomial in :math:`\theta` and
the states, of degree at most :math:`D = \sum_j d_j` and with integer coefficients whose content
is at most :math:`H`, :math:`\log_2 H \le \log_2 r! + \sum_j \log_2 \|P_j\|_1`, the norms
bounded by those of the :math:`N_v` and :math:`F_k` cleared to integers. A kernel of the values
makes it vanish at the point. At most :math:`\lceil \log_2 H / 29 \rceil` of the primes divide the
content, and the primes drawn from number at least :math:`N` by :math:`x / \ln x < \pi(x) < 1.25506\,
x / \ln x` :cite:p:`Rosser1962`; a row where a denominator vanishes is drawn again, which conditions on an
event of probability :math:`a \ge 1 - (r + 6) \deg \prod_k F_k / (q - 1)`. So a trial shows a
kernel with probability at most :math:`\varepsilon = D / ((2^{29} - 1)\, a) + \lceil \log_2 H / 29
\rceil / N` by :cite:p:`Schwartz1980`, and :math:`k = 5` independent trials, each with a kernel of one
vector, with probability at most :math:`\varepsilon^k`.

(c) The coefficients. Where a coefficient :math:`c_{j_0}` is a monomial in the parameters, the
others over it are Laurent polynomials. Axis lines through a random point, of 24 points at the
start, show such a :math:`j_0`: the quotient of every other entry by it has a power of :math:`t`
as denominator on every line, which gives the range of the exponents of each entry per parameter.
The entries with the fewest exponent vectors in their ranges are interpolated by Ben-Or and Tiwari
:cite:p:`BenOr1988` modulo two primes and rebuilt over :math:`\mathbb{Q}`, those whose gradients raise
the rank first, until the rank of the gradients of all entries found on the lines, then as many
more. At every trial of (b) each of them equals the entry of the kernel over entry :math:`j_0`. A
wrong function :math:`w = M / (L\, \theta^a)` would make :math:`M C_{j_0} - L\, \theta^a C_j`, with
:math:`C` the cofactors of the first :math:`r - 1` rows, a nonzero polynomial that vanishes at the
point, so a trial passes with probability at most :math:`\varepsilon_w`, as :math:`\varepsilon`
with the degree and the height of :math:`w` added, and :math:`k` trials with
:math:`\varepsilon_w^k`.

These functions join the :math:`f_i` of the count, and where they
are counted, the bound of the count is that over :math:`\mathbb{Q}` plus :math:`\varepsilon^k` and
:math:`\varepsilon_w^k` per function counted. ``certificate["pinned"]["relation"]`` lists the
functions, the monomial :math:`j_0`, the trials, the kernels, :math:`D`, :math:`\log_2 H` and the
probabilities, ``relation_bound`` their sum and ``count_bound_q`` the bound over
:math:`\mathbb{Q}` alone.

.. _verified-count:

The verified count
------------------

Where the exact count is not found, the count is taken again over
:math:`\mathbb{Q}`, by the rational parametrization of msolve, at an integer point
:math:`\theta^\ast` uniform in :math:`\{1, \dots, 2^b\}^n`. A count :math:`k` there bounds the
fiber:

.. math::
   :label: eq-count-bound

   \Pr\big[|T| > k\big] \le \frac{(N_u + 2D)\, D}{2^b},

with :math:`N_u` the number of unknowns and :math:`D` the Bezout number below; :math:`b \ge 64`
is chosen so that the bound is at most :math:`10^{-6}`, up to :math:`b = 256`
(``count_bits``, ``bezout``). The bound holds for a model without events and switches, with every
start a coordinate, a number, a rational function of the coordinates or a jet of an unknown input
(*Unknown inputs* below), at orders no lower than those of the rank criterion; otherwise, or where the count differs, the count stays under the
axiom of :ref:`certificates` and ``count_bound_why`` names the reason. msolve is taken to be
correct. The count over :math:`\mathbb{Q}` calls msolve per block, plain and, where that fails, with a
separating linear form. Before a count, modulo a prime or over :math:`\mathbb{Q}`, a variable :math:`v` that an
equation :math:`a v + b\, m` sets to a monomial :math:`m` free of :math:`v` is substituted, a
coordinate only for a monomial in coordinates: on the variety :math:`v` is a function of the other
unknowns, so the substitution maps the points one to one onto those of the smaller system, and
points with distinct coordinates keep distinct coordinates. The count, and the bound below, which
is computed for the system before the substitution, do not change. Equations of the fiber with more
than 400000 terms, or with a power, product or sum of series of more terms while they are built, or
with more than 10000 unknowns, are not counted: the count stops at that order, and ``limits`` names
the limit.

*The system.* The unknowns :math:`u = (x, z, w)` are the coordinates, the derivatives of the
states at the start and the variable of the saturation. The Taylor coefficients :math:`b(\theta)`
of the observables are rational functions; written over one denominator :math:`q(\theta)`, the
equations of the fiber become :math:`F_1, \dots, F_s \in \mathbb{Q}[\theta, u]`: the equations of
the model, the output equations :math:`\sum_l E_l(u)\, q\, b_{k-l} - q M_k(u)` and
:math:`w\, s(u) - 1`, with :math:`s` the product of the coordinates and the denominators at the
start. Their degrees in :math:`\theta` follow from the recursion of the series run on degree
bounds instead of values, a division adding the numerator of the divisor to the denominators.
:math:`V = V(F) \subset \mathbb{A}^{n + N_u}` has degree at most :math:`D = \prod_i \deg F_i`, by
the Bezout inequality :math:`\deg(V \cap W) \le \deg V \deg W` :cite:p:`Heintz1983`.

*Step 1: the points of T are isolated.* For :math:`\sigma \in T` the point :math:`P_\sigma =
(\theta, \sigma, z(\sigma), 1/s)` lies on :math:`V`. The Jacobian of :math:`F` in :math:`u` has
full column rank there: the equation :math:`\mathrm{den}(x)\, z_0 = \mathrm{num}(x)` of a start
rational in the coordinates has the coefficient :math:`\mathrm{den}(\sigma) \ne 0`, the equation of
the derivative of order :math:`k + 1` of a state has the coefficient :math:`(k + 1) D_i(x_0)`,
nonzero by the saturation, and no derivative of higher order, so the derivatives are local
functions of :math:`x`; then the output equations have the Jacobian :math:`-E_0 \cdot \partial_x
\Psi_N` up to a triangular factor, and the rank criterion gives :math:`\operatorname{rank}
\partial_x \Psi_N = n` at a random point modulo a prime, so at every point whose coordinates are
algebraically independent over :math:`\mathbb{Q}`, as those of :math:`\sigma` are. So
:math:`P_\sigma` is an isolated point of :math:`V` over :math:`K = \mathbb{Q}(\theta)` and lies on
the union :math:`V_0` of the components of :math:`V` of dimension :math:`n` that dominate the
coordinates, which has at most :math:`D` points over :math:`\theta`.

*Step 2: polynomials of bounded degree.* For each unknown :math:`u_j` the closure of the image of
:math:`V_0` under :math:`(\theta, u) \mapsto (\theta, u_j)` is a hypersurface :math:`V(E_j)`,
:math:`E_j \in \mathbb{Q}[\theta, U]` square-free, of degree at most :math:`D`, since a linear map
does not raise the degree. A linear form :math:`\lambda(x)` with integer coefficients separates the
at most :math:`D` values of :math:`x` on :math:`V_0` over :math:`\theta`; the image under
:math:`(\theta, u) \mapsto (\theta, \lambda(x))` is :math:`V(E_\lambda)`, of degree at most
:math:`D`, square-free in :math:`U` over :math:`K`, so :math:`R_\lambda =
\operatorname{Res}_U(E_\lambda, \partial_U E_\lambda) \in \mathbb{Q}[\theta]` is nonzero, of degree
at most :math:`(2D - 1) D`.

*Step 3: specialization.* Let :math:`a_j`, :math:`a_\lambda` be the leading coefficients in
:math:`U` and :math:`\Delta = a_\lambda R_\lambda \prod_j a_j`, nonzero of degree at most
:math:`(N_u + 2D) D`; by :cite:p:`Schwartz1980` :math:`\Delta(\theta^\ast) = 0` with probability at most
the right side of :eq:`eq-count-bound`. Suppose :math:`\Delta(\theta^\ast) \ne 0`. The map
:math:`\theta \mapsto \theta^\ast` extends to a place of :math:`\bar K` with residue field
:math:`\bar{\mathbb{Q}}` (Chevalley's extension theorem). Every :math:`u_j` of a point of
:math:`V_0` is a root of :math:`E_j` with a leading coefficient nonzero at :math:`\theta^\ast`, so
finite at the place, and the point maps to a solution of :math:`F(\theta^\ast, u) = 0` with
:math:`w` finite, so away from the saturation. The values :math:`\lambda(x)` are the roots of
:math:`E_\lambda`, and with :math:`a_\lambda R_\lambda \ne 0` at :math:`\theta^\ast` they stay
distinct. The series at :math:`\theta^\ast` is the same recursion as over :math:`K`; without a
division by zero, :math:`q(\theta^\ast) \ne 0` and the values are those of :math:`b`, so the system
solved is :math:`F(\theta^\ast, \cdot)` up to nonzero factors. The substitutions of the fixed
variables, the derivatives that leave with their equation and the saturation by one variable per
block keep the values of :math:`x`. The system solved then has at least :math:`|T|` points with
distinct :math:`x`, and the count of msolve, the degree of the rational parametrization, at least
the number of distinct solutions, is at least :math:`|T|`. Counted by stages, each point of a stage is
exact and rational, and the sum over these points of the degrees of the rest is at least the number of
points with distinct :math:`x`, again at least :math:`|T|`.

*A smaller variety.* The same steps hold for :math:`V' \subset \mathbb{A}^{n} \times
\mathbb{A}^{n + 1}` in :math:`(\theta, x, w)`, cut out by :math:`n` output equations
:math:`\mathrm{num}_i(x)\, \mathrm{den}_i(\theta) - \mathrm{num}_i(\theta)\, \mathrm{den}_i(x)` whose
gradients in :math:`x` are independent at a random point modulo a prime, and by :math:`w\, s'(x) =
1`, :math:`s'` the product of the coordinates, the denominators of the starts and the denominators of
the right-hand sides and observables at the start. A point of :math:`T` is an isolated point of
:math:`V'` over :math:`K`, since these gradients are independent at every point with algebraically
independent coordinates. A point of :math:`T` satisfies every output equation, so the specialization of the point,
with :math:`s' \ne 0` there, gives the derivatives at the start and a solution of the system solved.
With :math:`D' = (1 + \deg s') \prod_i \deg(\mathrm{num}_i\, \mathrm{den}_i)` and :math:`n + 1`
unknowns in place of :math:`D` and :math:`N_u`, the smaller of the two bounds is used.

*Pins.* A function :math:`f = a/b` that the search of the exact count proves to lie in :math:`F`
takes the same value at every point of :math:`T`: with :math:`f = A(c)/B(c)` and :math:`c(\sigma) =
c(\theta)`, :math:`f(\sigma) = f(\theta)`. The count, the counts at the sample points and the
verified count therefore add the equation :math:`a(x)\, b(\theta) - b(x)\, a(\theta)` per such
function, of degree at most :math:`2 \max(\deg a, \deg b)` in :math:`(\theta, x)`, which only removes
points outside :math:`T`; a coordinate so proven becomes the linear equation :math:`x_i =
\theta_i` and is substituted. Every point of :math:`T` satisfies these equations, so Steps 1 to 3
hold for :math:`V` with them, and the smaller variety takes the :math:`p` pinned coordinates, each of
degree one, with :math:`n - p` output equations whose gradients are independent modulo the pinned
columns, :math:`D' = (1 + \deg s') \prod_i \deg(\mathrm{num}_i\, \mathrm{den}_i)` over these
:math:`n - p` equations.

*Unknown inputs.* The jets :math:`v` of the inputs that the output coefficients read (*Unknown
inputs in the count* below) are coordinates of the point, drawn uniform in :math:`\{1, \dots,
2^b\}` with :math:`\theta^\ast`, and :math:`K = \mathbb{Q}(\theta, \hat v)`; the jets above them are
zero in the system and at the point. The unknowns :math:`u` add these jets and the derivatives of
the chains, and :math:`D` takes the degrees in :math:`(\theta, \hat v, u)`, the series run on
degree bounds with the jets of the point as variables. Step 1 needs the Jacobian of the output
coefficients in :math:`(x, v)` of full column rank :math:`n + q`, :math:`q` the number of jets
read; it is checked at a random point modulo a prime, and a minor that is not zero there is not
zero over :math:`\mathbb{Q}`, so the rank holds at every point with algebraically independent
coordinates. A point :math:`(\sigma, v_\sigma)` with the output of :math:`(\theta, \hat v)` is
one: :math:`(\theta, \hat v)` is an isolated solution of :math:`b(\cdot) = b(\sigma, v_\sigma)`,
so algebraic over :math:`\mathbb{Q}(\sigma, v_\sigma)`, which therefore has transcendence degree
:math:`n + q`. The derivatives of the states and of the chains are local functions of :math:`(x,
v)` as before, the chain equations having the coefficient :math:`k + 1`. Steps 2 and 3 hold as
stated with :math:`\theta` replaced by :math:`(\theta, \hat v)`; the bound of
Equation :eq:`eq-count-bound` keeps the same form, since it does not depend on the number of
coordinates of the point. The smaller variety is not used, since the output equations of that variety are in
:math:`x` alone.

.. _several-experiments:

Several experiments
-------------------

The fiber of one experiment needs the Taylor coefficients up to an order
near the number of coordinates over the number of observables, and the degrees of the equations grow
with the order. Experiments that share the parameters need lower orders: :math:`K` copies of the
model with the parameters :math:`\theta` shared and starts :math:`x_0^1, \dots, x_0^K` of their own,
each with another draw of the known inputs. Let :math:`T(\theta, x_0)` be the fiber of one
experiment, :math:`T_K` the fiber of the :math:`K` experiments at :math:`(\theta, x_0^1, \dots,
x_0^K)`, and :math:`\pi` the projection onto :math:`(\theta', x_0'^1)`; always :math:`\pi(T_K)
\subseteq T(\theta, x_0^1)`. The route takes a model with one condition and no events, switches or
clock in the equations, every state with a free start: a coordinate that nothing else reads
:cite:p:`OvchinnikovPogudinThompson2023,Ovchinnikov2022`.

*Proposition 1.* If :math:`\dot x = f(x, \theta, u)` at a constant input :math:`u` has no
nonconstant rational first integral over :math:`\mathbb{Q}(\theta, u)`, then :math:`\pi(T_K) =
T(\theta, x_0^1)` at a generic point.

*Proof.* Let :math:`\kappa` be the jets of the known inputs at the start, :math:`\Theta'(\theta,
x_0, \kappa)` the finite set of the :math:`\theta'` of :math:`T(\theta, x_0)`, and :math:`e` the
coefficients of :math:`\prod_{\theta' \in \Theta'} (Z - \lambda(\theta'))` for a linear form
:math:`\lambda`. :math:`T` is cut out by finitely many output equations over :math:`\mathbb{Q}(
\theta, x_0, \kappa)` and is stable under the automorphisms of the algebraic closure over this
field, so :math:`e \in \mathbb{Q}(\theta, x_0, \kappa_0, \dots, \kappa_M)` for some :math:`M`. The
experiment that starts at time :math:`s` from :math:`x(s)`, with the jets :math:`\kappa(s)` of the
inputs there, has the output shifted by :math:`s`, for :math:`\theta` and for every :math:`\theta'
\in \Theta'`; so :math:`\Theta'` is constant along the flow of :math:`X = f \cdot \partial_x +
\sum_j \kappa_{j+1} \partial_{\kappa_j}`, and :math:`X e = 0`. In :math:`X e` only
:math:`\kappa_{M+1} \partial_{\kappa_M} e` reads :math:`\kappa_{M+1}`, so :math:`e` does not read
:math:`\kappa_M`, and down to :math:`\kappa_0`: :math:`f(x, \theta, \kappa_0) \cdot \partial_x e =
0` for every :math:`\kappa_0`, a rational first integral at a constant input, so :math:`e \in
\mathbb{Q}(\theta)`. :math:`\Theta'` therefore does not depend on the starts or the inputs: for
:math:`(\theta', x_0'^1) \in T(\theta, x_0^1)` each experiment :math:`k` has an :math:`x_0'^k` with
:math:`(\theta', x_0'^k) \in T(\theta, x_0^k)`, and these form a point of :math:`T_K` over
:math:`(\theta', x_0'^1)`. :math:`\square`

*Conservation laws.* Where linear forms :math:`\ell(x) = C(\theta)\, x` in the states, :math:`C` of
rank :math:`c` with entries rational in the parameters, satisfy :math:`C f = 0` identically in the
states and inputs, symident finds a basis of them over :math:`\mathbb{Q}(\theta)` and takes the
premise on the level sets: if the system restricted to :math:`\ell = \gamma` has no nonconstant
rational first integral over :math:`\mathbb{Q}(\theta, \gamma, u)`, every rational first integral
:math:`h` of :math:`f` is a function of :math:`\ell`, since :math:`h` restricted to a generic level set
is constant there. The pairs :math:`(\theta', \ell_{\theta'}(x_0'))` over :math:`T(\theta, x_0)` are
then, as in the proof, functions of :math:`(\theta, \ell_\theta(x_0))` alone, :math:`\ell_{\theta'}` being
constant along the flow at :math:`\theta'`; so the experiments start on the level set of the first:
per law, a pivot state of the reduced echelon form of :math:`C` starts at the value that makes
:math:`\ell(x_0^k) = \ell(x_0^1)`, a rational function of the coordinates, and Proposition 1 holds for the
fiber of these experiments.

The twisted oscillator :math:`\dot x_1 = (\omega + x_3) x_2`, :math:`\dot x_2 = -\omega x_1`,
:math:`\dot x_3 = 0`, :math:`h = (x_2, x_3)` of :cite:p:`OvchinnikovPogudinThompson2023` (Example 2.7)
shows that the premise is needed: :math:`x_3` is a first integral, one experiment leaves
:math:`\omega` and :math:`-(\omega + x_3)`, and two experiments with different :math:`x_3` leave
:math:`\omega` alone.

*The premise at one point.* A system :math:`\dot x = F(x)` analytic at an equilibrium :math:`x^\ast`
has at most as many functionally independent rational first integrals near :math:`x^\ast` as the
dimension of the span of :math:`\{k \in \mathbb{Z}^n \setminus 0 : \sum_i k_i \lambda_i = 0\}`,
:math:`\lambda` the eigenvalues of :math:`DF(x^\ast)` :cite:p:`CongLlibreZhang2011` (Theorem 1): none where
the eigenvalues are linearly independent over :math:`\mathbb{Z}`. The parameters (the constant
inputs and known constants among them) where :math:`F`, cleared of denominators, has a nonconstant
rational first integral :math:`A/B` with :math:`\deg A, \deg B \le d` form a closed set :math:`S_d`
defined over :math:`\mathbb{Q}`: :math:`A/B` is one exactly where :math:`A \wedge B \ne 0` and
:math:`B\, L A - A\, L B = 0`, :math:`L` the Lie derivative, a condition linear in :math:`A \wedge
B` and so closed on the Grassmannian of planes of polynomials of degree at most :math:`d`, whose
projection onto the parameters is closed since the Grassmannian is projective. A point
:math:`\theta^\ast`, rational or algebraic, outside every :math:`S_d` puts a generic point outside
every :math:`S_d`: the premise of Proposition 1 holds.

*The test.* symident takes the constant inputs at random values and the equilibrium first: the
states at random values away from zero, the parameters in products of parameters at random values
until the balances are linear in the others, and those solved over the field :math:`F` of the point
(:math:`\mathbb{Q}`, or :math:`\mathbb{Q}(\alpha)` below), in a random order of the parameters; a
point that forces a parameter to zero is not taken. Where the balances have no such solution, it
draws :math:`\theta^\ast`, solves :math:`f = 0` and :math:`\ell = \gamma^\ast` at random
:math:`\gamma^\ast` with msolve over :math:`\mathbb{Q}`, and takes each Galois orbit of the at most
120 equilibria as a number field :math:`F = \mathbb{Q}[s]/(W)`, :math:`W` monic over
:math:`\mathbb{Z}` of degree at most 12. :math:`J = Df(x^\ast)` over :math:`F` splits into diagonal
blocks, the strongly connected components of the nonzero entries. :math:`CJ = 0` gives :math:`J` the
eigenvalue 0 at least :math:`c` times, and the eigenvalues of :math:`J` on the tangent space of the
level set are the others. With :math:`\chi_j` the characteristic polynomial of block :math:`j`
without the factor :math:`z^{m_j}`, :math:`\sum_j m_j = c`, of degree :math:`n_j`, these eigenvalues
are linearly independent over :math:`\mathbb{Q}` if

(a) the Galois group of each :math:`\chi_j` over :math:`F` acts 2-transitively on the roots,

(b) the splitting fields :math:`L_j` of the blocks with :math:`n_j \ge 2` are pairwise distinct, and

(c) the traces :math:`t_j` of the blocks are linearly independent over :math:`\mathbb{Q}`.

*Proof.* The group :math:`G` of the compositum over :math:`F` permutes the eigenvalues, so the
:math:`F`-linear relations among them form a submodule :math:`R` of :math:`\bigoplus_j F^{n_j}`.
:math:`F^{n_j}` is the trivial module plus :math:`V_j`, the vectors of sum zero, which is absolutely
irreducible since the permutation character of a 2-transitive group is the trivial character plus an
irreducible one :cite:p:`Isaacs1976` (Chapter 5); :math:`G` acts on :math:`V_j` faithfully through
:math:`\mathrm{Gal}(L_j/F)`, so by (b) the :math:`V_j` are pairwise non-isomorphic. :math:`R` is the
sum of the intersections of :math:`R` with the isotypic parts. :math:`R \supseteq V_j` would make
two roots of :math:`\chi_j` equal, which is irreducible by (a) and so separable; so :math:`R` lies
in the span of the block sums, and a relation over :math:`\mathbb{Q}` is :math:`\sum_j c_j t_j = 0`
with :math:`c_j \in \mathbb{Q}`, which (c) excludes. :math:`\square` For one block this is a case of
:cite:p:`Girstmair1999`: with the Galois group :math:`S_n`, the only linear relation among the roots is a
vanishing trace.

(a) and (b) follow from Frobenius elements. At a prime :math:`\mathfrak{p} = (p, s - \tau)` of
degree one, with :math:`W` square-free modulo :math:`p` (so that :math:`\mathbb{Z}[s]` is maximal at
:math:`p`), the entries of :math:`J` integral at :math:`p` and :math:`\chi_j` square-free modulo
:math:`\mathfrak{p}`, the degrees of the irreducible factors of :math:`\chi_j` modulo
:math:`\mathfrak{p}` are the cycle type of a Frobenius element on the roots (Dedekind,
:cite:p:`Neukirch1999`, Chapter I). Reduction commutes with the characteristic polynomial, so it is taken
of :math:`J` modulo :math:`\mathfrak{p}`. The multiplicity of the eigenvalue 0 of a block over
:math:`F` is at most the one modulo :math:`\mathfrak{p}`, and the multiplicities add up to at least
:math:`c`; a prime where those modulo :math:`\mathfrak{p}` add up to :math:`c` gives the
:math:`m_j`. An :math:`n_j`-cycle makes the group transitive, an :math:`(n_j - 1)`-cycle then
2-transitive; a cycle type with one 2-cycle and odd cycles otherwise gives a transposition, and the
group is :math:`S_{n_j}` (``group``). Equal splitting fields would let one Frobenius element act on
both blocks, with the same order: a prime at which the orders differ proves (b). The primes are read
from 10007 on, at most 3000 prime ideals. Rational traces of two blocks are always dependent over
:math:`\mathbb{Q}`, and so are traces that an identity of the balances ties with a factor rational
in the states; where (c) fails with :math:`r` blocks, the point is drawn again in
:math:`\mathbb{Q}(\alpha)`, :math:`\alpha` a root of an irreducible polynomial of degree :math:`r`,
the states of a chosen equilibrium as well. The test refuses, with the reason, where the eigenvalue
0 is more frequent than the laws give (a state without dynamics is a law) or a single trace is 0,
the equilibria form a family, or the primes read do not show the cycles.
``info["discrete"]["experiments"]["first_integrals"]`` holds :math:`\theta^\ast`, :math:`F`, the
equilibrium and per block the primes of the cycles.

*The count.* For :math:`K = 2, \dots, \min(8, \lceil n_\theta / n_y \rceil + 1)`, :math:`n_\theta`
the coordinates other than the starts and :math:`n_y` the observables, the stacked model gets the
orders of the rank criterion; the :math:`K` with the lowest highest order is taken, the fewest
experiments among equal ones, and the orders rise as in the count of one experiment. The count
modulo :math:`p` is that of the distinct values of a random linear form in :math:`(\theta', x_0'^1)`
on the points of each block (the projection of a product is the product of the projections); a
coordinate outside the projection is substituted as a jet is, a coordinate inside only for a
monomial in the coordinates inside, and a count by stages adds the counts of the rest over the
points of a stage, at least the number of distinct projections.

*The verified count.* :eq:`eq-count-bound` holds for the :math:`K` experiments as one model, with
the projection: in Step 2, :math:`\lambda` is a linear form in :math:`(\theta', x_0'^1)` that
separates the projections of the at most :math:`D` points of :math:`V_0` over :math:`\mathbb{Q}(
\theta, x_0^1, \dots, x_0^K)`, and in Step 3 the values of :math:`\lambda` stay distinct at the
random point, so the solutions there have at least :math:`|\pi(T_K)|` distinct projections, with
:math:`D` and :math:`N_u` those of the stacked system. Over :math:`\mathbb{Q}` a block of at most 64
points counts the distinct values of the projected coordinates on the points of the rational
parametrization: with :math:`w` the square-free eliminating polynomial, :math:`\mathbb{Q}[t]/(w)` is
the algebra of functions on the points, and the dimension of the subalgebra the projected
coordinates generate, found by exact linear algebra, is the number of their distinct values; a
larger block counts all points. Either count bounds :math:`|\pi(T_K)|`, and by Proposition 1 it
bounds :math:`|T|`.

*The orbit.* The exact maps form a group :math:`G` that acts on :math:`T` without fixed points at a
generic point: an element that fixes a point of :math:`T` fixes a point with algebraically
independent coordinates and is the identity. So :math:`T` is a union of orbits of :math:`|G|`
points, :math:`|G|` divides :math:`|T|`, and a count below :math:`2|G|` proves :math:`|T| = |G|`:
``count_final``, ``complete: "verified"`` with ``count_bound`` where the count over
:math:`\mathbb{Q}` is below :math:`2|G|`, else ``"axiom"``. The certificate lists the route
(``route``: the number of experiments, the orders, the count and the premise) and adds the premise
and the divisibility to ``proven``. ``experiments`` of :meth:`~symident.Symmetries.discrete` takes
the route first, with ``True`` or a number of experiments; by default it runs where a short count of
one experiment finds none, after the exact count and before the full count. Where the count of one experiment fails too, the count of the experiments stands for it
without maps. A fixed start (a known initial value, a start at rest, :math:`\nu(0) = 1` of the
relaxed model, a start fixed by a reduction), events, conditions, switches or the clock in the
equations leave the route out, and ``info["discrete"]["experiments"]["why"]`` names the reason.

.. _monodromy:

Monodromy and certified points
------------------------------

Where the exact maps have an orbit smaller than the count, or no count exists, points of the fiber
are proposed numerically at a random rational point :math:`\theta^\ast`. The coefficients are
evaluated by the recursion of the series in complex floating point, with gradients in the
coordinates, never expanded. :math:`n` coefficients :math:`c_I` whose gradients are independent at a
random point modulo a prime, lowest order first, form a square system :math:`c_I(x) = w`. The
coordinates range over :math:`\mathbb{C}^n`, which is irreducible, so the monodromy group of
:math:`c_I` acts transitively on the fiber of :math:`c_I` over a value off the branch locus
:cite:p:`Duff2019`: from :math:`x = \theta^\ast` over :math:`w_0 = c_I(\theta^\ast)`, the known points are
tracked along the triangle :math:`w_0 \to w_1 \to w_2 \to w_0`, :math:`w_1, w_2` random complex
vectors, by an Euler predictor and a Newton corrector with step control, and the end points join
the known ones, until five triangles add none or as many points of the fiber as the count are
known. This runs where :math:`c_I` has at most 200 points modulo a prime at a random point, since
the fiber of :math:`c_I` holds the fiber of the model several times over; otherwise, or where it
stops short, the points are the roots of the rational parametrization of msolve over
:math:`\mathbb{Q}` at :math:`\theta^\ast`, taken numerically. A point is kept where every coefficient
up to the order of the count agrees with the one at :math:`\theta^\ast` to a relative
:math:`10^{-7}`. ``info["discrete"]["monodromy"]`` lists the number of points, the source, the
number of points of :math:`c_I` modulo the prime, the boxes below and the maps.

*Proposition* (Krawczyk :cite:p:`Krawczyk1969,Moore1977,Rump2010`). Let :math:`\phi: U \to \mathbb{C}^n`
be holomorphic, :math:`X \subset U` a product of closed complex rectangles :math:`X_i` with radii
:math:`r_i > 0`, :math:`m \in X`, :math:`Y` a complex :math:`n \times n` matrix and :math:`J(X)` a
matrix of rectangles that holds :math:`\partial \phi(\xi)` for every :math:`\xi \in X`. Let
:math:`E = I - Y J(X)` and :math:`K = m - Y \phi(m) + E\, (X - m)` in rectangle arithmetic. If
:math:`K_i` lies in the interior of :math:`X_i` and :math:`\sum_j |E_{ij}|\, r_j < r_i` for every
:math:`i`, with :math:`|E_{ij}|` the largest modulus on the rectangle, then :math:`\phi` has exactly one
zero in :math:`X`, and :math:`\partial \phi` is invertible there.

*Proof.* Let :math:`k(x) = x - Y \phi(x)`. For :math:`x, x' \in X` the segment between them lies in
the convex :math:`X`, and :math:`\phi_l(x) - \phi_l(x') = \int_0^1 \partial \phi_l(x' + s(x - x'))\, ds\,
(x - x')`; the integral lies in the closed convex row :math:`J_l(X)`. So :math:`k(x) - k(x') = (I -
Y \bar J)(x - x')` with every row of :math:`\bar J` in :math:`J(X)`, the entries of :math:`I - Y
\bar J` lie in those of :math:`E`, and in the norm :math:`\max_i |v_i| / r_i` the map :math:`k` is
a contraction on :math:`X`. With :math:`x' = m`, :math:`k(x) \in K \subset X`. By Banach's fixed
point theorem :math:`k` has exactly one fixed point :math:`x^\ast` in :math:`X`. The row sums give
:math:`\|I - Y \bar J\| < 1` for every such :math:`\bar J`, so :math:`Y \bar J` and :math:`Y` are
invertible: :math:`Y \phi(x^\ast) = 0` gives :math:`\phi(x^\ast) = 0`, every zero of :math:`\phi` in
:math:`X` is a fixed point of :math:`k`, and :math:`\partial \phi(x^\ast)` is invertible. :math:`\square`

For a kept point :math:`\hat x`, :math:`\phi = c_{I'} - c_{I'}(\theta^\ast)` with :math:`n` rows
:math:`I'` of the coefficients up to the order of the count, chosen by a QR decomposition with
pivoting of the Jacobian at :math:`\hat x` with rows of unit norm, :math:`m = \hat x`, :math:`Y` a
floating-point inverse of the Jacobian at :math:`\hat x` and radii :math:`10^{-10}` and then
:math:`10^{-7}` times :math:`1 + |\hat x_i|`. :math:`\phi(m)` and :math:`J(X)` are evaluated by the same
recursion in rectangle arithmetic: every operation in IEEE double precision is correctly rounded, so
the result moved by one floating-point number outward encloses the exact one, and the enclosures
hold the exact values for every point of the operands; :math:`\theta^\ast` and the constants enter
as enclosures of the rationals. Boxes with disjoint rectangles in some coordinate hold distinct
zeros, so the boxes kept, pairwise disjoint, prove that many simple zeros of :math:`\phi`
(``certified``). These are zeros of the square system: a point of the fiber needs every
coefficient equal, an identity that no enclosure shows, since an overdetermined system loses the
solutions under any perturbation. The certified boxes are therefore no lower bound of the count;
the lower bound is the orbit of the exact maps. Where the enclosures of the recursion grow past the
radii, the point stays without a box.

A map is read from a point :math:`\hat x \ne \theta^\ast`: each image coordinate is matched to
:math:`c\, \theta_j^{\pm 1}` or :math:`\theta_i + d` with Gaussian rationals :math:`c, d` of
denominators up to 12 that reproduce :math:`\hat x_i`, and a combination is kept where the
coefficients at two random points and their images agree to a relative :math:`10^{-7}`. Nothing is
concluded from these numbers: a map proposed so is ``exact`` once the identities of :ref:`maps`
hold over :math:`\mathbb{Q}(i)(x, \theta)`, and then joins the group :math:`G`. A map
over :math:`\mathbb{Q}(i)` whose image of a real point is not real never meets :math:`\Theta`. For
such maps the theorem of the exact count holds as stated: the proof uses that the orbit of
:math:`\theta` lies in :math:`T` and has :math:`|G|` points, that :math:`P_\alpha \in F[Z]`, whose
coefficients are checked to lie in :math:`\mathbb{Q}(\theta)`, and that :math:`L = F(\alpha)`.

.. _certificates:

Certificates
------------

``info["discrete"]["certificate"]`` states what the count rests on, and ``complete`` sums it up:

``"exact"``
   the field fixed by the exact maps lies in the field of the output coefficients
   (:ref:`exact-count`), so the fiber at a generic point is the orbit of the exact maps; no random
   step enters.
``"verified"``
   a count over :math:`\mathbb{Q}` at a random integer point equals the orbit, directly
   (:ref:`verified-count`) or for several experiments (:ref:`several-experiments`), with the
   probability ``count_bound`` that the fiber at a generic point has more points.
``"axiom"``
   the count holds under an axiom: that the count modulo the prime at a random point is the count
   over :math:`\mathbb{C}` at a generic point, or a premise of the rewrites (``premise``), such
   as the algebraic independence of the starts of new coordinates; ``count_bound_why`` names the
   reason.
``"no"``
   the count exceeds the orbit of the exact maps: further points exist at the sample point, and
   the maps between them are not proven.
``"open"``
   no count or no orbit was reached; ``limits`` names the work limits that stopped the analysis.

``count_final`` is true where the exact maps account for every point of the count. Each map has a
``level``: ``exact`` with a map of the states (:ref:`maps`), ``checked`` where the coefficients
agree at random points. ``preserves_domain`` and ``domain_conditions`` say whether the map keeps
:math:`\Theta`, and ``count_domain`` gives the number of points of the fiber in
:math:`\Theta` where it is exact. The rewritten model and the result in it are listed under
``rewrites``, ``relaxed`` and ``start_powers``.
