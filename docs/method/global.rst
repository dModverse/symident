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
locally identifiable, so the *fiber* :math:`\Psi_N^{-1}(\Psi_N(\theta^\ast))` is finite. It is
the solution set of the polynomial system that equates the Taylor coefficients of the
observables up to order :math:`N` with their values at :math:`\theta^\ast`, away from zeros of
coordinates and denominators. As in SIAN :cite:p:`Hong2020`, the derivatives of the states at the
start are unknowns of the system, tied by the Taylor coefficients of :math:`D(x)\,\dot x = N(x)`
and :math:`E(x)\, g = M(x)`. msolve :cite:p:`Berthomieu2021` counts the points over the algebraic
closure of :math:`\mathbb{F}_p`; the count is the number of distinct values of a random linear form
in the coordinates, so points that differ only in the derivatives of the states, or that are
multiple, count once. It bounds the size of the fiber of the model, and more equations can only
lower it. The orders rise until the count is the same at two successive steps or equals the orbit
of the proven maps, which no higher order changes.

The system splits into blocks without a shared variable, whose counts multiply. A block may be
counted in *stages*: with :math:`P_A` the equations that read only the variables :math:`A`, the count
is the sum over the points of :math:`P_A` of the counts of the rest. This holds for every
:math:`A`; the block triangular form of :cite:p:`Dulmage1958` proposes one.

.. _fiber-features:

Conditions, events, resting states and switches
-----------------------------------------------

The conditions are stacked as in the local analysis: each has separate derivatives, the
coordinates are shared. After an event the gaps are :math:`\varepsilon\, \Delta t_i^0` as in
:ref:`conditions-events`, and every coefficient after the event is a power series in
:math:`\varepsilon`, kept up to the gap order. Under ``equilibrate`` the start of a condition is
the rational solution of the balances over :math:`\mathbb{Q}(\theta)`, and none without one. With
switches (:ref:`switching`), each allowed regime enters as a copy whose start is free on a
switching surface :math:`\gamma_j = 0`, an equation of the fiber. A map is ``exact`` with a map of
the states in every regime that takes each surface to itself, :math:`\gamma_j(\varphi(x),
\sigma(\theta)) = c\, \gamma_j(x, \theta)` with :math:`c` free of the states and positive on
:math:`\Theta`. A switch on the parameters alone is not handled. A map reconstructed from the fiber
is checked on coefficients of higher order; a map that fails shows a fiber too large at that
order, and the orders rise.

.. _rational-form:

Rational form of the fiber
--------------------------

The fiber needs a model rational in the states and coordinates. Rewrites that are bijective on
the real values in question keep the fiber: a coordinate that enters only through :math:`b^{c\theta}`
or :math:`e^{c/\theta}` with rational :math:`c` is replaced by that power, an observable
:math:`h` with rational :math:`e^{c h}` by :math:`e^{c h}`, an odd power :math:`r^{p/q}` by
:math:`r`, a factor :math:`e^{a t}` by a state :math:`E` with :math:`\dot E = a E`, :math:`E(0) =
1`, and a positive state read only through :math:`\log x` and powers by :math:`L = \log x`. Maps
are reported in the original coordinates. ``info["discrete"]["rewrites"]`` lists the rewrites.

**Relaxed model.** A term :math:`\varphi(\psi)` of the states that remains, :math:`\varphi` an
exponential, a logarithm or a power :math:`b^\mu` with non-integer exponent and :math:`\psi`
rational, becomes :math:`W \nu` (:math:`W + \nu` for the logarithm), innermost first, with a new
state :math:`\nu` and a new coordinate :math:`W = \varphi(\psi_0)`, :math:`\psi_0` the value at the
start:

.. math::

   \nu = e^{\psi - \psi_0},\ \Big(\frac{b}{b_0}\Big)^{\mu},\ \log \frac{\psi}{\psi_0}, \qquad
   \dot\nu = \nu\, \dot\psi,\ \ \mu\, \nu\, \frac{\dot b}{b},\ \ \frac{\dot\psi}{\psi}, \qquad
   \nu(0) = 1,\ 1,\ 0 .

A term of the coordinates alone becomes a new coordinate :math:`P`, and a coordinate that no
equation reads afterwards is dropped where one :math:`P` determines it on the positive reals. The
*relaxed model* is rational in :math:`\tilde\theta = (\theta, W, P)`, and the output of the model
is the relaxed output on the graph :math:`\Gamma` of :math:`\theta \mapsto \omega(\theta)`, the
starts of the new coordinates. ``info["discrete"]["relaxed"]`` lists the terms, the starts and the
dropped coordinates.

*Proposition.* Suppose (a) every direction of the relaxed model that moves a :math:`W` is exact, a
scaling or a shift of the :math:`W` it moves by constants, and moves no coordinate read by a start
:math:`\omega`, with an invertible matrix of the weights and shifts on the :math:`W` these
directions move, (b) no direction moves a coordinate read by the start of a :math:`W` that no
scaling moves, and (c) these starts are algebraically independent over :math:`\mathbb{Q}(\theta)`.
Then at a generic point the fiber of the model has at most :math:`k` connected components,
:math:`k` the count of the relaxed model modulo the continuous directions, and exactly :math:`k`
where :math:`k = 1` or where the exact maps of the relaxed model that fix every :math:`W` and every
coordinate read by a start have an orbit of :math:`k` points.

*Proof.* By (c) a generic :math:`\theta` lies on a generic orbit of the relaxed model, and a point
:math:`\theta'` of the fiber gives the point :math:`(\theta', \omega(\theta'))` of the relaxed
fiber. On a relaxed orbit :math:`O`, the :math:`W` that no scaling moves and, by (b), what their
starts read are constant; by (a) a unique group element :math:`s(q)`, continuous in :math:`q`,
brings :math:`q \in O` to the starts of the moved :math:`W`. So :math:`O \cap \Gamma` is empty or
the connected image of :math:`O` under :math:`q \mapsto s(q)\, q`. A map that fixes the :math:`W`
and what the starts read takes :math:`\Gamma` to itself, so an orbit of :math:`k` points meets
:math:`k` relaxed orbits on :math:`\Gamma`. :math:`\square`

For exponentials alone, (c) holds where the Jacobian of the exponents :math:`\psi_0` in the
coordinates has full rank, by the lemma of Kolchin and Ostrowski in :ref:`non-rational`, and for
logarithms alone likewise with :math:`\mathrm{d} \log \psi_0`. Otherwise (c) is a premise, named in
``certificate["premise"]``, and the certificate is ``axiom``. Where (a) or (b) fails,
``certificate["relaxed_why"]`` names the direction, the count is not reported, and the certificate
is ``open``.

**Free powers of a state at 0.** A power :math:`x^{c_0 + c\alpha}` of a state that starts at 0,
with a positive coordinate :math:`\alpha`, has no start to relax. With :math:`k` and :math:`c_x` as
in :ref:`singular-starts`, :math:`x^\alpha = Z N`, :math:`Z = c_x^\alpha` and :math:`N = (x /
c_x)^\alpha` a series in :math:`t` and :math:`t^{k\alpha}` with coefficients polynomial in
:math:`\alpha`. :math:`Z` cancels, is absorbed by a scaling of the new coordinates, or becomes a new
coordinate under the proposition above.

**Lemma.** Let :math:`c_x` have one sign on the domain, and let :math:`d`, the coefficient of the
term :math:`t^{v_h + k\alpha}`, :math:`v_h` an integer, of least exponent among the terms of an observable :math:`h` that read
:math:`N`, have one sign on the domain. Then every point :math:`\theta'` of the domain with the
output of a generic :math:`\theta` has :math:`\alpha' = \alpha`.

*Proof.* The expansions converge, and :math:`e = v_h + k\alpha \notin \mathbb{Q}` for generic
:math:`\alpha`. If :math:`e' = v_h + k\alpha' \ne e`, the smaller exponent, or :math:`e` where
:math:`k\alpha'` is an integer, has a nonzero coefficient on one side only, which the independence
of the powers of :math:`t` (:ref:`singular-starts`) excludes. :math:`\square`

Where one sign is not shown, the condition is a premise and the count is ``axiom``. The fiber then
has the equation :math:`\alpha' = \alpha`, and the exact count is not taken
(``info["discrete"]["start_powers"]``).

**Resting states inside a term.** Under ``equilibrate`` a resting value inside a transcendental
term solves a transcendental balance. Such resting values :math:`R_H` become coordinates of a
*resting chart*, with the terms at rest as leaves :math:`W` and the other resting states and one
coordinate :math:`p_h` per state of :math:`H` solved from the balances where these are linear:

.. math::

   (\theta \setminus p,\ R_H) \mapsto \big(\theta,\ x^\ast\big), \qquad
   p = p(\theta \setminus p, R_H, W), \quad x^\ast_H = R_H, \quad x^\ast_N = x^\ast_N(\theta, R_H, W).

The chart is a bijection onto the points of the resting set where the coefficients do not vanish,
so the fiber in the chart is the fiber of the model; it is relaxed as above, and (c) follows from
the rank of the exponents. The counts in the domain do not hold the declared sign of :math:`p`
(``info["limits"]``).

**Unknown inputs.** An unknown input enters the fiber through the jets at the start, unknowns that
are no coordinates (:ref:`inputs`); points that differ only in the jets count once. An observable
of *relative degree* :math:`r` in the input :math:`w`, the least order whose Lie derivative reads a
jet of :math:`w`, has

.. math::

   L^{r + m} h = \partial_w (L^r h)\, w^{(m)} + (\text{terms in } w, \dots, w^{(m - 1)}),
   \qquad \partial_w L^r h \ne 0,

so the Taylor coefficient of order :math:`n` reads exactly the jets up to :math:`n - r`, and higher
jets are set to zero without changing the coordinates of the fiber. A map :math:`\sigma` is ``exact`` where a map :math:`\psi(x, \theta, w)` of the input
values, with :math:`\varphi` reading no input, satisfies

.. math::

   F(\varphi(x), \sigma(\theta), \psi) = D\varphi(x)\, F(x, \theta, w), \qquad
   h(\varphi(x), \sigma(\theta), \psi) = h(x, \theta, w), \qquad
   \varphi(x_0(\theta)) = x_0(\sigma(\theta))

over :math:`\mathbb{Q}(x, \theta, w)`; then :math:`(\varphi(x(t)), \psi(x(t), \theta, w(t)))`
solves the model at :math:`\sigma(\theta)` with the same output. The model has one condition, no
events, switches or ``equilibrate``.

.. _maps:

Maps between the points of the fiber
------------------------------------

*Permutations* of the coordinates are tested first: a permutation that keeps every Taylor
coefficient maps each coordinate onto one with the same set of coefficients that depend on it, which
limits the candidates.

Where the other points of the fiber are rational functions :math:`\sigma_j(\theta)`, they are
reconstructed without matching points across samples: for a coordinate :math:`\theta_i`, the
polynomial :math:`\prod_j (Z - \sigma_{j,i}(\theta))` is invariant under permutations of the
points, so the coefficients are rational in :math:`\theta`, interpolated from the rational
univariate representation of msolve, and a linear factor over :math:`\mathbb{Q}(\theta)` is a
rational :math:`\sigma_{j,i}`. A map is ``checked`` when the coefficients agree at random rational
points, and ``exact`` with a map :math:`\varphi` of the states such that

.. math::

   F(\varphi(x), \sigma(\theta)) = D\varphi(x)\, F(x, \theta), \qquad
   h(\varphi(x), \sigma(\theta)) = h(x, \theta), \qquad
   \varphi(x_0(\theta)) = x_0(\sigma(\theta))

over :math:`\mathbb{Q}(x, \theta)`, with :math:`\varphi_{\text{after}}(E(x, \theta)) =
E(\varphi_{\text{before}}(x), \sigma(\theta))` at each event. Candidates for :math:`\varphi` are
read from the initial values, from the observables and from affine maps on the states that
:math:`\sigma` reaches; the identities decide.

The ``exact`` maps generate a group :math:`G`; at a generic point the orbit is :math:`|G|` points
of the fiber. When :math:`|G|` equals the count, the count is final (``count_final``).
``complete`` is ``"exact"`` where the fiber is proven to be the orbit (:ref:`exact-count`),
``"verified"`` where a count over :math:`\mathbb{Q}` equals the orbit with the probability
``count_bound`` of more points, else ``"axiom"``: that the count modulo the prime at a random point
is the count over :math:`\mathbb{C}` at a generic point (``count_bound_why``).

A map with a positivity certificate for each image coordinate preserves :math:`\Theta`; a map
with an image coordinate negative on :math:`\Theta` never meets it; otherwise the conditions on
:math:`\theta` are listed. Where the fiber is the orbit and every element of :math:`G` either
preserves :math:`\Theta` or never meets it, ``count_domain`` is the exact number of points in
:math:`\Theta`. Keeping in each orbit the point least in the first coordinate a map moves gives a
*fundamental domain*, a set of inequalities such as :math:`k_{01} < k_{12}`. Otherwise msolve
isolates the real points at sample points; the number of points in :math:`\Theta` is constant only
between the hypersurfaces on which two points meet or a point crosses the boundary of
:math:`\Theta`, so it is reported per point and never taken as a global verdict.

.. _exact-count:

The exact count
---------------

Let :math:`L = \mathbb{Q}(\theta)`, :math:`G` the group of the ``exact`` maps acting on :math:`L`
by :math:`\sigma(f) = f \circ \sigma`, and :math:`c` the Taylor coefficients of the observables at
the start of each condition, with the known inputs at the values drawn for them. Let :math:`T` be
the fiber at a generic point :math:`\theta` for these inputs, the values :math:`\sigma` over
:math:`\overline{L}` with the output of :math:`\theta`.

*Theorem.* Let :math:`F = \mathbb{Q}(c)` and :math:`\alpha \in L` with :math:`P_\alpha(Z) =
\prod_{\sigma \in G} (Z - \sigma(\alpha)) \in F[Z]` and :math:`L = F(\alpha)`. Then :math:`T` is the
orbit of :math:`\theta` under :math:`G`, of :math:`|G|` points.

*Proof.* The coefficients of the output generate a field :math:`\mathbb{Q}(\Psi) \supseteq F`. A
point :math:`\sigma \in T` has algebraically independent coordinates, so :math:`\theta \mapsto
\sigma` defines an embedding of :math:`L` over :math:`\mathbb{Q}(\Psi)`, and every such embedding
gives a point of :math:`T`; so :math:`|T| = [L : \mathbb{Q}(\Psi)]`. The orbit lies in :math:`T`,
and :math:`|T| \le [L : F] = [F(\alpha) : F] \le |G|`. :math:`\square`

The test has no random step: random points only propose candidates, each checked exactly.
Membership in :math:`F` is proven as a tower: :math:`f = A(w)/B(w)` lies in :math:`F` when
:math:`w` are output coefficients or functions already shown in :math:`F`, by the identity
:math:`A(w(\theta))\, \mathrm{den}_f(\theta) = \mathrm{num}_f(\theta)\, B(w(\theta))` with
:math:`B(w(\theta)) \ne 0`, checked as polynomials in :math:`\theta`. The functions shown in
:math:`F` are:

- *Coefficients of input-output relations.* A vector :math:`\lambda(\theta)` with :math:`\sum_j
  \lambda_j \phi_j(g, u) = 0` along every trajectory, :math:`\phi_j` monomials in the derivatives
  of the observables and the known inputs, lies in the kernel of the matrix :math:`M(c)` of the
  Taylor coefficients of the :math:`\phi_j`. The relation is checked exactly as a polynomial
  identity in states, inputs and coordinates. Where the rank of :math:`M` at one point is :math:`r
  - s` and :math:`s` such vectors are proven, they are the basis of the kernel over
  :math:`\mathbb{Q}(\theta)`, and the entries lie in :math:`F` by Cramer's rule.
- *Coefficients of projections.* The *projection* of an observable :math:`g_i` is the polynomial
  relation of :math:`g_i^{(h_i)}` with the algebraically independent derivatives of all observables
  below their orders :math:`h_r`, the generator of the ideal of these relations :cite:p:`Dong2023`.
  :cite:p:`Dong2023` computes projections by resultants over :math:`\mathbb{Q}(\theta)`; here the
  support is proposed by kernels at random points modulo a prime, bounded by counts of points on the
  Newton polytope, and the coefficients are interpolated as rational functions, sparse after
  :cite:p:`BenOr1988`. Every proposing step can only make the search fail; the identity check
  decides, and the coefficients lie in :math:`F` as in the item before.
- *Coefficients of relations by elimination.* The states are eliminated as in
  :cite:p:`Hong2020`, one at a time by resultants over :math:`\mathbb{Z}[\theta]`. Every polynomial
  of this process vanishes along every trajectory, and of the irreducible factors one does, since
  the power series along a trajectory form a domain and the closed sets where a given factor
  vanishes cover the parameter space; the product of the factors that vanish at three random
  trajectory points therefore vanishes along every trajectory, without a random step. A remaining
  equation of degree one in a removed state gives the start of that state in :math:`F`.
- the coordinates that :math:`G` fixes, the coefficients of :math:`P_\alpha` and traces over
  :math:`G`, each shown as a rational function of the functions before.

:math:`\alpha` is a moved coordinate with :math:`|G|` distinct images, or a combination of a few.
:math:`L = F(\alpha)` is shown coordinate by coordinate, each as a rational function of functions in
:math:`F`, of :math:`\alpha` and of the coordinates shown before, the start of a state also by a
relation of degree one in that start along every trajectory. The search ends at the work limits
``search_entries``, ``msolve_entries`` and ``elimination_terms``; past a limit the certificate names
what was not reached. Constants of the model that are no coordinates are known symbols
:math:`\kappa`: :math:`\mathbb{Q}` is replaced by :math:`\mathbb{Q}(\kappa)`, and the theorem holds
for generic values of the constants. ``info["discrete"]["certificate"]["exact"]`` lists the
functions, the relations and :math:`\alpha`.

**Pinned count.** Where the search stops before :math:`L = F(\alpha)` is shown, the functions
:math:`f_1, \dots, f_m` proven to lie in :math:`F` are counted instead. Every point of :math:`T`
has the values :math:`f_i(\theta)`, so :math:`T` lies in the fiber of :math:`\theta \mapsto
(f(\theta), c_0(\theta), \dots, c_N(\theta))`, at orders lower than those of the outputs alone.
:math:`|G|` divides :math:`|T|` (*The orbit* in :ref:`several-experiments`), so a count :math:`k`
over :math:`\mathbb{Q}` with :math:`|G| \le k < 2|G|` gives :math:`|T| = |G|`, with the bound of
:ref:`verified-count`. ``certificate["pinned"]`` lists the functions, the orders, the counts and
the bound.

**Coefficients of a large projection, verified.** Where the elimination gives no relation, a few
coefficients of a projection join the :math:`f_i` with an error bound. Let :math:`S` be the support
of :math:`r` monomials :math:`\phi_j`. (a) The Taylor coefficients of the :math:`\phi_j` have rank
:math:`r - 1` at a random point modulo a prime, so at most one relation on :math:`S` holds, with
coefficients in :math:`F`.

(b) The relation exists, verified. A trial takes :math:`r + 6` random states and a random
:math:`\theta` modulo a prime :math:`q` uniform among the primes from :math:`2^{29}` to the largest
engine prime. Without a relation, the determinant of :math:`r` rows of the cleared values
:math:`P_j` is a nonzero polynomial of degree at most :math:`D = \sum_j \deg P_j` with content at
most :math:`H`, :math:`\log_2 H \le \log_2 r! + \sum_j \log_2 \|P_j\|_1`. With :math:`N` the number
of primes in the range, bounded by :math:`x / \ln x < \pi(x) < 1.25506\, x / \ln x`
:cite:p:`Rosser1962`, and :math:`a \ge 1 - (r + 6) \deg \prod_k F_k / (q - 1)` the probability that
no denominator vanishes, a trial shows a kernel with probability at most

.. math::

   \varepsilon = \frac{D}{(2^{29} - 1)\, a} + \frac{\lceil \log_2 H / 29 \rceil}{N}

by :cite:p:`Schwartz1980`, and :math:`k = 5` independent trials with probability at most
:math:`\varepsilon^k`.

(c) A coefficient :math:`w` interpolated as a Laurent polynomial over a monomial coefficient
:math:`c_{j_0}` passes every trial; a wrong one would make a nonzero cofactor polynomial vanish, so
:math:`k` trials pass with probability at most :math:`\varepsilon_w^k`, :math:`\varepsilon_w` as
:math:`\varepsilon` with the degree and height of :math:`w` added.

The bound of the count is that over :math:`\mathbb{Q}` (``count_bound_q``) plus
:math:`\varepsilon^k` and :math:`\varepsilon_w^k` per function counted (``relation_bound``).

.. _verified-count:

The verified count
------------------

Where the exact count is not found, the count is taken again over :math:`\mathbb{Q}`, by the
rational parametrization of msolve, at an integer point :math:`\theta^\ast` uniform in
:math:`\{1, \dots, 2^b\}^n`. A count :math:`k` there bounds the fiber:

.. math::
   :label: eq-count-bound

   \Pr\big[|T| > k\big] \le \frac{(N_u + 2D)\, D}{2^b},

with :math:`N_u` the number of unknowns and :math:`D` the Bezout number below; :math:`b \ge 64` is
chosen so that the bound is at most :math:`10^{-6}` (``count_bits``, ``bezout``). The bound holds
for a model without events and switches, with every start a coordinate, a number, a rational
function of the coordinates or a jet of an unknown input, at orders no lower than those of the
rank criterion; otherwise the count stays under the axiom of :ref:`certificates`, and
``count_bound_why`` names the reason. msolve is taken to be correct.

*The system.* The unknowns :math:`u` are the coordinates, the derivatives of the states at the
start and the variable :math:`w` of the saturation :math:`w\, s(u) = 1`, :math:`s` the product of
the coordinates and the denominators at the start. Cleared over one denominator, the equations
:math:`F_1, \dots, F_s \in \mathbb{Q}[\theta, u]` define :math:`V` of degree at most :math:`D =
\prod_i \deg F_i` by the Bezout inequality :cite:p:`Heintz1983`.

*Step 1: the points of T are isolated.* For :math:`\sigma \in T` the Jacobian of :math:`F` in
:math:`u` has full column rank at the point of :math:`\sigma`: the equations of the derivatives are
triangular with nonzero diagonal, and the output equations have the Jacobian :math:`-E_0 \cdot
\partial_x \Psi_N` up to a triangular factor, of rank :math:`n` at every point with algebraically
independent coordinates by the rank criterion. So the point lies on the union :math:`V_0` of the
components of :math:`V` that dominate the coordinates, at most :math:`D` points over
:math:`\theta`.

*Step 2: polynomials of bounded degree.* The image of :math:`V_0` under :math:`(\theta, u) \mapsto
(\theta, u_j)` is a hypersurface :math:`V(E_j)` of degree at most :math:`D`, and so is that under a
separating linear form :math:`\lambda(x)`, square-free over :math:`\mathbb{Q}(\theta)`, with
discriminant :math:`R_\lambda \in \mathbb{Q}[\theta]` nonzero of degree at most :math:`(2D - 1) D`.

*Step 3: specialization.* Let :math:`\Delta = a_\lambda R_\lambda \prod_j a_j` with the leading
coefficients :math:`a`, nonzero of degree at most :math:`(N_u + 2D) D`; by :cite:p:`Schwartz1980`
:math:`\Delta(\theta^\ast) = 0` with probability at most the right side of :eq:`eq-count-bound`.
Otherwise, by Chevalley's extension theorem, every point of :math:`V_0` specializes to a solution at
:math:`\theta^\ast` away from the saturation, and the values of :math:`\lambda` stay distinct, so
the system solved has at least :math:`|T|` points with distinct :math:`x`, and the count of msolve
is at least :math:`|T|`. A count by stages adds exact counts over rational points and gives the
same bound.

*A smaller variety.* The same steps hold for the variety in :math:`(\theta, x, w)` cut out by
:math:`n` output equations whose gradients in :math:`x` are independent at a random point and by
the saturation, with :math:`D' = (1 + \deg s') \prod_i \deg(\mathrm{num}_i\, \mathrm{den}_i)` and
:math:`n + 1` unknowns; the smaller of the two bounds is used.

*Pins.* A function :math:`f = a/b` proven to lie in :math:`F` takes the same value at every point
of :math:`T`, so the equation :math:`a(x)\, b(\theta) - b(x)\, a(\theta)` removes points outside
:math:`T` only, and Steps 1 to 3 hold with it.

*Unknown inputs.* The jets :math:`v` that the output coefficients read are drawn with
:math:`\theta^\ast`, and :math:`\theta` is replaced by :math:`(\theta, \hat v)`. Step 1 needs the
Jacobian of the output coefficients in :math:`(x, v)` of full column rank, checked at a random
point. Equation :eq:`eq-count-bound` keeps the same form; the smaller variety is not used.

.. _several-experiments:

Several experiments
-------------------

The fiber of one experiment needs Taylor coefficients up to an order near the number of
coordinates over the number of observables, and the degrees grow with the order. :math:`K`
experiments that share :math:`\theta` and have starts :math:`x_0^1, \dots, x_0^K` of their own
need lower orders :cite:p:`OvchinnikovPogudinThompson2023,Ovchinnikov2022`. Let :math:`T_K` be the
fiber of the :math:`K` experiments and :math:`\pi` the projection onto :math:`(\theta',
x_0'^1)`; always :math:`\pi(T_K) \subseteq T(\theta, x_0^1)`. The route takes a model with one
condition, no events, switches or clock in the equations, and every start a free coordinate.

*Proposition 1.* If :math:`\dot x = f(x, \theta, u)` at a constant input :math:`u` has no
nonconstant rational first integral over :math:`\mathbb{Q}(\theta, u)`, then :math:`\pi(T_K) =
T(\theta, x_0^1)` at a generic point.

*Proof.* The coefficients :math:`e` of :math:`\prod_{\theta'} (Z - \lambda(\theta'))` over the
:math:`\theta'` of :math:`T(\theta, x_0)` are rational in :math:`\theta`, :math:`x_0` and the input
jets :math:`\kappa_0, \dots, \kappa_M`. Starting the experiment later along the trajectory shifts
the output for :math:`\theta` and every :math:`\theta'`, so :math:`X e = 0` for :math:`X = f \cdot
\partial_x + \sum_j \kappa_{j+1} \partial_{\kappa_j}`. Only :math:`\kappa_{M+1}
\partial_{\kappa_M} e` reads :math:`\kappa_{M+1}`, so :math:`e` reads no jet, and :math:`f \cdot
\partial_x e = 0` makes :math:`e` a first integral, hence :math:`e \in \mathbb{Q}(\theta)`. The set
of :math:`\theta'` therefore does not depend on the start, and each :math:`(\theta', x_0'^1)` extends
to a point of :math:`T_K`. :math:`\square`

The premise is needed: the twisted oscillator of :cite:p:`OvchinnikovPogudinThompson2023`
(Example 2.7) has a first integral and a second point for one experiment only. Under linear
conservation laws :math:`C(\theta) f = 0` the premise is taken on the level sets, and the
experiments start on the level set of the first.

*The premise at one point.* A system analytic at an equilibrium :math:`x^\ast` has at most as many
functionally independent rational first integrals near :math:`x^\ast` as the dimension of the span
of the integer relations :math:`\sum_i k_i \lambda_i = 0` among the eigenvalues of
:math:`DF(x^\ast)` :cite:p:`CongLlibreZhang2011`. The parameters with a rational first integral of
degree at most :math:`d` form a closed set :math:`S_d` defined over :math:`\mathbb{Q}`, by a
condition linear on a Grassmannian, which is projective. A point outside every :math:`S_d` puts a
generic point outside every :math:`S_d`: the premise holds.

*The test.* symident draws an equilibrium over a number field :math:`F` and takes :math:`J =
Df(x^\ast)` in diagonal blocks. With :math:`\chi_j` the characteristic polynomial of block
:math:`j` without the eigenvalue 0 of the laws, the eigenvalues on the level set are linearly
independent over :math:`\mathbb{Q}` if

(a) the Galois group of each :math:`\chi_j` over :math:`F` acts 2-transitively on the roots,

(b) the splitting fields of the blocks of degree at least 2 are pairwise distinct, and

(c) the traces :math:`t_j` of the blocks are linearly independent over :math:`\mathbb{Q}`.

*Proof.* The :math:`F`-linear relations among the eigenvalues form a module for the Galois group of
the compositum. By (a) each block splits into the trivial module and an absolutely irreducible
:math:`V_j` :cite:p:`Isaacs1976`, pairwise non-isomorphic by (b); :math:`V_j` in the relations would
make two roots equal. So a relation lies in the span of the block sums, :math:`\sum_j c_j t_j = 0`,
which (c) excludes. :math:`\square` For one block this is a case of :cite:p:`Girstmair1999`.

(a) and (b) follow from Frobenius elements: at a prime of degree one where :math:`\chi_j` is
square-free, the degrees of the factors modulo the prime are the cycle type of a Frobenius element
(Dedekind, :cite:p:`Neukirch1999`). Where the test cannot decide, it refuses with the reason;
``info["discrete"]["experiments"]["first_integrals"]`` holds the point, the equilibrium and the
primes.

*The count.* Equation :eq:`eq-count-bound` holds for the stacked model with the projection:
:math:`\lambda` in Step 2 separates the projections of the points of :math:`V_0`, so the solutions
at the random point have at least :math:`|\pi(T_K)|` distinct projections, which by Proposition 1
bounds :math:`|T|`.

*The orbit.* An element of :math:`G` that fixes a point of :math:`T` fixes a point with
algebraically independent coordinates and is the identity. So :math:`T` is a union of orbits of
:math:`|G|` points, :math:`|G|` divides :math:`|T|`, and a count below :math:`2|G|` proves
:math:`|T| = |G|`: ``count_final``, ``complete: "verified"`` with ``count_bound`` where the count
over :math:`\mathbb{Q}` is below :math:`2|G|`, else ``"axiom"``. The certificate adds the premise and the
divisibility to ``proven``; ``experiments`` of :meth:`~symident.Symmetries.discrete` selects the
route.

.. _monodromy:

Monodromy and certified points
------------------------------

Where the exact maps have an orbit smaller than the count, points of the fiber are proposed
numerically at a random rational point :math:`\theta^\ast`. :math:`n` coefficients :math:`c_I` with
independent gradients form a square system :math:`c_I(x) = w`; :math:`\mathbb{C}^n` is irreducible,
so the monodromy group acts transitively on the fiber of :math:`c_I` over a value off the branch
locus :cite:p:`Duff2019`, and tracking the known points around random loops in :math:`w` finds the
others. A point is kept where every coefficient up to the order of the count agrees
(``info["discrete"]["monodromy"]``).

*Proposition* (Krawczyk :cite:p:`Krawczyk1969,Moore1977,Rump2010`). Let :math:`\phi: U \to \mathbb{C}^n`
be holomorphic, :math:`X \subset U` a product of closed complex rectangles :math:`X_i` with radii
:math:`r_i > 0`, :math:`m \in X`, :math:`Y` a complex :math:`n \times n` matrix and :math:`J(X)` a
matrix of rectangles that holds :math:`\partial \phi(\xi)` for every :math:`\xi \in X`. Let
:math:`E = I - Y J(X)` and :math:`K = m - Y \phi(m) + E\, (X - m)` in rectangle arithmetic. If
:math:`K_i` lies in the interior of :math:`X_i` and :math:`\sum_j |E_{ij}|\, r_j < r_i` for every
:math:`i`, then :math:`\phi` has exactly one zero in :math:`X`, and :math:`\partial \phi` is
invertible there.

*Proof.* :math:`k(x) = x - Y \phi(x)` maps :math:`X` into :math:`K \subset X` by the mean value
theorem and is a contraction in the norm :math:`\max_i |v_i| / r_i`, so it has one fixed point; the
row sums make :math:`Y` invertible, so the fixed points of :math:`k` are the zeros of :math:`\phi`.
:math:`\square`

The enclosures are evaluated in rectangle arithmetic rounded outward, so disjoint boxes prove that
many simple zeros of the square system (``certified``). A point of the fiber needs every
coefficient equal, an overdetermined identity that no enclosure shows, so the certified boxes are no
lower bound of the count; the lower bound is the orbit of the exact maps. A map read from a point
is ``exact`` only once the identities of :ref:`maps` hold over :math:`\mathbb{Q}(i)(x, \theta)`.

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
