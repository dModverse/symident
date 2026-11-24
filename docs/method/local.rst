Local analysis
==============

The local analysis decides the rank of the observability-identifiability matrix (OIM) and returns
the kernel, the non-identifiable directions at generic coordinates. The OIM is evaluated over
finite fields :math:`\mathbb{F}_p`; the order is certified sufficient where a proven upper bound
of the rank is reached.

.. _oim:

Lie derivatives and the observability-identifiability matrix
------------------------------------------------------------

The *Lie derivatives* of the output are

.. math::
   :label: eq-lie

   L_F^0 h = h, \qquad L_F^{k+1} h = F\big(L_F^k h\big).

For analytic models they determine the output: near :math:`t_0` it is the Lie series

.. math::
   :label: eq-lie-series

   g(t; \theta) = \sum_{k \ge 0} \frac{(t - t_0)^k}{k!}\, L_F^k h\big(z_0(\theta)\big),

and elsewhere the analytic continuation of this series, so equal Lie derivatives at :math:`t_0` give equal
outputs :cite:p:`Pohjanpalo1978`. The Lie derivatives up to the *Lie order* :math:`N`,
differentiated with respect to :math:`\theta`,

.. math::
   :label: eq-oim

   \Psi_N(\theta) = \big(h, L_F^1 h, \dots, L_F^N h\big)\big(z_0(\theta)\big), \qquad
   \mathrm{OIM}_N(\theta) = \frac{\partial \Psi_N}{\partial \theta},

give the *observability-identifiability matrix* (OIM) :cite:p:`Villaverde2016`. The *rows of
order* :math:`k` are the gradients :math:`\partial_\theta\, L_F^k h_i(z_0(\theta))`,
:math:`i = 1, \dots, n_y`. A generator leaves :math:`\Psi_N` unchanged, so the value of the generator lies in
:math:`\ker \mathrm{OIM}_N(\theta)` for every :math:`N`. Conversely, where the rank of
:math:`\mathrm{OIM}_N` is the largest over all orders and all :math:`\theta`, which holds for
generic :math:`\theta`, every vector of the kernel is the value of a generator: the kernels
have constant dimension there, so a smooth vector field with values in them extends any
kernel vector, and along the flow of that field every Lie derivative, and by Equation :eq:`eq-lie-series`
the output, stays constant. The model is therefore locally structurally identifiable when
:math:`\mathrm{OIM}_N` has rank :math:`d` for some :math:`N`, and the *non-identifiable
directions* are the vectors of the kernel.

For the example of Equation :eq:`eq-example` the first Lie derivatives are

.. math::

   L_F^0 h = s B, \qquad L_F^1 h = s\,(k_1 A - k_2 B), \qquad
   L_F^2 h = s\,\big(k_2^2 B - k_1 (k_1 + k_2) A\big),

evaluated at :math:`(A, B) = (A_0, B_0)`. The rows of :math:`\mathrm{OIM}_N` reach rank four,
and the kernel is spanned by the value of the generator above.

.. _finite-fields:

Ranks over finite fields
------------------------

symident evaluates :math:`\mathrm{OIM}_N` exactly over :math:`\mathbb{F}_p` in a compiled
*modular engine*: the Lie derivatives as power series in time truncated at order :math:`N`,
the derivatives with respect to :math:`\theta` as dual numbers, all rows at a point in one pass
:cite:p:`Sedoglavic2002`. A point can only underestimate the generic rank, and a random point does so
with small probability :cite:p:`Schwartz1980,Zippel1979`. The rank of the result is taken at the
base point, as the largest over the four engine primes. Randomness enters through the points
uniform modulo the verification prime: one point that checks the rank of the *saturation*,
i.e., of the order chosen, and the points that check closed forms.

.. _sufficient-lie-order:

Sufficient Lie order
--------------------

The Lie order :math:`N` has to stop, and a rank that stays constant for a few orders is not
final at a special start, such as an initial value fixed to a number, a resting state or the
state after a dose. For :math:`\dot x = \theta`, :math:`h = x^m` and :math:`x_0 = 0`, every
row below order :math:`m` vanishes, and only the row of order :math:`m` shows that
:math:`\theta` is identifiable. symident therefore stops at a proven upper bound of the rank,
the *ceiling* below :cite:p:`HermannKrener1977`.

Let :math:`D_k` be the matrix of the gradients :math:`\partial_z L_F^j h_i`, :math:`j \le k`,
:math:`i \le n_y`, and :math:`S` an irreducible variety of points :math:`z` that is invariant
under :math:`F`. By the chain rule,
:math:`\mathrm{OIM}_k(\theta) = D_k\big(z_0(\theta)\big)\, \partial_\theta z_0(\theta)`.

**Lemma.** At a generic point of :math:`S`, once :math:`\operatorname{rank} D_{k+1} =
\operatorname{rank} D_k` for one :math:`k`, the rank stays the same for every later order.

*Proof.* Let :math:`\Omega_k` be the span of the rows of :math:`D_k` over the rational
functions on :math:`S`. A row is a one-form :math:`\omega = \sum_i \omega_i\, \mathrm{d}z_i`,
and the Lie derivative of :math:`\omega` along :math:`F` is
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
matrix rational in :math:`\theta`, so at generic :math:`\theta` it is at least the value at
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
resting variety of the condition (:ref:`steady-states`), on which :math:`f = 0`; a generic point of it
is a random coordinate value with the resting state solved there (the first of 24 draws with a
root). In both cases the rank at the start is taken at the base point. Where the start does
not reach :math:`\rho_S`, e.g., on the level set of a nonlinear conserved quantity, no order
is proven sufficient. The order then stops after a *plateau*, three orders without change of
rank (``SYMIDENT_LIEPLATEAU``), and the result says that the Lie order is not proven
sufficient.

**Conditions and events.** The rows of several conditions are stacked, and their row space
stops changing once that of every condition has stopped, so every condition is expanded to
a separate order. An event moves the start of the next segment: a state replaced by a value of
the coordinates is removed from :math:`U`, a value added keeps :math:`U`, and an event that
multiplies a state or reads the states adds that state's direction to :math:`U`. The
proposition extends to the power series of :ref:`conditions-events` and is checked at a
vanishing gap.

**Random points.** :math:`U` and :math:`\rho_S` are taken at points uniform modulo the
verification prime :math:`p`. They are wrong only where a polynomial vanishes: the state part
of :math:`F` at the step that ends :math:`U`, or a minor of size :math:`\rho_S` of the rows of
an order, both composed with the start. Since the generic ranks rise strictly up to their final
value, orders up to the number of columns plus one reveal a wrong plateau. A point is the first of
the draws without a pole of the starts, up to 8, or on a resting variety the first with a root, up
to 24. The probability of a wrong certificate is bounded in :ref:`order-bound`.

**Without a stopping rule.** A direction that is an exact symmetry, i.e., a scaling under
which every expression is homogeneous (:ref:`scalings`) or a field with the identities of
:ref:`proofs-over-q`, lies in the kernel at every order, so a rank that together with such
independent directions gives :math:`d` is proven for all orders. ``engine="symbolic"`` computes
the Lie derivatives over the rational functions and confirms the order by an exact rank:
where the rows at the initial values reach the generic rank, a nonvanishing minor makes every
later row a combination of earlier ones. Where no order is proven sufficient, a *saturation
guard* (``verify=True``) takes the rank at the base point again six orders further
(``SYMIDENT_VERIFY_MARGIN``) and the gap order of :ref:`conditions-events` two orders further.
If the rank grows, the detection is repeated from that order, up to three times; if it still
grows, the rank is a lower bound, ``info["verification"]`` says where, and a reduction is
refused.

.. _order-bound:

Error probability of the order
------------------------------

The random choices are the verification prime :math:`p`, drawn per detection from the seed of the detection, and
points uniform modulo :math:`p`, each from a separate seed. All error bounds of symident rest on
the Schwartz-Zippel lemma :cite:p:`Schwartz1980,Zippel1979`: a nonzero polynomial of degree
:math:`D` in :math:`m` variables over a field vanishes at a point drawn uniformly from
:math:`A^m`, for a finite subset :math:`A` of the field, with probability at most :math:`D / |A|`.

A wrong certificate of the Lie or gap order needs a polynomial to vanish at one of the random
points of :ref:`sufficient-lie-order` or :ref:`conditions-events`: the step that ends the
subspace :math:`U`, or a minor of the rows of an order. The degree of the polynomial is that of the rational system
the modular engine evaluates, with exponentials, free powers and trigonometric functions as
separate states, multiplied by the degree :math:`\deg z_0` of the start, and on a resting variety
by the Bezout number of the balances, Equation :eq:`eq-rest-bound`. The probabilities add up to
``info["order_bound"]``, each multiplied by the number of points drawn until the point was taken,
since the point taken is wrong only if one of them is; ``info["order_draws"]`` is the largest such
number, and the points of the gap order are drawn once. ``info["order_bound_why"]`` names the
reason for a missing bound.

.. _non-rational:

Non-rational terms
------------------

The modular engine needs rational functions. Elementary functions are therefore recast into
rational form :cite:p:`Savageau1987`, type by type in a fixed order and innermost first within a type;
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

Under ``equilibrate`` a power :math:`b^\mu` of a positive rational number :math:`b` is
rewritten first: next to a power :math:`c^\mu` of a symbol
:math:`c` with the same exponent, :math:`c` stands for :math:`b c` throughout,
:math:`(b c)^\mu` being a power of one symbol, and the result is mapped back. Otherwise
:math:`b` is a constant with :math:`\log b` taken independent of the other constants, as for
logarithms of numbers in log coordinates (named in ``info["limits"]``).

A new state starts at :math:`\varphi(\psi)` of the initial state. Where this depends on the
coordinates, a transcendental :math:`\varphi` becomes a separate input of the modular engine,
a *leaf*, drawn independently of the argument and tied to it by relation rows
:math:`\mathrm{d}\nu = \varphi'(\psi)\, \mathrm{d}\psi`: a transcendental graph lies in no proper
variety, so a random point on it is generic. With several transcendental constants, such as
:math:`e` and :math:`\pi`, this needs their algebraic independence, which follows from
Schanuel's conjecture and is not proven. An algebraic :math:`\varphi`, a root or rational power,
is exact: through :math:`L = \log \psi` when :math:`\psi` is affine in one coordinate, else by
the root modulo the prime at each sample point, over primes in which every root of the model
exists. Logarithms, fractional powers and bases of free powers are real for a positive argument.
The rank and the kernel are taken at generic points, where every argument is nonzero, so the
detection needs no sign. An argument that the declared signs do not make positive on
:math:`\Theta`, also in initial and event values, is named in ``info["limits"]``. An observable
:math:`\kappa \sum_i a_i \log \rho_i + \beta`, with rational :math:`\rho_i` and constants
:math:`\kappa, a_i, \beta`, is strictly monotone in :math:`\prod_i \rho_i^{a_i} e^{\beta/\kappa}`
and is analyzed as that product.

An event after the start reads the states just before it. There the state propagated across the
gaps is a power series in the gap lengths whose constant term :math:`\bar x` is the state at zero
gaps, the initial state with the earlier events of the condition applied. A transcendental
:math:`\varphi(\psi(x))` in an event value is the leaf :math:`\varphi(\bar\psi)`,
:math:`\bar\psi = \psi(\bar x)`, times or plus a series in :math:`\psi(x) - \bar\psi`,

.. math::

   e^{\psi} = e^{\bar\psi}\, e^{\psi - \bar\psi}, \qquad
   \log \psi = \log \bar\psi + \log\Bigl(1 + \frac{\psi - \bar\psi}{\bar\psi}\Bigr), \qquad
   \psi^\alpha = e^{\alpha \log \psi}, \qquad
   \sin \psi = \sin \bar\psi \cos(\psi - \bar\psi) + \cos \bar\psi \sin(\psi - \bar\psi),

and since :math:`\psi(x) - \bar\psi` has no constant term, each series ends at the truncation
order; :math:`\cos` and :math:`\tan` follow from the addition theorems. An event on a state
moves the new states of the recast with it, :math:`\nu` to
:math:`\varphi` of :math:`\psi` after the event, so a dose added to a state under a free power
moves :math:`B` and :math:`L_b`, and in log coordinates :math:`L = \log x` becomes
:math:`\log(e^{L} + d)`. Under ``equilibrate`` :math:`\bar x` is the resting state, a
coordinate, and an event at the start takes the leaves at it alone.

**Terms of resting states.** Under ``equilibrate`` a transcendental term
:math:`\varphi(\psi(x, \theta))` of the states is a leaf :math:`W` in the balances as well, the
same leaf as the initial value of the new state :math:`\nu`. The balances are then rational in
the resting states, the coordinates and the leaves, and the relation rows
:math:`\mathrm{d}W = \varphi'(\psi)\, \mathrm{d}\psi` tie each leaf to the state columns and the
coordinates of the joint kernel (:ref:`steady-states`). The leaves are drawn independently of the
resting states, and the resting states are solved from the balances at the drawn values. Let
:math:`M = \{(\theta, x, W) : f(x, \theta, W) = 0,\ W = \varphi(\psi(x, \theta))\}` be the
resting set and :math:`A` the component of the algebraic set :math:`\{f(x, \theta, W) = 0\}` on
which the drawn points lie. Rank and kernel are rational in :math:`(\theta, x, W)`, so their
generic values on :math:`A` are those on :math:`M` when :math:`M` is Zariski dense in :math:`A`.
This holds under two conditions. First, let :math:`H` be the states and :math:`\theta_A` the
coordinates inside the arguments :math:`\psi`. The resting values :math:`x_H` vary independently
of :math:`\theta_A`, i.e., the tangent space of the resting set projects onto the coordinates
:math:`(\theta_A, x_H)`. Then :math:`(\theta_A, x_H)` and further coordinates form a chart of
:math:`M` in which every argument is rational. Second, no nontrivial integer combination of the
arguments is constant, which the recast ensures by one leaf per basis term, as for leaves of
coordinates. The density then follows from a classical lemma, the exponential case of the
Kolchin-Ostrowski theorem :cite:p:`Kolchin1968`, which extends the case of integrals :cite:p:`Ostrowski1946`
to exponentials of integrals: exponentials :math:`e^{\psi_1}, \dots, e^{\psi_n}` of rational
functions :math:`\psi_i \in \mathbb{C}(z)` are algebraically independent over
:math:`\mathbb{C}(z)` if and only if no nontrivial integer combination :math:`\sum_i n_i \psi_i`
is constant. Proof of the direction used: an algebraic relation reads
:math:`\sum_a c_a(z)\, e^{\langle a, \psi \rangle} = 0` with rational :math:`c_a \neq 0` and
distinct integer vectors :math:`a`, whose exponents differ pairwise by nonconstant functions. On
a generic line through the domain this stays so, with one variable :math:`t`. Take a relation
with the fewest terms, divide by the last term and differentiate in :math:`t`. The constant term
drops, and the other terms :math:`r_j e^{h_j}` become :math:`(r_j' + r_j h_j') e^{h_j}`, a
shorter relation, so every :math:`r_j' + r_j h_j' = 0`. Then :math:`e^{h_j}` is a constant over
:math:`r_j`, a rational function, which a nonconstant rational :math:`h_j` excludes. Logarithms
of rational functions no product of integer powers of which is constant, such as distinct
irreducible factors, are algebraically independent in the same way, and a half-angle tangent
:math:`\tan(\psi/2)` is rational in :math:`e^{i\psi}`. Ax's theorem :cite:p:`Ax1971`, a functional form
of Schanuel's conjecture, generalizes the lemma. The image of :math:`M` is therefore dense in
the space of the chart and the leaves, which has the dimension of :math:`A` where the resting
Jacobian is invertible, and :math:`M` is dense in :math:`A`. The first condition is checked at
the base point; where it fails, an ``InputError`` names the states. An example is
:math:`\dot x = k - x e^{x/k}`, whose resting value no coordinate outside the exponent moves.

A state that enters the balances only through leaves, such as :math:`x` in :math:`\dot x = p - k
\sin x`, leaves the resting Jacobian singular with the leaves drawn. The resting value of such a
state is drawn, and one leaf of the state that enters the balance becomes a resting unknown,
solved from the balance; the points remain points of :math:`A`.
``info["settings"]["rest_leaves"]`` lists the resting states inside transcendental terms with
their leaves, and under ``drawn`` the states whose resting value is drawn. A leaf of a resting
state is one coordinate for all conditions, so conditions with different resting models are
refused for such a state. A root or rational power of an expression in the states in the resting
model has no leaf and is refused.

.. _singular-starts:

Singular starts
---------------

The Lie series of Equation :eq:`eq-lie-series` needs an output that is analytic at the start.
Two kinds of terms are not, although the output is analytic for :math:`t > t_0`: a quotient
whose numerator and denominator both vanish at the start, such as the fraction of a species in
one compartment when every compartment starts empty, and a free power :math:`x^\alpha` of a state
that starts at 0. In both cases the output has an expansion at :math:`t_0` whose
coefficients are rational in :math:`\theta`, and these coefficients take the place of the Lie
derivatives. In the following, :math:`t_0 = 0`.

**Quotients at 0/0.** Let the right-hand side be analytic at the start and let the observable
read the states through a quotient :math:`N/D` whose denominator vanishes at :math:`z_0`. The
states are power series in :math:`t`, and so are :math:`N` and :math:`D`. Let :math:`v` be the
smallest order with :math:`D_v(\theta) \not\equiv 0`. Then

.. math::
   :label: eq-laurent

   \frac{N}{D} = t^{-v}\, \frac{N}{D_v + D_{v+1} t + \dots},

a Laurent series whose coefficients are rational in :math:`\theta` where :math:`D_v(\theta) \ne
0`, and a power series when :math:`N` vanishes to order :math:`v` as well. The engine evaluates
every reciprocal outside the right-hand sides whose operand starts at 0 in this form, after the
states, with the dual numbers inside the coefficients: an operand :math:`a = t^{v}(U + M)`,
with :math:`U` a power series of nonzero constant value and :math:`M` the terms below
:math:`t^v`, which lie in the dual part and the monomials of the gap lengths, has the inverse
:math:`t^{-v} U^{-1} \sum_e (-M U^{-1})^e`, a finite sum since :math:`M` is nilpotent. Each
division costs :math:`v` orders, so the states are expanded further until every output is known
to the Lie order. Rows are the gradients of the coefficients of the output, and the
argument of :ref:`oim` holds for them
unchanged, since it uses only that the coefficients determine the output near the start.
The valuation :math:`v` is taken at the base point, where :math:`D_k(\theta) = 0` for
:math:`k < v` holds together with the gradient; a point where a coefficient that is not
identically zero vanishes gives the expansion of that point, whose rank is a lower bound. A pole
of the output itself is refused, and so is a reciprocal of an operand that starts at 0 inside
the right-hand sides.

The stopping rule of :ref:`sufficient-lie-order` extends to these outputs. Let :math:`u(t) =
\partial_{z_0} g(t)` be the gradient of the output with respect to the start, a Laurent
series, possibly with a pole: away from the start the quotient is defined, and a start moved
off :math:`D = 0` gives a denominator of order 0.

**Lemma.** Let :math:`U_\infty` be the span of the coefficients of :math:`u`. Then
:math:`\dim U_\infty \le \rho_S`.

*Proof.* :math:`\dim U_\infty` is the number of functions among the components
:math:`u_i(t)` that are linearly independent over :math:`\mathbb{R}`, which equals the rank of
their Wronskian matrix :math:`\big(u^{(j)}(t)\big)_{j \ge 0}` over the meromorphic functions.
Along the solution :math:`\varphi_t(z_0)`, the chain rule gives :math:`u^{(j)}(t) = \partial_z
L_F^j h\big(\varphi_t(z_0)\big)\, \Phi(t)` with the invertible fundamental matrix :math:`\Phi(t)
= \partial \varphi_t / \partial z_0`. For small :math:`t > 0`, :math:`\varphi_t(z_0)` is a point
of :math:`S`, which is invariant under the flow, where the output is defined, so by the lemma of
:ref:`sufficient-lie-order` the rank is at most :math:`\rho_S`. :math:`\square`

The rows of the coordinates are the coefficients of :math:`\partial_\theta g = u\,
\partial_\theta z_0`. If the coefficients of :math:`u` up to order :math:`N`, those of negative
order included, reach rank :math:`\rho_S` at the start, they span :math:`U_\infty`, and the
rows up to order :math:`N` span all rows: the order is proven sufficient, as in the proposition
there. The coefficients at the base point are those of generic :math:`\theta` unless the base
point lies where a leading coefficient vanishes; this event is not counted in
``info["order_bound"]``.

**Free powers of a state at 0.** Let a state :math:`x` start at 0 in every condition and enter
through powers :math:`x^{c_0 + c\alpha}`, with integers :math:`c_0 \ge 0`, :math:`c > 0` and a
coordinate :math:`\alpha > 0` as exponent. Generically :math:`x = c_x t^k (1 + O(t))` with
:math:`k \ge 1` and :math:`c_x \neq 0`, so :math:`x^\alpha` behaves as :math:`t^{k\alpha}`, which
is not analytic at 0 for :math:`\alpha \notin \mathbb{N}`. The recast :math:`B = x^\alpha` of
:ref:`non-rational` needs :math:`x \ne 0` at the start. Instead, with the formal variable
:math:`S = (c_x t^k)^\alpha`,

.. math::
   :label: eq-start-power

   x^\alpha = S\, Q, \qquad Q = \Big(\frac{x}{c_x t^k}\Big)^{\alpha},

where :math:`Q` is a power series in :math:`t` with constant term 1. Every state and output
becomes a power series in :math:`t` and :math:`S`. A coefficient of :math:`t^n S^m`, kept with
the *index* :math:`n` (the exponent of :math:`t` at :math:`\alpha = 1`, so :math:`n \ge km`),
stands for the function :math:`t^{n - km + km\alpha}`. Differentiation in :math:`t` maps it to
:math:`\lambda_{n,m} = n + km(\alpha - 1)` times the coefficient of index :math:`n - 1`, so the
states follow from :math:`\lambda_{n+1,m}\, x_{n+1,m} = f_{n,m}`, and :math:`Q` from the
derivation of the index, :math:`n Q_n = A_0^{-1} \sum_{j=1}^{n} (\alpha j + j - n)\, A_j Q_{n-j}`
with :math:`A = x / (c_x t^k)` and :math:`Q_0 = A_0^\alpha`, where :math:`A_0 - 1` is nilpotent.
:math:`Q_n` needs :math:`x` up to index :math:`n + k`, which is the index of :math:`S Q_n`, so the
recursion runs order by order as for analytic outputs. The coefficients :math:`d_{n,m}` of an
output are rational in :math:`\theta` where no :math:`\lambda` vanishes, which holds for
:math:`\alpha > 0`: :math:`\lambda_{n,m} \ge \min(1, \alpha)\, n`. The same bound shows that the
series converges for small :math:`t > 0`: the recursion divides by at least :math:`\min(1, \alpha)`
times the divisor of a Taylor series, so Cauchy's majorant of the right-hand side divided by
:math:`\min(1, \alpha)` dominates it.

**Lemma.** For distinct real numbers :math:`q_1, \dots, q_r`, the functions :math:`t^{q_j}` and
:math:`t^{q_j} \log t` are linearly independent over :math:`\mathbb{R}` on every interval
:math:`(0, \varepsilon)`, also as terms of convergent sums whose exponents have no accumulation
point.

*Proof.* Let :math:`\sum_j (a_j + b_j \log t)\, t^{q_j} = 0` with :math:`q_1 < q_2 < \dots` and
:math:`j` the first index with :math:`(a_j, b_j) \neq 0`. Divided by :math:`t^{q_j}`, the other
terms tend to 0 as :math:`t \to 0^+`, since :math:`t^\delta \log t \to 0` for :math:`\delta > 0`,
while :math:`a_j + b_j \log t` does not. :math:`\square`

For :math:`\alpha \notin \mathbb{Q}`, distinct :math:`(n, m)` give distinct exponents
:math:`n - km + km\alpha`, and with several powers the exponents :math:`n + \sum_i k_i m_i
(\alpha_i - 1)` are distinct when :math:`1, \alpha_1, \dots, \alpha_K` are linearly independent
over :math:`\mathbb{Q}`; this holds for generic coordinates if the exponents are distinct
coordinates, and two powers of states at 0 with the same exponent are refused. The gradient of
the output is

.. math::

   \partial_\theta g = \sum_{n,m} c_x^{m\alpha} \Big(\partial_\theta d_{n,m}
   + m\, d_{n,m} \big(\alpha\, \partial_\theta c_x / c_x + \log c_x\, \partial_\theta \alpha\big)
   + km\, d_{n,m} \log t\; \partial_\theta \alpha\Big)\, t^{n - km + km\alpha},

so by the lemma a direction lies in the kernel if and only if it annihilates the vectors in
parentheses without the logarithm of :math:`t` and, separately, the vectors :math:`km\, d_{n,m}\,
\partial_\theta \alpha`. Once a coefficient with :math:`m > 0` is nonzero, the latter span
:math:`\partial_\theta \alpha`, and the transcendental term :math:`\log c_x\, \partial_\theta
\alpha` lies in their span. The rows are therefore :math:`\partial_\theta d_{n,m} + m\,
d_{n,m}\, \alpha\, \partial_\theta c_x / c_x`, rational in :math:`\theta`, and one row
:math:`\partial_\theta \alpha`. With several powers, the rows of the logarithm are
:math:`\sum_i k_i m_i\, \partial_\theta \alpha_i`; the engine refuses an expansion in which the
vectors :math:`(k_i m_i)_i` of the nonzero coefficients do not span every power that occurs.
The engine computes :math:`d_{n,m}` with :math:`S` held fixed in the dual numbers and adds the
term of :math:`c_x`, the leading coefficient of :math:`x`, from the dual of :math:`c_x`. The ranks
are taken at random points, where :math:`\alpha` is generic.

Truncation at the Lie order :math:`N` keeps the indices :math:`n \le N`, and each power of
:math:`S` counts :math:`k` orders, so the monomials have total degree at most :math:`N/k` in the
:math:`S`. Every coefficient computed is exact, and the rank grows to the final value with
:math:`N`. The stopping rule does not apply: the rows :math:`\partial_z L_F^j h` do not exist at
the start, and a start moved off :math:`x = 0` changes the type of the expansion, so no start
reaches the ceiling. The order stops after the plateau and the saturation guard, the result
says that the Lie order is not proven sufficient, and ``info["limits"]`` names the power. Events,
``equilibrate`` and a state that starts at 0 in one condition only are refused for such powers.
