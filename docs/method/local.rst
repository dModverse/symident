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
*modular engine*: the Lie derivatives as power series in time truncated at order :math:`N`, the
derivatives with respect to :math:`\theta` as dual numbers :cite:p:`Sedoglavic2002`. A point can
only underestimate the generic rank, and a random point does so with small probability
:cite:p:`Schwartz1980,Zippel1979`. The rank is taken at the base point, as the largest over the
engine primes, and checked at points uniform modulo the verification prime.

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
subspace :math:`U` of the states, :math:`S = \{z_0(\theta) + (b, 0, 0) : \theta \in \Theta,\ b \in
U\}`, with :math:`U` the smallest subspace for which :math:`F` stays in :math:`U \times \{0\}
\times \mathbb{R}` on :math:`S`, built by adding the state part of :math:`F` at random points of
:math:`S`. This covers a start at rest, states that stay at zero, fixed initial values, and linear
relations the dynamics keeps. Under ``equilibrate`` :math:`S` is the resting variety of the
condition (:ref:`steady-states`). In both cases the rank at the start is taken at the base point.
A continued condition rests at the base point as a power series only and is ranked at a resting
point drawn forward instead: random states and parameters, the turnover rates and the states of
the plan solved from the balances. Such points are generic on one irreducible component of the
resting variety, the closure of the points where the balances are regular in these unknowns. The
continued condition lies on it where the balances are regular in them along the continuation, a
rank over :math:`\mathbb{F}_p((\varepsilon))` that a truncated series can only lower; otherwise the
order is not certified. The draw has no error bound, and ``info["order_bound_why"]`` says so.
Where the start does not reach :math:`\rho_S`, e.g., on the level set of a nonlinear conserved
quantity, the order stops after a *plateau* of three orders without change of rank
(``SYMIDENT_LIEPLATEAU``), and the result says that the Lie order is not proven sufficient.

**Conditions and events.** Every condition is expanded to a separate order. An event that replaces
a state by a value of the coordinates removes that state from :math:`U`, an event that multiplies a
state or reads the states adds it, and the proposition extends to the power series of
:ref:`conditions-events`. :math:`U` and :math:`\rho_S` are taken at points uniform modulo the
verification prime; the error is bounded in :ref:`order-bound`.

**Without a stopping rule.** A direction that is an exact symmetry (:ref:`scalings`,
:ref:`proofs-over-q`) lies in the kernel at every order, so a rank that together with such
independent directions gives :math:`d` is proven for all orders. ``engine="symbolic"`` confirms
the order by an exact rank over the rational functions. Where no order is proven sufficient, a
*saturation guard* (``verify=True``) takes the rank again at higher Lie and gap orders
(``SYMIDENT_VERIFY_MARGIN``) and repeats the detection where it grows; if it still grows, the rank
is a lower bound, ``info["verification"]`` says where, and a reduction is refused.

.. _order-bound:

Error probability of the order
------------------------------

The random choices are the verification prime :math:`p`, drawn per detection from the seed of the detection, and
points uniform modulo :math:`p`, each from a separate seed. All error bounds of symident rest on
the Schwartz-Zippel lemma :cite:p:`Schwartz1980,Zippel1979`: a nonzero polynomial of degree
:math:`D` in :math:`m` variables over a field vanishes at a point drawn uniformly from
:math:`A^m`, for a finite subset :math:`A` of the field, with probability at most :math:`D / |A|`.

A wrong certificate of the Lie or gap order needs a polynomial to vanish at one of the random
points of :ref:`sufficient-lie-order` or :ref:`conditions-events`. The degree of the polynomial is
that of the rational system the modular engine evaluates, multiplied by the degree :math:`\deg z_0`
of the start, and on a resting variety by the Bezout number of the balances, Equation
:eq:`eq-rest-bound`. The probabilities, each multiplied by the number of points drawn until the
point was taken, add up to ``info["order_bound"]``; ``info["order_bound_why"]`` names the reason
for a missing bound.

.. _non-rational:

Non-rational terms
------------------

The modular engine needs rational functions. Elementary functions are therefore recast into
rational form :cite:p:`Savageau1987`, innermost first. A term :math:`\nu = \varphi(\psi)` with
:math:`\psi` rational in the states becomes a new state with right-hand side :math:`\varphi'(\psi)\,
\sum_j \partial_{x_j} \psi \, f_j`, which is rational in :math:`\nu` and :math:`\psi` for

.. math::

   \varphi(\psi) = \psi^\alpha:\ \ \frac{\alpha\, \nu}{\psi}, \qquad
   \exp(\psi):\ \ \nu, \qquad \log(\psi):\ \ \frac{1}{\psi}, \qquad
   \arctan(\psi):\ \ \frac{1}{1 + \psi^2}, \qquad \operatorname{artanh}(\psi):\ \ \frac{1}{1 -
   \psi^2},

and for the other inverse functions contains a square root, which is recast in turn.
Trigonometric functions go through :math:`\tan(\psi/2)`. A *free power* :math:`b^\mu`, with a
parameter :math:`\mu` as exponent, becomes a state :math:`B = b^\mu` with companion :math:`L_b =
\log b`, so that :math:`\mu` enters through :math:`\partial B / \partial \mu = B L_b`. A nesting the
recast cannot order, such as :math:`\sin(\exp x)`, is refused.

A new state starts at :math:`\varphi(\psi)` of the initial state. Where this depends on the
coordinates, a transcendental :math:`\varphi` becomes a separate input of the engine, a *leaf*,
drawn independently of the argument and tied to it by relation rows :math:`\mathrm{d}\nu =
\varphi'(\psi)\, \mathrm{d}\psi`: a transcendental graph lies in no proper variety, so a random
point on it is generic. With several transcendental constants, such as :math:`e` and :math:`\pi`,
this needs their algebraic independence, which follows from Schanuel's conjecture and is not
proven. An algebraic :math:`\varphi`, a root or rational power, is exact. Rank and kernel are taken
at generic points, where every argument is nonzero, so the detection needs no sign; an argument that
the declared signs do not make positive on :math:`\Theta` is named in ``info["limits"]``.

**Terms of resting states.** Under ``equilibrate`` a transcendental term :math:`\varphi(\psi(x,
\theta))` is a leaf :math:`W` in the balances as well, tied to states and coordinates by
:math:`\mathrm{d}W = \varphi'(\psi)\, \mathrm{d}\psi`, and the resting states are solved at the drawn
leaves. Let :math:`M = \{(\theta, x, W) : f(x, \theta, W) = 0,\ W = \varphi(\psi(x, \theta))\}` be
the resting set and :math:`A` the component of :math:`\{f(x, \theta, W) = 0\}` on which the drawn
points lie. Rank and kernel are rational, so their generic values on :math:`A` are those on
:math:`M` when :math:`M` is Zariski dense in :math:`A`. This holds when the resting values of the
states inside the arguments vary independently of the coordinates inside them, which gives a chart
of :math:`M` with rational arguments, and no nontrivial integer combination of the arguments is
constant, which the recast ensures. The density then follows from the exponential case of the
Kolchin-Ostrowski theorem :cite:p:`Kolchin1968,Ostrowski1946`:

**Lemma.** Exponentials :math:`e^{\psi_1}, \dots, e^{\psi_n}` of rational functions :math:`\psi_i
\in \mathbb{C}(z)` are algebraically independent over :math:`\mathbb{C}(z)` if and only if no
nontrivial integer combination :math:`\sum_i n_i \psi_i` is constant.

*Proof of the direction used.* A relation reads :math:`\sum_a c_a(z)\, e^{\langle a, \psi \rangle}
= 0` with rational :math:`c_a \ne 0` and exponents that differ pairwise by nonconstant functions,
also on a generic line. In a relation with the fewest terms, divided by the last term and
differentiated, every :math:`r_j' + r_j h_j' = 0`, so :math:`e^{h_j}` is a constant over a rational
function, which a nonconstant rational :math:`h_j` excludes. :math:`\square`

Logarithms of rational functions no product of integer powers of which is constant are
algebraically independent in the same way, and :math:`\tan(\psi/2)` is rational in
:math:`e^{i\psi}`; Ax's theorem :cite:p:`Ax1971` generalizes the lemma. The first condition is
checked at the base point; where it fails, an ``InputError`` names the states, as for :math:`\dot x
= k - x e^{x/k}`. ``info["settings"]["rest_leaves"]`` lists the resting states inside
transcendental terms.

.. _singular-starts:

Singular starts
---------------

The Lie series of Equation :eq:`eq-lie-series` needs an output that is analytic at the start.
Two kinds of terms are not, although the output is analytic for :math:`t > t_0`: a quotient whose
numerator and denominator both vanish at the start, and a free power :math:`x^\alpha` of a state
that starts at 0. In both cases the output has an expansion at :math:`t_0 = 0` whose coefficients
are rational in :math:`\theta`, and these take the place of the Lie derivatives.

**Quotients at 0/0.** Let the observable read the states through a quotient :math:`N/D` whose
denominator vanishes at :math:`z_0`, and :math:`v` the smallest order with :math:`D_v(\theta) \not
\equiv 0`. Then

.. math::
   :label: eq-laurent

   \frac{N}{D} = t^{-v}\, \frac{N}{D_v + D_{v+1} t + \dots},

a Laurent series whose coefficients are rational in :math:`\theta` where :math:`D_v(\theta) \ne 0`.
The rows are the gradients of these coefficients; the argument of :ref:`oim` holds unchanged, since
it uses only that the coefficients determine the output near the start. A pole of the output
itself is refused.

The stopping rule of :ref:`sufficient-lie-order` extends to these outputs. Let :math:`u(t) =
\partial_{z_0} g(t)`, a Laurent series.

**Lemma.** Let :math:`U_\infty` be the span of the coefficients of :math:`u`. Then :math:`\dim
U_\infty \le \rho_S`.

*Proof.* :math:`\dim U_\infty` is the rank of the Wronskian matrix :math:`(u^{(j)}(t))_{j \ge 0}`.
By the chain rule :math:`u^{(j)}(t) = \partial_z L_F^j h(\varphi_t(z_0))\, \Phi(t)` with the
invertible fundamental matrix :math:`\Phi(t)`, and for small :math:`t > 0`, :math:`\varphi_t(z_0)`
is a point of :math:`S` where the output is defined, so the rank is at most :math:`\rho_S`.
:math:`\square`

If the coefficients of :math:`u` up to order :math:`N`, those of negative order included, reach
rank :math:`\rho_S` at the start, they span :math:`U_\infty`, and the order is proven sufficient.
The event that the base point lies where a leading coefficient vanishes is not counted in
``info["order_bound"]``.

**Free powers of a state at 0.** Let a state :math:`x` start at 0 in every condition and enter
through powers :math:`x^{c_0 + c\alpha}`, with integers :math:`c_0 \ge 0`, :math:`c > 0` and a
coordinate :math:`\alpha > 0`. Generically :math:`x = c_x t^k (1 + O(t))` with :math:`k \ge 1` and
:math:`c_x \ne 0`. With the formal variable :math:`S = (c_x t^k)^\alpha`,

.. math::
   :label: eq-start-power

   x^\alpha = S\, Q, \qquad Q = \Big(\frac{x}{c_x t^k}\Big)^{\alpha},

where :math:`Q` is a power series in :math:`t` with constant term 1. Every state and output becomes
a power series in :math:`t` and :math:`S`; a coefficient of :math:`t^n S^m`, the *index* :math:`n`
and the *layer* :math:`m`, stands for :math:`t^{n - km + km\alpha}`, and differentiation multiplies
it by :math:`\lambda_{n,m} = n + km(\alpha - 1) \ge \min(1, \alpha)\, n`. So the coefficients
:math:`d_{n,m}` are rational in :math:`\theta` for :math:`\alpha > 0`, and the series converges for
small :math:`t > 0` by a Cauchy majorant.

**Lemma.** For distinct real numbers :math:`q_j`, the functions :math:`t^{q_j}` and :math:`t^{q_j}
\log t` are linearly independent over :math:`\mathbb{R}` on every interval :math:`(0,
\varepsilon)`, also as terms of convergent sums whose exponents have no accumulation point.

*Proof.* In a relation, divide by the term of least exponent with a nonzero coefficient
:math:`a_j + b_j \log t`; the other terms tend to 0 as :math:`t \to 0^+`, and :math:`a_j + b_j
\log t` does not. :math:`\square`

For :math:`\alpha \notin \mathbb{Q}` distinct :math:`(n, m)` give distinct exponents, and with
several powers this holds where the exponents are distinct coordinates; two powers of states at 0
with the same exponent are refused. The gradient of the output is

.. math::

   \partial_\theta g = \sum_{n,m} c_x^{m\alpha} \Big(\partial_\theta d_{n,m}
   + m\, d_{n,m} \big(\alpha\, \partial_\theta c_x / c_x + \log c_x\, \partial_\theta \alpha\big)
   + km\, d_{n,m} \log t\; \partial_\theta \alpha\Big)\, t^{n - km + km\alpha},

so by the lemma the rows are :math:`\partial_\theta d_{n,m} + m\, d_{n,m}\, \alpha\,
\partial_\theta c_x / c_x`, rational in :math:`\theta`, and one row :math:`\partial_\theta \alpha`,
which also absorbs the transcendental term :math:`\log c_x\, \partial_\theta \alpha`. The stopping
rule does not apply: a start moved off :math:`x = 0` changes the type of the expansion, so no start
reaches the ceiling. The order stops after the plateau and the saturation guard, the Lie order is
reported as not proven sufficient, and ``info["limits"]`` names the power. Events,
``equilibrate`` and a state that starts at 0 in one condition only are refused for such powers.
