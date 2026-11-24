Method
======

The method chapters state the mathematics behind symident: the notion of identifiability it
decides, how the rank of the observability-identifiability matrix is computed and when the order
is sufficient, how non-identifiable directions are obtained in closed form and proven, how a model
is reduced to identifiable parameters, and how the parameter values with the same output are
counted. Every statement that rests on random points comes with an error probability or is named
in the result.

.. toctree::
   :maxdepth: 1

   method/local
   method/proofs
   method/reduction
   method/global
   method/conditions
   method/references

Notation and terms
------------------

**Model.** symident analyzes models given by ordinary differential equations (ODEs),

.. math::
   :label: eq-model

   \dot x = f(x, u, \eta), \qquad x(t_0) = x_0(\eta),

with states :math:`x \in \mathbb{R}^{n_x}`, parameters :math:`\eta`, start time :math:`t_0`,
inputs :math:`u = u(t, \eta)`, i.e., given functions of time such as a dose set by an event,
and an observation function :math:`h = (h_1, \dots, h_{n_y})` with one component per
observable. The *coordinates* :math:`\theta \in \Theta \subseteq \mathbb{R}^d` are the
estimated parameters and the free initial values; they give the parameters by a
*parametrization* :math:`\eta = \eta(\theta)` and the initial value :math:`x_0(\theta)`, and
:math:`\Theta` is an open set, the *parameter space*. The observed quantities along the
solution,

.. math::

   g(t; \theta) = h\big(x(t), \eta(\theta)\big),

form the *output*. Right-hand sides, initial values and observables are rational
functions, or they are made rational by auxiliary states (see :ref:`non-rational`). A
*direction* at :math:`\theta` is a vector :math:`\xi \in \mathbb{R}^d` of rates of change of
the coordinates.

**Extended state.** Adding parameters and time to the states, :math:`z = (x, \eta, t)` with
:math:`\dot\eta = 0` and :math:`\dot t = 1`, makes the model the autonomous vector field
:math:`F = f\, \partial_x + \partial_t` :cite:p:`Tunali1987`. :math:`F(\varphi)` is the rate of
change of a function :math:`\varphi(z)` along a solution, and the *start* is
:math:`z_0(\theta) = (x_0(\theta), \eta(\theta), t_0)`. A set :math:`S` of points :math:`z` is
*invariant* under :math:`F` when :math:`F` is tangent to :math:`S` at each point of :math:`S`, so that a
solution that starts in :math:`S` stays in :math:`S`.

**Conditions and events.** A *condition* is one experiment; conditions share :math:`\theta`
and differ in fixed values, inputs or renamings. An *event* changes a state or an input
instantaneously at a given time and splits a condition into *segments*.

**Algebra.** A *variety* is the common zero set of finitely many polynomials; it is
*irreducible* when it is not the union of two smaller varieties. A statement holds for
*generic* points of an irreducible variety :math:`V` when it holds outside a proper subvariety
of :math:`V`; on :math:`\Theta` the exceptions have measure zero. The *generic rank* of a
matrix of rational functions on :math:`V` is the rank at generic points of :math:`V`, which is
the largest rank on :math:`V`. The *degree* of a polynomial is the total degree, that of a
vector of rational functions the largest degree of the numerators and of the common
denominator; to *clear* an expression is to multiply it by the denominator. The degree
:math:`\deg V` of a variety of dimension :math:`m` is the number of points of :math:`V` on a generic
affine subspace of complementary dimension.

**Primes.** Ranks are computed over finite fields :math:`\mathbb{F}_p`. The four *engine
primes* :math:`p_1, \dots, p_4`, close to :math:`1.5 \cdot 10^9`, are fixed. The *verification
prime* :math:`p`, a prime below :math:`1.4 \cdot 10^9` distinct from them, is drawn from the
seed of each detection. A point *uniform modulo* :math:`p` has independent coordinates, each
uniform in :math:`\{0, \dots, p - 1\}`, drawn from a separate seed. The *base point*
:math:`\theta_b` has the coordinates 2, 3, 5, ..., the first primes.

**Closed forms.** A *closed form* of a direction gives each component :math:`\xi_i` as a
rational expression of the coordinates. The *level* (``exact``, ``verified``, ``checked``,
``open``) says how far it is established (:ref:`levels`), and the *route* how it
was obtained (:ref:`general-directions`).

The example is two compartments, a substance :math:`A` that is converted into
:math:`B` with rate :math:`k_1`, a loss of :math:`B` with rate :math:`k_2`, and an observation
of :math:`B` up to an unknown scale :math:`s`,

.. math::
   :label: eq-example

   \dot A = -k_1 A, \qquad \dot B = k_1 A - k_2 B, \qquad h = s\, B,

with free initial values :math:`A_0` and :math:`B_0`, so that
:math:`\theta = (k_1, k_2, s, A_0, B_0)` and :math:`d = 5`.

Structural identifiability and Lie symmetries
---------------------------------------------

The model is *locally structurally identifiable* at :math:`\theta` if no nearby
:math:`\tilde\theta \ne \theta` has the same output. *Structural* means that the question
is decided by the equations alone, as if the observables were known exactly and continuously,
and for generic :math:`\theta`.

A continuous family of coordinate values with the same output is the orbit of a *Lie
symmetry*, a local one-parameter group :math:`\Phi_\lambda` of transformations of
:math:`\Theta` with

.. math::
   :label: eq-symmetry

   g\big(t; \Phi_\lambda(\theta)\big) = g(t; \theta)

:cite:p:`Yates2009,Merkt2015,Massonis2020`. The group is described by the *generator*, the
vector field

.. math::
   :label: eq-generator

   X = \sum_{i=1}^d \xi_i(\theta)\, \partial_{\theta_i}, \qquad
   \xi(\theta) = \partial_\lambda \Phi_\lambda(\theta)\big|_{\lambda = 0},

whose value :math:`\xi(\theta)` at :math:`\theta` is a direction. The integral curve of
:math:`X` through :math:`\theta` is the *orbit* :math:`\mathcal{O}(\theta)`. The coordinates
with :math:`\xi_i \not\equiv 0` form the *support* of the generator. A function :math:`I(\theta)`
with :math:`X(I) = 0` is an *invariant* of :math:`X`; it is constant on orbits, and the
output depends on :math:`\theta` only through invariants. With :math:`r` generators whose
values are linearly independent at generic :math:`\theta`, :math:`d - r` functionally
independent combinations of the coordinates are *identifiable*. In Equation :eq:`eq-example`,
multiplying both initial values by a factor and dividing :math:`s` by it leaves :math:`g`
unchanged; the generator is

.. math::

   X = A_0\, \partial_{A_0} + B_0\, \partial_{B_0} - s\, \partial_s,

and :math:`k_1`, :math:`k_2`, :math:`s A_0` and :math:`s B_0` are invariants of :math:`X`.
