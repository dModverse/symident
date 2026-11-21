# symident 0.17.0

## Steady states

* `equilibrate` supports every non-rational term the recast handles: exponentials and powers
  `b^x` of states, logarithms, trigonometric and hyperbolic functions and their inverses of
  expressions in the resting states, and exponentials of the clock. A term at rest becomes a leaf
  drawn independently of the resting states, which are solved from the balances at the drawn
  values.
* This is exact where the resting set is Zariski dense in the algebraic set of the balances
  (docs, "Terms of resting states"). The density holds under two conditions: the resting values
  of the states inside the terms vary independently of the coordinates inside them, and no
  integer combination of the arguments is constant. The first condition is checked at the base
  point, and an `InputError` names the states where it fails.
* A state that enters the balances only through leaves has the resting value drawn and one leaf
  solved from the balance. `info["settings"]["rest_leaves"]` lists the resting states inside
  non-rational terms with their leaves, and under `drawn` the states whose resting value is drawn.
* Refused at rest: conditions with different resting models where a state at rest enters a
  non-rational term, and roots or rational powers of expressions in the states.
* Before the analysis, msolve solves the resting balances at three random parameter values.
  Balances without a complex solution at all three raise an `InputError`. Where no real solution
  lies in the declared domain at some of the values, a warning is given, and `info["resting"]`
  holds the counts. This is a check at random points, not a proof, and it does not consider
  stability. Resting models with leaves of non-rational terms or more than twelve resting states
  are not checked.

## Reduction

* A block of directions that, in linear coordinates and logarithms of positive coordinates, moves
  every coordinate at a rate that depends only on coordinates the block does not move is a linear
  flow, reduced before the invariant search. The orbits are affine subspaces. Pins on coordinates
  whose minor of the components has one sign give a global section with `injective` and `reach`
  proven. An offset that enters only the observables is pinned first, an initial value last. The
  other moved coordinates keep their names and stand for the invariants.
* The functional independence of invariants is decided by a Jacobian minor in interval arithmetic
  at a rational point, which needs no exact powers of rationals for exponential invariants.
* A direction of a model with unknown inputs whose components depend on states or inputs is
  reported as no vector field on the parameters, with the unknown inputs that absorb it.
* A direction without closed form after a detection with `reconstruct=True` is reported at level
  open, without the advice to rerun the reconstruction.
* A root with the imaginary unit of a triangular invariant system is dropped as a branch without a
  real section, and the solution is marked incomplete.

## Detection

* An observable that is 0/0 at the start, such as a quotient of sums of states that all start at
  0, is analyzed. The kernel takes a reciprocal of a series without constant value outside the
  right-hand sides as a Laurent series, expands the states further by the orders the division
  costs, and uses the coefficients of the prediction as rows. The stopping rule of the Lie order
  holds for these rows, with the coefficients along the states, poles included, in place of the
  Lie derivatives at the start (docs, "Singular starts").
* A free power `x^n` of a state that starts at 0, such as a Hill term of a species absent at the
  start, is analyzed. With `x = c t^k + ...`, the kernel expands the model in `t` and
  `S = (c t^k)^n`. The coefficients give the rows, and the terms in `log t` give a row for the
  exponent; this is exact for generic exponents `n > 0`. The Lie order is not proven sufficient
  there, and `info["limits"]` says so. Supported are powers `x^(c0 + c*n)` with integers
  `c0 >= 0` and `c > 0`, one exponent per state and one state per exponent, without events and
  `equilibrate`, and with the state at 0 in every condition.
* Exact scalings may move the resting states of some conditions only, such as a total that one
  parameter scales in part of the experiments together with the scale factors of the observables
  there. The parameter weights are those that extend to the states of every condition, found
  condition by condition from the left null space of the state columns. Each condition is checked
  with the state weights of that condition.
* A logarithm, fractional power or base of a free exponent whose argument the declared signs do
  not make positive is analyzed. Rank and kernel are taken at generic values, where every argument
  is nonzero, so the detection needs no sign. `info["limits"]` names the argument.
* The rank bound that proves the gap order sufficient stops once a symbolic Lie derivative grows
  past the size cap, measured on the expression tree before the expansion.
* The rewrite of a direction from resting states into coordinates tries each set of solved
  equations and coordinates once and stops after a fixed number of steps. The direction then stays
  in the resting states.
* The forward route fits an entry up to degree 4, the degree of a leaf of a non-rational term that
  is solved from a resting balance. A degree whose fit needs more points than the bank holds is
  skipped, with the higher ones.
* The closed forms from Taylor jets bound the size of a jet from the expression tree and print the
  jet only where the bound does not already exceed the size limit.
* In a model with switches, a direction whose free column is a fresh coordinate of a switch
  surface stays open with the kernel support: the closed forms that the switch rule reads hold no
  fresh coordinate.

## Large models

* Each chain of segments (a condition with events, or a copy of a condition in a regime of a
  switch) is built over the coordinates that the tapes of the chain read, for the rank profiles,
  the kernels of the saturation and of the closed forms, and the ranks of the Lie-order
  certificate. The other columns of the rows are zero, so ranks and kernels are unchanged.
* The stacked kernel over all chains is the reduced form of the stacked reduced rows of the chains,
  with the series rows reduced together. Where a chain changes the rows at the end of the gaps, the
  joint build is used.
* The jets of a chain are built once per point, prime and gap order and kept unreduced, each row
  with the Lie order. The coefficients at an order do not depend on the order the jets are taken
  to, so the kernel at a lower order reduces the rows up to that order. The rank profiles of the
  saturation, the kernel at the order the scan stops at, the lowered caps and the stacked kernels
  read the same jets. The kernel of a chain alone is kept for the point, prime and orders it was
  taken at.
* The next rank profile goes to the order at which the rank reaches the columns of the chain at the
  current rate, at most four times the last order. The rank profiles of the chains run in
  parallel, one thread per chain.
* With recast atoms (free exponents, exponentials, trigonometric functions), each chain of the
  saturation is taken to the saturated Lie order of that chain. The relation rows of the atoms join
  the stacked rows and do not change the rows of a chain.
* A chain of one segment has no gap: the rows at gap order 1 or more are those at gap order 0 with
  zero rows of the gap monomials.
* The certificate of the Lie order starts the blocks with the most lanes over their segments and
  the highest saturated orders first, and takes the generic and the start ranks of a block side by
  side. The ranks at an order do not depend on the order taken, so the orders 16, 32, 64, ... of
  the sequence are checked on prefixes of ranks taken once. The first ranks of a block go a tenth
  past the saturated order, later ones to the order at which the ranks reach their plateau at the
  current rate. A block whose generic ranks have reached their plateau takes only the start ranks
  to higher orders.
* The free columns whose kernel vectors leave the span of the scalings are found from one reduced
  form of all kernel vectors side by side. The minimal-support gauge and the selection of
  independent directions test their vectors against a span kept in reduced form.
* Whether two regimes of a switch give different observable jets is decided from the jets to a few
  orders first, doubling the order up to the full one only while they agree.
* Conditions and segments that repeat the same entries are handled once per distinct text or
  expression: the form checks of the arguments, the parsing of model lines, the search for
  coordinates that enter only as `b^theta`, the logarithmic chart, the trigonometric recast, the
  tests for non-rational terms, the zero sets of the regimes of a switch, the degree bounds of the
  certificate and the substitutions of the tape compiler. The error bound of the projection of a
  model with switches parses the parameter rules of a copy once.
* The degrees of the field do not change under a renaming of the symbols and are shared by the
  copies of a model that differ only in names. `detect()` copies dicts and lists of strings of the
  model directly, not through `copy.deepcopy`.
* The Taylor coefficients of values that are constant in time, the parameters and what the tape
  forms from them alone, are not formed past order 0, and a product with one takes a single term.
* The Taylor values of a column block take an eighth of the physical memory over the cores,
  between 256 MB and 1 GB (`SYMIDENT_BLOCKMB` overrides it).

## PEtab

* A table that starts with a byte order mark keeps the name of the first column.
* A parameter that a later period sets to the value it already has, in every experiment, gets no
  input: the terms of that input leave the model, so a parameter that only such terms hold is
  found exactly as without effect.
* States that start at 0 in every experiment, that no event sets and whose rates vanish where all
  of them are 0 stay at 0, also through a preequilibration that starts from these values. They
  leave the model, and so do the parameters that only their terms hold.
* An estimated parameter that only terms vanishing in every experiment hold gives a warning and is
  listed in `m.petab["without_effect"]`.
* A parameter value that names another parameter, such as an SBML initial assignment `k = k_syn`,
  takes the value of that parameter in each experiment, also where a condition changes it.

## Other

* An expression that holds a Python keyword as a name, such as `and` or `lambda`, raises an
  `InputError` that names the keyword.

# symident 0.16.1

* `install_msolve()` links msolve with `-pthread`, which the thread pool of FLINT needs. On
  systems with glibc older than 2.34, such as RHEL 8 and its derivatives, pthread is not part of
  libc, and the link failed with undefined references to `pthread_create`. CI builds msolve on
  AlmaLinux 8.

# symident 0.16.0

## Global analysis

* `discrete()` rewrites the model into rational form before the fiber, with the same fiber: a
  coordinate that enters only through `exp(theta)` or `b^theta` is read as that power, positive,
  and the maps, fixed sets and fundamental domains are reported in the given coordinates, a
  discrete scaling as a translation; an observable under `log`, `log10`, `exp` or an odd power is
  replaced by the argument; `exp(a*time)` becomes a state `E` with `E' = a*E`, `E(0) = 1`.
  `info["discrete"]["rewrites"]` lists each rewrite with the reason it keeps the fiber, and a term
  that stays non-rational is named in the error.
* `discrete()` tests candidates before the count: permutations of coordinates on which the same
  output coefficients depend, modulo a prime at a random point, the transpositions within each
  class and the products of two that fail alone. A candidate joins the maps of the fiber and is
  proven like them. `info["discrete"]["candidates"]` and `info["discrete"]["timing"]` give the
  numbers and the seconds of each stage, and each map names where it was found in `found_by`.
* A map of the states for the proof may be affine in the states that the map of the coordinates
  reaches, the identity elsewhere, with coefficients solved from the identities; each symmetry
  holds the maps in `state_maps`.
* `info["discrete"]["certificate"]` states what a result rests on: the invariance identities per
  map, the size of the orbit of the group the exact maps generate, the count, the prime, and the
  axiom that the count modulo the prime at a random point is the count over the complex numbers
  at a generic point. `count_final` holds when the orbit has as many points as the count, also
  where the reconstruction of the fiber runs out of budget; the counts in the domain then come from
  the orbit.
* The polynomial system of the fiber holds each monomial as pairs of variable and exponent, and
  the substitution of the variables that a linear equation fixes and the removal of the jets that
  one equation holds alone run in the C++ kernel modulo a prime, in Python over Q.
* The count of the fiber is the number of distinct values of a random linear form in the
  coordinates on the points, from the reduced Gröbner basis of msolve and the multiplication by the
  form on the quotient, without the parametrization of the points; points that differ only in the
  derivatives of the states count once. The parametrization is computed only to reconstruct maps.
* Every observable starts at the highest order of the rank criterion plus one, which also covers an
  observable whose coefficients add no rank for some orders before they do, such as a state at rest
  at the end of a chain. The orders stop rising where the count equals the orbit of the proven
  candidates, and the maps of the fiber are not reconstructed when the candidates account for the
  count.
* The counts in the domain end after the budget and one call of msolve; the points left without a
  count are named in `limits`.

## Reduction

* `reduce()` reduces translations of logarithmic coordinates, those that a trafo passes only
  through `exp(theta)` or `b^theta`, as scalings of `exp(theta)`, with the pins at zero, exactly;
  a block of several such translations no longer goes through the section search. The detection
  lists these coordinates in `info["log_coordinates"]`.

## Lie-order certificate

* Integer powers on the tape are built by squaring, about `2*log2(n)` products for `x^n`. Rates
  with unround constants put exponentials over a large common denominator `N` of `exp(V/N)`, and
  the powers of that leaf reach `N` (#1, reported by Mio Heinrich).
* The degrees of the certificate past a total degree of 10000 come from the expression tree: a
  sum has at most the largest degree of the terms, a product the sum, a power `k` times that of the
  base. The common denominator is then the product of the distinct denominators. No dense
  polynomial of such a degree is built (#1).
* A Lie-order certificate whose Schwartz-Zippel bound reaches 1 reports no error bound:
  `info["order_bound"]` is `None` and `info["order_bound_why"]` says that the bound reaches 1 at
  the verification prime.

# symident 0.15.0

## Global analysis

* `detect(mode="global")` analyzes local identifiability, reduces the directions found and counts
  the values of the coordinates with the output of a generic one, with the rational maps between
  them; `global_options` passes the settings of `Symmetries.discrete()`. A setting the global
  analysis does not handle warns and leaves the reason in `info["discrete"]["unsupported"]`.
  `mode="local"` is the default; `discrete=` is deprecated. The JSON interface takes `"mode"` and
  `"global_options"`, and the printout of a result shows the global analysis.
* `discrete()` handles several conditions, which share the coordinates, and events at given or
  estimated times: after an event the gaps are eps times the given gaps, as in the local analysis,
  and the coefficients are power series in eps up to a gap order that rises with the order. A map
  of the states proves a symmetry through the events. A map of the fiber that changes the
  coefficients four orders further up raises the orders.
* Under `equilibrate`, `discrete()` starts each condition at the resting state where the balances
  have one rational closed form in the coordinates. States that no observable depends on are left
  out of the fiber.
* `discrete()` starts one order above the rank criterion and ends at a count of one, which a higher
  order cannot change; a derivative in one equation only leaves the system with it; msolve runs with
  its probabilistic linear algebra over GF(p) and with the separating form only where it fails
  without; where the maps account for every point counted, the real count at a sample point is
  read from them. The counts do not change.
* `discrete()` handles switches on the states, the inputs or the clock: each condition from its
  start, and each allowed regime as a copy starting on a switching surface to an allowed
  neighbor, as in the local analysis. A map is proven with a map of the states in every regime
  that takes each switching surface to itself.

## Directions and reduction

* Each direction has a `kind` by the geometry of its orbits: `"scaling"` (a line in logarithmic
  coordinates), `"translation"` (a constant generator, a line in linear coordinates) or `"curved"`,
  also in `to_dict()` and the JSON interface. The printout groups the directions by it, and the
  labels X1, X2, ... of the report and of the reduction count scalings, translations and curved
  directions in this order.
* The documentation of the reduction calls the invariants, as coordinates on the orbit space, a
  chart; `alternatives` and `partial` are described in these terms.
* `reduce()` gives a chart per branch of a discrete symmetry, each side `s q > 0` of its fixed set
  `q = 0`: free values in the positive orthant map one to one onto the side, chained with the
  section, with `injective` and `reach` proven or listed under `unproven`, and the fixed set as the
  boundary. `reduce(branch="k01 > k12")` takes one as the trafo; without it the branches are
  alternatives. The choice of the solved coordinate and the names of the branches do not depend on
  the names or the order of the parameters. The JSON interface takes `"branch"`.
* `reduce(timeout=)` limits the wall time of the section search of each block to 600 seconds by
  default; `timeout=math.inf` lets it run to its end. A block that reaches the limit is named in
  `Reduction.limits`.

## Proofs, sections and fixes

* A free exponent of a parameter, b^n with b and n parameters, keeps n as a coordinate; before, n
  vanished into the recast atom and a scaling of b alone was reported. A parameter in an exponent is
  analyzed as base^theta only for a numeric base.
* A trafo writes Euler's number as `exp(1)`: the model reads `E` as a symbol, so a section that pins
  a logarithm at one failed the check over Q and the closed loop.
* A section under sign conditions is searched over every gauge coordinate, not only the one the
  first matching of carriers leaves, and the section without a square root, with `injective`
  proven and with the fewest sign conditions is kept. A system of invariants past the degree cap
  for one gauge coordinate may be solved for another.
* `injective` is proven for a section that covers part of the domain as well.
* The detailed summary of a section under sign conditions names the conditions.
* `discrete()` handles initial values that are rational functions of the coordinates, also those
  that a section sets in the reduced model.
* `discrete()` raises the order past a fiber without a finite count, where one count may take a
  quarter of `budget`, and returns without the counts in the domain where no order gives a
  finite count.
* `discrete()` on a model whose reduced model is not rational names the section that makes it so.
* The Lie order of a model with known inputs is proven sufficient: the order certificate takes
  the ranks on the states and coordinates, without the columns of the input jets, whose ranks rose
  by one per order and never reached a ceiling (docs/method.rst, "Unknown and known inputs").
  Models with events at several times and steady states are not covered yet. A certificate that
  ends at the jet chains of the inputs is a note in `info["limits"]`.
* The degree bounds of a model with known inputs are computed over the symbols each expression
  holds, which shortens the detection of models with long input jet chains.
* The narrow route tries every minimal support of the kernel while a direction stays open, not
  only the one the column order picks, so the levels and the rank no longer depend on the names
  of the parameters.
* A direction whose field Y the flow keeps up to a factor, [Y, F] = lam Y, or up to directions
  proven before, is proven exact, also without a field that commutes with F, such as a
  translation of two initial values. Through an event the derivative of the event map has to take
  the field into the span after it.
* Directions of models with unknown inputs are proven exact: the jets of the input, the input
  included, are the prolongation of the input component.
* The components of a field are written in the recast states of free powers (b**(c n) as E**c)
  and the logarithm of a recast enters the identities with its field; a zero test splits
  powers with symbolic exponents into powers of independent atoms.
* `reduce()` warns for a direction of level `"open"` or `"checked"` and for a failed saturation
  check, and reduces on the closed forms (an open direction stays in place); `allow_open=False`
  and `allow_checked=False` make them errors as before.
* The rank modulo unknown inputs is proven where an input cancels in the first Lie derivatives:
  the order at which an input first enters an observable is proven over Q, and jet j of the input
  enters from that order plus j on, so a jet column that cannot enter no longer counts against full
  rank.
* A state under a logarithm, a fractional power or a free exponent that starts at zero or below
  raises a ValueError; before, the start log(0) became a coordinate named `zoo`.
* `discrete()` proves a map of a model with given initial values where the states have to move:
  the state map is solved from the observables and their Lie derivatives where these are linear in
  it.
* The section search factors over Q with the roots of earlier branches as symbols, not over an
  algebraic number field, and the triangular elimination stops at the work budget.
* Under `equilibrate`, the states that hold the free resting values of the conserved totals are
  chosen by the structure of the model (not at zero, a coefficient without parameters, in the
  fewest totals, observed), not by their names or order. Where several choices tie, up to six are
  analyzed and the result with the best rank status, rank, levels and kinds is kept, so the result
  does not depend on the names or the order of the states and totals.
  `info["settings"]["free_initial"]` holds the states taken.
* `free_initial` takes every set of one state per total with independent coefficients, also for
  totals that share a species and in every order of `totals`. A state given twice, a state in no
  total or not a state, an input, the base of a free power, a set the totals do not leave
  independent and a state at zero at every resting state raise an `InputError`; before, some
  were ignored and others raised a singular resting Jacobian.
* A state of a total that rests at zero on one branch of the resting set, where a factor of other
  states vanishes on the other, is named in `info["branches"]`; `free_initial` naming it takes the
  branch away from zero.
* `reduce_cq` applies an event on a species of a total: the total changes by the coefficient of the
  species times the change of the species, read before the event, and an event on an eliminated
  species changes only the totals. `reduce_cq` keeps the states at zero of the model before the
  elimination, eliminates unobserved species first, solves the eliminated species jointly from the
  totals and names them in `info["settings"]["eliminated"]`. Rank and span of the directions are
  those without `reduce_cq`.
* The kernel checks of closed forms at random points read the resting states solved there, so a
  closed form with a component on a resting state is verified, not checked.
* A field of the invariant-field route is taken only where it leaves the resting value of a pivot
  in place; the parameter part of such a field alone is not a direction.
* Without an error bound of the invariant fields, verified closed forms with an error bound
  verify the rank.
* Under `equilibrate`, the exact scalings bound the rank in every case, so a rank at that bound
  proves the Lie order sufficient without random points, with and without `reduce_cq`.
* The totals read from a network or from the ODEs are nonnegative combinations where these span
  the conservation laws, the same for every order of the species. Under `reduce_cq`, a total with
  coefficients of both signs is named in `info["limits"]`.
* docs/method.rst, "Steady states", states the tangency rows of the reduced resting Jacobian, the
  choice of the pivots, `reduce_cq` with events, and the rank at the declared resting start.

# symident 0.14.0

## Discrete symmetries

* `detect(discrete=True)` and `Symmetries.discrete()` search discrete symmetries: other points of
  the domain with the output of a given one. A model with directions is reduced first, and the
  reduced model is analyzed. The search needs msolve, whose threads `cores` sets, and
  reports the counts without closed forms past `budget` seconds.
* The count holds the points over the algebraic closure modulo a prime and, at rational sample
  points, the real points and those in the declared domain, per point: the number in the domain
  may differ between regions. The order of the Taylor coefficients rises from the Lie order of
  the rank until the count is stable. The model is prolonged as in SIAN, with the derivatives
  of the states at the start as unknowns, an order per observable from the rank of its Taylor
  coefficients, the series of known inputs as numbers, the variables a linear equation fixes
  substituted, and the system split into blocks without a shared variable, whose fibers
  multiply.
* Where the other points are rational functions sigma of the coordinates, sigma is
  reconstructed from the characteristic polynomials of the coordinates over the fiber, factored
  over Q(theta), and the other coordinates on each branch. A sigma is `checked` on the output
  at random points and `exact` with a map of the states over Q(x, theta). Each lists whether it
  maps the domain into itself, never, or under listed conditions, and the factors on which it
  fixes theta. `count_final` holds when exact maps account for every point.
* `Symmetries.solution_count()` no longer counts points where a coordinate or a denominator
  vanishes: msolve reads no brackets, and the saturation was passed as a bracketed product.
* `info["discrete"]["fundamental_domain"]` gives inequalities that keep one point of each
  orbit, such as `k01 < k12`, and `Reduction.discrete` repeats it for a reduction of the same
  result.

# symident 0.13.6

* An event value that reads the states through `exp()`, `log()`, a non-integer power, `sin()`,
  `cos()` or `tan()` is expanded around the state at zero gaps: the function at that state is a
  leaf, the rest a series without constant term that the kernel sums to the truncation order.
  Such a value at a later event was evaluated at the initial values before, and a trigonometric
  one was refused.
* Events on a state under a free exponent move the power and its logarithm with the state, at the
  start and later, also under `equilibrate`; before, both kept their values from before the event.
  In log coordinates an added or multiplied dose on a state analyzed as `log(x)` is supported, as
  is an initial value whose logarithm does not split into logarithms of coordinates, such as
  `B*V + D`. Any event on a state inside an exponent or the argument of a trigonometric function
  is supported.
* Under `equilibrate` the relation rows of an exponential leaf that reads a resting state are
  scaled like the state columns.
* A free power takes an exponent with a rational offset, `V^(1 - n)`.
* A leaf that only a relation of another leaf reads is one coordinate; before, it entered twice,
  and the second copy, free of relations, failed the check of every reconstructed direction.

# symident 0.13.5

* The order bound multiplies each term by the number of points drawn until its random point was
  taken: the first without a pole of the starts, or under `equilibrate` the first with a resting
  state modulo the verification prime. `info["order_draws"]` holds the largest such number.
* The degree bounds of the order certificate and of the kernel check take the product of the
  distinct denominators of the right-hand sides as common denominator, in place of their least
  common multiple; on models with many denominators the detection is several times faster.
* The method page states the conditions of the invariant fields with the brackets and event
  images as identities over the rational functions, whose coefficients may have poles, and proves
  the kernel at every order and gap length through the functions the fields annihilate.

# symident 0.13.4

* Under `equilibrate`, a closed direction without exact proof is `verified` with an error bound
  where the roots the solves take lie on one component of each resting variety over Q(theta):
  balances solved by linear elimination, or interior roots conjugate over Q(theta). The bound
  takes the degree of the kernel identity on the resting variety times the Bezout numbers of the
  resting balances it reads. Undecided branches, resting states continued along a line at the
  check points, and a power or exponential in the balances leave the direction `checked`, with
  the reason in `info["bound"]`.
* The Lie order certificate takes the generic rank on a set that holds the start of every value
  of the coordinates: the starts shifted by the subspace of states that the field keeps, at
  random values of the coordinates, or under `equilibrate` the resting variety at a random
  resting point. A start that reaches this rank proves the order sufficient for generic values,
  at which the closed forms are checked, and the order bound covers the random points. Under
  `equilibrate` the certificate has an error bound.
* A check point at rest counts where every state is nonzero, unless linear elimination solves the
  balances. The bound at rest needs a resting Jacobian invertible at one resting point, and it
  covers the closed forms of the routes checked at random points; a static subsystem or a
  forward fit without exact proof stays `checked`.
* A gauged detection reports the general directions of its second pass in `info["proof"]`,
  `info["attempts"]` and `info["bound"]`.
* The homotopy route evaluates a closed form at the resting states solved at the check point.
* `switches()` refuses `True` and `False` for the path switches `MSOLVE` and `CACHE`.
* The user guide is split into chapters, and the method page states the stopping rule of the Lie
  order, the error bounds, the routes of the closed forms, invariant fields, gauges, conserved
  totals and the choice of pins.

# symident 0.13.3

* The exact proof of a direction expands each identity in a polynomial ring over Q with a cap on
  the terms of every polynomial and on the term products of one multiplication; past the cap the
  identity is undecided and the direction keeps the level of its modular check. An identity with a
  nonzero value modulo a prime at a random point is refuted without expansion.
* The state components that an initial value set by parameters gives a direction are solved from
  the parametrization of that value by those parameters, rational in the states also where the
  initial value holds a square root.

# symident 0.13.2

* The directions under `equilibrate` are the same for every naming and order of the
  parameters: the forward route solves another rate where the kernel refutes a form, takes
  the forms outside the span of the directions found, and the parametrization of the resting
  states tries each parameter first in turn.
* A direction through a conserved total, or of a model linear in its states, is proven exact
  where the state components solve the identities: the total reads an initial value on all
  its states, and the components of a linear model are solved from the identities.
* The linear conservation laws of `odes` are the `totals` when omitted, as for `reactions`.
* A symbolic power with a product base, as `(2*K)^n`, is split into powers of the factors.
* Under `equilibrate`, a power `b^n` of a positive rational `b` joins the base of `K^n`, as in
  `(2*K)^n`, or is held as a generic constant, with `log(b)` independent of the other
  constants; the second case is named in `info["limits"]`.
* Under `equilibrate`, a direction through the resting value of a subsystem that no event
  reaches is proven exact with events after or at the start, where the states that follow
  need components affine in them.
* The text of `info["rank_proof"]` is `proven symmetries` whenever the exact directions fill the
  kernel, whichever routes found them.
* `reduce` leaves a direction unreduced when its invariants read a resting state.
* A wrong model or argument raises `InputError`, a `ValueError`, that names it: the shape of
  the stoichiometry, an empty, unparsable or zero-dividing expression, a keyword or a state in
  `parameters`, an incomplete event, the length of a condition column, a total that is not
  conserved, `detect()` and `reduce()` arguments. An ignored `trafo` entry under
  `equilibrate` is also an entry in `info["limits"]`.
* `reduce` does not solve a system of invariants of degree above 32 in the carriers; the
  block keeps its invariants, and `Reduction.limits` names the cap.
* A total that the right-hand sides do not conserve raises an `InputError`, also without
  `reduce_cq` and `equilibrate`. The totals that a stoichiometry gives are kept where the
  equations conserve them exactly, as for a PEtab model with compartments of different sizes.
* Under `equilibrate` without `reduce_cq`, the state that holds the free resting value of a
  moiety is taken from the states of its total until the directions have closed forms; a
  direction that stays open names `reduce_cq` in `info["limits"]`.
* The user guide and the method page list the reasons in `info["proof"]`.

* A general direction is proven exact where a `trafo` sets initial values through other
  parameters, as `A = T - B`: the state components follow from the initial-value identities,
  solved for those parameters.

# symident 0.13.1

* A rank under `equilibrate` is proven where the resting Jacobian is invertible at the base
  point. Where the resting state there needs msolve and msolve is missing, the rank is
  verified, and `info["limits"]` names the reason; the analysis itself does not need msolve
  for that check.
* The report of a reduction on a declared domain names that domain.
* Wheels for Python 3.15.

# symident 0.13.0

## Certificates and levels

* A certified Lie order and the gap-order bound hold except with probability at most
  `info["order_bound"]`, which the error bound of verified results includes; where no bound
  applies, `info["order_bound_why"]` names the reason. `info["gap_bound"]` is the bound that
  proved the gap order; an event time that holds a symbol gets none.
* The certificate frees a state that an event value reads. A certified rank below the scanned
  one leaves the order uncertified, named in `info["limits"]`.
* Ranks modulo unknown inputs, resting states and switch surfaces are proven where the projected
  columns have full rank, else verified with the probability in `info["projection"]`. The
  printout says what a verified rank or a lower bound rests on.
* Verification points are uniform modulo the verification prime, and the error bound counts the
  points drawn (`info["bound"]["draws"]`).
* An exact scaling holds for the initial values, the events and the resting model as well; an
  exact lift needs every parameter component constant along the flow.
* The chains of known inputs are lengthened while the Lie order reaches past them
  (`info["known_inputs"]`).

## Switches and unknown inputs

* An unknown input may enter a model with switches: it is continuous at every surface and
  analytic between two switches. Each condition needs a separate input; a switch on the
  input, an event after the start and an initial value or event that reads it are refused.
* A coordinate is non-identifiable only if it is so in every allowed regime; the state at a
  switch is any point of the surface.
* A state that starts at zero and that every regime and event keeps at zero is taken out of the
  model first (`info["zero_states"]`).

## Resting states

* Where a drained reaction or the resting balances have several solutions, `info["branches"]`
  names the one taken.
* Under `equilibrate`, a state without dynamics that is not an input rests at the initial value,
  and an event at the start applies on top of it.
* A logarithm, a fractional power and the base of a free exponent need a positive argument on
  the declared domain, in the right-hand sides, observables, initial values and event values;
  otherwise an error names the symbols. With `reduce_cq` the check reads the model before the
  elimination.

## Reduction

* A non-polynomial atom such as `log(x)` and an AM-GM certificate count as positive only where
  proven; an entry or a pin without a certificate is listed under `unproven`.
* Without declared bounds, a block whose sections all need a sign condition is reduced on part
  of the domain, with the conditions in `sign_conditions`; such a section ranks last.
* Alternative reductions are ranked by their level first.
* Under `equilibrate`, `reduce` lifts a direction that leaves the resting equations unchanged,
  with state components 0, for resting systems of up to eight states.

## Model and PEtab

* The conserved totals of a reaction network are integer combinations of the species; the
  totals of an ODE model may have parameter coefficients, such as volumes in `A + V*B`.
* `pi` and Euler's number are numbers in the angles of trigonometric functions.
* `read_petab()` renames ids the expression language cannot read (`m.petab["renamed"]`), takes
  the time of an SBML event per experiment, and reads problems with thousands of reactions and
  experiments.
* The verbose printout splits the elapsed time into rank, reconstruction and proofs.

# symident 0.12.0

## Unknown inputs

* `Model(unknown_inputs=)` declares unknown functions of time: symbols in the right-hand
  sides and observables without an equation. A direction may change them along with the
  parameters, while `rank` and `dim` count the parameters and initial values only. An exact
  direction lists the change of each input in `input_change`, printed under
  "input change:".
* The inputs enter as chains of jets whose initial values are nuisance coordinates of the
  kernel. `info["unknown_inputs"]` lists the jets, the Lie order up to which they are exact
  and the rank of the jet columns; `reconst_control(input_jet_order=)` sets their number.
  A rank that is neither full nor matched by exact directions is a lower bound.
* A string in `conditions` renames an unknown input per condition. Across an event the
  input stays analytic, so a jump of a state can separate parameters that the input would
  otherwise absorb. Under `equilibrate` an unknown or known input is zero at rest and takes
  its jets from the start on. Switches, an event or a `trafo` entry for an unknown input and
  `engine="symbolic"` are refused.

## Known inputs and discrete symmetries

* `Model(known_inputs=)` declares known functions of time of no assumed form: jet chains
  whose initial values are known symbols.
* `Symmetries.solution_count()` counts the values of the coordinates with the output of a
  random point modulo a prime (msolve), over the algebraic closure; more than one is a
  candidate discrete symmetry. It reads the clock and known inputs, substitutes the `trafo`
  into the initial values as the detection does, takes a one-row grid and one-element
  lists, returns None for a model that is not rational and refuses switches and a result
  under `equilibrate` or `reduce_cq`.

## Detection

* A closed form reads every name of the model as a symbol, also `E`, `I`, `S`, `N`, `rf`
  or `beta`. A kernel check that evaluates only part of the entries stays `"checked"`.
* An event value after the start reads the states just before the event; one at the start
  reads the initial values before the events. A state without dynamics that an event sets
  to another state is analyzed as a state.
* When the saturation guard sees the rank grow past the Lie order used, the saturation is
  repeated from past that order.
* A free exponent of a number enters through the primes of the number; exponent terms that
  are rational combinations of each other up to a constant share their leaves, and a
  logarithm inside an exponent is a leaf of its own.
* A symbol in two different logarithms, or with a free power beside an affine logarithm,
  gets no logarithmic coordinate; auxiliary states take its powers and logarithms.
* `positive` accepts expressions such as `"1-x"`.
* Condition columns and constant states keep their logarithms and free powers, which become
  numbers per condition; `0^n` is 0 for a positive exponent `n`.
* A free power of a state with a given initial value starts at the power of that value.
* The reset of a root event enters the stacked rows of a model with switches. When the
  rule of the switches leaves directions open without candidates, the report says so.
* A switch that the default `positive=True` decides through the sign of a state warns.
* `regimes` names a switching surface in any algebraic form.
* `info["timing"]` holds the seconds at each stage of the analysis.
* `time` in an initial value reads the clock at the start, in an event value at the time of
  the event. The model has the clock wherever a right-hand side, observable, `trafo` value
  or condition cell reads `time`; under `equilibrate` the clock is an input that rests at
  zero and counts from the start, and `exp()` of it is refused by name.
* A `trafo` value or condition cell enters the equations as one operand, in parentheses.
* An event time that a condition cell or a per-condition `trafo` sets is resolved per
  condition, also to the start.
* `scalings_only` holds the initial values, event values and times, per-condition `trafo`
  entries, one-row grids and known inputs as the full detection does and reports its
  coordinates; event times in parameters are supported.
* `gauge` keeps the unknown inputs.
* `reduce_cq` reads the `totals` of an ODE model; a total that the right-hand sides do not
  conserve is an error. A moiety species with an initial value in `trafo` stays a state
  where another species of the moiety can be eliminated.
* A right-hand side that the `trafo` sets to zero rests like a constant state.
* A name in `known_inputs`, `inputs`, `fixed`, `parameters`, a condition column or an event
  `var` that the model does not hold is an error; inputs and event variables are states. A
  name in `positive` outside the model warns.
* An input starts at zero: an initial value of an input in `trafo` or a condition column is
  an error that names the event to set it with. `Model(inputs=)` documents it.
* A `trafo` entry that replaces a symbol a condition column sets warns.
* `detect()` keeps a copy of the model, so later changes of the model leave the result as
  it is.
* Under `equilibrate`, `info["bound"]` says that the error bound does not cover a start at
  rest.
* A Lie derivative past the size cap leaves the gap order a lower bound; an exponential
  at the start takes a generic value in the gap bound.
* The scaling rows expand each rational right-hand side once, in sparse polynomial
  arithmetic.

## Equilibrate

* A state with a signed net rate such as `-k*(x - u)` rests at zero, on the reaction and
  the ODE path and per condition; a state whose right-hand side vanishes at rest keeps
  its resting value as a coordinate.
* Exponentials of parameters outside the resting model are supported.
* A joint solve that fails names the condition.
* A full-rank result lists the parameter coordinates only.

## Reduction

* `reduce()` refuses a direction of level `"checked"` and a result whose saturation check
  failed, unless `allow_checked=True`.
* The closed loop applies the trafo to symbolic condition cells. It fails when a trafo
  entry has no effect, and is reported as not run when the detection refuses the reduced
  model; the summary line says when it does not confirm the reduction. An entry for a
  symbol of the grid goes into its cells, an entry that the model trafo consumes stays
  out of the composed trafo, and under `equilibrate` a numeric pin of a resting value goes
  into `fixed` (`Reduction.trafo` documents it).
* Section candidates start with pins of one coordinate; balances through fixed symbols
  follow those between coordinates and displace none of them. The reason a block is not
  reduced is printed.
* A declared positive expression such as `"1-a"` bounds the domain of the reduction: a
  section that leaves the bound counts as partial, and pins in the middle of a bound are
  tried first.
* `Symmetries.reduce()` returns a copy of the reduction computed first for the same
  arguments.
* `reduce()` and `reduce_json()` take the declared positive coordinates from the detection
  without a warning for the states and fixed names among them.
* A refused `"checked"` direction names why it is checked: without an error bound, or on
  part of its entries.

## PEtab

* The reader drops species without dynamics, events or inputs that start at one value,
  and sets a total that cancels below rounding to zero. The initial values of inputs
  under preequilibration raise no warning.

# symident 0.11.0

## PEtab

* `read_petab()` and `Model.from_petab()` read a PEtab problem, format version 1 or 2,
  into a `Model` (extra `symident[petab]`). Volumes that change in time dilute the
  concentrations, and a reset volume keeps the amounts of its species.

## Model

* `Model(equilibrate=, positive=, reduce_cq=)` set the defaults of `detect()` for the
  model; `read_petab()` sets them from the problem. `reduce(positive=)` defaults to the
  setting of the detection.
* A condition starts at t = 0, or at its earliest event when that comes before.

## Levels and assumptions

* New level `"checked"`: a closed form checked at random points without an error bound.
  A `"verified"` direction always has its probability in `info["bound"]`, and a rank
  without a probability is a lower bound. A closed form whose kernel check can evaluate
  only part of its entries is `"checked"` as well. `Symmetries.unproven()` returns the
  verified and checked directions.
* Directions of a model with switches that the rule of the switches does not prove are no
  longer reported as symmetries: they are listed under `info["switching"]["candidates"]`,
  and the printout says how many more directions there may be.
* New `info["assumptions"]` and `Reduction.assumptions`: the declared `domain` and the
  sides of switches that `regimes` declares at the start (`starts`), together with the
  `error_bound` of the verification.
* The start sides that `regimes` declares move from
  `info["switching"]["conditions"][i]["assumptions"]` to `["declared"]`.

## Reduction

* More section claims are proven: that every orbit meets a section at most once, for any
  number of generators (`blocks[i]["injective"]`), and that every level set of the
  invariants is one orbit, so that every orbit meets the section (`blocks[i]["reach"]`),
  also under partial coverage.
* A section on a face of the parameter space is proven to give the output of every orbit
  in the limit, also under `equilibrate`. A face whose resting state is only checked at
  random points has `blocks[i]["face_level"]` `"checked"`, and so has its block.
* A section claim that is not proven is named under `blocks[i]["unproven"]`, makes the
  block `"open"` and gives a warning. This replaces the level `"assumed"` and
  `blocks[i]["assumptions"]`; `reduce(verbose=)` is removed.
* `blocks[i]["new_parameters"]`, `blocks[i]["solved_domain"]` and `choices` in `blocks[i]`
  and `family[i]` replace `survivor_meaning`, `carrier_domain` and `admissible`.
* The section search also tries a product of two coordinates at one. A search that a
  budget cuts short is named in `limits`.
* `zero_limits` reports "no" only when every branch of the face equations is solved.
* Ranks and signs in the reduction are exact; no floating point remains.

## Detection

* `equilibrate`: conditions at rest that share parameters are continued as power series
  from a common start instead of being solved jointly with msolve; `SYMIDENT_NODIAG`
  switches this off. `info["series_certified"]` says whether the series order is proven
  sufficient; otherwise the rank is a lower bound. msolve uses the cores of the call.
* Condition-specific parameters in log coordinates map back to their names.
* `switches()` also sets `CACHE`, `MSOLVE` and the switches of the msolve build;
  `install_msolve()` takes `MSOLVE_VERSION` and `JOBS` as its defaults. `LIBS_CACHE` is
  removed, `SYMIDENT_CACHE` sets the cache. `ROUTEDIAG` replaces `ROBUSTDIAG`, and
  `BLOCKMB` is listed.

# symident 0.10.1

* On Windows the msolve builds in the cache are found as `msolve.exe`.

# symident 0.10.0

## Levels and proofs

* Every direction has a `level`: `"exact"` (proven over Q), `"verified"` (checked at random
  points modulo a large prime) or `"open"` (no closed form). `Symmetry.certified` is gone;
  `Symmetries.unproven()` returns the verified directions.
* Closed forms are proven over Q through a lifting field, also per regime, in the recast
  coordinates of symbolic exponents, for polynomial outputs in t and on resting states.
  `reconst_control(exact_proof=, exact_work_cap=)` controls the proof of combined regimes.
  `Symmetry.state_components` holds the state components of an exact direction.
* `info["rank_status"]` says whether the rank is proven, verified or a lower bound, and why.
  A rank from the invariant fields is verified (`info["rank_fields"]`), checked at random
  points modulo the drawn verification prime; `info["rank_fields_bound"]` bounds the
  probability that a wrong field passes.
* `info["bound"]` is the probability that a wrong closed form passes the kernel check, also
  through events after the start; it is given only where the Lie and gap orders are proven.
* Under events after the start, the gap order is raised to a proven rank bound
  (`info["gap_certified"]`, `info["gap_certificate"]`); otherwise the rank is a lower bound
  with an entry in `info["limits"]`.
* `engine="symbolic"` proves its Lie order, its closed forms and its rank
  (`info["lie_certificate"]`).
* `info["proof"]` says per closed direction why the exact proof does not apply.
* Models with switches list the directions the rule does not prove as `"verified"`
  (`info["switching"]["uncertified"]`).

## Detection

* The homotopy route runs without `equilibrate`, through events after the start, with
  radicals in initial values and in recast coordinates (`reconst_control(homotopy_degree=)`).
* Entries past `degree_cap` are probed up to `reconst_control(degree_cap_max=)`.
  Directions of minimal support are found at any width.
* Under `equilibrate` with reactions, only drained species rest at zero and a rate is zero
  at rest only when it vanishes exactly. A held moiety pivot rests at its parameter. The
  weights that find drained species solve their linear program exactly over Q.
* Results do not depend on `cores` or `PYTHONHASHSEED`; `reconst_control(seed=)` seeds every
  random choice. Work budgets replace clocks (`SYMIDENT_JETWORK`, `work_cap`).
* `info["attempts"]`, `info["errors"]`, `info["limits"]`, `info["branches"]` and
  `info["settings"]` record routes tried, exceptions, budgets reached, resting state
  branches and settings; `summary()` shows them. `SYMIDENT_DEBUG` raises the exceptions.
* `symident.switches()` sets switches for a `with` block; `SYMIDENT_DIAG` turns on all
  diagnostics. Unused route switches are removed; an unknown `SYMIDENT_*` name warns.

## Model strings

* The expression functions of cppDE parse: `exp2`, `cbrt`, `root`, `pow`, the inverse
  reciprocal trigonometric and hyperbolic functions and `atanh`. `floor()`, `ceiling()`
  and `round()` are refused; `erf()` and `erfc()` need `engine="symbolic"`.
* `pi` is the number and cannot name a symbol. `==` and `!=` decide by identity. Names such
  as `N`, `S`, `E`, `I` or `beta` stay model symbols. A switch on parameters alone needs
  its side in `regimes`.
* The printout reads `X = Σᵢ ξ(i) ∂ᵢ`; the docs write θ for the parameters and ξ for the
  components of a generator.

## Reduction

* `reduce()` reads the levels: an `"open"` direction is an error unless `allow_open=True`,
  a `"verified"` one warns. The reduction has a level per block and overall (`"exact"`,
  `"verified"`, `"assumed"` with the named assumptions).
* `reduce()` checks its trafo over Q (`Reduction.checks`) and reruns the detection on the
  transformed model (`check=True`). `Reduction.errors` lists failing sections.
* Signs, invariant independence and ranks are decided exactly or with guaranteed precision;
  floats only order candidates. A scaling pinned at a coordinate not declared positive
  covers part of the orbits (`coverage = "partial"`).
* The section search stops after `reduce(work_cap=4000)` candidates; `timeout` defaults to
  infinity. Reached limits are listed in `Reduction.limits`.

# symident 0.9.3

* `install_msolve()` builds GMP and FLINT for the generic architecture instead of the CPU it
  runs on. The binary no longer stops with an illegal instruction on another CPU sharing the
  cache, or on a virtual machine that hides instruction sets its CPU model implies.
* The reduction report lists each section as its equation. `summary(detailed=True)`
  explains each section.
* The summary no longer shows how a closed form was found.
* The block lines of `summary()` are aligned in columns and show the section as its
  equation, with the number of possible choices.

# symident 0.9.2

* `log(x, b)` and `atan2(y, x)` parse: the commas between the arguments were dropped.
* `atanh()` of an argument whose logarithms cancel in part, such as `atanh(x/(1 + x))`,
  no longer stops with a `KeyError`.

# symident 0.9.1

* Wheels for Python 3.14.
* Linux wheels are built on manylinux_2_28 and need glibc 2.28 or newer; on the older
  manylinux2014 base, installing the wheel built SciPy from source.
* The tests pass on Windows: they read their data as UTF-8, and the msolve tests that run a
  shell script are skipped there.

# symident 0.9.0

First release.

* `Model`, `detect()` and `reduce()`: non-identifiable directions of ODE models with
  observables, conditions, events and inputs, as exact generators where they close, and the
  reparametrization that removes them.
* Observability rank over finite fields in a C++ kernel, with a certified Lie order, exact
  scaling symmetries and rank proofs by scalings or invariant fields.
* Elementary functions of states (logarithms, inverse trigonometric and hyperbolic
  functions, powers of sums, nested) recast into rational form; their transcendental
  initial values become generic leaves.
* Steady-state constraints (`equilibrate`) through the joint resting-state system.
  Directions that no sampling route closes are fitted along parameter lines through the
  continued resting state, in the parameters and, where needed, the resting states
  (`reconst_control(homotopy=)`).
* Switches in right-hand sides and observables: `piecewise()` in SBML order, `ifelse()`,
  comparisons, `&&`, `||`, `!`, `Heaviside()`, `sign()`, `abs()`, `max()` and `min()`, and
  events with a `root`. A direction is reported when the observables of every regime
  that the declared signs do not exclude stay unchanged along it, it maps every surface
  of a visible switch to itself and keeps the events; a switch hidden from the
  observables may move. `Model(regimes=)` restricts the analysis to the regimes an
  experiment visits. The rank is certified between the stacked rows of the regimes and
  directions proven symbolically, and an error is raised where they do not meet.
* Reduction by scaling transversals, module reduction and invariants (monomial, polynomial,
  rational, Darboux, exponential), with positivity certificates and face sections.
* `reduce()` returns the best reparametrization and lists the others under `alternatives`
  (`reduce(alternatives=True)`); dependent directions are named under `dependent`.
* `install_msolve()` builds msolve, used for coupled steady states, into a per-user cache.
* `symident.rjson`: JSON interface for host languages.
