# symident 0.9.0

First release.

* `Model`, `detect()` and `reduce()`: non-identifiable directions of ODE models with
  observables, conditions, events and inputs, as exact generators where they close, and the
  reparametrisation that removes them.
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
* `reduce()` returns the best admissible chart and lists the others under `alternatives`
  (`reduce(alternatives=True)`); dependent directions are named under `dependent`.
* `install_msolve()` builds msolve, used for coupled steady states, into a per-user cache.
* `symident.rjson`: JSON interface for host languages.
