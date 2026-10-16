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
* `reduce()` returns the best admissible chart and lists the others under `alternatives`
  (`reduce(alternatives=True)`); dependent directions are named under `dependent`.
* `install_msolve()` builds msolve, used for coupled steady states, into a per-user cache.
* `symident.rjson`: JSON interface for host languages.
