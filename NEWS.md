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
