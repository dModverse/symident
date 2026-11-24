# symident

[![CI](https://github.com/dModverse/symident/actions/workflows/ci.yml/badge.svg)](https://github.com/dModverse/symident/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/symident)](https://pypi.org/project/symident/)
[![docs](https://github.com/dModverse/symident/actions/workflows/docs.yml/badge.svg)](https://dmodverse.github.io/symident/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<!-- description-start -->

Structural identifiability of ODE models by Lie symmetries over finite fields, local and global.

The **local analysis** computes the rank of the observability-identifiability matrix over
F<sub>p</sub> in a C++ kernel, with the Lie order certified sufficient, and returns the
non-identifiable directions as exact generators. A rank and a closed form are proven over ℚ or
verified at random points with an error bound. The model is then reduced to identifiable
parameters, with sections certified on the declared domain.

The **global analysis** counts the parameter values in the domain with the output of a generic
one and returns the maps between them, the discrete symmetries, as rational maps. The count is
certified exact, by the field of the output coefficients, or verified over ℚ with an error bound.

Models have observables, conditions, events, known and unknown inputs, switches and resting
states; PEtab problems are read directly, readouts at rest included. Large models are analyzed
by kernels per chain of states over the columns each chain reads, with the jets reused across
orders.

symident continues the Lie-symmetry analysis of dMod (Merkt, Timmer and Kaschek,
[Phys. Rev. E 92, 012920, 2015](https://doi.org/10.1103/PhysRevE.92.012920)); the
[method chapters](https://dmodverse.github.io/symident/method.html) state the mathematics and
the work it builds on.

<!-- description-end -->

## Installation

```sh
pip install symident
```

Wheels exist for Linux, macOS on Apple silicon and Windows and Python 3.10 to 3.15; they depend
on NumPy, SciPy, SymPy and python-flint. The global analysis and steady states with coupled
balances need [msolve](https://msolve.lip6.fr), which `symident.install_msolve()` builds. The
[installation page](https://dmodverse.github.io/symident/install.html) covers the build from
source, the PEtab extra, msolve, and the use from R through
[dMod2](https://dmodverse.github.io/dMod2/).

## Example

The two-compartment model with an input into the first compartment, observed in the second, is
locally identifiable, but exchanging the elimination rate `k01` and the return rate `k12` keeps
the output:

```python
import symident as si

m = si.Model({"x1": "-(k21 + k01)*x1 + k12*x2 + u", "x2": "k21*x1 - k12*x2"}, {"y": "x2"}, known_inputs=["u"])
res = si.detect(m, mode="global")
print(res)
# > Result:  structurally locally identifiable (rank 5 / 5)
# >
# > Global: values with the output of a generic one, counted at random points
# >   2 over the algebraic closure, 1 or 2 in the domain (order 5, stable)
# >   exactly 2 values at a generic point, proven
# >   sigma1 (exact): k01 -> k12, x1 -> (k01*x2 - k12*x2 + k21*x1)/k21, k12 ->
# >   k01; in the domain where (k01*x2 - k12*x2 + k21*x1)/k21 > 0
# >   fundamental domain: k01 < k12
```

A model with non-identifiable directions prints them as generators, and `res.reduce()` returns
the reparametrization to identifiable parameters; the
[quickstart](https://dmodverse.github.io/symident/quickstart.html) shows both.

## Interface

|  |  |
| -- | -- |
| `Model(odes, observables, ...)` | ODEs or a reaction network with observables, conditions, events, inputs and switches |
| `detect(model, ...)` | rank, identifiability and the non-identifiable directions (`Symmetries` of `Symmetry`) |
| `detect(model, mode="global", global_options=...)` | also the count of the values with the same output and the discrete symmetries |
| `reduce(result, ...)`, `result.reduce()` | reparametrization that removes the directions (`Reduction`) |
| `result.discrete(...)` | the global analysis on a result of `detect()` |
| `read_petab(path)`, `Model.from_petab(path)` | a `Model` from a PEtab problem, format version 1 or 2 (`pip install "symident[petab]"`) |
| `reconst_control(...)` | settings of the saturation and the reconstruction |
| `switches(...)` | diagnostic and route switches for a `with` block |
| `install_msolve()` | builds msolve, needed for the global analysis and coupled steady states |
| `PEtabError` | raised for a PEtab problem the reader does not support |

Both results print a report; `summary()` adds the computation details and
`to_dict()` returns the raw result.

## Contact

Simon Beyer, simon.beyer@fdm.uni-freiburg.de. Requests for further functions in model
equations, questions and bug reports are welcome by e-mail or as an
[issue](https://github.com/dModverse/symident/issues); the supported functions are listed
in the [guide](https://dmodverse.github.io/symident/guide/models.html#expressions).

## License

MIT
