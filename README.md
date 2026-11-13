# symident

[![wheels](https://github.com/dModverse/symident/actions/workflows/wheels.yml/badge.svg)](https://github.com/dModverse/symident/actions/workflows/wheels.yml)
[![docs](https://img.shields.io/badge/docs-dmodverse.github.io%2Fsymident-blue)](https://dmodverse.github.io/symident/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

<!-- description-start -->

Structural identifiability of ODE models by Lie symmetries over finite fields.

symident finds the non-identifiable directions of an ODE model with observables,
conditions, events and inputs, returns them as exact generators where possible, and
reduces the model to identifiable parameters. The rank of the OIM is computed over F<sub>p</sub> in
a C++ kernel; closed forms are reconstructed by sparse interpolation and proven over ℚ, or
checked at random points modulo a large prime, with an error bound where one applies.

symident continues the Lie-symmetry analysis of dMod (Merkt, Timmer and Kaschek,
[Phys. Rev. E 92, 012920, 2015](https://doi.org/10.1103/PhysRevE.92.012920)); the
[method page](https://dmodverse.github.io/symident/method.html) states the method and the
work it builds on.

<!-- description-end -->

## Installation

```sh
pip install symident
```

Wheels exist for Linux, macOS on Apple silicon and Windows and Python 3.10 to 3.15. The
[installation page](https://dmodverse.github.io/symident/install.html) covers the build
from source, the PEtab extra, [msolve](https://msolve.lip6.fr) for coupled steady states,
and the use from R through dMod2.

## Example

```python
import symident as si

m = si.Model({"A": "-k1*A", "B": "k1*A - k2*B"}, {"y": "s*B"})
res = si.detect(m, reconstruct=True)
print(res)
#> Result:  rank 4 / 5  |  1 non-identifiable direction
#>
#> Generators  X = Σᵢ ξ(i) ∂ᵢ
#>
#> Scalings:
#>   X₁  (exact)
#>       ξ(A) :  + A
#>       ξ(B) :  + B
#>       ξ(s) :  - s
#>
#> exact = proven over Q.
res.symmetries[0].generator
#> {'A': 'A', 'B': 'B', 's': '-s'}
print(res.reduce())
#> Reduced 1 of 1 direction.
#>
#> Trafo (non-identity entries)
#>   A = 1
#>
#> Sections:
#>   {X₁}  A = 1
#>   summary(detailed=True) explains each section.
```

## Interface

| | |
|---|---|
| `Model(odes, observables, ...)` | ODEs or a reaction network with observables, conditions, events, inputs and switches |
| `detect(model, ...)` | rank, identifiability and the non-identifiable directions (`Symmetries` of `Symmetry`) |
| `reduce(result, ...)`, `result.reduce()` | reparametrization that removes the directions (`Reduction`) |
| `read_petab(path)`, `Model.from_petab(path)` | a `Model` from a PEtab problem, format version 1 or 2 (`pip install "symident[petab]"`) |
| `reconst_control(...)` | settings of the saturation and the reconstruction |
| `switches(...)` | diagnostic and route switches for a `with` block |
| `install_msolve()` | builds msolve, needed for coupled steady states and `solution_count()` |
| `PEtabError` | raised for a PEtab problem the reader does not support |

Both results print a report; `summary()` adds the computation details and
`to_dict()` returns the raw result.

## Contact

Simon Beyer, simon.beyer@fdm.uni-freiburg.de. Requests for further functions in model
equations, questions and bug reports are welcome by e-mail or as an
[issue](https://github.com/dModverse/symident/issues); the supported functions are listed
in the [user guide](https://dmodverse.github.io/symident/guide/models.html#expressions).

## License

MIT
