# symident

[![wheels](https://github.com/dModverse/symident/actions/workflows/wheels.yml/badge.svg)](https://github.com/dModverse/symident/actions/workflows/wheels.yml)
[![docs](https://img.shields.io/badge/docs-dmodverse.github.io%2Fsymident-blue)](https://dmodverse.github.io/symident/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Structural identifiability of ODE models by Lie symmetries over finite fields.

symident finds the non-identifiable directions of an ODE model with observables,
conditions, events and inputs, returns them as exact generators where possible, and
reduces the model to identifiable parameters. The rank of the OIM is computed over F_p in
a C++ kernel; closed forms are reconstructed by sparse interpolation and proven over Q, or
checked at random points modulo a large prime, with an error bound where one applies.

## Installation

```sh
pip install symident
```

Wheels are built for Linux (x86_64, aarch64, glibc 2.28 or newer; tested on Debian
and Fedora), macOS (x86_64, arm64) and Windows (x86_64), for Python 3.10 to 3.14. Elsewhere
pip builds from source, which needs a C++17 compiler and CMake; with OpenMP the kernel runs
on several threads. The development version installs with
`pip install git+https://github.com/dModverse/symident`.

Steady-state analyses whose resting states need a coupled nonlinear solve use
[msolve](https://msolve.lip6.fr); on Linux and macOS `symident.install_msolve()` builds it
into a per-user cache. The [installation page](https://dmodverse.github.io/symident/install.html)
covers Windows and other setups.

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
| `detect(model, ...)` | rank, identifiability and the non-identifiable directions (`Symmetries`) |
| `reduce(result, ...)`, `result.reduce()` | reparametrization that removes the directions (`Reduction`) |
| `read_petab(path)`, `Model.from_petab(path)` | a `Model` from a PEtab problem, format version 1 or 2 (`pip install "symident[petab]"`) |
| `reconst_control(...)` | settings of the saturation and the reconstruction |
| `switches(...)` | diagnostic and route switches for a `with` block |

Both results print a report; `summary()` adds the computation details and
`to_dict()` returns the raw result.

## Contact

Simon Beyer, simon.beyer@fdm.uni-freiburg.de. Requests for further functions in model
equations, questions and bug reports are welcome by e-mail or as an
[issue](https://github.com/dModverse/symident/issues); the supported functions are listed
in the [user guide](https://dmodverse.github.io/symident/guide.html#expressions).

## License

MIT
