# symident

[![wheels](https://github.com/dModverse/symident/actions/workflows/wheels.yml/badge.svg)](https://github.com/dModverse/symident/actions/workflows/wheels.yml)
[![docs](https://img.shields.io/badge/docs-dmodverse.github.io%2Fsymident-blue)](https://dmodverse.github.io/symident/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Structural identifiability of ODE models by Lie symmetries over finite fields.

symident finds the non-identifiable directions of an ODE model with observables,
conditions and events, returns them as exact generators where possible, and reduces
the model to identifiable parameters. The observability rank is computed modulo
primes in a C++ kernel; closed forms are reconstructed by sparse interpolation and
verified at fresh primes.

## Installation

```sh
pip install git+https://github.com/dModverse/symident
```

This builds from source and needs a C++17 compiler and CMake; with OpenMP the kernel runs
on several threads. Once symident is on PyPI, `pip install symident` installs a wheel.
Wheels are built for Linux (x86_64, aarch64; tested on Ubuntu, Debian and Fedora), macOS
(x86_64, arm64) and Windows (x86_64).

Steady-state analyses (`equilibrate=True`) whose resting state is a coupled nonlinear
polynomial system use [msolve](https://msolve.lip6.fr). On Linux and macOS

```python
import symident
symident.install_msolve()
```

builds it with GMP, MPFR and FLINT into a per-user cache (needs a C compiler, `make`, `m4`
and `curl` or `wget`), where symident finds it. Alternatively point `SYMIDENT_MSOLVE` at an
msolve executable; on Windows that is the only way.

## Example

```python
import symident as si

m = si.Model({"A": "-k1*A", "B": "k1*A - k2*B"}, {"y": "s*B"})
res = si.detect(m, reconstruct=True)
print(res)
#> Result:  rank 4 / 5  |  1 non-identifiable direction
res.symmetries[0].generator
#> {'A': 'A', 'B': 'B', 's': '-s'}
print(res.reduce())
#> Reduced 1 of 1 direction.
#> Trafo (non-identity entries)
#>   A = 1
```

## Interface

| | |
|---|---|
| `Model(odes, observables, ...)` | ODEs or a reaction network, observables, `trafo`, `conditions`, `events`, `inputs`, `fixed` |
| `detect(model, ...)` | rank, identifiability and the non-identifiable directions (`Symmetries`) |
| `reduce(result, ...)`, `result.reduce()` | reparametrisation that removes the directions (`Reduction`) |
| `reconst_control(...)` | settings of the saturation and the reconstruction |

Both results print a report; `summary()` adds the computation details and
`to_dict()` returns the raw result.

## Contact

Simon Beyer, simon.beyer@fdm.uni-freiburg.de. Requests for further functions in model
equations, questions and bug reports are welcome by e-mail or as an
[issue](https://github.com/dModverse/symident/issues); the supported functions are listed
in the [user guide](https://dmodverse.github.io/symident/guide.html#expressions).

## License

MIT
