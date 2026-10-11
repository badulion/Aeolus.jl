# Installation

Aeolus.jl requires **Julia 1.10** or newer (developed and tested on 1.12).

## From a local clone

```julia
julia> using Pkg
julia> Pkg.develop(path = "/path/to/Aeolus.jl")
julia> Pkg.instantiate()   # only needed the first time
```

or, if you cloned the repo and want to use it in-place:

```bash
$ cd Aeolus.jl
$ julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

## Dependencies

Aeolus keeps its dependency footprint deliberately small at this stage:

| Package     | Purpose                                          |
| :---------- | :----------------------------------------------- |
| `FFTW`      | Direct FFT-based Poisson solver.                 |
| `LinearAlgebra` (stdlib) | `mul!` for planned FFT execution.   |
| `Printf`    | Formatted output in examples.                    |

Test extras: `Random`, `Statistics`.

## Running tests

```bash
$ julia --project=. test/run.jl                          # fast tier, in parallel
$ AEOLUS_TEST_TIER=full julia --project=. test/run.jl    # full tier, in parallel
$ julia --project=. test/runtests.jl                     # serial (also Pkg.test())
```

The fast tier (the default) skips the validation mini-simulations; the
full tier runs everything. See [Running the tests](testing.md) for the
tiers, the runner's options and measured times.

## Running examples

```bash
$ julia --project -m Aeolus run examples/taylor_green_vortex/taylor_green_vortex.toml
$ julia --project -m Aeolus run examples/buoyant_convection/buoyant_convection.toml
$ julia --project -m Aeolus run examples/scalar_advection_2d/scalar_advection_2d.toml
```

An example is a TOML file (and its input drivers) run by `aeolus run`; the
`.jl` files beside some of them only read the run folder's output and plot
it. The manufactured-solution checks of the Poisson solvers are tests
(`test/runtests.jl`).

## Building the docs

```bash
$ julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
$ julia --project=docs docs/make.jl
```

The generated site lives in `docs/build/`.

## Optional: benchmarks

The benchmark harness lives in `benchmarks/`. See the
[Benchmarks](benchmarks.md) page for how to run it.
