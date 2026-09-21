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

```julia
julia> using Pkg
julia> Pkg.test("Aeolus")
```

Full suite runs in ~10 s on a modern laptop.

## Running examples

```bash
$ julia --project=. examples/taylor_green_vortex.jl
$ julia --project=. examples/buoyant_convection.jl
$ julia --project=. examples/poisson_manufactured.jl
$ julia --project=. examples/scalar_advection_2d.jl
```

## Building the docs

```bash
$ julia --project=docs -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
$ julia --project=docs docs/make.jl
```

The generated site lives in `docs/build/`.

## Optional: benchmarks

The benchmark harness lives in `benchmarks/`. See the
[Benchmarks](benchmarks.md) page for how to run it.
