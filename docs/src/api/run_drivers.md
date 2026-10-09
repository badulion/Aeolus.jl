# Example run types

`Aeolus.RunDrivers` (EX-d, #441) holds the run types of the example cases
that the bare `simulation` driver ([`Aeolus.SimulationDriver`](@ref)) does
not cover. An example is a TOML file in an `examples/**/configs/`
directory (plus its static or dynamic input driver), run with

```bash
julia --project -m Aeolus run examples/<dir>/configs/<case>.toml
```

Each run type writes its numbers to `output/results.toml` (and, where a
figure needs profiles or series, CSV files) in the run folder; the `.jl`
files beside the configs read those files and plot them, and run no
simulation code. `aeolus options <driver>` prints a driver's sections; the
generated option reference is under *Reference*.

| driver | example configs | formerly |
| :-- | :-- | :-- |
| `advection` | `examples/configs/scalar_advection_2d.toml` | `examples/scalar_advection_2d.jl` |

```@autodocs
Modules = [Aeolus.RunDrivers, Aeolus.RunDrivers.AdvectionDriver]
```
