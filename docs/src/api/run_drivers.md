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
| `cbl` | `examples/sgs/configs/cbl_tke.toml`, `examples/nesting/configs/cbl_inflow_statistics.toml` | `examples/sgs/cbl_tke.jl`, `examples/nesting/cbl_inflow_statistics.jl`, `test/tke_cbl_driver.jl` |
| `inflow_fetch` | `examples/nesting/configs/inflow_fetch.toml` | `examples/nesting/inflow_fetch.jl`, `test/inflow_fetch_harness.jl` |

The bare `simulation` driver also gained the couplings of the kernel and
coupled examples (`[forcing]`, `[surface]`, `[microphysics]`, `[tracer]`,
`[particles]`, `[obstacles]`; [`Aeolus.SimulationDriver`](@ref)).

```@autodocs
Modules = [Aeolus.RunDrivers, Aeolus.RunDrivers.AdvectionDriver, Aeolus.RunDrivers.CBLDriver,
           Aeolus.RunDrivers.InflowFetchDriver]
```
