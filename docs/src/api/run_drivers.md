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
| `advection` | `examples/scalar_advection_2d/scalar_advection_2d.toml` | `examples/scalar_advection_2d.jl` |
| `cbl` | `examples/cbl_tke/cbl_tke.toml`, `examples/cbl_inflow_statistics/cbl_inflow_statistics.toml` | `examples/sgs/cbl_tke.jl`, `examples/nesting/cbl_inflow_statistics.jl`, `test/tke_cbl_driver.jl` |
| `inflow_fetch` | `examples/inflow_fetch/inflow_fetch.toml` | `examples/nesting/inflow_fetch.jl`, `test/inflow_fetch_harness.jl` |
| `shear_layer` | `examples/kelvin_helmholtz/kelvin_helmholtz.toml` | `examples/transition/kelvin_helmholtz.jl` |
| `flat_plate` | `examples/flat_plate_t3a/flat_plate_t3a.toml` | `examples/transition/flat_plate_t3a.jl` |
| `sgs_decay` | `examples/rotating_stratified_decay/rotating_stratified_decay.toml` | `examples/sgs/rotating_stratified_decay.jl`, `test/fixtures/sgs_validation.jl` |
| `obstacle_channel` | `examples/surface_mounted_cube/surface_mounted_cube.toml`, `surface_mounted_cube_closures.toml` (the cube in a static driver) | `examples/sgs/surface_mounted_cube.jl` |

The bare `simulation` driver also gained the couplings of the kernel and
coupled examples (`[forcing]`, `[surface]`, `[microphysics]`, `[tracer]`,
`[particles]`, `[obstacles]`; [`Aeolus.SimulationDriver`](@ref)).

```@autodocs
Modules = [Aeolus.RunDrivers, Aeolus.RunDrivers.AdvectionDriver, Aeolus.RunDrivers.CBLDriver,
           Aeolus.RunDrivers.InflowFetchDriver, Aeolus.RunDrivers.ObstacleChannelDriver,
           Aeolus.RunDrivers.ShearLayerDriver, Aeolus.RunDrivers.FlatPlateDriver,
           Aeolus.RunDrivers.SGSDecayDriver]
```
