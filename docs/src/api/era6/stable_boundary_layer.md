# The `sbl` run type — StableBoundaryLayer

`Aeolus.StableBoundaryLayer` is the `sbl` run type (`[run] driver =
"sbl"`): the stable-boundary-layer cases of Era VI, configured from TOML
only. It was the example driver `examples/sbl/` (CFG-d, #359) and moved
into `src/` with EX-b (#439): examples are configuration plus analysis of
run output. One run type covers four kinds, `[case] kind`:

| kind | builds and runs | shipped configs (`examples/sbl/configs/`) |
| :-- | :-- | :-- |
| `homogeneous` | [`sbl_case`](@ref Aeolus.StableBoundaryLayer.sbl_case), [`sbl_run!`](@ref Aeolus.StableBoundaryLayer.sbl_run!): the periodic SBL LES | `gabls1`, `gabls2`, `cases99`, `gabls3`, `gabls4_exp3`; sweeps `gabls1_closures`, `cases99_closures`, `gabls3_closures`, `gabls4_closures` |
| `stratified_channel` | a neutral spin-up, then `stratified_channel_run` | `stratified_channel_re180`; sweep `stratified_channel_sweep` |
| `relaminarisation_channel` | `relam_spinup`, then `relam_run` | `relaminarisation`; sweep `relaminarisation_sweep` |
| `collapse_channel` | the neutral half channel, then `collapse_run` | `collapse_channel`; sweep `collapse_channel_sweep` |

```bash
julia --project -m Aeolus run examples/sbl/configs/gabls1_closures.toml     # one run folder per closure
julia --project examples/sbl/gabls1.jl runs/gabls1_*                        # reads the run folders: table, figure
```

A run writes into its run folder `output/statistics.toml` (the averaged
profiles and scalars, or the channel result), `output/series.csv` (every
`[averaging] series_interval`) and, with `[towers] below > 0`,
`output/towers.csv` (the CASES-99 hockey-stick bins). The scripts of
`examples/sbl/` only read those files. The physics, the configuration and
the validation are on [Stable boundary layer](../../theory/stable_boundary_layer.md);
the options on [Options: driver `sbl`](../../reference/options/sbl.md).

The module is not re-exported by `Aeolus`: `using Aeolus.StableBoundaryLayer`
for the builders (the tests use them directly).

```@autodocs
Modules = [Aeolus.StableBoundaryLayer]
```
