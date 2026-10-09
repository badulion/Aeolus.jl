# Validation run types

`Aeolus.ValidationRuns` holds the run types of the validation cases
(EX-c, #440). A validation case is a TOML file in
`examples/validation/configs/`, run with

```
julia --project -m Aeolus run examples/validation/configs/<case>.toml
```

Its numbers are diagnostics of the run, in the run folder's
`output/results.toml` (the `dns` run type: `output/dns_apriori.toml`).
The scripts in `examples/validation/` only read those files and print the
tables or draw the figures; they take the root of the runs as their
argument (default `runs/`) and use the newest run of each case
(`examples/validation/read_runs.jl`). No script builds or steps a model.

| `[run] driver` | config(s) | script reading its output |
| :-- | :-- | :-- |
| `channel` | `neutral_abl.toml`, `log_layer_mismatch.toml` | `neutral_abl.jl`, `log_layer_mismatch.jl` |
| `column` | `dycoms_rf01.toml`, `parisfog_2011.toml`, `spectral_broadening.toml` (and `examples/kernels/configs/cloud_spinup.toml`) | `dycoms_rf01_radiation.jl`, `parisfog_2011_radiation_fog.jl`, `spectral_broadening.jl` |
| `cbc_decay` | `cbc_decay.toml` | `cbc_decay.jl` |
| `patchy_surface` | `patchy_surface.toml` | `patchy_surface_les.jl` |
| `dns` | `apriori_sgs.toml`, `dns256.toml` | `apriori_sgs.jl` |
| `plant_canopy` | `canopy_profiles.toml` | `canopy_profiles.jl` |
| `dynamic_roughness` | `dynamic_roughness_cubes.toml` | `dynamic_roughness_cubes.jl` |

What moved where, and what the runs now record:

  * The harnesses the run types are built on are the ones the module tests
    use: the Comte-Bellot & Corrsin benchmark ([`cbc_benchmark`](@ref
    Aeolus.ValidationRuns.cbc_benchmark), formerly
    `test/cbc_benchmark_driver.jl`; its table is `src/data/cbc1971_table3.txt`),
    the filtered-DNS fixture and its a-priori record ([`dns_decay`](@ref
    Aeolus.ValidationRuns.dns_decay), [`dns_apriori_finish`](@ref
    Aeolus.ValidationRuns.dns_apriori_finish), formerly `test/fixtures/`),
    the patchy-surface LES ([`patchy_les`](@ref Aeolus.ValidationRuns.patchy_les))
    and the RF01 column ([`rf01_radiation`](@ref Aeolus.ValidationRuns.rf01_radiation)).
  * `channel` averages the streamwise spectra E11(k1) over the statistics
    window at the heights `[spectra] levels` (what `neutral_abl.jl` sampled
    with a callback).
  * `cbc_decay` records CBC's table 3 and its interpolant
    ([`cbc_reference`](@ref Aeolus.ValidationRuns.cbc_reference)).
  * `column` records the column density and layer depths of RF01, the
    fog-top and fog-layer longwave terms of ParisFog, and the parcel's
    temperature and Ornstein–Uhlenbeck width
    ([`parcel_ou_sigma_r2`](@ref Aeolus.ValidationRuns.parcel_ou_sigma_r2)).
  * `apriori_sgs.toml` runs on `dns` (the `dns_decay` driver ran the same
    DNS and is gone); the 256³ record is `aeolus run dns256.toml`, with
    `--set 'apriori.out="test/fixtures/dns256_apriori.toml"'` to refresh the
    fixture (`dns256_generate.jl` is gone).
  * `plant_canopy` and `dynamic_roughness` are new: the canopy column and
    the cube-array roughness ran inside their scripts before.

The option reference of each run type is generated from its schema
(`aeolus options <driver>`).

```@autodocs
Modules = [Aeolus.ValidationRuns]
```
